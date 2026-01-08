# Copyright © 2025 Implementation based on NVIDIA OmniVinci
# MLX implementation following Apple MLX conventions

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache
import librosa
import soundfile as sf


@dataclass
class OmniVinciModelArgs(BaseModelArgs):
    """Configuration for OmniVinci model."""
    
    model_type: str = "omnivinci"
    
    # LLM backbone parameters
    hidden_size: int = 4096
    head_dim: int = 128
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rope_theta: float = 50000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    attention_bias: bool = False
    layer_norm_bias: bool = False
    
    # Vision encoder parameters
    vision_hidden_size: int = 1024
    vision_num_layers: int = 24
    vision_num_heads: int = 16
    vision_patch_size: int = 14
    vision_image_size: int = 336
    num_vision_tokens: int = 576  # (336/14)^2
    
    # Audio encoder parameters
    audio_hidden_size: int = 1024
    audio_num_layers: int = 12
    audio_num_heads: int = 16
    audio_encoder_dim: int = 768
    audio_sampling_rate: int = 16000
    num_audio_tokens: int = 1000  # Max audio tokens per chunk
    
    # OmniAlignNet parameters
    omni_align_hidden_dim: int = 2048
    omni_align_num_layers: int = 2
    
    # Temporal Embedding Grouping parameters
    temporal_group_size: int = 8  # Number of frames per temporal group
    max_temporal_groups: int = 64  # Maximum temporal groups
    
    # Constrained Rotary Time Embedding parameters
    time_embedding_dim: int = 128
    max_time_position: int = 3600  # Maximum time in seconds
    
    # Projection dimensions
    vision_projection_dim: int = 4096
    audio_projection_dim: int = 4096


class VisionEncoder(nn.Module):
    """Vision encoder using ViT-style architecture."""
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=args.vision_hidden_size,
            kernel_size=args.vision_patch_size,
            stride=args.vision_patch_size,
            bias=True
        )
        
        # Position embedding
        num_patches = (args.vision_image_size // args.vision_patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, args.vision_hidden_size)    
        
        # Transformer blocks
        self.layers = [
            VisionTransformerBlock(args)
            for _ in range(args.vision_num_layers)
        ]
        
        self.norm = nn.LayerNorm(args.vision_hidden_size, eps=args.layer_norm_eps)
        
    def __call__(self, images: mx.array) -> mx.array:
        """
        Args:
            images: (B, C, H, W) tensor of images
        Returns:
            (B, num_patches, hidden_size) tensor of vision features
        """
        B = images.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(images)  # (B, hidden_size, H/P, W/P)
        x = x.reshape(B, self.args.vision_hidden_size, -1)  # (B, hidden_size, num_patches)
        x = x.transpose(0, 2, 1)  # (B, num_patches, hidden_size)
        
        # Add position embedding
        positions = mx.arange(x.shape[1])
        x = x + self.position_embedding(positions)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)


class VisionTransformerBlock(nn.Module):
    """Transformer block for vision encoder."""
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        
        self.attention = nn.MultiHeadAttention(
            args.vision_hidden_size,
            args.vision_num_heads,
            bias=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(args.vision_hidden_size, args.vision_hidden_size * 4),
            nn.GELU(),
            nn.Linear(args.vision_hidden_size * 4, args.vision_hidden_size)
        )
        self.norm1 = nn.LayerNorm(args.vision_hidden_size, eps=args.layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.vision_hidden_size, eps=args.layer_norm_eps)
        
    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """Audio encoder using transformer architecture."""
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        
        # Audio feature extraction (simplified - in practice would use Whisper or similar)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, args.audio_encoder_dim // 4, kernel_size=10, stride=5),
            nn.GELU(),
            nn.Conv1d(args.audio_encoder_dim // 4, args.audio_encoder_dim // 2, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv1d(args.audio_encoder_dim // 2, args.audio_encoder_dim, kernel_size=4, stride=2),
            nn.GELU()
        )
        
        # Projection to hidden size
        self.projection = nn.Linear(args.audio_encoder_dim, args.audio_hidden_size)
        
        # Position embedding
        self.position_embedding = nn.Embedding(args.num_audio_tokens, args.audio_hidden_size)
        
        # Transformer blocks
        self.layers = [
            AudioTransformerBlock(args)
            for _ in range(args.audio_num_layers)
        ]
        
        self.norm = nn.LayerNorm(args.audio_hidden_size, eps=args.layer_norm_eps)
        
    def __call__(self, audio: mx.array) -> mx.array:
        """
        Args:
            audio: (B, T) tensor of audio waveform
        Returns:
            (B, num_tokens, hidden_size) tensor of audio features
        """
        B = audio.shape[0]
        
        # Extract features
        audio = audio.reshape(B, 1, -1)  # (B, 1, T)
        x = self.feature_extractor(audio)  # (B, encoder_dim, T')
        x = x.transpose(0, 2, 1)  # (B, T', encoder_dim)
        
        # Project to hidden size
        x = self.projection(x)  # (B, T', hidden_size)
        
        # Add position embedding
        seq_len = min(x.shape[1], self.args.num_audio_tokens)
        x = x[:, :seq_len, :]
        positions = mx.arange(seq_len)
        x = x + self.position_embedding(positions)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)


class AudioTransformerBlock(nn.Module):
    """Transformer block for audio encoder."""
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        
        self.attention = nn.MultiHeadAttention(
            args.audio_hidden_size,
            args.audio_num_heads,
            bias=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(args.audio_hidden_size, args.audio_hidden_size * 4),
            nn.GELU(),
            nn.Linear(args.audio_hidden_size * 4, args.audio_hidden_size)
        )
        self.norm1 = nn.LayerNorm(args.audio_hidden_size, eps=args.layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.audio_hidden_size, eps=args.layer_norm_eps)
        
    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class OmniAlignNet(nn.Module):
    """
    OmniAlignNet: Aligns vision and audio embeddings in a shared omni-modal space.
    Uses cross-modal attention to strengthen alignment between modalities.
    """
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(args.vision_hidden_size, args.omni_align_hidden_dim)
        self.audio_proj = nn.Linear(args.audio_hidden_size, args.omni_align_hidden_dim)
        
        # Cross-modal alignment layers
        self.alignment_layers = [
            CrossModalAlignmentLayer(args)
            for _ in range(args.omni_align_num_layers)
        ]
        
        self.norm = nn.LayerNorm(args.omni_align_hidden_dim, eps=args.layer_norm_eps)
        
    def __call__(
        self,
        vision_features: mx.array,
        audio_features: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Args:
            vision_features: (B, V, vision_hidden_size)
            audio_features: (B, A, audio_hidden_size)
        Returns:
            Aligned vision and audio features in shared space
        """
        # Project to common space
        vision_emb = self.vision_proj(vision_features)
        audio_emb = self.audio_proj(audio_features)
        
        # Apply cross-modal alignment
        for layer in self.alignment_layers:
            vision_emb, audio_emb = layer(vision_emb, audio_emb)
        
        vision_emb = self.norm(vision_emb)
        audio_emb = self.norm(audio_emb)
        
        return vision_emb, audio_emb


class CrossModalAlignmentLayer(nn.Module):
    """Cross-modal attention layer for aligning vision and audio."""
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        dim = args.omni_align_hidden_dim
        
        # Vision attending to audio
        self.vision_cross_attn = nn.MultiHeadAttention(dim, 8, bias=True)
        self.vision_norm = nn.LayerNorm(dim, eps=args.layer_norm_eps)
        
        # Audio attending to vision
        self.audio_cross_attn = nn.MultiHeadAttention(dim, 8, bias=True)
        self.audio_norm = nn.LayerNorm(dim, eps=args.layer_norm_eps)
        
        # Feed-forward networks
        self.vision_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.audio_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.vision_mlp_norm = nn.LayerNorm(dim, eps=args.layer_norm_eps)
        self.audio_mlp_norm = nn.LayerNorm(dim, eps=args.layer_norm_eps)
        
    def __call__(
        self,
        vision_emb: mx.array,
        audio_emb: mx.array
    ) -> Tuple[mx.array, mx.array]:
        # Vision attends to audio
        vision_out = self.vision_cross_attn(
            self.vision_norm(vision_emb),
            self.vision_norm(audio_emb),
            self.vision_norm(audio_emb)
        )
        vision_emb = vision_emb + vision_out
        vision_emb = vision_emb + self.vision_mlp(self.vision_mlp_norm(vision_emb))
        
        # Audio attends to vision
        audio_out = self.audio_cross_attn(
            self.audio_norm(audio_emb),
            self.audio_norm(vision_emb),
            self.audio_norm(vision_emb)
        )
        audio_emb = audio_emb + audio_out
        audio_emb = audio_emb + self.audio_mlp(self.audio_mlp_norm(audio_emb))
        
        return vision_emb, audio_emb


class TemporalEmbeddingGrouping(nn.Module):
    """
    Temporal Embedding Grouping: Groups vision and audio embeddings by timestamps
    to capture relative temporal alignment.
    """
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        self.group_size = args.temporal_group_size
        
    def __call__(
        self,
        vision_emb: mx.array,
        audio_emb: mx.array,
        vision_timestamps: Optional[mx.array] = None,
        audio_timestamps: Optional[mx.array] = None
    ) -> mx.array:
        """
        Groups vision and audio embeddings based on timestamps.
        
        Args:
            vision_emb: (B, V, D) vision embeddings
            audio_emb: (B, A, D) audio embeddings
            vision_timestamps: (B, V) timestamps for vision frames
            audio_timestamps: (B, A) timestamps for audio chunks
            
        Returns:
            Grouped embeddings: (B, G, D) where G = num_groups * (vision + audio per group)
        """
        B = vision_emb.shape[0]
        
        if vision_timestamps is None:
            # Assume uniform temporal spacing
            num_vision = vision_emb.shape[1]
            vision_timestamps = mx.arange(num_vision).reshape(1, -1).repeat(B, axis=0)
        
        if audio_timestamps is None:
            num_audio = audio_emb.shape[1]
            audio_timestamps = mx.arange(num_audio).reshape(1, -1).repeat(B, axis=0)
        
        # Determine temporal groups
        max_time = max(vision_timestamps.max(), audio_timestamps.max())
        num_groups = int((max_time / self.group_size).item()) + 1
        num_groups = min(num_groups, self.args.max_temporal_groups)
        
        # Group embeddings
        grouped_embeddings = []
        
        for g in range(num_groups):
            group_start = g * self.group_size
            group_end = (g + 1) * self.group_size
            
            # Find vision embeddings in this group
            vision_mask = (vision_timestamps >= group_start) & (vision_timestamps < group_end)
            # Find audio embeddings in this group
            audio_mask = (audio_timestamps >= group_start) & (audio_timestamps < group_end)
            
            # Concatenate vision and audio for this group
            for b in range(B):
                v_indices = mx.where(vision_mask[b])[0]
                a_indices = mx.where(audio_mask[b])[0]
                
                if len(v_indices) > 0:
                    grouped_embeddings.append(vision_emb[b, v_indices])
                if len(a_indices) > 0:
                    grouped_embeddings.append(audio_emb[b, a_indices])
        
        # Concatenate all grouped embeddings
        if grouped_embeddings:
            result = mx.concatenate(grouped_embeddings, axis=0)
            # Reshape to (B, total_tokens, D)
            result = result.reshape(B, -1, result.shape[-1])
        else:
            # Fallback: simple concatenation
            result = mx.concatenate([vision_emb, audio_emb], axis=1)
        
        return result


class ConstrainedRotaryTimeEmbedding(nn.Module):
    """
    Constrained Rotary Time Embedding: Encodes absolute temporal information
    using constrained rotary positional encoding.
    """
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.time_embedding_dim
        
        # Constrained frequency bands for time
        inv_freq = 1.0 / (10000 ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim))
        self.inv_freq = inv_freq
        
    def __call__(self, embeddings: mx.array, timestamps: mx.array) -> mx.array:
        """
        Apply rotary time embedding to the embeddings.
        
        Args:
            embeddings: (B, L, D) embeddings
            timestamps: (B, L) timestamps in seconds
            
        Returns:
            Time-encoded embeddings: (B, L, D)
        """
        B, L, D = embeddings.shape
        
        # Ensure timestamps are in valid range
        timestamps = mx.clip(timestamps, 0, self.args.max_time_position)
        
        # Compute time-based frequencies
        freqs = mx.outer(timestamps.reshape(-1), self.inv_freq)  # (B*L, dim/2)
        emb = mx.concatenate([mx.sin(freqs), mx.cos(freqs)], axis=-1)  # (B*L, dim)
        emb = emb.reshape(B, L, self.dim)
        
        # Apply to embeddings (assuming first 'dim' dimensions)
        if D >= self.dim:
            # Rotate the first time_embedding_dim dimensions
            x1 = embeddings[:, :, :self.dim // 2]
            x2 = embeddings[:, :, self.dim // 2:self.dim]
            cos = emb[:, :, self.dim // 2:]
            sin = emb[:, :, :self.dim // 2]
            
            rotated = mx.concatenate([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ], axis=-1)
            
            result = mx.concatenate([rotated, embeddings[:, :, self.dim:]], axis=-1)
        else:
            # If embedding dim is smaller, just add time embedding
            result = embeddings + emb[:, :, :D]
        
        return result


class Attention(nn.Module):
    """Standard multi-head attention for LLM backbone."""
    
    def __init__(self, args: OmniVinciModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        
        if (head_dim * n_heads) != dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {dim}"
                f" and `num_heads`: {n_heads})."
            )
        self.scale = head_dim**-0.5

        attention_bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is None:
            queries = self.rope(queries)
            keys = self.rope(keys)
        else:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """Feed-forward network for transformer blocks."""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block for LLM backbone."""
    
    def __init__(self, args: OmniVinciModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads

        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        h = x + attn_h
        
        mlp_h = self.mlp(self.post_attention_layernorm(h))
        return h + mlp_h


class OmniVinciModel(nn.Module):
    """Main OmniVinci model combining all components."""
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        
        # Modality encoders
        self.vision_encoder = VisionEncoder(args)
        self.audio_encoder = AudioEncoder(args)
        
        # OmniAlignNet for cross-modal alignment
        self.omni_align_net = OmniAlignNet(args)
        
        # Temporal grouping
        self.temporal_grouping = TemporalEmbeddingGrouping(args)
        
        # Constrained rotary time embedding
        self.time_embedding = ConstrainedRotaryTimeEmbedding(args)
        
        # Projection to LLM dimension
        self.vision_projection = nn.Linear(
            args.omni_align_hidden_dim,
            args.vision_projection_dim
        )
        self.audio_projection = nn.Linear(
            args.omni_align_hidden_dim,
            args.audio_projection_dim
        )
        
        # Text embeddings
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        
        # LLM backbone layers
        self.layers = [
            TransformerBlock(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        
        self.norm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

    def encode_vision(self, images: mx.array) -> mx.array:
        """Encode vision inputs."""
        return self.vision_encoder(images)
    
    def encode_audio(self, audio: mx.array) -> mx.array:
        """Encode audio inputs."""
        return self.audio_encoder(audio)
    
    def align_modalities(
        self,
        vision_features: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        vision_timestamps: Optional[mx.array] = None,
        audio_timestamps: Optional[mx.array] = None
    ) -> mx.array:
        """
        Align vision and audio features using OmniAlignNet and temporal grouping.
        
        Returns:
            Aligned and temporally grouped embeddings ready for LLM
        """
        if vision_features is None and audio_features is None:
            raise ValueError("At least one of vision or audio features must be provided")
        
        # Create dummy features if one modality is missing
        if vision_features is None:
            B, A, _ = audio_features.shape
            vision_features = mx.zeros((B, 1, self.args.vision_hidden_size))
        if audio_features is None:
            B, V, _ = vision_features.shape
            audio_features = mx.zeros((B, 1, self.args.audio_hidden_size))
        
        # Apply OmniAlignNet
        vision_aligned, audio_aligned = self.omni_align_net(
            vision_features, audio_features
        )
        
        # Apply temporal grouping
        grouped_emb = self.temporal_grouping(
            vision_aligned,
            audio_aligned,
            vision_timestamps,
            audio_timestamps
        )
        
        # Apply constrained rotary time embedding
        if vision_timestamps is not None or audio_timestamps is not None:
            # Create combined timestamps for grouped embeddings
            # Simplified: assume uniform distribution
            B, L, _ = grouped_emb.shape
            timestamps = mx.linspace(0, self.args.max_time_position, L)
            timestamps = timestamps.reshape(1, -1).repeat(B, axis=0)
            grouped_emb = self.time_embedding(grouped_emb, timestamps)
        
        return grouped_emb

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        vision_inputs: Optional[mx.array] = None,
        audio_inputs: Optional[mx.array] = None,
        vision_timestamps: Optional[mx.array] = None,
        audio_timestamps: Optional[mx.array] = None,
        cache=None,
    ):
        """
        Forward pass through OmniVinci model.
        
        Args:
            input_ids: Text token IDs
            vision_inputs: Vision features (images)
            audio_inputs: Audio waveforms
            vision_timestamps: Timestamps for vision frames
            audio_timestamps: Timestamps for audio chunks
            cache: KV cache for generation
        """
        embeddings_list = []
        
        # Process multimodal inputs
        if vision_inputs is not None or audio_inputs is not None:
            # Encode modalities
            vision_features = None
            audio_features = None
            
            if vision_inputs is not None:
                vision_features = self.encode_vision(vision_inputs)
            if audio_inputs is not None:
                audio_features = self.encode_audio(audio_inputs)
            
            # Align and group
            multimodal_emb = self.align_modalities(
                vision_features,
                audio_features,
                vision_timestamps,
                audio_timestamps
            )
            
            # Project to LLM dimension (use vision projection as default)
            multimodal_emb = self.vision_projection(multimodal_emb)
            embeddings_list.append(multimodal_emb)
        
        # Process text inputs
        if input_ids is not None:
            text_emb = self.embed_tokens(input_ids)
            embeddings_list.append(text_emb)
        
        # Concatenate all embeddings
        if not embeddings_list:
            raise ValueError("No inputs provided")
        
        h = mx.concatenate(embeddings_list, axis=1)

        # Create cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create attention mask
        mask = create_attention_mask(h, cache[0])

        # Apply transformer layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    """
    Complete OmniVinci model with language modeling head.
    """
    
    def __init__(self, args: OmniVinciModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = OmniVinciModel(args)
        self.args = args

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        vision_inputs: Optional[mx.array] = None,
        audio_inputs: Optional[mx.array] = None,
        vision_timestamps: Optional[mx.array] = None,
        audio_timestamps: Optional[mx.array] = None,
        cache=None,
    ):
        """Forward pass with language modeling head."""
        out = self.model(
            input_ids=input_ids,
            vision_inputs=vision_inputs,
            audio_inputs=audio_inputs,
            vision_timestamps=vision_timestamps,
            audio_timestamps=audio_timestamps,
            cache=cache
        )
        
        # Language modeling head (weight tying with embeddings)
        out = self.model.embed_tokens.as_linear(out)
        return out

    def make_cache(self):
        """Create KV cache for generation."""
        return [KVCache() for _ in range(self.args.num_hidden_layers)]

    @property
    def layers(self):
        return self.model.layers
    
    def encode_vision(self, images: mx.array) -> mx.array:
        """Convenience method to encode vision."""
        return self.model.encode_vision(images)
    
    def encode_audio(self, audio: mx.array) -> mx.array:
        """Convenience method to encode audio."""
        return self.model.encode_audio(audio)