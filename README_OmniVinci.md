# OmniVinci MLX Implementation

A complete MLX implementation of NVIDIA's OmniVinci model for omni-modal understanding of vision, audio, and text.

## Overview

OmniVinci is a state-of-the-art multimodal LLM that achieves joint understanding across:
- **Vision**: Images and video frames
- **Audio**: Natural sounds and speech
- **Text**: Natural language

This implementation follows the Apple MLX framework conventions and is based on the architecture described in the [OmniVinci paper](https://arxiv.org/abs/2510.15870).

## Key Features

### 1. **OmniAlignNet**
Cross-modal alignment mechanism that strengthens the relationship between vision and audio embeddings in a shared omni-modal latent space using bidirectional cross-attention.

### 2. **Temporal Embedding Grouping (TEG)**
Groups vision and audio embeddings by timestamps to capture relative temporal alignment between modalities, ensuring synchronized understanding of temporal relationships.

### 3. **Constrained Rotary Time Embedding (CRTE)**
Encodes absolute temporal information using constrained rotary positional encoding, allowing the model to understand when events occur in absolute time.

### 4. **Multi-Modal Architecture**
- Vision Encoder: ViT-style architecture for processing images
- Audio Encoder: Transformer-based encoder for audio waveforms
- LLM Backbone: Transformer architecture for unified understanding

## Architecture Components

```
Input (Vision/Audio/Text)
    ↓
Vision Encoder → Vision Features
Audio Encoder  → Audio Features
    ↓
OmniAlignNet (Cross-Modal Alignment)
    ↓
Temporal Embedding Grouping
    ↓
Constrained Rotary Time Embedding
    ↓
Projection → LLM Dimension
    ↓
Text Embeddings (if text input)
    ↓
Transformer Layers (LLM Backbone)
    ↓
Output Logits
```

## Installation

```bash
# Install MLX
pip install mlx

# Install other dependencies
pip install numpy pillow librosa
```

## Usage

### Basic Text-Only Inference

```python
import mlx.core as mx
from omnivinci_mlx import Model, OmniVinciModelArgs

# Initialize model
args = OmniVinciModelArgs()
model = Model(args)

# Text input
input_ids = mx.array([[1, 2, 3, 4, 5]])  # Token IDs
cache = model.make_cache()

# Forward pass
logits = model(input_ids=input_ids, cache=cache)
```

### Vision + Text Inference

```python
import mlx.core as mx
from omnivinci_mlx import Model, OmniVinciModelArgs
from PIL import Image
import numpy as np

# Load and preprocess image
image = Image.open("example.jpg").resize((336, 336))
image_array = np.array(image).transpose(2, 0, 1)  # (C, H, W)
vision_inputs = mx.array(image_array).reshape(1, 3, 336, 336)

# Text prompt
input_ids = mx.array([[1, 2, 3]])  # "Describe this image"

# Forward pass with vision + text
model = Model(OmniVinciModelArgs())
cache = model.make_cache()

logits = model(
    input_ids=input_ids,
    vision_inputs=vision_inputs,
    cache=cache
)
```

### Audio + Text Inference

```python
import mlx.core as mx
import librosa

# Load audio
audio_waveform, sr = librosa.load("example.wav", sr=16000)
audio_inputs = mx.array(audio_waveform).reshape(1, -1)

# Text prompt
input_ids = mx.array([[1, 2, 3]])  # "What sounds are in this audio?"

# Forward pass
logits = model(
    input_ids=input_ids,
    audio_inputs=audio_inputs,
    cache=cache
)
```

### Vision + Audio + Text (Full Omni-Modal)

```python
import mlx.core as mx

# Vision input (video frames)
vision_inputs = mx.random.normal((1, 3, 336, 336))  # Single frame
vision_timestamps = mx.array([[0.0]])  # Time 0 seconds

# Audio input
audio_inputs = mx.random.normal((1, 16000 * 5))  # 5 seconds of audio
audio_timestamps = mx.linspace(0, 5, 1000).reshape(1, -1)  # Audio timestamps

# Text prompt
input_ids = mx.array([[1, 2, 3]])

# Forward pass with all modalities
logits = model(
    input_ids=input_ids,
    vision_inputs=vision_inputs,
    audio_inputs=audio_inputs,
    vision_timestamps=vision_timestamps,
    audio_timestamps=audio_timestamps,
    cache=cache
)
```

## Model Configuration

### OmniVinciModelArgs Parameters

```python
@dataclass
class OmniVinciModelArgs(BaseModelArgs):
    # Core LLM parameters
    hidden_size: int = 4096              # LLM hidden dimension
    num_hidden_layers: int = 32          # Number of transformer layers
    num_attention_heads: int = 32        # Number of attention heads
    intermediate_size: int = 14336       # FFN intermediate size
    vocab_size: int = 256000             # Vocabulary size
    
    # Vision encoder
    vision_hidden_size: int = 1024       # Vision encoder hidden size
    vision_num_layers: int = 24          # Vision transformer layers
    vision_patch_size: int = 14          # Patch size for ViT
    vision_image_size: int = 336         # Input image size
    
    # Audio encoder
    audio_hidden_size: int = 1024        # Audio encoder hidden size
    audio_num_layers: int = 12           # Audio transformer layers
    audio_sampling_rate: int = 16000     # Audio sampling rate
    
    # OmniAlignNet
    omni_align_hidden_dim: int = 2048    # Alignment space dimension
    omni_align_num_layers: int = 2       # Number of alignment layers
    
    # Temporal grouping
    temporal_group_size: int = 8         # Frames per temporal group
    max_temporal_groups: int = 64        # Maximum temporal groups
    
    # Time embedding
    time_embedding_dim: int = 128        # Time embedding dimension
    max_time_position: int = 3600        # Max time in seconds (1 hour)
```

## Advanced Features

### Custom Temporal Grouping

```python
# Control temporal grouping granularity
args = OmniVinciModelArgs(
    temporal_group_size=4,  # Smaller groups = finer temporal resolution
    max_temporal_groups=128  # More groups = longer videos
)
```

### Multi-Frame Video Processing

```python
# Process multiple video frames with timestamps
num_frames = 32
vision_inputs = mx.random.normal((1, 3, 336, 336 * num_frames))
vision_timestamps = mx.linspace(0, 10, num_frames).reshape(1, -1)  # 10 seconds

logits = model(
    input_ids=input_ids,
    vision_inputs=vision_inputs,
    vision_timestamps=vision_timestamps,
    cache=cache
)
```

### Encoding Only (Feature Extraction)

```python
# Extract vision features only
vision_features = model.encode_vision(vision_inputs)  # (B, num_patches, vision_hidden_size)

# Extract audio features only
audio_features = model.encode_audio(audio_inputs)  # (B, num_tokens, audio_hidden_size)
```

## Performance Considerations

### Memory Optimization

1. **Gradient Checkpointing**: Use gradient checkpointing for training large models
2. **KV Cache**: Use the provided KV cache for efficient generation
3. **Batch Processing**: Process multiple samples in batches when possible

### Efficiency Tips

```python
# Use smaller models for faster inference
args = OmniVinciModelArgs(
    hidden_size=2048,           # Reduce from 4096
    num_hidden_layers=16,       # Reduce from 32
    vision_num_layers=12,       # Reduce from 24
    audio_num_layers=6          # Reduce from 12
)
```

## Training

To train the model:

```python
import mlx.optimizers as optim

# Initialize model and optimizer
model = Model(OmniVinciModelArgs())
optimizer = optim.AdamW(learning_rate=1e-4)

# Training loop
def loss_fn(model, batch):
    logits = model(
        input_ids=batch['input_ids'],
        vision_inputs=batch.get('vision_inputs'),
        audio_inputs=batch.get('audio_inputs'),
        vision_timestamps=batch.get('vision_timestamps'),
        audio_timestamps=batch.get('audio_timestamps')
    )
    # Compute cross-entropy loss
    return mx.mean(nn.losses.cross_entropy(logits, batch['labels']))

# Training step
loss, grads = mx.value_and_grad(loss_fn)(model, batch)
optimizer.update(model, grads)
```

## Citation

If you use this implementation, please cite the original OmniVinci paper:

```bibtex
@article{ye2025omnivinci,
  title={OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM},
  author={Ye, Hanrong and Yang, Chao-Han Huck and Goel, Arushi and others},
  journal={arXiv preprint arXiv:2510.15870},
  year={2025}
}
```

## License

This implementation follows the Apache 2.0 license of the original OmniVinci project.

## Acknowledgments

- Original OmniVinci by NVIDIA Research
- Apple MLX framework
- Based on the architecture described in [arXiv:2510.15870](https://arxiv.org/abs/2510.15870)

## Implementation Notes

### Differences from Original

This MLX implementation makes some adaptations:
1. Simplified audio feature extraction (original uses Whisper-style encoder)
2. Adapted to MLX's computational model
3. Optimized for Apple Silicon hardware

### Future Enhancements

- [ ] Add pre-trained weight loading
- [ ] Implement quantization support
- [ ] Add streaming inference
- [ ] Support for longer videos
- [ ] Multi-GPU training support

## Support

For issues or questions:
1. Check the [OmniVinci GitHub](https://github.com/NVlabs/OmniVinci)
2. Review MLX documentation
3. Open an issue with details about your use case
