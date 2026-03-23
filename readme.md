# EfficientVSR — Joint Video Spatiotemporal Enhancement

A lightweight hybrid CNN-Transformer network that performs 
**2x Video Frame Interpolation (VFI)** and **2x Super Resolution (SR)** 
simultaneously in a single forward pass.

## Results

| Method | PSNR | SSIM |
|--------|------|------|
| Bicubic Baseline | 25.13 dB | 0.7509 |
| **Our Model** | **31.25 dB** | **0.9088** |
| **Improvement** | **+6.13 dB** | **+0.1579** |

## Architecture

- Shared DenseBlock Encoder
- Temporal Branch with Deformable Convolution
- Spatial Branch with SE Channel Attention
- Cross-Frame Attention with RoPE (borrowed from LLaMA)
- Gated FFN (borrowed from LLaMA)
- CBAM Fusion + PixelShuffle 2x Upsampling

## Dataset

Vimeo-90K Triplet Dataset  
- 20,000 training triplets  
- 2,000 test triplets  

## Training

Staged training — 3 stages over 60 epochs:
- Stage 1 (1-10): SR branch only
- Stage 2 (11-20): VFI branch only  
- Stage 3 (21-60): Full joint model

## Project Structure

project/
├── dataset.py       # Vimeo dataloader
├── model.py         # EfficientVSR architecture
├── train.py         # Training pipeline
├── evaluate.py      # Evaluation + visualizations
└── README.md

## Requirements

pip install torch torchvision scikit-image matplotlib numpy pillow


## Run Training

python train.py


## Run Evaluation

python evaluate.py