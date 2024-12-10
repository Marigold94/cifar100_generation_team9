# cifar100_generation_team9
# CIFAR-100 Conditional BigGAN Implementation

This repository contains an implementation of a Conditional BigGAN trained on the CIFAR-100 dataset. The implementation includes spectral normalization, conditional batch normalization, and other modern GAN training techniques.

## Features

- Conditional BigGAN architecture with spectral normalization
- Mixed precision training for improved performance
- Comprehensive evaluation metrics (FID, Inception Score, Intra-FID)
- Detailed visualization and analysis tools
- Class-conditional image generation
- Support for superclass-based generation

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
scikit-learn>=0.24.0
pandas>=1.3.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cifar100-biggan.git
cd cifar100-biggan
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── evaluation_metrics.py   # Evaluation metrics (FID, IS, Intra-FID)
├── gan_training.py        # Main training script
├── improved_cifar_gan.py  # GAN model architecture
├── gan_visualization.py   # Visualization and analysis tools
└── inception_model.py     # Inception model for evaluation
```

## Usage

### Training

To train the model with default parameters:

```bash
python3 gan_training.py
```

You can modify the random seed in `gan_training.py` by changing the value in the following section:

```python
if __name__ == "__main__":
    torch.manual_seed(42)  # Change 42 to your desired seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  # Change 42 to the same seed
```

The default seed is set to 42, but you can change it to any integer value for different random initializations. Make sure to set both `torch.manual_seed()` and `torch.cuda.manual_seed_all()` to the same value for consistent results.

The script will automatically:
- Download and prepare the CIFAR-100 dataset
- Initialize the GAN models
- Start the training process
- Save checkpoints and generate samples periodically
- Perform evaluation and save metrics

### Directory Structure

During training, the following directory structure will be created:

```
gan_results/
├── samples/           # Generated image samples
├── superclass/        # Superclass-specific generations
├── metrics/          # Evaluation metrics
└── checkpoints/      # Model checkpoints
```

## Model Architecture

The implementation uses a modified BigGAN architecture with:
- Conditional Batch Normalization in the generator
- Spectral Normalization in both generator and discriminator
- Projection Discriminator for conditioning
- Residual blocks with up/downsampling
- Global sum pooling in the discriminator

## Training Details

- Batch size: 128
- Learning rates: 
  - Generator: 1e-4
  - Discriminator: 4e-4
- Optimizer: Adam (β1=0.0, β2=0.999)
- Number of discriminator iterations per generator iteration: 5
- Mixed precision training enabled
- Automatic batch size scaling based on available GPU memory

## Evaluation Metrics

The model is evaluated using:
- Fréchet Inception Distance (FID)
- Inception Score (IS)
- Intra-FID for class-conditional generation quality
- t-SNE visualization of feature spaces

## Hardware Requirements

- GPU with at least 8GB VRAM recommended
- CPU training is supported but not recommended due to computational intensity
- At least 16GB system RAM recommended

## Acknowledgments

This implementation incorporates techniques and insights from:
- BigGAN (Brock et al., 2018)
- Spectral Normalization (Miyato et al., 2018)
- Projection Discriminator (Miyato & Koyama, 2018)

## License

MIT License
