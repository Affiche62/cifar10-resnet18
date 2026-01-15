# CIFAR-10 Classification with PyTorch Lightning

A PyTorch Lightning project achieving 93%+ accuracy on CIFAR-10 using a modified ResNet18 architecture.

## Features

- **PyTorch Lightning** framework for clean training code
- **Modified ResNet18** architecture optimized for 32x32 CIFAR-10 images
- **Advanced data augmentation** with RandomCrop and RandomHorizontalFlip
- **OneCycleLR** learning rate scheduler for optimal convergence
- **Automatic mixed precision** training (FP16) for faster training
- **TensorBoard logging** for monitoring metrics

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py          # Hyperparameters and configuration
├── data.py            # CIFAR-10 LightningDataModule
├── model.py           # Modified ResNet18 model
├── lit_module.py      # LightningModule with training logic
├── main.py            # Training entry script
└── requirements.txt   # Project dependencies
```

## Configuration

Key hyperparameters in `config.py`:

- **Batch Size**: 128
- **Max Epochs**: 50
- **Learning Rate**: 0.1 (with OneCycleLR)
- **Optimizer**: SGD with Momentum (0.9)
- **Weight Decay**: 5e-4
- **Data Augmentation**: RandomCrop(32, padding=4), RandomHorizontalFlip(0.5)

## Usage

### Training

```bash
python main.py
```

### Monitor Training

```bash
tensorboard --logdir logs/
```

## Model Architecture

The ResNet18 is modified for CIFAR-10:

1. **First Conv Layer**: Changed from 7x7 with stride 2 to 3x3 with stride 1
2. **Removed MaxPool**: Replaced with Identity to preserve spatial information on 32x32 images
3. **Output Layer**: Modified for 10 classes (CIFAR-10)

## Expected Results

- **Training Time**: ~30-40 minutes on single GPU
- **Test Accuracy**: 93%+ (typically 93-95%)
- **Peak Accuracy**: Reached around epoch 30-40

## Data Augmentation

Training transformations:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(0.5)
- Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

## Checkpoints

Best models are saved in `./checkpoints/` with validation accuracy in the filename.

## Credits

Based on best practices for CIFAR-10 classification with modern deep learning techniques.
