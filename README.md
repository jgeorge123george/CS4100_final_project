# EMNIST Character Recognition Project

A PyTorch-based machine learning project for training and evaluating models on the EMNIST (Extended Modified NIST) dataset for handwritten character recognition.

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── MNIST_data.py                # Legacy MNIST data loader
├── test_drawing.py             # Drawing interface for testing
├── data/                       # Dataset storage
│   └── EMNIST/
└── src/                        # Main source code
    ├── model.py                # CNN model architecture
    ├── trainer.py              # Model training script
    ├── data_loader.py          # EMNIST data loading utilities
    ├── evaluate.py             # Single image prediction
    ├── evaluate_test.py        # Test set evaluation
    ├── check_emnist_classes.py # Dataset analysis
    ├── utils.py                # Utility functions
    ├── requirements.txt        # Dependencies
    └── data/                   # Local dataset cache
        ├── EMNIST/
        └── MNIST/
```

## Features

- **Multiple EMNIST Splits Support**: Train on different character sets
  - `balanced`: 47 classes (digits + balanced letters)
  - `byclass`: 62 classes (digits + uppercase + lowercase)
  - `digits`: 10 classes (digits only)
  - `letters`: 26 classes (uppercase letters)
  - And more...

- **Flexible Training**: Configurable epochs, learning rate, and data splits
- **Model Evaluation**: Test set accuracy and single image prediction
- **Cross-Platform**: Automatic device detection (CUDA/MPS/CPU)
- **Visualization**: Training loss curves and prediction confidence plots

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Train a Model

Train on the default EMNIST balanced split for 10 epochs:

```bash
python src/trainer.py
```

Train with custom parameters:

```bash
python src/trainer.py [epochs] [split]
```

Example:
```bash
python src/trainer.py 20 balanced  # 20 epochs on balanced split
python src/trainer.py 15 digits    # 15 epochs on digits only
```

### 2. Evaluate on Test Set

Evaluate the trained model on the test dataset:

```bash
python src/evaluate_test.py
```

Evaluate specific split or model:

```bash
python src/evaluate_test.py [split] [model_path]
```

Example:
```bash
python src/evaluate_test.py balanced
python src/evaluate_test.py digits model_state_emnist_digits.pt
```

### 3. Predict Single Images

Predict a single image file:

```bash
python src/evaluate.py test_img.png
```

With custom split or model:

```bash
python src/evaluate.py image.png [split] [model_path]
```

Example:
```bash
python src/evaluate.py my_drawing.png balanced
python src/evaluate.py letter.png letters model_state_emnist_letters.pt
```

## Model Architecture

The `ImageClassifier` uses a CNN architecture:
- 3 Convolutional layers (32, 64, 64 filters)
- ReLU activations
- Fully connected output layer
- Configurable number of classes based on EMNIST split

## Output Files

Training generates:
- `model_state_emnist_[split].pt`: Saved model weights and metadata
- `training_loss_curve_emnist_[split].png`: Training loss visualization

## EMNIST Splits

| Split | Classes | Description |
|-------|---------|-------------|
| balanced | 47 | Balanced digits and letters |
| byclass | 62 | All digits, uppercase, and lowercase |
| digits | 10 | Digits 0-9 only |
| letters | 26 | Uppercase letters A-Z |
| bymerge | 47 | Unbalanced merge of similar characters |
| mnist | 10 | MNIST-compatible subset |

## Device Support

The project automatically detects and uses the best available device:
- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2) GPUs  
- **CPU**: Fallback for compatibility

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:
- PyTorch
- torchvision
- matplotlib
- Pillow

## Example Workflow

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Train a model on balanced EMNIST for 15 epochs
python src/trainer.py 15 balanced

# Evaluate the trained model
python src/evaluate_test.py balanced

# Test on a custom image
python src/evaluate.py my_handwritten_letter.png balanced
```

The model will display prediction confidence and show the top 5 most likely characters for single image predictions.
