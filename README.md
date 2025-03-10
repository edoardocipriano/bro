# Tumor Classification with CNN

This project implements a Convolutional Neural Network (CNN) for classifying tumor images using PyTorch.

## Project Structure

```
.
├── src/
│   ├── model.py         # CNN model architecture and training functions
│   ├── data_utils.py    # Dataset and data loading utilities
│   └── main.py          # Main script to run training
├── checkpoints/         # Directory for saved models
└── README.md            # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- datasets (Hugging Face)
- tqdm
- matplotlib
- numpy
- Pillow

You can install the required packages using:

```bash
pip install torch torchvision datasets tqdm matplotlib numpy pillow
```

## Dataset

The project uses the "youngp5/tumors" dataset from Hugging Face, which contains tumor images for binary classification.

## Model

The model is a Convolutional Neural Network (TumorCNN) with the following architecture:
- 4 convolutional blocks with batch normalization and max pooling
- Fully connected layers with dropout for regularization
- Binary classification output (tumor/non-tumor)

## Training

To train the model with default parameters:

```bash
python src/main.py
```

### Command Line Arguments

- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of epochs to train (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (L2 penalty) (default: 1e-4)
- `--num_classes`: Number of output classes (default: 2)
- `--dropout_rate`: Dropout rate (default: 0.5)
- `--num_workers`: Number of data loading workers (default: 4)
- `--patience`: Early stopping patience (default: 5)
- `--save_dir`: Directory to save model checkpoints (default: 'checkpoints')
- `--no_cuda`: Disable CUDA acceleration

Example with custom parameters:

```bash
python src/main.py --batch_size 64 --num_epochs 30 --learning_rate 0.0005
```

## Features

- CUDA acceleration for faster training
- Progress bars with tqdm
- Early stopping to prevent overfitting
- Learning rate scheduling
- Training history visualization
- Model checkpointing

## Results

After training, the model will:
1. Save the trained model to the checkpoints directory
2. Generate a plot of training and validation metrics (loss and accuracy)
3. Evaluate the model on the test set if available

The training history plot will be saved as `training_history.png`. 