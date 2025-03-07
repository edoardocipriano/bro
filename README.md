# Tumor Classification CNN

A convolutional neural network (CNN) for classifying tumor images using PyTorch.

## Project Structure

- `model.py`: Contains the CNN architecture with BatchNorm and Dropout layers
- `data_utils.py`: Utilities for loading and preprocessing the tumor dataset
- `train.py`: Training script with progress bars and visualization

## Model Architecture

The CNN model includes:
- 4 convolutional blocks with BatchNorm and MaxPooling
- 2 fully connected layers with BatchNorm and Dropout
- Kaiming initialization for weights

## Usage

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the model:
```
python train.py
```

3. Optional arguments:
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--dropout_rate`: Dropout probability (default: 0.5)

## Dataset

This project uses the "youngp5/tumors" dataset from Hugging Face, which contains tumor images for binary classification.

## Results

After training, the model will:
- Save the trained model to `tumor_cnn_model.pth`
- Generate a visualization of training metrics in `training_history.png`