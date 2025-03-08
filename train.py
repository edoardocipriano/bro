import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
import os

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device=None, save_dir='checkpoints'):
    """
    Train the tumor classification model
    
    Args:
        model: PyTorch model to train
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer for training
        num_epochs: Number of epochs to train for
        device: Device to train on (default: None, will use CUDA if available)
        save_dir: Directory to save model checkpoints (default: 'checkpoints')
        
    Returns:
        model: Trained model with best weights
        history: Dictionary containing training history (losses and accuracies)
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Start training timer
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Wrap dataloader with tqdm for progress bar
            dataloader_with_progress = tqdm(dataloaders[phase], desc=f"{phase}")
            
            # Iterate over data
            for inputs, labels in dataloader_with_progress:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar with current loss
                dataloader_with_progress.set_postfix(loss=loss.item())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Deep copy the model if best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Save the best model checkpoint
                    checkpoint_path = os.path.join(save_dir, f'best_model_acc_{best_acc:.4f}.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': epoch_acc,
                        'val_loss': epoch_loss,
                    }, checkpoint_path)
                    print(f"Saved new best model with accuracy: {best_acc:.4f} to {checkpoint_path}")
        
        # Save model at the end of each epoch (optional)
        epoch_checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': history['train_loss'][-1],
            'val_loss': history['val_loss'][-1],
            'train_acc': history['train_acc'][-1],
            'val_acc': history['val_acc'][-1],
        }, epoch_checkpoint_path)
        
        print()
    
    # Training complete
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save final best model
    final_model_path = os.path.join(save_dir, 'best_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': best_acc,
    }, final_model_path)
    print(f"Saved final best model to {final_model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Dictionary containing training history
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], label='Training Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['train_loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show() 