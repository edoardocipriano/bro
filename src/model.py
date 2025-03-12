import torch
import torch.nn as nn
import torch.nn.functional as F

class TumorCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        """
        CNN for tumor image classification with BatchNorm and Dropout
        
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            dropout_rate: Dropout probability (default: 0.5)
        """
        super(TumorCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling to handle different input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

def get_model(num_classes=2, dropout_rate=0.5):
    """
    Factory function to create and initialize a tumor classification model
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        
    Returns:
        Initialized model
    """
    model = TumorCNN(num_classes=num_classes, dropout_rate=dropout_rate)
            
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=10, device=None, early_stopping_patience=5):
    """
    Train the tumor classification model
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on (will use CUDA if available when None)
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        Trained model and training history dictionary
    """
    import time
    from tqdm import tqdm
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on {device}")
    model = model.to(device)
    
    # Initialize history dictionary to store metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in train_pbar:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': train_loss / train_total, 
                'acc': 100 * train_correct / train_total
            })
        
        # Calculate epoch statistics
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # No gradients needed for validation
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': val_loss / val_total, 
                        'acc': 100 * val_correct / val_total
                    })
            
            # Calculate epoch statistics
            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = 100 * val_correct / val_total
            
            # Update learning rate scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Store current learning rate before step
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(epoch_val_loss)
                    # Check if learning rate changed and print the new value
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
                else:
                    # Store current learning rate before step
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step()
                    # Check if learning rate changed and print the new value
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != old_lr:
                        print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            
            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                early_stopping_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    model.load_state_dict(best_model_state)
                    break
        else:
            epoch_val_loss = 0
            epoch_val_acc = 0
            
            # Update learning rate scheduler if provided
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Store current learning rate before step
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                # Check if learning rate changed and print the new value
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
        # Record history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% - "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")
    
    # Load best model if early stopping was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training history
    """
    import matplotlib.pyplot as plt
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_loader, criterion, device=None):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Test loss and accuracy
    """
    from tqdm import tqdm
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # No gradients needed for evaluation
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Evaluating")
        
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            test_pbar.set_postfix({
                'loss': test_loss / test_total, 
                'acc': 100 * test_correct / test_total
            })
    
    # Calculate final statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100 * test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.2f}%")
    
    return test_loss, test_acc
