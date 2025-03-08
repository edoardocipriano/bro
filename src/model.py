import torch
import torch.nn as nn
import torch.nn.functional as F

# class TumorCNN(nn.Module):
#     def __init__(self, num_classes=2, dropout_rate=0.5):
#         """
#         CNN for tumor image classification with BatchNorm and Dropout
        
#         Args:
#             num_classes: Number of output classes (default: 2 for binary classification)
#             dropout_rate: Dropout probability (default: 0.5)
#         """
#         super(TumorCNN, self).__init__()
        
#         # First convolutional block
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Second convolutional block
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Third convolutional block
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Fourth convolutional block
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Calculate the size of the flattened features
#         # Input: 224x224 -> After 4 pooling layers: 14x14
#         self.fc1 = nn.Linear(256 * 14 * 14, 512)
#         self.bn5 = nn.BatchNorm1d(512)
#         self.dropout1 = nn.Dropout(dropout_rate)
        
#         self.fc2 = nn.Linear(512, 128)
#         self.bn6 = nn.BatchNorm1d(128)
#         self.dropout2 = nn.Dropout(dropout_rate)
        
#         self.fc3 = nn.Linear(128, num_classes)
        
#     def forward(self, x):
#         # First block
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
#         # Second block
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
#         # Third block
#         x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
#         # Fourth block
#         x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
#         # Flatten
#         x = x.view(x.size(0), -1)
        
#         # Fully connected layers
#         x = F.relu(self.bn5(self.fc1(x)))
#         x = self.dropout1(x)
        
#         x = F.relu(self.bn6(self.fc2(x)))
#         x = self.dropout2(x)
        
#         x = self.fc3(x)
        
#         return x

class TumorCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        """
        Lightweight CNN for tumor image classification
        Designed for faster training while maintaining good performance
        
        Args:
            num_classes: Number of output classes (default: 2 for binary classification)
            dropout_rate: Dropout probability (default: 0.3)
        """
        super(TumorCNN, self).__init__()
        
        # Reduce input size with initial pooling to speed up computation
        self.initial_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First convolutional block - reduced channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block - reduced channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block - reduced channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling instead of multiple FC layers
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Single fully connected layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Initial pooling to reduce computation
        x = self.initial_pool(x)
        
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def get_model(num_classes=2, dropout_rate=0.5, model_type='light'):
    """
    Factory function to create and initialize a tumor classification model
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        model_type: Type of model to use ('original' or 'light')
        
    Returns:
        Initialized model
    """
    model = TumorCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Initialize weights
    
    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.01)
    #         nn.init.constant_(m.bias, 0)
            
    return model
