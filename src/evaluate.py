import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def evaluate_model(model, dataloader, criterion, device=None):
    """
    Evaluate the model on a given dataset
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        criterion: Loss function
        device: Device to evaluate on (default: None, will use CUDA if available)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader.dataset)
    
    # Convert predictions and labels to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_evaluation_results(results, save_dir):
    """
    Save evaluation results to files
    
    Args:
        results: Dictionary containing evaluation results
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics to text file
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Loss: {results['loss']:.4f}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                f.write(f"\nClass: {class_name}\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.4f}\n")
    
    # Plot and save confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)

def predict_single_image(model, image, device=None, transform=None):
    """
    Make prediction for a single image
    
    Args:
        model: Trained PyTorch model
        image: PIL Image or tensor
        device: Device to evaluate on (default: None, will use CUDA if available)
        transform: Transforms to apply to the image (if image is PIL)
        
    Returns:
        predicted_class, confidence_score
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Apply transform if provided and if image is not already a tensor
    if transform is not None and not isinstance(image, torch.Tensor):
        image = transform(image)
    
    # Add batch dimension if not present
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def load_model_for_inference(model_path, model_class, device=None):
    """
    Load a trained model for inference
    
    Args:
        model_path: Path to the saved model checkpoint
        model_class: The model class to instantiate
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model 