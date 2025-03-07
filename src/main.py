import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_model, save_evaluation_results
import os
import json
from datetime import datetime

def main():
    # Configuration
    config = {
        'batch_size': 32,
        'num_workers': 4,
        'num_classes': 9,
        'learning_rate': 0.001,
        'num_epochs': 25,
        'dropout_rate': 0.5,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiments/tumor_classification_{timestamp}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        # Convert device to string for JSON serialization
        config_to_save = {**config, 'device': str(config['device'])}
        json.dump(config_to_save, f, indent=4)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print("Initializing model...")
    model = get_model(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"Starting training on {config['device']}...")
    model, history = train_model(
        model=model,
        dataloaders={'train': train_loader, 'val': val_loader},
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_dir=os.path.join(experiment_dir, 'checkpoints')
    )
    
    # Save training history
    history_path = os.path.join(experiment_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc']
        }, f, indent=4)
    
    if test_loader is not None:
        print("\nEvaluating model on test set...")
        test_results = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=config['device']
        )
        
        # Save evaluation results
        eval_dir = os.path.join(experiment_dir, 'evaluation')
        save_evaluation_results(test_results, eval_dir)
        
        print(f"\nTest Loss: {test_results['loss']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    else:
        print("\nNo test set available for evaluation.")
    
    print(f"\nExperiment completed. Results saved in: {experiment_dir}")

if __name__ == "__main__":
    main() 