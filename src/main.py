import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_model, save_evaluation_results
import os
import json
import argparse
from datetime import datetime

def run_training(model_type='light', batch_size=32, num_workers=4, num_classes=9, 
                learning_rate=0.001, num_epochs=15, dropout_rate=0.5):
    """Run a single training with the specified configuration"""
    # Configuration
    config = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_classes': num_classes,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'dropout_rate': dropout_rate,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'model_type': model_type,
        'pin_memory': True,  # Enable pinned memory for faster data transfer
        'prefetch_factor': 2  # Prefetch factor for DataLoader
    }
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiments/tumor_classification_{model_type}_{timestamp}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        # Convert device to string for JSON serialization
        config_to_save = {**config, 'device': str(config['device'])}
        json.dump(config_to_save, f, indent=4)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor']
    )
    
    print(f"Initializing {model_type} model...")
    model = get_model(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        model_type=config['model_type']
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
    
    print(f"\nTraining completed. Results saved in: {experiment_dir}")
    
    return model, history, experiment_dir

def main():
    """Main function to run the tumor classification project"""
    parser = argparse.ArgumentParser(description='Tumor Classification')
    parser.add_argument('--model', type=str, default='light', choices=['original', 'light'],
                        help='Model type to use: original (larger) or light (faster)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs to train for (default: 15)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate (default: 0.5)')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes (default: 9)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    args = parser.parse_args()
    
    print(f"Running training with {args.model} model...")
    run_training(
        model_type=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        dropout_rate=args.dropout_rate
    )

if __name__ == "__main__":
    main() 