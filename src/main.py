import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_model, save_evaluation_results
from experiment import Experiment
import os
import json
import argparse
from datetime import datetime

def run_single_training(model_type='light'):
    """Run a single training with default configuration"""
    # Configuration
    config = {
        'batch_size': 32,
        'num_workers': 4,
        'num_classes': 9,
        'learning_rate': 0.001,
        'num_epochs': 25,
        'dropout_rate': 0.5,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'model_type': model_type,
        'use_amp': True,  # Enable automatic mixed precision
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
        save_dir=os.path.join(experiment_dir, 'checkpoints'),
        use_amp=config['use_amp']
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

def run_hyperparameter_search(model_type='light'):
    """Run hyperparameter search using the Experiment class"""
    # Define parameter grid for hyperparameter search
    param_grid = {
        'batch_size': [16, 32],
        'learning_rate': [0.001, 0.0001],
        'optimizer': ['Adam', 'SGD'],
        'num_epochs': [15],
        'dropout_rate': [0.3, 0.5],
        'num_classes': [9],
        'model_type': [model_type],
        'optimizer_params': [
            {},  # Default parameters for Adam
            {'momentum': 0.9}  # Parameters for SGD
        ]
    }
    
    # Create and run experiment
    experiment = Experiment(f'tumor_classification_{model_type}_hyperparameter_search')
    results = experiment.run_experiment(param_grid)
    
    # Evaluate best model
    test_results = experiment.evaluate_best_model()
    
    print("\nHyperparameter search completed.")
    print(f"Best configuration: {experiment.best_config}")
    print(f"Best validation accuracy: {experiment.best_accuracy:.4f}")
    
    return experiment

def main():
    """Main function to run the tumor classification project"""
    parser = argparse.ArgumentParser(description='Tumor Classification')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'hyperparameter_search'],
                        help='Mode to run: single training or hyperparameter search')
    parser.add_argument('--model', type=str, default='light', choices=['original', 'light'],
                        help='Model type to use: original (larger) or light (faster)')
    args = parser.parse_args()
    
    if args.mode == 'single':
        print(f"Running single training with {args.model} model...")
        run_single_training(model_type=args.model)
    else:
        print(f"Running hyperparameter search with {args.model} model...")
        experiment = run_hyperparameter_search(model_type=args.model)
        
        # Load best model for further use if needed
        best_model, best_config = experiment.load_best_model()
        print(f"Best model loaded from: {experiment.best_model_path}")

if __name__ == "__main__":
    main() 