import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from train import train_model
from data_utils import get_dataloaders
import itertools
import json
import os
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Tuple

class Experiment:
    def __init__(self, experiment_name: str, base_save_dir: str = 'experiments'):
        """
        Initialize an experiment for hyperparameter tuning
        
        Args:
            experiment_name: Name of the experiment
            base_save_dir: Base directory to save experiment results
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(base_save_dir, f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize results tracking
        self.results = []
        self.best_accuracy = 0.0
        self.best_config = None
        self.best_model_path = None
        
    def generate_hyperparameter_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all possible combinations of hyperparameters
        
        Args:
            param_grid: Dictionary of hyperparameter names and their possible values
            
        Returns:
            List of dictionaries containing hyperparameter combinations
        """
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def run_single_training(self, config: Dict[str, Any]) -> Tuple[float, str]:
        """
        Run a single training with given hyperparameters
        
        Args:
            config: Dictionary containing hyperparameter configuration
            
        Returns:
            best_accuracy: Best validation accuracy achieved
            model_path: Path to the saved model
        """
        # Create model directory
        model_dir = os.path.join(self.save_dir, f"model_{len(self.results)}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Get dataloaders
        train_loader, val_loader, _ = get_dataloaders(
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        
        # Initialize model
        model = get_model(
            num_classes=config.get('num_classes', 2),
            dropout_rate=config.get('dropout_rate', 0.5)
        )
        
        # Initialize optimizer
        optimizer_class = getattr(optim, config['optimizer'])
        optimizer = optimizer_class(
            model.parameters(),
            lr=config['learning_rate'],
            **config.get('optimizer_params', {})
        )
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        trained_model, history = train_model(
            model=model,
            dataloaders={'train': train_loader, 'val': val_loader},
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['num_epochs'],
            save_dir=model_dir
        )
        
        # Get best validation accuracy
        best_val_acc = max(history['val_acc'])
        
        # Save training history
        history_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc']
        })
        history_df.to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
        
        return best_val_acc, os.path.join(model_dir, 'best_model_final.pth')
    
    def run_experiment(self, param_grid: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Run experiment with all hyperparameter combinations
        
        Args:
            param_grid: Dictionary of hyperparameter names and their possible values
            
        Returns:
            DataFrame containing results of all runs
        """
        configs = self.generate_hyperparameter_combinations(param_grid)
        print(f"Running experiment with {len(configs)} different configurations")
        
        for config in configs:
            print("\nTraining with configuration:")
            print(json.dumps(config, indent=2))
            
            # Run training
            accuracy, model_path = self.run_single_training(config)
            
            # Store results
            result = {
                'config': config,
                'accuracy': accuracy,
                'model_path': model_path
            }
            self.results.append(result)
            
            # Update best model if necessary
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = config
                self.best_model_path = model_path
                print(f"\nNew best model found! Accuracy: {accuracy:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {**r['config'], 'accuracy': r['accuracy'], 'model_path': r['model_path']}
            for r in self.results
        ])
        
        # Save results
        results_df.to_csv(os.path.join(self.save_dir, 'experiment_results.csv'), index=False)
        
        # Save best configuration
        with open(os.path.join(self.save_dir, 'best_config.json'), 'w') as f:
            json.dump({
                'best_accuracy': self.best_accuracy,
                'best_config': self.best_config,
                'best_model_path': self.best_model_path
            }, f, indent=4)
        
        return results_df
    
    def load_best_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load the best model from the experiment
        
        Returns:
            model: Best performing model
            config: Configuration used for the best model
        """
        if self.best_model_path is None:
            raise ValueError("No best model found. Run the experiment first.")
        
        # Load best configuration
        model = get_model(
            num_classes=self.best_config.get('num_classes', 2),
            dropout_rate=self.best_config.get('dropout_rate', 0.5)
        )
        
        # Load model weights
        checkpoint = torch.load(self.best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, self.best_config

# Example usage:
"""
# Define parameter grid
param_grid = {
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'optimizer': ['Adam', 'SGD'],
    'num_epochs': [10],
    'dropout_rate': [0.3, 0.5],
    'optimizer_params': [
        {},  # Default parameters for Adam
        {'momentum': 0.9}  # Parameters for SGD
    ]
}

# Create and run experiment
experiment = Experiment('tumor_classification_v1')
results = experiment.run_experiment(param_grid)

# Load best model
best_model, best_config = experiment.load_best_model()
""" 