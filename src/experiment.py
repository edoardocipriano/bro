import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from train import train_model
from data_utils import get_dataloaders
from evaluate import evaluate_model, save_evaluation_results, plot_confusion_matrix
import itertools
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Experiment '{experiment_name}' initialized. Results will be saved to {self.save_dir}")
        print(f"Using device: {self.device}")
        
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
            # Convert device to string for JSON serialization if present
            config_to_save = {**config}
            if 'device' in config_to_save and not isinstance(config_to_save['device'], str):
                config_to_save['device'] = str(config_to_save['device'])
            json.dump(config_to_save, f, indent=4)
        
        # Get dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 4)
        )
        
        # Initialize model
        model = get_model(
            num_classes=config.get('num_classes', 2),
            dropout_rate=config.get('dropout_rate', 0.5)
        )
        
        # Initialize optimizer
        optimizer_name = config.get('optimizer', 'Adam')
        optimizer_class = getattr(optim, optimizer_name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            **config.get('optimizer_params', {})
        )
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss()
        
        # Train model
        print(f"\nTraining model with configuration:")
        print(json.dumps(config_to_save, indent=2))
        
        trained_model, history = train_model(
            model=model,
            dataloaders={'train': train_loader, 'val': val_loader},
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config.get('num_epochs', 25),
            device=self.device,
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
        
        # Plot and save training history
        self._plot_training_history(history, save_path=os.path.join(model_dir, 'training_history.png'))
        
        # Evaluate on test set if available
        if test_loader is not None:
            print("\nEvaluating model on test set...")
            test_results = evaluate_model(
                model=trained_model,
                dataloader=test_loader,
                criterion=criterion,
                device=self.device
            )
            
            # Save evaluation results
            eval_dir = os.path.join(model_dir, 'evaluation')
            save_evaluation_results(test_results, eval_dir)
            
            print(f"Test Loss: {test_results['loss']:.4f}")
            print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        
        return best_val_acc, os.path.join(model_dir, 'best_model_final.pth')
    
    def _plot_training_history(self, history, save_path=None):
        """
        Plot training and validation accuracy/loss
        
        Args:
            history: Dictionary containing training history
            save_path: Path to save the plot
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
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
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
        
        for i, config in enumerate(configs):
            print(f"\n[Experiment {i+1}/{len(configs)}]")
            
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
            best_config_to_save = {**self.best_config}
            if 'device' in best_config_to_save and not isinstance(best_config_to_save['device'], str):
                best_config_to_save['device'] = str(best_config_to_save['device'])
            
            json.dump({
                'best_accuracy': self.best_accuracy,
                'best_config': best_config_to_save,
                'best_model_path': self.best_model_path
            }, f, indent=4)
        
        # Plot results comparison
        self._plot_experiment_results(results_df)
        
        print(f"\nExperiment completed. Results saved in: {self.save_dir}")
        print(f"Best model accuracy: {self.best_accuracy:.4f}")
        
        return results_df
    
    def _plot_experiment_results(self, results_df):
        """
        Plot comparison of different experiment configurations
        
        Args:
            results_df: DataFrame containing experiment results
        """
        plt.figure(figsize=(12, 6))
        
        # Sort by accuracy
        sorted_results = results_df.sort_values('accuracy', ascending=False)
        
        # Plot bar chart of accuracies
        plt.bar(range(len(sorted_results)), sorted_results['accuracy'], color='skyblue')
        plt.xlabel('Configuration Index')
        plt.ylabel('Validation Accuracy')
        plt.title('Comparison of Model Configurations')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add configuration details as text
        for i, (_, row) in enumerate(sorted_results.iterrows()):
            config_text = ", ".join([f"{k}: {v}" for k, v in row.items() 
                                    if k not in ['accuracy', 'model_path', 'optimizer_params']])
            plt.text(i, row['accuracy'] - 0.05, f"{row['accuracy']:.4f}", 
                     ha='center', va='bottom', rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'experiment_comparison.png'))
        plt.close()
    
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
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model, self.best_config
    
    def evaluate_best_model(self, test_loader=None):
        """
        Evaluate the best model on the test set
        
        Args:
            test_loader: DataLoader for test data. If None, will create one using the best config.
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.best_model_path is None:
            raise ValueError("No best model found. Run the experiment first.")
        
        # Load best model
        model, config = self.load_best_model()
        
        # Get test dataloader if not provided
        if test_loader is None:
            _, _, test_loader = get_dataloaders(
                batch_size=config.get('batch_size', 32),
                num_workers=config.get('num_workers', 4)
            )
            
            if test_loader is None:
                raise ValueError("No test set available and no test_loader provided.")
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate model
        print("\nEvaluating best model on test set...")
        test_results = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=self.device
        )
        
        # Save evaluation results
        eval_dir = os.path.join(self.save_dir, 'best_model_evaluation')
        save_evaluation_results(test_results, eval_dir)
        
        print(f"Test Loss: {test_results['loss']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        
        return test_results

# Example usage:
"""
# Define parameter grid
param_grid = {
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001],
    'optimizer': ['Adam', 'SGD'],
    'num_epochs': [10],
    'dropout_rate': [0.3, 0.5],
    'num_classes': [9],
    'optimizer_params': [
        {},  # Default parameters for Adam
        {'momentum': 0.9}  # Parameters for SGD
    ]
}

# Create and run experiment
experiment = Experiment('tumor_classification_v1')
results = experiment.run_experiment(param_grid)

# Evaluate best model
test_results = experiment.evaluate_best_model()

# Load best model for inference
best_model, best_config = experiment.load_best_model()
""" 