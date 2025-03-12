import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Import our modules
from model import get_model, train_model, plot_training_history, evaluate_model
from data_utils import get_dataloaders

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train tumor classification model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA acceleration')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (fewer workers, smaller batches)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Debug mode overrides
    if args.debug:
        args.num_workers = 0
        args.batch_size = min(8, args.batch_size)
        print("Debug mode enabled: workers=0, batch_size=", args.batch_size)
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Get dataloaders
        print("Loading data...")
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=not args.no_cuda and torch.cuda.is_available()  # Only pin memory if using CUDA
        )
        
        # Create model
        print("Creating model...")
        model = get_model(num_classes=args.num_classes, dropout_rate=args.dropout_rate)
        
        # Print model summary
        print(f"Model created with {args.num_classes} output classes")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3
            # verbose parameter is deprecated, use scheduler.get_last_lr() to access learning rates
        )
        
        # Train the model
        print("Starting training...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            device=device,
            early_stopping_patience=args.patience
        )
        
        # Plot training history
        print("Plotting training history...")
        plot_training_history(history)
        
        # Evaluate on test set if available
        if test_loader is not None:
            print("Evaluating on test set...")
            test_loss, test_acc = evaluate_model(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device
            )
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.save_dir, f"tumor_model_{timestamp}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'args': vars(args)
        }, save_path)
        print(f"Model saved to {save_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 