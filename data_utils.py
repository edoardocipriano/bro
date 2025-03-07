from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

ds = load_dataset("youngp5/tumors")

class TumorDataset(Dataset):
    def __init__(self, hf_dataset, split="train", transform=None):
        """
        Args:
            hf_dataset: Hugging Face dataset
            split: Which split to use (train, validation, test)
            transform: Optional transform to be applied on images
        """
        self.dataset = hf_dataset[split]
        self.transform = transform if transform is not None else transforms.ToTensor()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Assuming the dataset has 'image' and 'label' fields
        # Adjust these field names based on the actual dataset structure
        item = self.dataset[idx]
        
        # Convert image to PIL if it's not already
        if not isinstance(item['image'], Image.Image):
            image = Image.fromarray(item['image'])
        else:
            image = item['image']
            
        # Apply transformations
        image = self.transform(image)
        
        # Get label
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return image, label

def get_dataloaders(batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
        batch_size: Batch size for the dataloaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TumorDataset(ds, split="train", transform=train_transform)
    
    # Check if validation and test splits exist
    splits = ds.keys()
    
    val_dataset = None
    if "validation" in splits:
        val_dataset = TumorDataset(ds, split="validation", transform=val_test_transform)
    elif "val" in splits:
        val_dataset = TumorDataset(ds, split="val", transform=val_test_transform)
    
    test_dataset = None
    if "test" in splits:
        test_dataset = TumorDataset(ds, split="test", transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    return train_loader, val_loader, test_loader

# Example usage:
# train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)