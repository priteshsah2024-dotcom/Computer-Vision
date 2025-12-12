"""
CIFAR-10 Dataset Loader
This module handles loading and preprocessing the CIFAR-10 dataset.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def get_cifar10_data_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for training and validation
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store/download the dataset
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        classes: List of class names
    """
    
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Normalization for validation/test (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    return train_loader, test_loader, classes


if __name__ == "__main__":
    # Test the data loader
    print("Testing CIFAR-10 Data Loader...")
    train_loader, test_loader, classes = get_cifar10_data_loaders(batch_size=64)
    
    # Get a sample batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample labels: {labels[:10].tolist()}")
    print(f"Label names: {[classes[label] for label in labels[:10]]}")

