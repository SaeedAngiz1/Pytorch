"""
Data Loading Utilities
Helper functions for loading and preprocessing data in PyTorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np

def get_mnist_loaders(batch_size=64, data_dir='./data'):
    """
    Get MNIST data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
    
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def get_cifar10_loaders(batch_size=64, data_dir='./data'):
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load data
    
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

class CustomDataset(Dataset):
    """
    Custom dataset class for your own data.
    
    Example:
        dataset = CustomDataset(data, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

def create_dataloader(data, labels, batch_size=32, shuffle=True, transform=None):
    """
    Create a DataLoader from numpy arrays or lists.
    
    Args:
        data: Input data (numpy array or list)
        labels: Labels (numpy array or list)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        transform: Optional transform function
    
    Returns:
        DataLoader object
    """
    dataset = CustomDataset(data, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

