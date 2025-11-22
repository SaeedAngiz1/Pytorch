"""
Model Utilities
Helper functions for model operations like saving, loading, and evaluation.
"""

import torch
import torch.nn as nn
import os

def save_model(model, filepath, optimizer=None, epoch=None, loss=None, accuracy=None):
    """
    Save model and optionally optimizer state.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        loss: Optional loss value
        accuracy: Optional accuracy value
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath, optimizer=None, device=None):
    """
    Load model and optionally optimizer state.
    
    Args:
        model: PyTorch model (architecture must match)
        filepath: Path to the saved model
        optimizer: Optional optimizer to load state into
        device: Device to load model on
    
    Returns:
        Dictionary with loaded information (epoch, loss, accuracy if available)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    result = {}
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'epoch' in checkpoint:
        result['epoch'] = checkpoint['epoch']
    if 'loss' in checkpoint:
        result['loss'] = checkpoint['loss']
    if 'accuracy' in checkpoint:
        result['accuracy'] = checkpoint['accuracy']
    
    print(f"Model loaded from {filepath}")
    return result

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters, number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def evaluate_model(model, dataloader, criterion, device=None):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        Average loss, accuracy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def get_model_summary(model, input_size):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size tuple (e.g., (1, 28, 28) for MNIST)
    """
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(model)
    print("-" * 50)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)

def freeze_layers(model, num_layers=None):
    """
    Freeze layers of a model.
    
    Args:
        model: PyTorch model
        num_layers: Number of layers to freeze (from the beginning)
                     If None, freezes all layers
    """
    if num_layers is None:
        for param in model.parameters():
            param.requires_grad = False
    else:
        layers = list(model.children())
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    print(f"Frozen {num_layers if num_layers else 'all'} layers")

def unfreeze_layers(model):
    """Unfreeze all layers of a model."""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfrozen all layers")

