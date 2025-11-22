"""
Simple Neural Network Example
This script demonstrates how to create and use a basic neural network in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """A simple feedforward neural network."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_model(input_size=784, hidden_size=128, num_classes=10):
    """Create and return a SimpleNet model."""
    model = SimpleNet(input_size, hidden_size, num_classes)
    return model

def train_step(model, data, target, optimizer, criterion):
    """Perform one training step."""
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, target, criterion):
    """Evaluate the model."""
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1)
        accuracy = (pred == target).float().mean()
    return loss.item(), accuracy.item()

def main():
    print("=" * 50)
    print("Simple Neural Network Example")
    print("=" * 50)
    
    # Hyperparameters
    input_size = 784
    hidden_size = 128
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 5
    
    # Create model
    print("\n1. Creating Model:")
    print("-" * 30)
    model = create_model(input_size, hidden_size, num_classes)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n2. Model Setup Complete:")
    print("-" * 30)
    print(f"Loss function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr={learning_rate})")
    
    # Example with dummy data
    print("\n3. Training Example (with dummy data):")
    print("-" * 30)
    
    # Create dummy data
    dummy_data = torch.randn(batch_size, input_size)
    dummy_targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Input shape: {dummy_data.shape}")
    print(f"Target shape: {dummy_targets.shape}")
    
    # Training loop example
    for epoch in range(num_epochs):
        loss = train_step(model, dummy_data, dummy_targets, optimizer, criterion)
        
        if (epoch + 1) % 1 == 0:
            eval_loss, accuracy = evaluate(model, dummy_data, dummy_targets, criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("\n" + "=" * 50)
    print("Training example completed!")
    print("=" * 50)
    print("\nNote: This is a dummy example. For real training, use actual datasets like MNIST, CIFAR-10, etc.")

if __name__ == "__main__":
    main()

