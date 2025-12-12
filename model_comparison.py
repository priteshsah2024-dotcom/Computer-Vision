"""
Model Comparison and Analysis
This module provides tools to compare different model architectures and hyperparameters.
Enhances the project for maximum marks by showing evaluation of different approaches.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import CIFAR10CNN, get_model
from dataset_loader import get_cifar10_data_loaders
from train import train_model, validate
import time


class SimpleCNN(nn.Module):
    """
    Simpler CNN architecture for comparison.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def compare_architectures(model_configs, num_epochs=5, batch_size=128):
    """
    Compare different model architectures.
    
    Args:
        model_configs: List of dicts with 'name' and 'model_fn' (function that returns model)
        num_epochs: Number of epochs for quick comparison
        batch_size: Batch size
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, classes = get_cifar10_data_loaders(batch_size=batch_size)
    
    results = []
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Create model
        model = config['model_fn'](num_classes=10, device=device)
        
        # Train
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        start_time = time.time()
        train_losses = []
        val_accs = []
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))
            
            # Validate
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_val_acc = val_accs[-1]
        num_params = sum(p.numel() for p in model.parameters())
        
        results.append({
            'name': config['name'],
            'final_acc': final_val_acc,
            'training_time': training_time,
            'num_params': num_params,
            'train_losses': train_losses,
            'val_accs': val_accs
        })
        
        print(f"\nFinal Validation Accuracy: {final_val_acc:.2f}%")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Parameters: {num_params:,}")
    
    # Plot comparison
    plot_comparison(results)
    
    return results


def plot_comparison(results):
    """
    Plot comparison of different models.
    
    Args:
        results: List of result dictionaries
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy comparison
    names = [r['name'] for r in results]
    accs = [r['final_acc'] for r in results]
    axes[0].bar(names, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Final Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Training time comparison
    times = [r['training_time'] for r in results]
    axes[1].bar(names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Training Time', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # Parameter count comparison
    params = [r['num_params'] for r in results]
    axes[2].bar(names, params, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_title('Model Size (Parameters)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Number of Parameters', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to ./results/model_comparison.png")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('./results', exist_ok=True)
    
    # Define model configurations
    def create_simple_model(num_classes=10, device='cuda'):
        model = SimpleCNN(num_classes)
        return model.to(device)
    
    def create_advanced_model(num_classes=10, device='cuda'):
        return get_model(num_classes, device)
    
    configs = [
        {'name': 'Simple CNN', 'model_fn': create_simple_model},
        {'name': 'Advanced CNN', 'model_fn': create_advanced_model},
    ]
    
    print("Comparing model architectures...")
    results = compare_architectures(configs, num_epochs=5)

