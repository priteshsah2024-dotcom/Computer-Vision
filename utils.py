"""
Utility Functions
Helper functions for visualization and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    plt.close()


def visualize_data_samples(data_loader, classes, num_samples=16, save_path='data_samples.png'):
    """
    Visualize sample images from the dataset.
    
    Args:
        data_loader: DataLoader for the dataset
        classes: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Denormalize
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    images_denorm = images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images_denorm[i].permute(1, 2, 0).numpy()
        label = classes[labels[i].item()]
        
        axes[i].imshow(img)
        axes[i].set_title(f'{label}', fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('CIFAR-10 Dataset Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Data samples saved to: {save_path}")
    plt.close()

