"""
Evaluation Script for CIFAR-10 CNN
This script evaluates the trained model and computes metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset_loader import get_cifar10_data_loaders
from model import get_model


def evaluate_model(model, test_loader, device, classes):
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader for test data
        device: Device to run on
        classes: List of class names
    
    Returns:
        Dictionary containing metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Overall metrics (macro average)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results


def print_metrics(results, classes):
    """
    Print evaluation metrics.
    
    Args:
        results: Dictionary containing metrics
        classes: List of class names
    """
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall: {results['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score: {results['macro_f1'] * 100:.2f}%")
    
    print("\n" + "-" * 80)
    print("Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(classes):
        print(f"{class_name:<15} {results['precision'][i]:<12.4f} "
              f"{results['recall'][i]:<12.4f} {results['f1'][i]:<12.4f} "
              f"{len(results['labels'][results['labels'] == i]):<10}")
    
    print("=" * 80)


def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def plot_sample_predictions(model, test_loader, device, classes, num_samples=16, save_path='sample_predictions.png'):
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader for test data
        device: Device to run on
        classes: List of class names
        num_samples: Number of samples to visualize
        save_path: Path to save the plot
    """
    model.eval()
    
    # Get a batch of images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images[:num_samples])
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).to(device)
    images_denorm = images[:num_samples] * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
        true_label = classes[labels[i].item()]
        pred_label = classes[predicted[i].item()]
        confidence = probs[i][predicted[i]].item() * 100
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                         color=color, fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample predictions saved to: {save_path}")
    plt.close()


def main(checkpoint_path='./checkpoints/best_model.pth'):
    """
    Main evaluation function.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 test dataset...")
    _, test_loader, classes = get_cifar10_data_loaders(batch_size=128, num_workers=4)
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = get_model(num_classes=10, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy at checkpoint: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, classes)
    
    # Print metrics
    print_metrics(results, classes)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(results['confusion_matrix'], classes, 
                         save_path='./results/confusion_matrix.png')
    plot_sample_predictions(model, test_loader, device, classes,
                           save_path='./results/sample_predictions.png')
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs('./results', exist_ok=True)
    results = main()
