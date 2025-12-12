"""
Main Script for CIFAR-10 Classification Project
This script provides a unified interface to run the complete pipeline.
"""

import argparse
import os
import torch
 
from dataset_loader import get_cifar10_data_loaders
from model import get_model
from train import train_model
from evaluate import main as evaluate_main
from utils import plot_training_history, visualize_data_samples


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Classification')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'visualize', 'full', 'webcam', 'compare'],
                       help='Mode: train, evaluate, visualize, full pipeline, webcam, or compare models')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint for evaluation')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    if args.mode == 'train' or args.mode == 'full':
        print("\n" + "=" * 80)
        print("TRAINING MODE")
        print("=" * 80)
        model, history = train_model(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        
        # Plot training history
        plot_training_history(history, save_path='./results/training_history.png')
    
    if args.mode == 'evaluate' or args.mode == 'full':
        print("\n" + "=" * 80)
        print("EVALUATION MODE")
        print("=" * 80)
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Please train the model first or provide a valid checkpoint path.")
            return
        evaluate_main(args.checkpoint)
    
    if args.mode == 'visualize':
        print("\n" + "=" * 80)
        print("VISUALIZATION MODE")
        print("=" * 80)
        _, test_loader, classes = get_cifar10_data_loaders(batch_size=128)
        visualize_data_samples(test_loader, classes, save_path='./results/data_samples.png')
        print("Visualization complete!")
    
    if args.mode == 'webcam':
        print("\n" + "=" * 80)
        print("WEBCAM CLASSIFICATION MODE")
        print("=" * 80)
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Please train the model first.")
            return
        from webcam_classification import WebcamClassifier
        classifier = WebcamClassifier(args.checkpoint)
        classifier.run()
    
    if args.mode == 'compare':
        print("\n" + "=" * 80)
        print("MODEL COMPARISON MODE")
        print("=" * 80)
        from model_comparison import compare_architectures
        
        def create_simple_model(num_classes=10, device='cuda'):
            from model_comparison import SimpleCNN
            model = SimpleCNN(num_classes)
            return model.to(device)
        
        def create_advanced_model(num_classes=10, device='cuda'):
            from model import get_model
            return get_model(num_classes, device)
        
        configs = [
            {'name': 'Simple CNN', 'model_fn': create_simple_model},
            {'name': 'Advanced CNN', 'model_fn': create_advanced_model},
        ]
        compare_architectures(configs, num_epochs=5)


if __name__ == "__main__":
    main()

