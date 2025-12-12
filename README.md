# CIFAR-10 CNN Classification Project

This project implements a Convolutional Neural Network (CNN) for multi-class object classification on the CIFAR-10 dataset using PyTorch.

#Google Colab Link to Notebook
https://colab.research.google.com/drive/1yHvOi3_d_Ge3SyL53BT9OIJAjR0OkuM0?usp=sharing

## Project Structure

```
.
├── dataset_loader.py         # CIFAR-10 dataset loading and preprocessing
├── model.py                  # CNN architecture definition
├── train.py                  # Training script
├── evaluate.py               # Evaluation and metrics computation
├── utils.py                  # Utility functions for visualization
├── webcam_classification.py  # Real-time webcam/video classification (Optional Feature)
├── model_comparison.py       # Model architecture comparison tool
├── main.py                   # Main script to run the complete pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── TECHNICAL_REPORT.md       # 500-word technical report (Required)
└── ASSESSMENT_CHECKLIST.md   # Requirements verification checklist
```

## Features

- **CNN Architecture**: Custom CNN with 3 convolutional blocks, batch normalization, and dropout
- **Data Augmentation**: Random horizontal flips and random crops for better generalization
- **Multi-class Classification**: Classifies 10 object categories (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, and F1-score metrics
- **Visualizations**: Confusion matrix, sample predictions, and training history plots
- **GPU Acceleration**: Supports CUDA for faster training
- **Real-time Classification**: Webcam and video file classification (Optional feature for maximum marks)
- **Model Comparison**: Compare different CNN architectures and hyperparameters

## Setup Instructions

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional, but recommended)
- PyTorch with CUDA support (if using GPU)

### Installation

1. **Activate the virtual environment** (if using the provided venv):
   ```powershell
   cd "D:\Mubeen\anaya ml\Detection\VOC2007"
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU availability** (optional):
   ```bash
   python check_gpu.py
   ```

## Usage

### Training the Model

Train the CNN model from scratch:

```bash
python main.py --mode train --epochs 50 --batch_size 128 --lr 0.001
```

Or use the training script directly:

```bash
python train.py
```

### Evaluating the Model

Evaluate a trained model:

```bash
python main.py --mode evaluate --checkpoint ./checkpoints/best_model.pth
```

Or use the evaluation script directly:

```bash
python evaluate.py
```

### Full Pipeline

Run training and evaluation together:

```bash
python main.py --mode full --epochs 50
```

### Visualize Dataset

View sample images from the dataset:

```bash
python main.py --mode visualize
```

### Real-time Webcam Classification

Classify objects in real-time using your webcam (Optional feature):

```bash
python main.py --mode webcam --checkpoint ./checkpoints/best_model.pth
```

Or use the webcam script directly:

```bash
python webcam_classification.py --checkpoint ./checkpoints/best_model.pth
```

**Features:**
- Live webcam feed with real-time predictions
- Top-3 predictions displayed
- Confidence scores
- FPS counter
- Press 'q' to quit

### Classify Video File

Classify objects in a video file:

```bash
python webcam_classification.py --mode video --video path/to/video.mp4 --checkpoint ./checkpoints/best_model.pth --output output.mp4
```

### Compare Model Architectures

Compare different CNN architectures:

```bash
python main.py --mode compare
```

This will train and compare a simple CNN vs. the advanced CNN architecture.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- 50,000 training images
- 10,000 test images

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset will be automatically downloaded on first run.

## Model Architecture

The CNN architecture includes:
- **3 Convolutional Blocks**: Each with 2 conv layers, batch normalization, ReLU activation, max pooling, and dropout
- **2 Fully Connected Layers**: With batch normalization and dropout for regularization
- **Total Parameters**: ~1.2M trainable parameters

## Output Files

After training and evaluation, the following files will be generated:

- `./checkpoints/best_model.pth`: Best model checkpoint (highest validation accuracy)
- `./checkpoints/checkpoint_epoch_X.pth`: Periodic checkpoints every 10 epochs
- `./results/confusion_matrix.png`: Confusion matrix visualization
- `./results/sample_predictions.png`: Sample predictions with true/predicted labels
- `./results/training_history.png`: Training loss and accuracy curves
- `./results/model_comparison.png`: Model architecture comparison (if using compare mode)

## Performance Metrics

The evaluation script computes:
- Overall Accuracy
- Per-class Precision, Recall, and F1-score
- Macro-averaged metrics
- Confusion Matrix

## System Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but highly recommended)
- **Storage**: ~200MB for dataset and checkpoints

## Notes

- Training on CPU will be significantly slower than GPU
- The model uses data augmentation to improve generalization
- Best model is automatically saved based on validation accuracy
- All visualizations are saved in the `./results/` directory

## Documentation

### Technical Report

A 500-word technical report is included in `TECHNICAL_REPORT.md` covering:
- System design and architecture
- Object detection and classification techniques
- System performance evaluation
- Challenges encountered
- Potential improvements

### Assessment Checklist

See `ASSESSMENT_CHECKLIST.md` for a complete verification of all assignment requirements.

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size (e.g., `--batch_size 64`)
2. **Dataset download issues**: Check internet connection, dataset downloads automatically
3. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
4. **Webcam not working**: Ensure webcam is connected and not being used by another application
5. **Video classification errors**: Ensure video codec is supported (MP4 recommended)

## Assessment Requirements

This project meets all requirements for the Computer Vision and AI module (CMS22202):

✅ **Core Features:**
- Dataset selection (CIFAR-10)
- CNN architecture design
- Multi-class classification
- Model evaluation with metrics
- Optional: Real-time webcam classification

✅ **Submission Requirements:**
- Fully commented, runnable codebase
- 500-word technical report
- Setup instructions
- Ready for presentation and demo

✅ **Learning Outcomes:**
- LO_01: Evaluate AI principles and modern technologies
- LO_02: Develop AI-based system using suitable tools
- LO_03: Reflect on experience and potential uses
- LO_04: Evaluate various ML approaches

## License

This project is for educational purposes as part of the Computer Vision and AI module (CMS22202).


