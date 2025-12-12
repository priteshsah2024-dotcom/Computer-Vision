"""
Real-time Webcam Classification for CIFAR-10
This module provides real-time object classification using webcam feed.
Optional feature for maximum marks.
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time

from model import get_model


class WebcamClassifier:
    """
    Real-time webcam classifier for CIFAR-10 objects.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the webcam classifier.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = get_model(num_classes=10, device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded successfully!")
        
        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        print("Webcam initialized. Press 'q' to quit.")
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame for classification.
        
        Args:
            frame: OpenCV BGR frame
        
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        tensor = self.transform(pil_image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def classify_frame(self, frame):
        """
        Classify a single frame.
        
        Args:
            frame: OpenCV BGR frame
        
        Returns:
            predicted_class: Class name
            confidence: Confidence score
            all_probs: All class probabilities
        """
        # Preprocess
        tensor = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probs[0].cpu().numpy()
        
        return predicted_class, confidence_score, all_probs
    
    def draw_prediction(self, frame, predicted_class, confidence, all_probs):
        """
        Draw prediction on frame.
        
        Args:
            frame: OpenCV frame
            predicted_class: Predicted class name
            confidence: Confidence score
            all_probs: All class probabilities
        """
        # Main prediction text
        text = f"{predicted_class}: {confidence*100:.1f}%"
        color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255) if confidence > 0.3 else (0, 0, 255)
        
        # Draw main prediction
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw top 3 predictions
        top3_indices = np.argsort(all_probs)[-3:][::-1]
        y_offset = 60
        for i, idx in enumerate(top3_indices):
            class_name = self.classes[idx]
            prob = all_probs[idx] * 100
            cv2.putText(frame, f"{i+1}. {class_name}: {prob:.1f}%", 
                       (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self, show_top3=True):
        """
        Run real-time classification.
        
        Args:
            show_top3: Whether to show top 3 predictions
        """
        fps_counter = 0
        fps_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Classify frame
                predicted_class, confidence, all_probs = self.classify_frame(frame)
                
                # Draw prediction
                self.draw_prediction(frame, predicted_class, confidence, all_probs)
                
                # Calculate and display FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                
                cv2.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display instructions
                cv2.putText(frame, "Press 'q' to quit", 
                           (frame.shape[1] - 200, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('CIFAR-10 Real-time Classification', frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nStopping classification...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Webcam released.")


def classify_video(video_path, checkpoint_path, output_path=None, device='cuda'):
    """
    Classify objects in a video file.
    
    Args:
        video_path: Path to input video file
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save output video (optional)
        device: Device to run inference on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(num_classes=10, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess and classify
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            tensor = transform(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            predicted_class = classes[predicted.item()]
            confidence_score = confidence.item()
            
            # Draw prediction
            text = f"{predicted_class}: {confidence_score*100:.1f}%"
            color = (0, 255, 0) if confidence_score > 0.5 else (0, 165, 255)
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Write frame
            if out:
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
    
    finally:
        cap.release()
        if out:
            out.release()
        print(f"Video processing complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time CIFAR-10 Classification')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'video'],
                       help='Classification mode: webcam or video')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (for video mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (for video mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        classifier = WebcamClassifier(args.checkpoint)
        classifier.run()
    elif args.mode == 'video':
        if not args.video:
            print("Error: --video path required for video mode")
        else:
            classify_video(args.video, args.checkpoint, args.output)

