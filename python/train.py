#!/usr/bin/env python3
"""
YOLO Model Training Script using Ultralytics
"""

import argparse
import sys
from ultralytics import YOLO


def train_yolo(model_name, dataset, epochs, batch_size, device):
    print(f"=========================================")
    print(f"Starting YOLO Training Workflow")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device if device else 'Auto'}")
    print(f"=========================================")

    try:
        # Load a model
        # e.g., yolov8n.pt (YOLOv8 Nano) or yolo11n.pt (YOLOv11 Nano)
        model = YOLO(model_name)

        # Train the model
        results = model.train(
            data=dataset,
            epochs=epochs,
            batch=batch_size,
            device=device if device else None,
            project="runs/detect",
            name="train_experiment"
        )
        
        print("\nTraining completed successfully!")
        print(f"Results saved to runs/detect/train_experiment")
        return results

    except Exception as e:
        print(f"\nError during training: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model using Ultralytics.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model configuration or pretrained weights (e.g., yolov8n.pt, yolo11n.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco8.yaml",
        help="Path to dataset yaml file (default: coco8.yaml - a small built-in demo dataset)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run training on (e.g., 'cpu', '0' for CUDA GPU, or '' for auto-select)"
    )

    args = parser.parse_args()
    train_yolo(args.model, args.data, args.epochs, args.batch, args.device)
