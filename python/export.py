#!/usr/bin/env python3
"""
YOLO Model Exporting Script
"""

import argparse
import sys
from ultralytics import YOLO


def export_yolo(model_path, export_format):
    print(f"=========================================")
    print(f"Starting YOLO Model Export")
    print(f"Source Weights: {model_path}")
    print(f"Export Format: {export_format}")
    print(f"=========================================")

    try:
        # Load the PyTorch model weights
        model = YOLO(model_path)

        # Export the model
        print("Exporting model (this might download dependencies)...")
        exported_path = model.export(format=export_format)
        
        print("\nExport completed successfully!")
        print(f"Exported model path: {exported_path}")
        return exported_path

    except Exception as e:
        print(f"\nError during export: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained YOLO model to other formats.")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train_experiment/weights/best.pt",
        help="Path to trained PyTorch weights file (.pt) to export"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Export format (e.g., onnx, tflite, engine, openvino) (default: onnx)"
    )

    args = parser.parse_args()
    export_yolo(args.model, args.format)
