#!/usr/bin/env python3
"""
YOLO Object Detection Inference Script
"""

import argparse
import sys
import os
from ultralytics import YOLO


def run_detection(model_path, source, conf, save_results):
    print(f"=========================================")
    print(f"Running YOLO Inference")
    print(f"Model: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence Threshold: {conf}")
    print(f"Save Results: {save_results}")
    print(f"=========================================")

    if not os.path.exists(source) and not source.startswith(("http://", "https://")):
        print(f"Error: Source file '{source}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        # Load model
        model = YOLO(model_path)

        # Predict
        results = model.predict(
            source=source,
            conf=conf,
            save=save_results,
            project="runs/detect",
            name="predict_experiment"
        )

        print("\nPrediction completed successfully!")
        for res in results:
            print(f"Found {len(res.boxes)} object(s).")
            if save_results:
                print(f"Visual results saved to: {res.save_dir}")
                
        return results

    except Exception as e:
        print(f"\nError during inference: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection using a YOLO model.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO weights (.pt) or exported model file"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video, or URL to run inference on"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Inference confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated prediction outputs to runs/detect"
    )

    args = parser.parse_args()
    run_detection(args.model, args.source, args.conf, args.save)
