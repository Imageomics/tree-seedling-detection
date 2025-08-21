#!/usr/bin/env python3
"""
Tree Seedling Species Identification with FiftyOne and BioCLIP 2

This application provides a human-in-the-loop workflow for tree seedling species
identification using BioCLIP 2 for multi-level taxonomic classification.
"""

import fiftyone as fo
from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import time
import subprocess
import os
import torch


class SeedlingClassifier:
    """BioCLIP-based classifier for tree seedling taxonomic identification."""
    
    RANK_MAP = {
        'kingdom': Rank.KINGDOM,
        'phylum': Rank.PHYLUM, 
        'class': Rank.CLASS,
        'order': Rank.ORDER,
        'family': Rank.FAMILY,
        'genus': Rank.GENUS,
        'species': Rank.SPECIES
    }
    
    def __init__(self, confidence_threshold: float = 0.8, device: str = "cpu", custom_labels_file: Optional[str] = None):
        if custom_labels_file:
            # Load custom labels from CSV
            import csv
            custom_labels = []
            with open(custom_labels_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    custom_labels.append(row['label'])
            
            print(f"DEBUG: Loaded {len(custom_labels)} custom labels:")
            for i, label in enumerate(custom_labels):
                print(f"  {i}: '{label}'")
            
            self.classifier = CustomLabelsClassifier(custom_labels, device=device)
            self.use_custom_labels = True
            self.custom_labels = custom_labels
        else:
            self.classifier = TreeOfLifeClassifier(device=device)
            self.use_custom_labels = False
            self.custom_labels = None
        
        self.confidence_threshold = confidence_threshold
        self.device = device
        
    def classify_image(self, image_path: str) -> Dict[str, any]:
        """
        Classify tree seedling image across all taxonomic ranks or custom labels.
        Returns predictions with confidence scores for each rank.
        """
        if self.use_custom_labels:
            # Custom labels classification
            pred_list = self.classifier.predict(image_path)
            predictions = {'custom_labels': []}
            
            print(f"DEBUG: Got {len(pred_list)} predictions for {image_path}")
            for i, pred in enumerate(pred_list[:5]):  # Show top 5
                print(f"  {i}: {pred['classification']} ({pred['score']:.6f})")
                predictions['custom_labels'].append({
                    'label': pred['classification'],
                    'confidence': pred['score'],
                    'high_confidence': pred['score'] >= self.confidence_threshold
                })
            
            # Add remaining predictions without debug output
            for pred in pred_list[5:]:
                predictions['custom_labels'].append({
                    'label': pred['classification'],
                    'confidence': pred['score'],
                    'high_confidence': pred['score'] >= self.confidence_threshold
                })
            
            return predictions
        else:
            # Standard TreeOfLife classification
            predictions = {}
            
            for rank_name, rank_enum in self.RANK_MAP.items():
                pred_list = self.classifier.predict(image_path, rank_enum)
                if pred_list:
                    top_pred = pred_list[0]  # Get highest confidence prediction
                    predictions[rank_name] = {
                        'label': top_pred.get('species' if rank_name == 'species' else rank_name, ''),
                        'confidence': top_pred['score'],
                        'high_confidence': top_pred['score'] >= self.confidence_threshold,
                        'full_taxonomy': top_pred
                    }
                else:
                    predictions[rank_name] = {
                        'label': '',
                        'confidence': 0.0,
                        'high_confidence': False,
                        'full_taxonomy': {}
                    }
                    
            return predictions
    
    def get_most_granular_prediction(self, predictions: Dict[str, any]) -> Tuple[str, str, float]:
        """
        Returns the most granular high-confidence taxonomic prediction.
        Returns (rank, label, confidence)
        """
        if 'custom_labels' in predictions:
            # Custom labels mode - return highest confidence prediction
            if predictions['custom_labels']:
                best_pred = max(predictions['custom_labels'], key=lambda p: p['confidence'])
                return 'custom_label', best_pred['label'], best_pred['confidence']
            else:
                return 'custom_label', '', 0.0
        else:
            # Standard TreeOfLife mode
            rank_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            
            for rank in reversed(rank_order):
                if predictions[rank]['high_confidence']:
                    return rank, predictions[rank]['label'], predictions[rank]['confidence']
            
            # If no high-confidence prediction, return highest confidence overall
            best_rank = max(predictions.keys(), key=lambda r: predictions[r]['confidence'])
            return best_rank, predictions[best_rank]['label'], predictions[best_rank]['confidence']


class SeedlingDataset:
    """FiftyOne dataset management for tree seedling images."""
    
    def __init__(self, dataset_name: str = "tree_seedlings", device: str = "cpu", custom_labels_file: Optional[str] = None):
        self.dataset_name = dataset_name
        self.classifier = SeedlingClassifier(device=device, custom_labels_file=custom_labels_file)
        self._ensure_db_connection()
    
    def _ensure_db_connection(self):
        """Ensure FiftyOne database connection is working, with cleanup if needed."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Test basic FiftyOne database connection
                fo.list_datasets()
                return  # Success
            except Exception as e:
                print(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    # Clean up stale lock files
                    lock_file = os.path.expanduser("~/.fiftyone/var/lib/mongo/mongod.lock")
                    if os.path.exists(lock_file):
                        try:
                            os.remove(lock_file)
                            print("Removed stale MongoDB lock file")
                        except Exception:
                            pass
                    
                    # Wait before retry
                    time.sleep(2)
                else:
                    print("Failed to establish database connection after all retries")
                    print("Try running: fiftyone app stop && fiftyone app launch")
                    raise
        
    def create_dataset(self, image_dir: str, force_reprocess: bool = False) -> fo.Dataset:
        """Create FiftyOne dataset from image directory."""
        if fo.dataset_exists(self.dataset_name) and not force_reprocess:
            print(f"Loading existing dataset: {self.dataset_name}")
            dataset = fo.load_dataset(self.dataset_name)
            return dataset
        elif fo.dataset_exists(self.dataset_name) and force_reprocess:
            print(f"Deleting existing dataset to reprocess: {self.dataset_name}")
            fo.delete_dataset(self.dataset_name)
            
        print(f"Creating new dataset: {self.dataset_name}")
        dataset = fo.Dataset(self.dataset_name, persistent=True)
            
        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png")) + \
                     list(Path(image_dir).glob("*.jpeg"))
        
        samples = []
        for img_path in image_paths:
            sample = fo.Sample(filepath=str(img_path))
            
            # Get BioCLIP predictions
            predictions = self.classifier.classify_image(str(img_path))
            rank, label, confidence = self.classifier.get_most_granular_prediction(predictions)
            
            # Add predictions as sample fields
            sample["bioclip_predictions"] = predictions
            sample["suggested_rank"] = rank
            sample["suggested_label"] = label
            sample["suggested_confidence"] = confidence
            sample["needs_review"] = confidence < self.classifier.confidence_threshold
            # Extract filename from path
            import os
            image_name = os.path.basename(str(img_path))
            sample["image_name"] = image_name
            print(f"DEBUG: Processing {image_name}")
            
            samples.append(sample)
            
        dataset.add_samples(samples)
        return dataset
    
    def launch_app(self, dataset: fo.Dataset, port: int = 5151):
        """Launch FiftyOne app for human-in-the-loop labeling."""
        session = fo.launch_app(dataset, port=port)
        return session


def main():
    """Main application entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tree Seedling Species Identification")
    parser.add_argument("image_dir", help="Directory containing seedling images")
    parser.add_argument("--confidence", type=float, default=0.8, 
                       help="Confidence threshold for high-confidence predictions")
    parser.add_argument("--port", type=int, default=5151,
                       help="Port for FiftyOne app")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for BioCLIP inference (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--custom-labels", type=str, default=None,
                       help="Path to custom labels CSV file for constrained classification")
    parser.add_argument("--dataset-name", type=str, default="tree_seedlings",
                       help="Name for the FiftyOne dataset (default: tree_seedlings)")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing of images by deleting existing dataset")
    
    args = parser.parse_args()
    
    # Validate device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Initialize dataset
    custom_labels_file = getattr(args, 'custom_labels', None)
    seedling_dataset = SeedlingDataset(dataset_name=args.dataset_name, device=device, custom_labels_file=custom_labels_file)
    seedling_dataset.classifier.confidence_threshold = args.confidence
    
    if custom_labels_file:
        print(f"Using custom labels from: {custom_labels_file}")
    
    # Create dataset with BioCLIP predictions
    print(f"Processing images from {args.image_dir}...")
    dataset = seedling_dataset.create_dataset(args.image_dir, force_reprocess=args.force_reprocess)
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Launching FiftyOne app on port {args.port}...")
    
    # Launch FiftyOne GUI
    session = seedling_dataset.launch_app(dataset, port=args.port)
    
    print("FiftyOne app launched! Use the GUI to review and refine labels.")
    print("Press Ctrl+C to stop the application.")
    
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
    