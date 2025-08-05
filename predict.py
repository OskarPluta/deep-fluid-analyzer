#!/usr/bin/env python3
"""
Inference script for fluid segmentation model
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import get_validation_transforms, FluidDataset
from model import create_model, calculate_metrics


class FluidSegmentationInference:
    """Inference class for fluid segmentation"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (assuming resnet34 encoder, modify if needed)
        self.model = create_model(encoder_name="resnet34")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_validation_transforms()
        
        print(f"Model loaded successfully on {self.device}")
        if 'history' in checkpoint:
            print(f"Best validation Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")
    
    def predict_single_image(self, image_path: str, threshold: float = 0.5) -> tuple:
        """
        Predict segmentation mask for a single image
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary segmentation
            
        Returns:
            tuple: (original_image, predicted_mask, confidence_map)
        """
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        original_array = np.array(original_image)
        
        # Apply transforms
        transformed = self.transform(image=original_array)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Post-process output
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        return original_array, binary_mask, prob_map
    
    def predict_batch(self, image_paths: list, threshold: float = 0.5) -> list:
        """
        Predict segmentation masks for multiple images
        
        Args:
            image_paths: List of image paths
            threshold: Threshold for binary segmentation
            
        Returns:
            list: List of (original_image, predicted_mask, confidence_map) tuples
        """
        results = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = self.predict_single_image(image_path, threshold)
            results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, dataset_dir: str, split: str = "test") -> dict:
        """
        Evaluate model on a dataset split
        
        Args:
            dataset_dir: Path to dataset directory
            split: Dataset split to evaluate ('train', 'val', 'test')
            
        Returns:
            dict: Evaluation metrics
        """
        # Create dataset
        dataset = FluidDataset(
            frames_dir=Path(dataset_dir) / split / "frames",
            masks_dir=Path(dataset_dir) / split / "masks",
            transform=self.transform
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=4
        )
        
        # Evaluate
        total_dice = 0.0
        total_iou = 0.0
        total_pixel_acc = 0.0
        num_batches = 0
        
        print(f"Evaluating on {split} set...")
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)
                
                total_dice += metrics['dice']
                total_iou += metrics['iou']
                total_pixel_acc += metrics['pixel_accuracy']
                num_batches += 1
        
        # Average metrics
        avg_metrics = {
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches,
            'pixel_accuracy': total_pixel_acc / num_batches,
            'num_samples': len(dataset)
        }
        
        return avg_metrics
    
    def visualize_prediction(
        self, 
        original_image: np.ndarray,
        predicted_mask: np.ndarray,
        confidence_map: np.ndarray,
        ground_truth: np.ndarray = None,
        save_path: str = None
    ):
        """
        Visualize prediction results
        
        Args:
            original_image: Original input image
            predicted_mask: Binary predicted mask
            confidence_map: Confidence/probability map
            ground_truth: Ground truth mask (optional)
            save_path: Path to save visualization (optional)
        """
        # Setup subplot configuration
        num_plots = 3 if ground_truth is None else 4
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        if num_plots == 3:
            axes = [axes[0], axes[1], axes[2]]
        else:
            axes = [axes[0], axes[1], axes[2], axes[3]]
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(predicted_mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Confidence map
        im = axes[2].imshow(confidence_map, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Confidence Map')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Ground truth (if provided)
        if ground_truth is not None:
            axes[3].imshow(ground_truth, cmap='gray')
            axes[3].set_title('Ground Truth')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_overlay(
        self,
        original_image: np.ndarray,
        predicted_mask: np.ndarray,
        alpha: float = 0.3,
        color: tuple = (255, 0, 0)  # Red
    ) -> np.ndarray:
        """
        Create overlay of mask on original image
        
        Args:
            original_image: Original input image
            predicted_mask: Binary predicted mask
            alpha: Transparency of overlay
            color: Color of overlay (RGB)
            
        Returns:
            np.ndarray: Overlay image
        """
        overlay = original_image.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[predicted_mask > 0] = color
        
        # Blend images
        overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay


def main():
    parser = argparse.ArgumentParser(description='Run inference on fluid segmentation model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Output directory for predictions')
    parser.add_argument('--dataset-dir', type=str, help='Dataset directory for evaluation')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on dataset')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create inference pipeline
    inference = FluidSegmentationInference(args.model_path, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Evaluate on dataset if requested
    if args.evaluate and args.dataset_dir:
        print(f"\nEvaluating on {args.split} set...")
        metrics = inference.evaluate_on_dataset(args.dataset_dir, args.split)
        
        print(f"\nEvaluation Results:")
        print(f"Dice Score: {metrics['dice']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Number of samples: {metrics['num_samples']}")
        
        # Save metrics
        with open(output_dir / f'{args.split}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Process input images if provided
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            image_paths = [str(input_path)]
        elif input_path.is_dir():
            # Directory of images
            image_paths = [str(f) for f in input_path.glob("*.png")]
            image_paths.extend([str(f) for f in input_path.glob("*.jpg")])
            image_paths.extend([str(f) for f in input_path.glob("*.jpeg")])
        else:
            raise ValueError(f"Input path {input_path} does not exist")
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        # Process images
        results = inference.predict_batch(image_paths, args.threshold)
        
        # Save results
        for i, (image_path, (original, mask, confidence)) in enumerate(zip(image_paths, results)):
            image_name = Path(image_path).stem
            
            # Save mask
            mask_path = output_dir / f"{image_name}_mask.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            
            # Save confidence map
            conf_path = output_dir / f"{image_name}_confidence.png"
            plt.imsave(conf_path, confidence, cmap='jet', vmin=0, vmax=1)
            
            # Create and save overlay
            overlay = inference.create_overlay(original, mask)
            overlay_path = output_dir / f"{image_name}_overlay.png"
            Image.fromarray(overlay).save(overlay_path)
            
            # Create visualization if requested
            if args.visualize:
                viz_path = output_dir / f"{image_name}_visualization.png"
                inference.visualize_prediction(
                    original, mask, confidence, save_path=viz_path
                )
        
        print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
