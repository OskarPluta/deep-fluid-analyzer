#!/usr/bin/env python3
"""
Video inference script for fluid segmentation model
Real-time processing and visualization of video frames
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import get_validation_transforms
from model import create_model


class VideoFluidSegmentation:
    """Video inference class for fluid segmentation"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        """
        Initialize video inference pipeline
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load config to get encoder name
        import json
        import os
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        encoder_name = "resnet34"  # default fallback
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                encoder_name = config.get('encoder', 'resnet34')
                print(f"Using encoder: {encoder_name} (from config)")
        else:
            print(f"Config file not found, using default encoder: {encoder_name}")
        
        # Create model with correct encoder
        self.model = create_model(encoder_name=encoder_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_validation_transforms()
        
        print(f"Model loaded successfully on {self.device}")
        if 'best_dice' in checkpoint:
            print(f"Best validation Dice: {checkpoint['best_dice']:.4f}")
    
    def predict_frame(self, frame: np.ndarray, threshold: float = 0.5) -> tuple:
        """
        Predict segmentation mask for a single frame
        
        Args:
            frame: Input frame as numpy array (H, W, 3)
            threshold: Threshold for binary segmentation
            
        Returns:
            tuple: (predicted_mask, confidence_map)
        """
        # Apply transforms
        transformed = self.transform(image=frame)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Post-process output
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        return binary_mask, prob_map
    
    def create_visualization(
        self,
        original_frame: np.ndarray,
        predicted_mask: np.ndarray,
        confidence_map: np.ndarray,
        alpha: float = 0.3,
        display_size: tuple = (400, 300)
    ) -> np.ndarray:
        """
        Create visualization combining original frame, mask, and confidence
        
        Args:
            original_frame: Original input frame
            predicted_mask: Binary predicted mask
            confidence_map: Confidence/probability map
            alpha: Transparency for overlay
            display_size: Size for display (width, height)
            
        Returns:
            np.ndarray: Visualization image
        """
        # Resize original frame to display size
        display_w, display_h = display_size
        frame_resized = cv2.resize(original_frame, (display_w, display_h))
        
        # Resize mask and confidence to display size
        mask_resized = cv2.resize(predicted_mask.astype(np.uint8), (display_w, display_h))
        conf_resized = cv2.resize(confidence_map, (display_w, display_h))
        
        # Create overlay with red mask
        overlay = frame_resized.copy()
        red_mask = np.zeros_like(frame_resized)
        red_mask[mask_resized > 0] = [0, 0, 255]  # Red in BGR
        overlay = cv2.addWeighted(overlay, 1 - alpha, red_mask, alpha, 0)
        
        # Create confidence heatmap
        conf_colored = cv2.applyColorMap((conf_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create binary mask visualization
        mask_3ch = cv2.cvtColor((mask_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Create 2x2 grid layout
        top_row = np.hstack([frame_resized, overlay])
        bottom_row = np.hstack([mask_3ch, conf_colored])
        visualization = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Add text background for better visibility
        def add_text_with_background(img, text, pos, font, scale, color, thickness):
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            # Draw background rectangle
            cv2.rectangle(img, (pos[0] - 2, pos[1] - text_height - 2), 
                         (pos[0] + text_width + 2, pos[1] + baseline), (0, 0, 0), -1)
            # Draw text
            cv2.putText(img, text, pos, font, scale, color, thickness)
        
        # Labels for quadrants
        add_text_with_background(visualization, 'Original', (5, 20), font, font_scale, color, thickness)
        add_text_with_background(visualization, 'Overlay', (display_w + 5, 20), font, font_scale, color, thickness)
        add_text_with_background(visualization, 'Mask', (5, display_h + 20), font, font_scale, color, thickness)
        add_text_with_background(visualization, 'Confidence', (display_w + 5, display_h + 20), font, font_scale, color, thickness)
        
        return visualization
    
    def process_video_realtime(
        self,
        video_path: str,
        start_frame: int = 0,
        threshold: float = 0.5,
        save_output: bool = False,
        output_path: str = None,
        display_size: tuple = (400, 300)
    ):
        """
        Process video in real-time with live visualization
        
        Args:
            video_path: Path to input video
            start_frame: Frame number to start from
            threshold: Threshold for binary segmentation
            save_output: Whether to save output video
            output_path: Path for output video
            display_size: Size for display (width, height)
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Starting from frame: {start_frame}")
        print(f"  Display size: {display_size[0]}x{display_size[1]}")
        
        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Setup output video writer if saving
        out_writer = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Output will be 2x2 grid at display size
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (display_size[0] * 2, display_size[1] * 2))
        
        frame_count = start_frame
        processing_times = []
        
        print(f"\nStarting video processing...")
        print("Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
        print("Press '+' to increase threshold, '-' to decrease threshold")
        
        paused = False
        current_threshold = threshold
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video or failed to read frame")
                        break
                
                start_time = time.time()
                
                # Convert BGR to RGB for model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict
                mask, confidence = self.predict_frame(frame_rgb, current_threshold)
                
                # Create visualization
                visualization = self.create_visualization(
                    frame, mask, confidence, display_size=display_size
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Add frame info and threshold info
                info_text = f"Frame: {frame_count}/{total_frames} | Time: {processing_time:.3f}s | FPS: {1/processing_time:.1f}"
                threshold_text = f"Threshold: {current_threshold:.2f} | Fluid pixels: {np.sum(mask)}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                color = (255, 255, 255)
                thickness = 1
                
                # Add info at the bottom with background
                info_y = visualization.shape[0] - 25
                threshold_y = visualization.shape[0] - 8
                
                # Background for text
                cv2.rectangle(visualization, (0, info_y - 20), (visualization.shape[1], visualization.shape[0]), (0, 0, 0), -1)
                
                cv2.putText(visualization, info_text, (5, info_y), font, font_scale, color, thickness)
                cv2.putText(visualization, threshold_text, (5, threshold_y), font, font_scale, color, thickness)
                
                # Show visualization
                cv2.imshow('Fluid Segmentation - Video Processing', visualization)
                
                # Save frame if recording
                if out_writer:
                    out_writer.write(visualization)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"frame_{frame_count:06d}_segmentation.png"
                    cv2.imwrite(save_path, visualization)
                    print(f"Saved frame to {save_path}")
                elif key == ord('+') or key == ord('='):
                    current_threshold = min(1.0, current_threshold + 0.05)
                    print(f"Threshold increased to {current_threshold:.2f}")
                elif key == ord('-'):
                    current_threshold = max(0.0, current_threshold - 0.05)
                    print(f"Threshold decreased to {current_threshold:.2f}")
                
                if not paused:
                    frame_count += 1
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            if processing_times:
                avg_time = np.mean(processing_times)
                avg_fps = 1 / avg_time
                print(f"\nProcessing Statistics:")
                print(f"  Average processing time: {avg_time:.3f}s")
                print(f"  Average FPS: {avg_fps:.1f}")
                print(f"  Processed {len(processing_times)} frames")
    
    def process_video_batch(
        self,
        video_path: str,
        output_dir: str,
        start_frame: int = 0,
        end_frame: int = None,
        threshold: float = 0.5,
        save_frames: bool = True
    ):
        """
        Process video in batch mode (no real-time display)
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            start_frame: Frame number to start from
            end_frame: Frame number to end at (None for end of video)
            threshold: Threshold for binary segmentation
            save_frames: Whether to save individual frames
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = end_frame or total_frames
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        results = []
        
        print(f"Processing frames {start_frame} to {end_frame}...")
        
        for frame_idx in tqdm(range(start_frame, min(end_frame, total_frames))):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            mask, confidence = self.predict_frame(frame_rgb, threshold)
            
            # Save results if requested
            if save_frames:
                # Save original frame
                frame_path = output_path / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                
                # Save mask
                mask_path = output_path / f"mask_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                
                # Save confidence
                conf_path = output_path / f"confidence_{frame_idx:06d}.png"
                plt.imsave(conf_path, confidence, cmap='jet', vmin=0, vmax=1)
                
                # Save visualization
                visualization = self.create_visualization(frame, mask, confidence)
                viz_path = output_path / f"visualization_{frame_idx:06d}.png"
                cv2.imwrite(str(viz_path), visualization)
            
            # Store results
            results.append({
                'frame': frame_idx,
                'mask_area': np.sum(mask),
                'max_confidence': np.max(confidence),
                'mean_confidence': np.mean(confidence[mask > 0]) if np.any(mask) else 0.0
            })
        
        cap.release()
        
        # Save results summary
        with open(output_path / 'results_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch processing complete. Results saved to {output_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Video inference for fluid segmentation')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video')
    parser.add_argument('--start-frame', type=int, default=0, help='Frame number to start from')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at (for batch mode)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    parser.add_argument('--mode', type=str, choices=['realtime', 'batch'], default='realtime', 
                       help='Processing mode: realtime (live display) or batch (save to files)')
    parser.add_argument('--output-dir', type=str, default='video_output', help='Output directory for batch mode')
    parser.add_argument('--save-video', action='store_true', help='Save output video in realtime mode')
    parser.add_argument('--output-video', type=str, default='output_video.mp4', help='Output video path')
    parser.add_argument('--display-width', type=int, default=400, help='Display width for each quadrant')
    parser.add_argument('--display-height', type=int, default=300, help='Display height for each quadrant')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not Path(args.video_path).exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create video processor
    processor = VideoFluidSegmentation(args.model_path, device)
    
    display_size = (args.display_width, args.display_height)
    
    if args.mode == 'realtime':
        print("Starting real-time video processing...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Pause/Resume")
        print("  's' - Save current frame")
        print("  '+' - Increase threshold")
        print("  '-' - Decrease threshold")
        
        processor.process_video_realtime(
            video_path=args.video_path,
            start_frame=args.start_frame,
            threshold=args.threshold,
            save_output=args.save_video,
            output_path=args.output_video if args.save_video else None,
            display_size=display_size
        )
    
    elif args.mode == 'batch':
        print("Starting batch video processing...")
        results = processor.process_video_batch(
            video_path=args.video_path,
            output_dir=args.output_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            threshold=args.threshold
        )
        
        # Print summary
        if results:
            total_frames = len(results)
            frames_with_fluid = sum(1 for r in results if r['mask_area'] > 0)
            avg_confidence = np.mean([r['mean_confidence'] for r in results if r['mean_confidence'] > 0])
            
            print(f"\nProcessing Summary:")
            print(f"  Total frames processed: {total_frames}")
            print(f"  Frames with detected fluid: {frames_with_fluid}")
            print(f"  Detection rate: {frames_with_fluid/total_frames*100:.1f}%")
            print(f"  Average confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    main()
