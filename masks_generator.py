
import os
import sys
from pathlib import Path
import argparse

import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import get_validation_transforms
from model import create_model


def process_video_to_dataset(
    model_path: str,
    video_path: str,
    output_dir: str,
    frame_step: int = 30,
    threshold: float = 0.5,
    max_frames: int = None,
    start_frame: int = 0
):
    """
    Process video and save frames + masks in dataset format
    
    Args:
        model_path: Path to trained model checkpoint
        video_path: Path to input video
        output_dir: Output directory (will create 'images' and 'labels_binary' subdirs)
        frame_step: Extract every Nth frame (default: 30)
        threshold: Segmentation threshold (default: 0.5)
        max_frames: Maximum number of frames to extract (None = no limit)
        start_frame: Frame to start from (default: 0)
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    encoder_name = checkpoint.get('encoder_name', 'resnet34')
    model = create_model(encoder_name=encoder_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Setup transforms
    transform = get_validation_transforms()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Extracting every {frame_step} frames starting from frame {start_frame}")
    
    # Create output directories
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels_binary'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame indices to extract
    frame_indices = list(range(start_frame, total_frames, frame_step))
    if max_frames:
        frame_indices = frame_indices[:max_frames]
    
    print(f"Will extract {len(frame_indices)} frames")
    
    # Process frames
    extracted_count = 0
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames")):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            continue
        
        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms for inference
        transformed = transform(image=frame_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Generate mask
        with torch.no_grad():
            output = model(input_tensor)
        
        # Post-process
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary_mask = (prob_map > threshold).astype(np.uint8) * 255
        
        # Generate filename
        filename = f"frame_{frame_idx:06d}"
        
        # Save original frame (convert back to RGB for PIL)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil.save(images_dir / f"{filename}.png")
        
        # Save mask
        mask_pil = Image.fromarray(binary_mask, mode='L')
        mask_pil.save(labels_dir / f"{filename}_mask.png")
        
        extracted_count += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"Extracted {extracted_count} frames")
    print(f"Images saved to: {images_dir}")
    print(f"Masks saved to: {labels_dir}")
    print(f"Frame naming: frame_XXXXXX.png and frame_XXXXXX_mask.png")


def main():
    parser = argparse.ArgumentParser(description='Extract frames and masks from video using trained model')
    parser.add_argument('--model-path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--video-path', type=str, required=True, 
                       help='Path to input video')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory (will create images/ and labels_binary/ subdirs)')
    parser.add_argument('--frame-step', type=int, default=30, 
                       help='Extract every Nth frame (default: 30)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Segmentation threshold (default: 0.5)')
    parser.add_argument('--max-frames', type=int, default=None, 
                       help='Maximum number of frames to extract (default: no limit)')
    parser.add_argument('--start-frame', type=int, default=0, 
                       help='Frame to start from (default: 0)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    try:
        process_video_to_dataset(
            model_path=args.model_path,
            video_path=args.video_path,
            output_dir=args.output_dir,
            frame_step=args.frame_step,
            threshold=args.threshold,
            max_frames=args.max_frames,
            start_frame=args.start_frame
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())