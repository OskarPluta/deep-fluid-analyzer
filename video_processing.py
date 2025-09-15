import sys
from pathlib import Path
import json
import time
from typing import Optional, Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import get_validation_transforms
from model import create_model


class VideoProcessor:
    """Process entire videos and save frames + masks efficiently"""
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None, batch_size: int = 8):
        """
        Initialize the video processor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            batch_size: Number of frames to process simultaneously
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get encoder from checkpoint or use default
        encoder_name = checkpoint.get('encoder_name', 'resnet34')
        self.model = create_model(encoder_name=encoder_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_validation_transforms()
        
        print(f"Model loaded on {self.device}")
        if 'best_dice' in checkpoint:
            print(f"Model Dice score: {checkpoint['best_dice']:.4f}")
    
    def predict_frame(self, frame: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mask for a single frame
        
        Args:
            frame: Input frame as numpy array (H, W, 3) in RGB
            threshold: Threshold for binary segmentation
            
        Returns:
            tuple: (binary_mask, confidence_map) both as numpy arrays
        """
        # Apply transforms
        transformed = self.transform(image=frame)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Get probability map and binary mask
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        return binary_mask, prob_map
    
    def save_video_complete(
        self,
        video_path: str,
        output_file: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_step: int = 1,
        threshold: float = 0.5,
        include_confidence: bool = True,
        compression_level: int = 6
    ) -> Dict:
        """
        Save video segment with only masks and metadata (no frames) for efficiency
        
        Args:
            video_path: Path to input video
            output_file: Path for output .npz file
            start_frame: Frame to start from
            end_frame: Frame to end at (None = end of video)
            frame_step: Process every Nth frame (1 = all frames)
            threshold: Segmentation threshold
            include_confidence: Whether to save confidence maps
            compression_level: Compression level (0-9, higher = more compression)
            
        Returns:
            dict: Processing metadata and statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        end_frame = end_frame or total_frames
        frame_indices = list(range(start_frame, min(end_frame, total_frames), frame_step))
        
        print(f"Processing {len(frame_indices)} frames from {video_path}")
        print(f"Frame range: {start_frame} to {end_frame-1} (step: {frame_step})")
        
        # Prepare storage arrays (no frames_list to save memory)
        masks_list = []
        confidences_list = [] if include_confidence else None
        actual_frame_indices = []
        
        # Statistics
        stats = {
            'frames_processed': 0,
            'frames_with_fluid': 0,
            'total_fluid_pixels': 0,
            'processing_times': [],
            'file_size_mb': 0
        }
        
        # Process frames
        for frame_idx in tqdm(frame_indices, desc="Processing video frames"):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx}")
                continue
            
            start_time = time.time()
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate mask
            binary_mask, confidence_map = self.predict_frame(frame_rgb, threshold)
            
            processing_time = time.time() - start_time
            
            # Store data (only masks, not frames)
            masks_list.append(binary_mask)
            if include_confidence:
                confidences_list.append(confidence_map)
            actual_frame_indices.append(frame_idx)
            
            # Update statistics
            fluid_pixels = np.sum(binary_mask)
            stats['processing_times'].append(processing_time)
            stats['total_fluid_pixels'] += fluid_pixels
            if fluid_pixels > 0:
                stats['frames_with_fluid'] += 1
            stats['frames_processed'] += 1
        
        cap.release()
        
        if not masks_list:
            raise ValueError("No frames were successfully processed")
        
        # Convert to numpy arrays
        print("Converting to numpy arrays...")
        masks_array = np.array(masks_list, dtype=np.uint8)    # (N, H, W)
        frame_indices_array = np.array(actual_frame_indices, dtype=np.int32)
        
        # Prepare data dictionary (without frames)
        # Store metadata as individual fields rather than as an object
        save_data = {
            'masks': masks_array,
            'frame_indices': frame_indices_array,
            'threshold': threshold,
            'original_fps': fps,
            'original_total_frames': total_frames,
            'original_width': width,
            'original_height': height,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_step': frame_step,
            'video_path': str(video_path)
        }
        
        if include_confidence:
            confidences_array = np.array(confidences_list, dtype=np.float32)  # (N, H, W)
            save_data['confidences'] = confidences_array
        
        # Save with compression
        print(f"Saving to {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if compression_level > 0:
            np.savez_compressed(output_file, **save_data)
        else:
            np.savez(output_file, **save_data)
        
        # Calculate final statistics
        file_size = Path(output_file).stat().st_size / (1024**2)  # MB
        stats.update({
            'file_size_mb': file_size,
            'avg_processing_time': np.mean(stats['processing_times']),
            'avg_fps': 1.0 / np.mean(stats['processing_times']),
            'detection_rate': stats['frames_with_fluid'] / stats['frames_processed'],
            'avg_fluid_pixels_per_frame': stats['total_fluid_pixels'] / stats['frames_processed'],
            'final_shape': masks_array.shape,
            'compression_ratio': masks_array.nbytes / (file_size * 1024**2)
        })
        
        # Save metadata
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            # Convert numpy types for JSON
            json_stats = {}
            for k, v in stats.items():
                if isinstance(v, (np.floating, np.integer)):
                    json_stats[k] = float(v)
                elif k != 'processing_times':  # Skip the list of times
                    json_stats[k] = v
            
            # Add video metadata fields
            json_stats['original_fps'] = save_data['original_fps']
            json_stats['original_total_frames'] = save_data['original_total_frames']
            json_stats['original_resolution'] = (save_data['original_width'], save_data['original_height'])
            json_stats['start_frame'] = save_data['start_frame']
            json_stats['end_frame'] = save_data['end_frame'] 
            json_stats['frame_step'] = save_data['frame_step']
            json_stats['video_path'] = save_data['video_path']
            json_stats['threshold'] = threshold
            json_stats['include_confidence'] = include_confidence
            json_stats['frames_included'] = False  # Indicate we're not storing frames
            
            json.dump(json_stats, f, indent=2)
        
        print(f"Processed data saved successfully!")
        print(f"- File size: {file_size:.1f} MB")
        print(f"- Frames processed: {stats['frames_processed']}")
        print(f"- Detection rate: {stats['detection_rate']:.1%}")
        print(f"- Average processing FPS: {stats['avg_fps']:.1f}")
        print(f"- Compression ratio: {stats['compression_ratio']:.1f}x")
        
        return stats


class VideoLoader:
    """Load and work with saved video data"""
    
    @staticmethod
    def load_video_data(file_path: str, load_original_frames: bool = False) -> Dict[str, np.ndarray]:
        """
        Load saved video data
        
        Args:
            file_path: Path to .npz file
            load_original_frames: Whether to load original frames from video file
            
        Returns:
            dict: Video data with masks, etc. (frames if requested)
        """
        print(f"Loading video data from {file_path}")
        data = np.load(file_path, allow_pickle=True)
        
        # Convert to regular dict for easier access
        result = {}
        for key in data.files:
            result[key] = data[key]
        
        print(f"Loaded video data:")
        print(f"- Masks shape: {result['masks'].shape}")
        print(f"- Frame indices: {len(result['frame_indices'])} frames")
        
        if 'confidences' in result:
            print(f"- Confidences shape: {result['confidences'].shape}")
        
        # Check if frames are stored in the file
        has_frames = 'frames' in result
        if not has_frames:
            print("Note: Original frames not stored in file")
            
            # Load original frames if requested
            if load_original_frames:
                video_path = result['video_path'] if 'video_path' in result else None
                
                if not video_path:
                    print(f"Warning: Cannot find original video path in saved data")
                    return result
                    
                print(f"Loading original frames from {video_path}")
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Cannot open original video: {video_path}")
                else:
                    frames = []
                    frame_indices = result['frame_indices']
                    
                    for idx in tqdm(frame_indices, desc="Loading original frames"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            print(f"Warning: Failed to read frame {idx}")
                            # Use blank frame as placeholder
                            height, width = result['masks'][0].shape
                            frames.append(np.zeros((height, width, 3), dtype=np.uint8))
                    
                    cap.release()
                    result['frames'] = np.array(frames)
                    print(f"- Loaded original frames shape: {result['frames'].shape}")
        else:
            print(f"- Frames shape: {result['frames'].shape}")
        
        return result
    
    @staticmethod
    def get_frame_data(video_data: Dict, frame_index: int) -> Optional[Dict]:
        """
        Get data for specific frame by index in the saved array
        
        Args:
            video_data: Data from load_video_data()
            frame_index: Index in the saved array (0 to N-1)
            
        Returns:
            dict: Frame data or None if index invalid
        """
        if frame_index < 0 or frame_index >= len(video_data['masks']):
            return None
        
        result = {
            'mask': video_data['masks'][frame_index],
            'original_frame_number': video_data['frame_indices'][frame_index]
        }
        
        # Add frame if available
        if 'frames' in video_data:
            result['frame'] = video_data['frames'][frame_index]
        
        if 'confidences' in video_data:
            result['confidence'] = video_data['confidences'][frame_index]
        
        return result
    
    @staticmethod
    def get_frame_by_original_number(video_data: Dict, original_frame_number: int) -> Optional[Dict]:
        """
        Get data for specific frame by original video frame number
        
        Args:
            video_data: Data from load_video_data()
            original_frame_number: Original frame number from video
            
        Returns:
            dict: Frame data or None if frame not found
        """
        try:
            idx = np.where(video_data['frame_indices'] == original_frame_number)[0][0]
            return VideoLoader.get_frame_data(video_data, idx)
        except IndexError:
            return None
    
    @staticmethod
    def get_frames_range(video_data: Dict, start_idx: int = 0, end_idx: Optional[int] = None) -> Dict:
        """
        Get a range of frames
        
        Args:
            video_data: Data from load_video_data()
            start_idx: Start index
            end_idx: End index (None = to end)
            
        Returns:
            dict: Subset of video data
        """
        end_idx = end_idx or len(video_data['masks'])
        
        result = {
            'masks': video_data['masks'][start_idx:end_idx],
            'frame_indices': video_data['frame_indices'][start_idx:end_idx]
        }
        
        # Add frames if available
        if 'frames' in video_data:
            result['frames'] = video_data['frames'][start_idx:end_idx]
        
        if 'confidences' in video_data:
            result['confidences'] = video_data['confidences'][start_idx:end_idx]
        
        return result
    
    @staticmethod
    def calculate_statistics(video_data: Dict) -> Dict:
        """Calculate statistics from loaded video data"""
        masks = video_data['masks']
        frames_with_fluid = np.sum(np.any(masks.reshape(len(masks), -1), axis=1))
        total_fluid_pixels = np.sum(masks)
        
        stats = {
            'total_frames': len(masks),
            'frames_with_fluid': int(frames_with_fluid),
            'detection_rate': frames_with_fluid / len(masks),
            'total_fluid_pixels': int(total_fluid_pixels),
            'avg_fluid_pixels_per_frame': total_fluid_pixels / len(masks),
            'frame_shape': masks[0].shape,
            'data_size_mb': sum(arr.nbytes for arr in video_data.values()) / (1024**2)
        }
        
        if 'confidences' in video_data:
            confidences = video_data['confidences']
            # Only calculate confidence stats where masks > 0
            fluid_confidences = confidences[masks > 0]
            if len(fluid_confidences) > 0:
                stats.update({
                    'avg_confidence': float(np.mean(fluid_confidences)),
                    'max_confidence': float(np.max(fluid_confidences)),
                    'min_confidence': float(np.min(fluid_confidences))
                })
        
        return stats


# Convenience functions
def save_video_segment(
    model_path: str, 
    video_path: str, 
    output_file: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    **kwargs
) -> Dict:
    """Convenience function to save a video segment"""
    processor = VideoProcessor(model_path)
    return processor.save_video_complete(
        video_path, output_file, start_frame, end_frame, **kwargs
    )


def load_and_analyze_video(file_path: str) -> Tuple[Dict, Dict]:
    """Convenience function to load video and get statistics"""
    video_data = VideoLoader.load_video_data(file_path)
    stats = VideoLoader.calculate_statistics(video_data)
    return video_data, stats


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/fluid_segmentation_20250808_180354/best_model.pth"
    video_path = "video/problematyczny1.MP4"

    save_video_segment(
            model_path=model_path,
            video_path=video_path,
            output_file="saved_videos/video_masks_only.npz",
            start_frame=0,
            end_frame=None,  # Entire video
            frame_step=1,
            threshold=0.5,
            include_confidence=True
    )
    