import os
import random
from pathlib import Path
import shutil
from typing import Tuple, List
import re

class DatasetSplitter:
    def __init__(self, frames_dir: str, masks_dir: str, output_dir: str = "dataset"):
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Split dataset into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'frames').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Get all frame files directly from frames directory (support PNG/JPG/JPEG)
        frame_files = sorted([
            f
            for ext in ("*.png", "*.jpg", "*.jpeg")
            for f in self.frames_dir.glob(ext)
        ])
        
        print(f"Found {len(frame_files)} frame files")
        
        # Shuffle for random split
        random.shuffle(frame_files)
        
        # Calculate split indices
        total = len(frame_files)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # Split files
        splits = {
            'train': frame_files[:train_end],
            'val': frame_files[train_end:val_end],
            'test': frame_files[val_end:]
        }
        
        print(f"Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # Copy files to respective directories
        for split_name, files in splits.items():
            for frame_file in files:
                # Copy frame
                dst_frame = self.output_dir / split_name / 'frames' / frame_file.name
                shutil.copy2(frame_file, dst_frame)
                
                # Find and copy corresponding mask
                mask_file = self._find_corresponding_mask(frame_file)
                if mask_file:
                    dst_mask = self.output_dir / split_name / 'masks' / mask_file.name
                    shutil.copy2(mask_file, dst_mask)
                else:
                    print(f"Warning: No corresponding mask found for {frame_file.name}")
        
        print(f"Dataset split completed. Output saved to {self.output_dir}")
    
    def _find_corresponding_mask(self, frame_file: Path) -> Path:
        """Find corresponding mask file for a frame (supports PNG/JPG/JPEG)"""
        frame_name = frame_file.stem
        exts = (".png", ".jpg", ".jpeg")
        
        # New naming convention: frame_name + "_mask" with any extension
        for ext in exts:
            mask_file = self.masks_dir / f"{frame_name}_mask{ext}"
            if mask_file.exists():
                return mask_file
        
        # Fallback: try to find mask by extracting frame identifier
        frame_id = self._extract_frame_identifier(frame_name)
        if frame_id:
            candidates = [p for pat in ("*.png", "*.jpg", "*.jpeg") for p in self.masks_dir.glob(pat)]
            for mask_file in candidates:
                stem = mask_file.stem
                if stem == f"{frame_name}_mask" or frame_id in stem:
                    return mask_file
        
        return None
    
    def _extract_frame_identifier(self, filename: str) -> str:
        """Extract frame identifier from filename"""
        # For new format like '2beb73a8-frame_0013', extract the full identifier
        # This handles both the UUID part and frame number
        return filename

# Usage example
if __name__ == "__main__":
    splitter = DatasetSplitter(
        frames_dir="training_data/frames",
        masks_dir="training_data/masks",
        output_dir="dataset"
    )
    
    # Split into 70% train, 20% validation, 10% test
    splitter.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)