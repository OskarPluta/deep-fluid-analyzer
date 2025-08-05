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
        
        # Get all frame directories
        frame_dirs = [d for d in self.frames_dir.iterdir() if d.is_dir()]
        
        for frame_dir in frame_dirs:
            mask_dir = self.masks_dir / frame_dir.name
            if not mask_dir.exists():
                print(f"Warning: No corresponding mask directory for {frame_dir.name}")
                continue
                
            # Get all frame files
            frame_files = list(frame_dir.glob("*.png"))
            frame_files.sort()
            
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
            
            # Copy files to respective directories
            for split_name, files in splits.items():
                for frame_file in files:
                    # Copy frame
                    dst_frame = self.output_dir / split_name / 'frames' / frame_file.name
                    shutil.copy2(frame_file, dst_frame)
                    
                    # Find and copy corresponding mask
                    mask_file = self._find_corresponding_mask(frame_file, mask_dir)
                    if mask_file:
                        dst_mask = self.output_dir / split_name / 'masks' / mask_file.name
                        shutil.copy2(mask_file, dst_mask)
                    else:
                        print(f"Warning: No corresponding mask found for {frame_file.name}")
        
        print(f"Dataset split completed. Output saved to {self.output_dir}")
    
    def _find_corresponding_mask(self, frame_file: Path, mask_dir: Path) -> Path:
        """Find corresponding mask file for a frame"""
        frame_name = frame_file.stem
        
        # Try direct match first
        mask_files = list(mask_dir.glob("*.png"))
        
        for mask_file in mask_files:
            mask_name = mask_file.stem
            # Extract frame number from both files
            if self._extract_frame_number(frame_name) == self._extract_frame_number(mask_name):
                return mask_file
        
        return None
    
    def _extract_frame_number(self, filename: str) -> str:
        """Extract frame number from filename"""
        # Look for patterns like _000055 or _002688
        match = re.search(r'_(\d+)', filename)
        return match.group(1) if match else ""

# Usage example
if __name__ == "__main__":
    splitter = DatasetSplitter(
        frames_dir="training_data/frames",
        masks_dir="training_data/masks",
        output_dir="dataset"
    )
    
    # Split into 70% train, 20% validation, 10% test
    splitter.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)