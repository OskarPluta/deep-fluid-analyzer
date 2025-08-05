import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FluidDataset(Dataset):
    """Dataset for fluid segmentation task"""
    
    def __init__(
        self,
        frames_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            frames_dir: Directory containing frame images
            masks_dir: Directory containing mask images
            transform: Optional transforms to apply
            image_size: Target size for images (height, width)
        """
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all frame files
        self.frame_files = sorted([f for f in self.frames_dir.glob("*.png")])
        
        # Verify that corresponding masks exist
        self.valid_pairs = []
        for frame_file in self.frame_files:
            mask_file = self._find_corresponding_mask(frame_file)
            if mask_file and mask_file.exists():
                self.valid_pairs.append((frame_file, mask_file))
            else:
                print(f"Warning: No corresponding mask found for {frame_file.name}")
        
        print(f"Found {len(self.valid_pairs)} valid frame-mask pairs")
    
    def _find_corresponding_mask(self, frame_file: Path) -> Optional[Path]:
        """Find corresponding mask file for a frame"""
        frame_name = frame_file.stem
        
        # Extract frame number and video name from frame filename
        # Example: nowy2_MP4_frame_001502.png -> nowy2_MP4, 001502
        match = re.match(r'(.+)_frame_(\d+)', frame_name)
        if not match:
            return None
            
        video_name, frame_number = match.groups()
        
        # Construct expected mask filename
        # Example: mask_nowy2_MP4_001502.png
        mask_name = f"mask_{video_name}_{frame_number}.png"
        mask_file = self.masks_dir / mask_name
        
        return mask_file
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_file, mask_file = self.valid_pairs[idx]
        
        # Load frame image
        frame = Image.open(frame_file).convert('RGB')
        frame = np.array(frame)
        
        # Load mask image
        mask = Image.open(mask_file).convert('L')  # Convert to grayscale
        mask = np.array(mask)
        
        # Normalize mask to 0-1 range
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=frame, mask=mask)
            frame = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has the right shape for segmentation
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return frame, mask


def get_training_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Get training transforms with data augmentation"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Get validation transforms without augmentation"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_dataloaders(
    dataset_dir: str = "dataset",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    dataset_path = Path(dataset_dir)
    
    # Create datasets
    train_dataset = FluidDataset(
        frames_dir=dataset_path / "train" / "frames",
        masks_dir=dataset_path / "train" / "masks",
        transform=get_training_transforms(image_size),
        image_size=image_size
    )
    
    val_dataset = FluidDataset(
        frames_dir=dataset_path / "val" / "frames",
        masks_dir=dataset_path / "val" / "masks",
        transform=get_validation_transforms(image_size),
        image_size=image_size
    )
    
    test_dataset = FluidDataset(
        frames_dir=dataset_path / "test" / "frames",
        masks_dir=dataset_path / "test" / "masks",
        transform=get_validation_transforms(image_size),
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for frames, masks in train_loader:
        print(f"Frame batch shape: {frames.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
        print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
        break
