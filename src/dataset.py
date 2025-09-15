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
import cv2 as cv

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
        
        # Get all frame files (support PNG/JPG/JPEG)
        self.frame_files = sorted([
            f
            for ext in ("*.png", "*.jpg", "*.jpeg")
            for f in self.frames_dir.glob(ext)
        ])
        
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
        """Find corresponding mask file for a frame (supports PNG/JPG/JPEG)"""
        frame_name = frame_file.stem
        allowed_exts = (".png", ".jpg", ".jpeg")
        
        # New naming convention: frame_name + "_mask" with any extension
        # Example: 2beb73a8-frame_0013.png -> 2beb73a8-frame_0013_mask.(png|jpg|jpeg)
        for ext in allowed_exts:
            mask_file = self.masks_dir / f"{frame_name}_mask{ext}"
            if mask_file.exists():
                return mask_file
        
        # Fallback: Try old naming convention for backwards compatibility
        # Example frame: nowy2_MP4_frame_001502 -> mask_nowy2_MP4_001502.(png|jpg|jpeg)
        match = re.match(r'(.+)_frame_(\d+)', frame_name)
        if match:
            video_name, frame_number = match.groups()
            for ext in allowed_exts:
                mask_file = self.masks_dir / f"mask_{video_name}_{frame_number}{ext}"
                if mask_file.exists():
                    return mask_file
        
        # As a last resort, search by full identifier containment
        candidates = [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in self.masks_dir.glob(ext)]
        for cand in candidates:
            if frame_name in cand.stem or cand.stem.endswith(frame_name + "_mask"):
                return cand
        
        return None
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_file, mask_file = self.valid_pairs[idx]
        
        # Load frame image (supports PNG/JPG/JPEG)
        frame = Image.open(frame_file).convert('RGB')
        frame = np.array(frame)
        
        # Load mask image (supports PNG/JPG/JPEG)
        mask = Image.open(mask_file).convert('L')  # grayscale
        mask = np.array(mask)
        
        # Normalize mask to 0-1 range
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=frame, mask=mask)
            frame = transformed['image']
            mask = transformed['mask']
        
        # Binarize mask robustly (handles JPEG artifacts and interpolation)
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
        else:
            mask = (mask > 0.5).astype(np.float32)
        
        # Ensure mask has the right shape for segmentation (C,H,W)
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            # Convert to tensor if transforms didn't
            mask = torch.from_numpy(mask)
        
        # Ensure frame is tensor (in case transforms is None)
        if not isinstance(frame, torch.Tensor):
            frame = frame.astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            # Normalize similarly to validation transforms
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame = (frame - mean) / std
        
        return frame, mask


def get_training_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Training transforms without rotations/flips; zoom/pan via scale/shift, plus photometric and compression noise."""
    return A.Compose([
        # Keep target size fixed
        A.Resize(height=image_size[0], width=image_size[1]),
        # Translate/scale without rotation
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=cv.BORDER_REFLECT,
            p=0.5
        ),
        # Photometric augmentations
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        # Compression artifacts - fixed for newer albumentations
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        # Noise/blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 70.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.3)),
        ], p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        # Normalize and to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Get validation transforms without augmentation"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
