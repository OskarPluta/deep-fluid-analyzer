import os
import torch
import numpy as np
import cv2 as cv
import albumentations as A

from typing import Optional, Callable, Tuple, List
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from PIL import Image

class FluidDataset(Dataset):
    """Dataset for loading images of fluid and their corresponding binary masks."""
    def __init__(self,
                 frames_dir: str,
                  masks_dir: str,
                  transform: Optional[Callable]=None
                  ):
        """
        Args:
            frames_dir (str): Directory containing the input images.
            masks_dir (str): Directory containing the binary masks.
            transform: Albumentations transformation to apply to both images and masks.
            img_size (tuple[int, int]): Desired image size (height, width) after resizing.
        """
        self.frames_dir = Path(frames_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(self.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.valid_pairs = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = self.masks_dir / f'{base_name}_mask.png'
            if mask_file.exists():
                self.valid_pairs.append((img_file, os.path.basename(mask_file)))
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs.")

    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, index) -> tuple:
        img_file, mask_file = self.valid_pairs[index]

        image = Image.open(self.frames_dir / img_file).convert("RGB")
        image = np.array(image)

        mask = Image.open(self.masks_dir / mask_file).convert("L") # L = grayscale
        mask = np.array(mask)
        mask = mask.astype(np.float32) / 255.0

        # Ensure image and mask have the same dimensions
        if image.shape[:2] != mask.shape[:2]:
            # Resize mask to match image dimensions
            mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if isinstance(mask, torch.Tensor):
            mask = (mask > 0.5).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else: 
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask)

        if not isinstance(image, torch.Tensor):
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            # Normalize similarly to validation transforms
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std

        return image, mask
    
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
    ], is_check_shapes=False)

def get_validation_transforms(image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Validation transforms: resize, normalize, to tensor."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], is_check_shapes=False)

def create_dataloaders(
    dataset_dir: str,
    batch_size: int=8,
    num_workers: int=4,
    image_size: tuple[int, int]=(512, 512)
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """"Returns train, validation and test dataloaders."""

    dataset_path = Path(dataset_dir)

    train_dataset = FluidDataset(
        frames_dir=dataset_path / 'train' / 'images',
        masks_dir=dataset_path / 'train' / 'labels',
        transform=get_training_transforms(image_size=image_size)
    )

    val_dataset = FluidDataset(
        frames_dir=dataset_path / 'val' / 'images',
        masks_dir=dataset_path / 'val' / 'labels',
        transform=get_validation_transforms(image_size=image_size)
    )

    test_dataset = FluidDataset(
        frames_dir=dataset_path / 'test' / 'images',
        masks_dir=dataset_path / 'test' / 'labels',
        transform=get_validation_transforms(image_size=image_size)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def main():
    dataset_dir = "datasets"
    batch_size = 8
    num_workers = 2
    image_size = (512, 512)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Example: Iterate through one batch
    for images, masks in train_loader:
        print(f"Image batch shape: {images.size()}")
        print(f"Mask batch shape: {masks.size()}")
        break
    
    # Get a batch for visualization
    images, masks = next(iter(train_loader))
    
    # Convert tensors back to displayable format
    def denormalize_image(tensor_img):
        # Reverse the normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor_img * std + mean
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()
    
    # Show first 4 samples
    num_samples = min(4, images.size(0))
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        img = (img * 255).astype(np.uint8)
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        
        # Convert mask to displayable format
        mask = masks[i].squeeze().numpy()
        mask = (mask * 255).astype(np.uint8)
        
        # Create side-by-side visualization
        combined = np.hstack([img_bgr, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)])
        
        # Show image
        cv.imshow(f'Sample {i+1}: Image | Mask', combined)
        cv.waitKey(0)
    
    cv.destroyAllWindows()


if __name__ == "__main__": 
    main()



