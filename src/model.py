import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, Dict, Any


class FluidSegmentationModel(nn.Module):
    """U-Net model for fluid segmentation using segmentation_models_pytorch"""
    
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None
    ):
        """
        Initialize the segmentation model
        
        Args:
            encoder_name: Name of the encoder (backbone) network
            encoder_weights: Pre-trained weights for encoder
            in_channels: Number of input channels (3 for RGB)
            classes: Number of output classes (1 for binary segmentation)
            activation: Activation function for output ('sigmoid', 'softmax', None)
        """
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        dice_score = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        
        return 1 - dice_score


class CombinedLoss(nn.Module):
    """Combined Dice + BCE Loss"""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to predictions for Dice loss
        y_pred_sigmoid = torch.sigmoid(y_pred)
        
        dice = self.dice_loss(y_pred_sigmoid, y_true)
        bce = self.bce_loss(y_pred, y_true)
        
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate segmentation metrics"""
    
    # Apply sigmoid and threshold
    y_pred = torch.sigmoid(y_pred)
    y_pred_binary = (y_pred > threshold).float()
    
    # Flatten tensors
    y_pred_flat = y_pred_binary.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate metrics
    intersection = (y_pred_flat * y_true_flat).sum().item()
    union = y_pred_flat.sum().item() + y_true_flat.sum().item() - intersection
    
    # Dice coefficient
    dice = (2. * intersection) / (y_pred_flat.sum().item() + y_true_flat.sum().item() + 1e-6)
    
    # IoU (Jaccard index)
    iou = intersection / (union + 1e-6)
    
    # Pixel accuracy
    correct_pixels = (y_pred_flat == y_true_flat).sum().item()
    total_pixels = y_true_flat.numel()
    pixel_accuracy = correct_pixels / total_pixels
    
    return {
        'dice': dice,
        'iou': iou,
        'pixel_accuracy': pixel_accuracy
    }


def create_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    pretrained: bool = True
) -> FluidSegmentationModel:
    """
    Create a U-Net model for fluid segmentation
    
    Args:
        encoder_name: Encoder backbone ('resnet34', 'resnet50', 'efficientnet-b0', etc.)
        encoder_weights: Pre-trained weights ('imagenet', None)
        pretrained: Whether to use pre-trained weights
    
    Returns:
        FluidSegmentationModel instance
    """
    if not pretrained:
        encoder_weights = None
    
    model = FluidSegmentationModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,  # Binary segmentation
        activation=None  # We'll apply sigmoid in loss function
    )
    
    return model


# Available encoder options (popular choices)
AVAILABLE_ENCODERS = {
    'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
    'efficientnet': ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3'],
    'densenet': ['densenet121', 'densenet161', 'densenet169'],
    'vgg': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
    'mobilenet': ['mobilenet_v2'],
}


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(encoder_name="resnet34")
    model = model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test loss calculation
    dummy_target = torch.randn(2, 1, 512, 512).to(device)
    
    criterion = CombinedLoss()
    loss = criterion(output, dummy_target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test metrics
    metrics = calculate_metrics(output, dummy_target)
    print(f"Metrics: {metrics}")
