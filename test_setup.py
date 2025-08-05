#!/usr/bin/env python3
"""
Test script to verify dataset loading and model setup
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import create_dataloaders
from model import create_model


def test_dataset_loading():
    """Test dataset loading"""
    print("Testing dataset loading...")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_dir="dataset",
            batch_size=2,
            num_workers=0,  # Use 0 workers for testing
            image_size=(512, 512)
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test loading a batch
        for frames, masks in train_loader:
            print(f"✓ Successfully loaded batch")
            print(f"  Frame batch shape: {frames.shape}")
            print(f"  Mask batch shape: {masks.shape}")
            print(f"  Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
            print(f"  Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")
            break
            
        return True, (train_loader, val_loader, test_loader)
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False, None


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Using device: {device}")
        
        # Create model
        model = create_model(encoder_name="resnet34")
        model = model.to(device)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False, None


def visualize_samples(data_loaders, save_dir="test_samples"):
    """Visualize some samples from the dataset"""
    print(f"\nCreating sample visualizations...")
    
    Path(save_dir).mkdir(exist_ok=True)
    
    try:
        train_loader, val_loader, test_loader = data_loaders
        
        # Get a batch from training set
        for frames, masks in train_loader:
            # Denormalize frames for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_denorm = frames * std + mean
            frames_denorm = torch.clamp(frames_denorm, 0, 1)
            
            # Create visualization
            batch_size = min(4, frames.shape[0])
            fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
            
            for i in range(batch_size):
                # Show frame
                frame = frames_denorm[i].permute(1, 2, 0).numpy()
                axes[0, i].imshow(frame)
                axes[0, i].set_title(f'Frame {i+1}')
                axes[0, i].axis('off')
                
                # Show mask
                mask = masks[i, 0].numpy()
                axes[1, i].imshow(mask, cmap='gray')
                axes[1, i].set_title(f'Mask {i+1}')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/train_samples.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Sample visualization saved to {save_dir}/train_samples.png")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        return False


def main():
    """Main test function"""
    print("=== Deep Fluid Analyzer Test ===\n")
    
    # Test dataset loading
    dataset_success, data_loaders = test_dataset_loading()
    
    # Test model creation
    model_success, model = test_model_creation()
    
    # Create sample visualizations if dataset loaded successfully
    if dataset_success:
        viz_success = visualize_samples(data_loaders)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Dataset loading: {'✓ PASS' if dataset_success else '✗ FAIL'}")
    print(f"Model creation: {'✓ PASS' if model_success else '✗ FAIL'}")
    if dataset_success:
        print(f"Sample visualization: {'✓ PASS' if viz_success else '✗ FAIL'}")
    
    if dataset_success and model_success:
        print("\n🎉 All tests passed! You're ready to start training.")
        print("\nTo start training, run:")
        print("python train.py --epochs 50 --batch-size 8 --lr 1e-3")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        
        if not dataset_success:
            print("\nDataset loading failed. Make sure:")
            print("- The 'dataset' directory exists")
            print("- You have run the dataset_split.py script")
            print("- Frame and mask files are properly paired")
        
        if not model_success:
            print("\nModel creation failed. Make sure:")
            print("- All dependencies are installed")
            print("- PyTorch and segmentation_models_pytorch are available")


if __name__ == "__main__":
    main()
