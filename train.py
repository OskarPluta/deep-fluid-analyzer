#!/usr/bin/env python3
"""
Training script for fluid segmentation model
"""

import os
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime
import json

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset import create_dataloaders
from model import create_model, CombinedLoss, DiceLoss, calculate_metrics


class Trainer:
    """Training class for fluid segmentation model"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_dir: str = "checkpoints",
        encoder_name: str = "resnet34"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.encoder_name = encoder_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_pixel_accuracy': [],
            'learning_rate': []
        }
        
        self.best_dice = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> tuple:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_pixel_acc = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)
                
                total_loss += loss.item()
                total_dice += metrics['dice']
                total_iou += metrics['iou']
                total_pixel_acc += metrics['pixel_accuracy']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice"]:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        avg_pixel_acc = total_pixel_acc / num_batches
        
        return avg_loss, avg_dice, avg_iou, avg_pixel_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch,
            'encoder_name': self.encoder_name
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Dice: {self.best_dice:.4f}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice score plot
        axes[0, 1].plot(self.history['val_dice'], label='Val Dice', color='green')
        axes[0, 1].set_title('Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU plot
        axes[1, 0].plot(self.history['val_iou'], label='Val IoU', color='orange')
        axes[1, 0].set_title('Validation IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rate'], label='Learning Rate', color='red')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, num_epochs: int, save_frequency: int = 10):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss, val_dice, val_iou, val_pixel_acc = self.validate_epoch()
            
            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            self.history['val_pixel_accuracy'].append(val_pixel_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {val_dice:.4f}")
            print(f"Val IoU: {val_iou:.4f}")
            print(f"Val Pixel Acc: {val_pixel_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check if this is the best model
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.best_epoch = epoch
            
            # Save checkpoint
            if epoch % save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot training history
            if epoch % 5 == 0:
                self.plot_training_history()
        
        # Final checkpoint and plots
        self.save_checkpoint(num_epochs)
        self.plot_training_history()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total training time: {training_time/3600:.2f} hours")
        print(f"Best Dice score: {self.best_dice:.4f} at epoch {self.best_epoch}")
        
        # Save final history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for fluid segmentation')
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder backbone')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-frequency', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda') #if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"fluid_segmentation_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config['device'] = str(device)
    config['timestamp'] = timestamp
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saving checkpoints to: {save_dir}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"Creating model with encoder: {args.encoder}")
    model = create_model(encoder_name=args.encoder)
    model = model.to(device)
    
    # Create loss function
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        encoder_name=args.encoder
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs, save_frequency=args.save_frequency)


if __name__ == "__main__":
    main()
