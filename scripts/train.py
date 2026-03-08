import os
import sys
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

# Core PyTorch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup

# Adjust path to dynamically load sibling folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom User Modules (Type Ignore suppresses IDE 'Red Line' warnings since path is appended dynamically)
from utils.dataloader import OffroadDataset, CLASS_NAMES  # type: ignore
from utils.augmentations import get_training_augmentation, get_validation_augmentation  # type: ignore
from utils.losses import FocalLoss, get_weighted_ce_loss  # type: ignore

# Metrics from earlier script
from Dataset.Offroad_Segmentation_Scripts.train_segmentation import compute_iou, compute_dice, compute_pixel_accuracy, save_training_plots, save_history_to_file  # type: ignore

class SyntheticDesertDataset(Dataset):
    """
    A mock dataset simply to satisfy the DataLoader mechanism and validate the loop architecture.
    """
    def __init__(self, size=10, image_size=(384, 640), num_classes=10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Fake Image Tensor [3, H, W]
        img = torch.randn(3, self.image_size[0], self.image_size[1])
        # Fake Semantic Label Tensor [H, W] (Values 0-9)
        labels = torch.randint(0, self.num_classes, (self.image_size[0], self.image_size[1]), dtype=torch.long)
        return img, labels

def main():
    # ----- Configuration -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 15
    LEARNING_RATE = 2e-4
    HEIGHT, WIDTH = 384, 640
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Loss config
    USE_FOCAL_LOSS = True  # Set to False to use Weighted CE
    # Weights calculated from EDA (data/class_distribution.csv)
    # Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky
    
    # 🌲 PHASE 5 TUNING FIX: The Confusion Matrix proved the mit-b0 architecture critically fails on 'Trees'.
    # We are aggressively boosting the 'Trees' weight from 2.83 -> 15.00 to force the Kaiming He Focal Loss 
    # to brutally penalize the AdamW optimizer if it misses foliage shapes.
    CLASS_WEIGHTS = [3.56, 15.00, 1.68, 0.53, 9.10, 2.27, 128.34, 8.35, 0.41, 0.26]

    # Directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_dir = os.path.join(base_dir, 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(base_dir, 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(base_dir, 'models', 'segformer_b0')
    os.makedirs(output_dir, exist_ok=True)

    # ----- Data Loaders -----
    print("Initializing Synthetic Datasets and Loaders for Verification...")
    
    train_dataset = SyntheticDesertDataset(size=8, num_classes=NUM_CLASSES)
    val_dataset = SyntheticDesertDataset(size=2, num_classes=NUM_CLASSES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ----- Model Initialization -----
    print("Loading SegFormer mit-b0 Architecture...")
    # mit-b0 is lightweight and fast (good for the <50ms constraint)
    # If we had more VRAM/compute budget, we could use mit-b3 or mit-b5
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # ----- Loss & Optimizer -----
    if USE_FOCAL_LOSS:
        print("Using Focal Loss to prioritize hard/rare classes (Logs, Rocks)")
        weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
    else:
        print("Using Weighted Cross Entropy Loss")
        criterion = get_weighted_ce_loss(CLASS_WEIGHTS, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler (Cosine with warm-up)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )

    # ----- Training History -----
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou', 
                               'train_dice', 'val_dice', 'train_pixel_acc', 'val_pixel_acc']}
    best_iou = 0.0
    patience_counter = 0

    # ----- Training Loop -----
    print("\nStarting Training...")
    epoch_pbar = tqdm(range(EPOCHS), desc="Training Phase")
    
    for epoch in epoch_pbar:
        # --- TRAIN ---
        model.train()
        train_losses = []
        train_ious, train_dices, train_accs = [], [], []
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Segformer outputs logits
            outputs = model(pixel_values=images).logits
            
            # Upsample logits to match mask size
            upsampled_logits = nn.functional.interpolate(
                outputs, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            loss = criterion(upsampled_logits, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            # Metrics
            with torch.no_grad():
                train_ious.append(compute_iou(upsampled_logits, masks, NUM_CLASSES))
                train_dices.append(compute_dice(upsampled_logits, masks, NUM_CLASSES))
                train_accs.append(compute_pixel_accuracy(upsampled_logits, masks))

        # --- VALIDATE ---
        model.eval()
        val_losses = []
        val_ious, val_dices, val_accs = [], [], []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(pixel_values=images).logits
                upsampled_logits = nn.functional.interpolate(
                    outputs, 
                    size=masks.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
                
                loss = criterion(upsampled_logits, masks)
                val_losses.append(loss.item())
                
                val_ious.append(compute_iou(upsampled_logits, masks, NUM_CLASSES))
                val_dices.append(compute_dice(upsampled_logits, masks, NUM_CLASSES))
                val_accs.append(compute_pixel_accuracy(upsampled_logits, masks))

        # --- METRICS & LOGGING ---
        # Record averages
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(np.nanmean(train_ious))
        history['val_iou'].append(np.nanmean(val_ious))
        history['train_dice'].append(np.mean(train_dices))
        history['val_dice'].append(np.mean(val_dices))
        history['train_pixel_acc'].append(np.mean(train_accs))
        history['val_pixel_acc'].append(np.mean(val_accs))

        # Update progress bar
        epoch_pbar.set_postfix(
            t_loss=f"{history['train_loss'][-1]:.3f}",
            v_loss=f"{history['val_loss'][-1]:.3f}",
            v_iou=f"{history['val_iou'][-1]:.3f}"
        )

        # Save Best Model Strategy & Early Stopping
        current_val_iou = history['val_iou'][-1]
        
        # Initialize early stopping variables globally above this loop (handled in setup)
        if current_val_iou > best_iou:
            best_iou = current_val_iou
            patience_counter = 0 # Reset patience
            torch.save(model.state_dict(), os.path.join(output_dir, "best_segformer.pth"))
            print(f"\n[Epoch {epoch+1}] New Best Val IoU: {best_iou:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"\n[Epoch {epoch+1}] Val IoU did not improve. Patience: {patience_counter}/5")
            if patience_counter >= 5:
                print("\n[Early Stopping Triggered] Model is overfitting. Halting training.")
                break

    # Save final plotted results wrapper
    print("\nSaving final metrics and plots...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)
    
    # Save last epoch model
    torch.save(model.state_dict(), os.path.join(output_dir, "last_segformer.pth"))
    print("Training Complete! Validated Models and Plots are in:", output_dir)

if __name__ == "__main__":
    main()
