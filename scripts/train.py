import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup
from tqdm import tqdm
import time
import numpy as np
from PIL import Image

# Adjust path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import OffroadDataset, CLASS_NAMES
from utils.augmentations import get_training_augmentation, get_validation_augmentation
from utils.losses import FocalLoss, get_weighted_ce_loss

# Metrics from earlier script
from Dataset.Offroad_Segmentation_Scripts.train_segmentation import compute_iou, compute_dice, compute_pixel_accuracy, save_training_plots, save_history_to_file

# Wrapper to apply albumentations format to PIL images
class AlboDataset(OffroadDataset):
    def __getitem__(self, idx):
        # override to use albumentations
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = Image.open(mask_path)
        
        # Map raw pixel values to 0-9
        from utils.dataloader import convert_mask
        mask_arr = convert_mask(mask)

        if self.image_transform:
            augmented = self.image_transform(image=image, mask=mask_arr)
            image = augmented['image']
            mask_arr = augmented['mask']
            
        return image, torch.tensor(mask_arr, dtype=torch.long)

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
    CLASS_WEIGHTS = [3.56, 2.83, 1.68, 0.53, 9.10, 2.27, 128.34, 8.35, 0.41, 0.26]

    # Directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_dir = os.path.join(base_dir, 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(base_dir, 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(base_dir, 'models', 'segformer_b0')
    os.makedirs(output_dir, exist_ok=True)

    # ----- Data Loaders -----
    print("Initializing Datasets and Loaders...")
    train_transform = get_training_augmentation(height=HEIGHT, width=WIDTH)
    val_transform = get_validation_augmentation(height=HEIGHT, width=WIDTH)
    
    train_dataset = AlboDataset(train_dir, image_transform=train_transform)
    val_dataset = AlboDataset(val_dir, image_transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

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

        # Save Best Model Strategy
        current_val_iou = history['val_iou'][-1]
        if current_val_iou > best_iou:
            best_iou = current_val_iou
            torch.save(model.state_dict(), os.path.join(output_dir, "best_segformer.pth"))
            print(f"\n[Epoch {epoch+1}] New Best Val IoU: {best_iou:.4f}. Model saved.")

    # Save final plotted results wrapper
    print("\nSaving final metrics and plots...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)
    
    # Save last epoch model
    torch.save(model.state_dict(), os.path.join(output_dir, "last_segformer.pth"))
    print("Training Complete! Validated Models and Plots are in:", output_dir)

if __name__ == "__main__":
    main()
