import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import OffroadDataset, CLASS_NAMES
from utils.augmentations import get_visualization_augmentation

def visualize_augmentations():
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train'))
    
    # Load raw dataset
    dataset = OffroadDataset(train_dir)
    if len(dataset) == 0:
        print("No images found to test augmentation.")
        return
        
    image, mask = dataset[0] # Get first image
    
    # Apply Visualization Augmentations (No ToTensor so we can plot it)
    transform = get_visualization_augmentation(height=384, width=640)
    
    augmented = transform(image=np.array(image), mask=np.array(mask))
    aug_img = augmented['image']
    aug_mask = augmented['mask']
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Mask original
    axes[0, 1].imshow(np.array(mask), cmap='tab10', vmin=0, vmax=9)
    axes[0, 1].set_title('Original Mask')
    axes[0, 1].axis('off')
    
    # Augmented
    axes[1, 0].imshow(aug_img)
    axes[1, 0].set_title('Augmented Image (Dust/Glare + Dropout)')
    axes[1, 0].axis('off')
    
    # Augmented Mask
    axes[1, 1].imshow(aug_mask, cmap='tab10', vmin=0, vmax=9)
    axes[1, 1].set_title('Augmented Mask (Dropout Does Not Erase Mask)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmentation_sample.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved augmentation visualization to: {output_path}")

if __name__ == "__main__":
    visualize_augmentations()
