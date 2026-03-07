import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", 
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

class OffroadDataset(Dataset):
    def __init__(self, data_dir, image_transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        # Verify directories exist
        if not os.path.exists(self.image_dir) or not os.path.exists(self.masks_dir):
            self.data_ids = []
            print(f"Warning: {data_dir} does not contain Color_Images and Segmentation folders.")
        else:
            self.data_ids = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png') or f.endswith('.jpg')])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Convert mask to our 0-9 class IDs
        mask_arr = convert_mask(mask)
        mask = Image.fromarray(mask_arr)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            # if using standard ToTensor, it scales to [0, 1]. Multiply by 255 to get indices back
            if isinstance(mask, torch.Tensor):
                mask = (mask * 255).long()

        return image, mask
