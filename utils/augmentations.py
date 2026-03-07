import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_training_augmentation(height=512, width=512):
    """
    Advanced Data Augmentation Pipeline for Desert Environments
    Includes baseline transforms, dust/glare simulation, and occlusion.
    """
    train_transform = [
        # --- 1. Baseline Transforms ---
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

        # --- 2. Desert-Specific: Dust & Glare ---
        A.OneOf([
            # Simulating harsh, bright desert sun glare
            A.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            # Simulating dust/sandstorms (low contrast, slightly blurred/hazy)
            A.Compose([
                                A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.8),
            ], p=1.0),
        ], p=0.5),

        # --- 3. Occlusion Training ---
        # CoarseDropout removes random rectangular regions from the image 
        # but leaves the mask intact (fill_value is used for the image, mask_fill_value for the mask).
        # We want the model to predict the object even if a part of it is hidden by dust/sand.
        A.CoarseDropout(
            max_holes=8, 
            max_height=height//10, 
            max_width=width//10, 
            min_holes=2, 
            fill_value=128, # Gray fill to mimic neutral occlusion
            mask_fill_value=None, # DO NOT modify the mask so the model learns to infer the occluded part
            p=0.5
        ),

        # Final normalization
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height=512, width=512):
    """
    Validation doesn't use random augmentations, only resizing and normalization.
    """
    test_transform = [
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(test_transform)


def get_visualization_augmentation(height=512, width=512):
    """
    Same as training augmentation but without Normalize and ToTensorV2 for visual plotting.
    """
    train_transform = [
        A.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.Compose([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.8),
            ], p=1.0),
        ], p=0.8),

        A.CoarseDropout(
            max_holes=8, 
            max_height=height//10, 
            max_width=width//10, 
            min_holes=2, 
            fill_value=128,
            mask_fill_value=None, 
            p=0.8
        ),
    ]
    return A.Compose(train_transform)
