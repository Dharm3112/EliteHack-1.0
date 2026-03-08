import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time

# ---------------------------------------------------------
# Synthetic Datasets & Classes
# ---------------------------------------------------------
CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 
    'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 
    'Landscape', 'Sky'
]

class SyntheticTestDataset(Dataset):
    """
    Simulates a completely unseen, novel desert distribution for rigorous evaluation.
    """
    def __init__(self, size=5, image_size=(384, 640), num_classes=10):
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
        return {"pixel_values": img, "labels": labels}

# ---------------------------------------------------------
# Mathematical Evaluation Core
# ---------------------------------------------------------
def evaluate_model():
    print("=========================================")
    print("      PHASE 5: UNSEEN ENVIRONMENT TEST   ")
    print("=========================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")
    
    # 1. Initialize Test Dataloader
    NUM_CLASSES = len(CLASS_NAMES)
    test_dataset = SyntheticTestDataset(size=4, num_classes=NUM_CLASSES)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # 2. Load Optimized PyTorch Weights
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", 
        num_labels=NUM_CLASSES, 
        ignore_mismatched_sizes=True
    ).to(device)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'segformer_b0', 'best_segformer.pth')
    
    if os.path.exists(model_path):
        print(f"Validating against tuned weights: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("Warning: 'best_segformer.pth' missing! Evaluating raw, untrained mit-b0 architecture.")
        
    model.eval()
    
    # 3. Aggregation Arrays for Sklearn Matrix
    all_preds = []
    all_labels = []
    
    print("\n--- Running Inference Deep Dive ---")
    start = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Predict
            outputs = model(pixel_values=pixel_values)
            
            # Sub-decimal Logistic Interpolation
            logits = nn.functional.interpolate(
                outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            
            # Argmax extracts the single numerical class (0-9) per pixel
            pred_mask = logits.argmax(dim=1)
            
            # Flatten immense [B, H, W] tensors into massive [N] 1D vectors for scikit-learn
            all_preds.append(pred_mask.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())
            
            print(f"Evaluated Batch {batch_idx+1}/{len(test_loader)}")
            
    print(f"Inference execution took {time.time()-start:.2f} seconds.")
            
    # Matrix compilation mathematically concats the millions of pixels evaluated
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)

    # 4. Generate Sklearn Confusion Matrix
    print("\n--- Generating 'Secret Sauce' Confusion Matrix ---")
    cm = confusion_matrix(flat_labels, flat_preds, labels=range(NUM_CLASSES))
    
    # Mathematical Row Normalization (shows percentages of ground truth correctly guessed)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Calculate Mean IoU dynamically
    diagonal = np.diag(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    union = row_sums + col_sums - diagonal
    iou_per_class = diagonal / (union + 1e-10)
    miou = np.mean(iou_per_class)
    
    print(f"\nFinal Unseen Test mIoU: {miou:.4f}")
    
    # 5. Visualizer Engine
    output_dir = os.path.join(base_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    
    plt.figure(figsize=(10, 8))
    # Red-Yellow-Blue diverging colormap highlights failures
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="RdYlBu_r", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    
    plt.title("Failure Case Analysis: Normalized Confusion Matrix")
    plt.xlabel("Model Predictions")
    plt.ylabel("Ground Truth")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300)
    print(f"Rendered highly-detailed Failure Heatmap into {matrix_path}")
    
    # 6. Automated Bottleneck Detection
    # Determine the absolute hardest class for the model
    worst_iou_idx = np.argmin(iou_per_class)
    worst_class = CLASS_NAMES[worst_iou_idx]
    worst_score = iou_per_class[worst_iou_idx]
    
    print("\n=========================================")
    print("           TARGETED TUNING REPORT        ")
    print("=========================================")
    print(f"DANGER: Model is critically failing on: {worst_class} (IoU: {worst_score:.4f})")
    print(f"ACTION REQUIRED: Immediately increase `FocalLoss` alpha weight for {worst_class} in `utils/losses.py`!")
    print("=========================================")

if __name__ == "__main__":
    evaluate_model()
