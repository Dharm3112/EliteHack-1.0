import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss allows the model to focus on hard-to-classify and rare examples (like Logs/Rocks)
    by down-weighting the well-classified examples (like Sky/Landscape).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: [N, C, H, W] logits
        targets: [N, H, W] class indices
        """
        # Calculate Cross Entropy Loss per pixel
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none', ignore_index=self.ignore_index)
        
        # Calculate probabilities from logits
        pt = torch.exp(-ce_loss)
        
        # Focal Loss formula
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            # Create mask for valid pixels (not ignored)
            valid_mask = (targets != self.ignore_index)
            # Only mean over valid pixels
            if valid_mask.sum() > 0:
                return focal_loss[valid_mask].mean()
            return focal_loss.sum()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_weighted_ce_loss(weights_list, device, ignore_index=-100):
    """
    Returns standard CrossEntropyLoss but with class weights.
    weights_list: List of floats calculated during EDA.
    """
    if weights_list is not None:
        weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    else:
        weights = None
        
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
