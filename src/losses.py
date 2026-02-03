"""Custom loss functions for brain tumor segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class WeightedDiceLoss(nn.Module):
    """Weighted Dice Loss for multiple tumor classes.
    
    Weights:
    - TC (Tumor Core): 0.4
    - WT (Whole Tumor): 0.4
    - ET (Enhancing Tumor): 0.2
    """
    
    def __init__(self, weights=None, sigmoid=True, reduction='mean'):
        super().__init__()
        if weights is None:
            weights = [0.4, 0.4, 0.2]  # TC, WT, ET
        self.weights = torch.tensor(weights)
        self.dice_loss = DiceLoss(
            sigmoid=sigmoid,
            to_onehot_y=False,
            squared_pred=True,
            reduction=reduction
        )
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - binary targets
        """
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)
        
        # Compute dice for each channel
        loss = 0.0
        for i in range(pred.shape[1]):
            dice = self.dice_loss(pred[:, i:i+1], target[:, i:i+1])
            loss += self.weights[i] * dice
        
        return loss


class CombinedLoss(nn.Module):
    """Combined Dice + Focal Loss for robust training.
    
    Total Loss = alpha * DiceLoss + (1-alpha) * FocalLoss
    """
    
    def __init__(self, alpha=0.7, dice_weight=None, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.alpha = alpha
        self.weighted_dice = WeightedDiceLoss(weights=dice_weight)
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            sigmoid=True,
            reduction='mean'
        )
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) - logits
            target: (B, C, H, W, D) - binary targets
        """
        dice = self.weighted_dice(pred, target)
        focal = self.focal_loss(pred, target)
        
        combined = self.alpha * dice + (1 - self.alpha) * focal
        return combined


class MultiScaleLoss(nn.Module):
    """Loss from both coarse (large branch) and refined (small branch) outputs.
    
    Useful for LSNet to supervise both branches.
    """
    
    def __init__(self, coarse_weight=0.4, refined_weight=0.6):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.refined_weight = refined_weight
        self.criterion = CombinedLoss()
    
    def forward(self, coarse_logits, refined_logits, final_logits, target):
        """
        Args:
            coarse_logits: (B, C, H, W, D) - from large branch
            refined_logits: (B, C, H, W, D) - from small branch  
            final_logits: (B, C, H, W, D) - final fused output
            target: (B, C, H, W, D) - binary targets
        """
        loss_coarse = self.criterion(coarse_logits, target)
        loss_final = self.criterion(final_logits, target)
        
        # Weighted combination
        total_loss = (
            self.coarse_weight * loss_coarse +
            self.refined_weight * loss_final
        )
        
        return total_loss


def get_loss_fn(loss_type='combined', **kwargs):
    """Factory function to get loss function.
    
    Args:
        loss_type: 'weighted_dice', 'focal', 'combined', 'multiscale'
        **kwargs: additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'weighted_dice':
        return WeightedDiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(sigmoid=True, **kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'multiscale':
        return MultiScaleLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
