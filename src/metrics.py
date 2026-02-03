"""Custom metrics for evaluation."""

import torch
import numpy as np
from monai.metrics import DiceMetric
from scipy.spatial.distance import directed_hausdorff


class HausdorffDistance95Metric:
    """Hausdorff Distance 95th percentile metric.
    
    Measures the 95th percentile of distances between predicted and target boundaries.
    Lower is better. Range: [0, infinity)
    """
    
    def __init__(self, reduction='mean', spacing=None):
        self.reduction = reduction
        self.spacing = spacing  # tuple of voxel dimensions
        self.distances = []
    
    def compute_hd95(self, pred, target):
        """Compute HD95 between two binary masks.
        
        Args:
            pred: (H, W, D) binary mask
            target: (H, W, D) binary mask
            
        Returns:
            hd95 value (float)
        """
        if pred.sum() == 0 or target.sum() == 0:
            return float('nan')
        
        pred_points = torch.nonzero(pred, as_tuple=False).cpu().numpy()
        target_points = torch.nonzero(target, as_tuple=False).cpu().numpy()
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('nan')
        
        # Compute directed Hausdorff distances
        dist_pred_to_target = directed_hausdorff(pred_points, target_points)[0]
        dist_target_to_pred = directed_hausdorff(target_points, pred_points)[0]
        
        # Max gives regular Hausdorff distance
        hd = max(dist_pred_to_target, dist_target_to_pred)
        
        # 95th percentile: compute all pairwise distances
        try:
            dists = []
            for p in pred_points:
                min_dist = np.min(np.linalg.norm(target_points - p, axis=1))
                dists.append(min_dist)
            for t in target_points:
                min_dist = np.min(np.linalg.norm(pred_points - t, axis=1))
                dists.append(min_dist)
            
            hd95 = np.percentile(dists, 95)
            return hd95
        except:
            return hd  # fallback to standard Hausdorff
    
    def __call__(self, pred, target):
        """
        Args:
            pred: (B, C, H, W, D) or (C, H, W, D)
            target: (B, C, H, W, D) or (C, H, W, D)
        """
        # Add batch dim if needed
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        B, C, H, W, D = pred.shape
        
        for b in range(B):
            for c in range(C):
                p = (pred[b, c] > 0.5).float()
                t = (target[b, c] > 0.5).float()
                hd95 = self.compute_hd95(p, t)
                self.distances.append(hd95)
    
    def aggregate(self):
        """Aggregate HD95 distances."""
        if not self.distances:
            return float('nan')
        
        valid_dists = [d for d in self.distances if not np.isnan(d)]
        if not valid_dists:
            return float('nan')
        
        if self.reduction == 'mean':
            return np.mean(valid_dists)
        elif self.reduction == 'median':
            return np.median(valid_dists)
        elif self.reduction == 'max':
            return np.max(valid_dists)
        else:
            return np.mean(valid_dists)
    
    def reset(self):
        """Reset metric state."""
        self.distances = []


class SensitivitySpecificityMetric:
    """Sensitivity and Specificity per channel.
    
    Sensitivity = TP / (TP + FN)  (recall)
    Specificity = TN / (TN + FP)
    """
    
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.reset()
    
    def __call__(self, pred, target, threshold=0.5):
        """
        Args:
            pred: (B, C, H, W, D) or (C, H, W, D) - logits
            target: (B, C, H, W, D) or (C, H, W, D) - binary
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        pred_binary = (pred > threshold).float()
        
        B, C, H, W, D = pred_binary.shape
        
        for b in range(B):
            for c in range(C):
                p = pred_binary[b, c].flatten()
                t = target[b, c].flatten()
                
                tp = ((p == 1) & (t == 1)).sum().float()
                fp = ((p == 1) & (t == 0)).sum().float()
                tn = ((p == 0) & (t == 0)).sum().float()
                fn = ((p == 0) & (t == 1)).sum().float()
                
                # Sensitivity (True Positive Rate)
                sensitivity = tp / (tp + fn + 1e-10)
                
                # Specificity (True Negative Rate)
                specificity = tn / (tn + fp + 1e-10)
                
                self.sensitivities.append(float(sensitivity.item()))
                self.specificities.append(float(specificity.item()))
    
    def aggregate(self):
        """Aggregate metrics."""
        if not self.sensitivities:
            return {
                'sensitivity': float('nan'),
                'specificity': float('nan')
            }
        
        if self.reduction == 'mean':
            return {
                'sensitivity': np.mean(self.sensitivities),
                'specificity': np.mean(self.specificities)
            }
        elif self.reduction == 'median':
            return {
                'sensitivity': np.median(self.sensitivities),
                'specificity': np.median(self.specificities)
            }
        else:
            return {
                'sensitivity': np.mean(self.sensitivities),
                'specificity': np.mean(self.specificities)
            }
    
    def reset(self):
        """Reset metric state."""
        self.sensitivities = []
        self.specificities = []


class ComprehensiveMetrics:
    """Unified metrics computation."""
    
    def __init__(self):
        self.dice = DiceMetric(include_background=False, reduction='mean_batch')
        self.hd95 = HausdorffDistance95Metric()
        self.sens_spec = SensitivitySpecificityMetric()
    
    def __call__(self, pred, target):
        """Compute all metrics.
        
        Args:
            pred: (B, C, H, W, D) logits
            target: (B, C, H, W, D) binary targets
        """
        # Dice (needs post-processing)
        pred_binary = torch.sigmoid(pred)
        self.dice(y_pred=pred_binary, y=target)
        
        # HD95
        self.hd95(pred_binary, target)
        
        # Sensitivity/Specificity
        self.sens_spec(pred, target)
    
    def aggregate(self):
        """Get all aggregated metrics."""
        dice_val = self.dice.aggregate()
        hd95_val = self.hd95.aggregate()
        sens_spec_val = self.sens_spec.aggregate()
        
        return {
            'dice_tc': float(dice_val[0].item()) if dice_val.shape[0] > 0 else 0.0,
            'dice_wt': float(dice_val[1].item()) if dice_val.shape[0] > 1 else 0.0,
            'dice_et': float(dice_val[2].item()) if dice_val.shape[0] > 2 else 0.0,
            'dice_mean': float(dice_val.mean().item()),
            'hd95': hd95_val,
            'sensitivity': sens_spec_val['sensitivity'],
            'specificity': sens_spec_val['specificity'],
        }
    
    def reset(self):
        """Reset all metrics."""
        self.dice.reset()
        self.hd95.reset()
        self.sens_spec.reset()
