"""Baseline SegResNet model for BraTS."""

from monai.networks.nets import SegResNet


def get_model():
    """
    SegResNet for BraTS (baseline):
    - 4 input modalities (FLAIR, T1, T1c, T2)
    - 3 output channels (TC: Tumor Core, WT: Whole Tumor, ET: Enhancing Tumor)
    
    Architecture:
    - init_filters=8: Same as LSNet Large branch for fair comparison
    - 4 residual blocks with progressive downsampling
    - Dropout for regularization
    - Instance normalization for stability
    
    Parameters: ~27M
    """
    return SegResNet(
        in_channels=4,
        out_channels=3,
        init_filters=8,      # Same as LSNet Large branch
        out_channels_act='sigmoid',  # Sigmoid for multi-label (TC, WT, ET are independent)
        dropout_prob=0.2,
        norm='instance',     # Use instance norm (consistent with LSNet)
    )
