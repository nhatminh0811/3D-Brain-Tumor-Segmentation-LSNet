import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    MapTransform,
)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    BraTS label conversion:
    - TC (Tumor Core)       = label 1 OR 4
    - WT (Whole Tumor)     = label > 0
    - ET (Enhancing Tumor) = label 4
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))  # TC
            result.append(d[key] > 0)                                  # WT
            result.append(d[key] == 4)                                 # ET
            d[key] = torch.stack(result, dim=0).float()
        return d


def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[96, 96, 96],
            random_size=False,
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=0,
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
    ])


def get_val_transforms():
    """Deterministic transforms for validation (no augmentations)."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "nearest"),
        ),
        # ensure validation volumes have same spatial size as training crops
        CenterSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96]),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
    ])
