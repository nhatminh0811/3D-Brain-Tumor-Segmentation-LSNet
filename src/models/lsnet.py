import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNet
from typing import Sequence, Tuple
from typing import Optional


def _center_crop_indices(full: Sequence[int], crop: Sequence[int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    # returns slices for z,y,x ordering (we keep H,W,D ordering used in code)
    assert len(full) == len(crop) == 3
    # ensure integer sizes and clamp crop to full dimension
    crop = [int(c) for c in crop]
    starts = [max(0, (f - c) // 2) for f, c in zip(full, crop)]
    ends = [min(f, starts_i + c) for f, starts_i, c in zip(full, starts, crop)]
    return (starts[0], ends[0]), (starts[1], ends[1]), (starts[2], ends[2])


def _center_crop(t: torch.Tensor, crop_size: Sequence[int]) -> torch.Tensor:
    # t: (B, C, H, W, D)
    _, _, H, W, D = t.shape
    # ensure crop_size are ints and not larger than tensor dims
    crop_size = [int(c) for c in crop_size]
    crop_size = (min(crop_size[0], H), min(crop_size[1], W), min(crop_size[2], D))
    (zs, ze), (ys, ye), (xs, xe) = _center_crop_indices((H, W, D), crop_size)
    return t[:, :, zs:ze, ys:ye, xs:xe]


def _place_crop(full: torch.Tensor, crop: torch.Tensor, crop_size: Sequence[int]):
    # place crop into full tensor (in-place on a copy)
    out = full.clone()
    _, _, H, W, D = full.shape
    crop_size = [int(c) for c in crop_size]
    crop_size = (min(crop_size[0], H), min(crop_size[1], W), min(crop_size[2], D))
    (zs, ze), (ys, ye), (xs, xe) = _center_crop_indices((H, W, D), crop_size)
    out[:, :, zs:ze, ys:ye, xs:xe] = crop
    return out


def _bbox_from_mask(mask: torch.Tensor) -> Optional[Tuple[tuple, tuple, tuple]]:
    # mask: (B, H, W, D) or (H, W, D)
    single = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        single = True
    B, H, W, D = mask.shape
    bboxes = []
    for i in range(B):
        nz = torch.nonzero(mask[i], as_tuple=False)
        if nz.numel() == 0:
            bboxes.append(None)
            continue
        zs, ys, xs = nz.min(dim=0).values
        ze, ye, xe = nz.max(dim=0).values + 1
        bboxes.append(((int(zs), int(ze)), (int(ys), int(ye)), (int(xs), int(xe))))
    if single:
        return bboxes[0]
    return bboxes


def _center_of_mass(prob: torch.Tensor) -> tuple:
    # prob: (H, W, D) or (B, H, W, D) -> returns (z,y,x) or list per batch
    single = False
    if prob.dim() == 3:
        prob = prob.unsqueeze(0)
        single = True
    B, H, W, D = prob.shape
    coords = []
    grid_z = torch.arange(H, dtype=prob.dtype, device=prob.device)
    grid_y = torch.arange(W, dtype=prob.dtype, device=prob.device)
    grid_x = torch.arange(D, dtype=prob.dtype, device=prob.device)
    for i in range(B):
        p = prob[i]
        s = p.sum()
        if s.item() == 0:
            coords.append((H // 2, W // 2, D // 2))
            continue
        cz = (p.sum(dim=(1, 2)) * grid_z).sum() / s
        cy = (p.sum(dim=(0, 2)) * grid_y).sum() / s
        cx = (p.sum(dim=(0, 1)) * grid_x).sum() / s
        coords.append((int(cz.item()), int(cy.item()), int(cx.item())))
    if single:
        return coords[0]
    return coords


class LSNet(nn.Module):
    """Approximate LSNet: whole-volume "large" branch + focus "small" branch.

    Behavior implemented:
    - large branch processes the entire volume -> coarse logits
    - compute center crop of the input volume and corresponding coarse probs
    - small branch takes (crop_image || crop_coarse_prob) as input and predicts
      refined logits for the crop region
    - produce final logits by averaging coarse logits and the refined crop
      (placed back into full volume)

    This replicates the See Large / Focus Small idea (coarse-to-fine refinement)
    while staying compatible with the existing SegResNet-based pipeline.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3, crop_size_fraction: float = 0.35):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.crop_size_fraction = float(crop_size_fraction)
        # parameters for dynamic cropping and guidance
        self.crop_threshold = 0.3
        self.crop_padding = 8  # voxels to pad bbox
        self.top_k = 1

        # large branch: stronger context encoder
        self.large = SegResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=8,
            dropout_prob=0.2,
            norm='instance',
        )

        # small branch: receives image crop + coarse-prob crop (concat)
        # additional guidance channels (from coarse features)
        self.guidance_channels = 2
        # small conv block to produce richer feature-level guidance
        self.guidance_net = nn.Sequential(
            nn.Conv3d(out_channels, max(1, self.guidance_channels), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, self.guidance_channels), self.guidance_channels, kernel_size=1),
        )

        # small branch: receives image crop + coarse-prob crop + guidance channels
        self.small = SegResNet(
            in_channels=in_channels + out_channels + self.guidance_channels,
            out_channels=out_channels,
            init_filters=4,
            dropout_prob=0.2,
            norm='instance',
        )

        # learnable voxel-wise gating for fusion: input = [coarse, refined, guidance]
        self.fusion_gate = nn.Conv3d(out_channels * 2 + self.guidance_channels, out_channels, kernel_size=1)

        # predict residual (small branch will output residual to add to coarse logits)
        self.predict_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, D)
        B, C, H, W, D = x.shape

        # large coarse prediction
        coarse_logits = self.large(x)   
        coarse_prob = torch.sigmoid(coarse_logits)

        # determine crop size (floor to integers) and clamp to dims
        cz = max(1, int(H * self.crop_size_fraction))
        cy = max(1, int(W * self.crop_size_fraction))
        cx = max(1, int(D * self.crop_size_fraction))
        crop_size = (min(cz, H), min(cy, W), min(cx, D))

        # if crop covers the full volume, skip the small branch
        if crop_size == (H, W, D):
            return coarse_logits

        # dynamic crop selection based on coarse probability
        # compute a single-channel score map (max over classes)
        # coarse_prob: (B, C, H, W, D) -> score_map: (B, H, W, D)
        score_map = coarse_prob.max(dim=1).values
        # try bbox of thresholded mask (batch-aware)
        mask = score_map > self.crop_threshold
        bbox = _bbox_from_mask(mask)
        # handle batch dimension (training uses batch_size=1)
        if isinstance(bbox, list):
            bbox = bbox[0]

        # Initialize crop coordinates
        zs = ys = xs = 0
        
        if bbox is not None:
            # bbox is ((zs,ze),(ys,ye),(xs,xe)) for the first sample
            (zs, ze), (ys, ye), (xs, xe) = bbox
            # pad bbox
            zs = max(0, zs - self.crop_padding)
            ys = max(0, ys - self.crop_padding)
            xs = max(0, xs - self.crop_padding)
            ze = min(H, ze + self.crop_padding)
            ye = min(W, ye + self.crop_padding)
            xe = min(D, xe + self.crop_padding)
        else:
            # fallback to center-of-mass crop
            com = _center_of_mass(score_map)
            # _center_of_mass may return list for batch or single tuple
            if isinstance(com, list):
                cz_c, cy_c, cx_c = com[0]
            else:
                cz_c, cy_c, cx_c = com
            half_z = crop_size[0] // 2
            half_y = crop_size[1] // 2
            half_x = crop_size[2] // 2
            zs = max(0, cz_c - half_z)
            ys = max(0, cy_c - half_y)
            xs = max(0, cx_c - half_x)
            ze = min(H, zs + crop_size[0])
            ye = min(W, ys + crop_size[1])
            xe = min(D, xs + crop_size[2])

        # extract crop by indexing to preserve location
        img_crop = x[:, :, zs:ze, ys:ye, xs:xe]
        prob_crop = coarse_prob[:, :, zs:ze, ys:ye, xs:xe]
        coarse_logits_crop = coarse_logits[:, :, zs:ze, ys:ye, xs:xe]
        
        # actual crop size for placing back
        actual_crop_size = (ze - zs, ye - ys, xe - xs)

        # guidance channels from coarse logits (learned 1x1 conv)
        guidance_full = self.guidance_net(coarse_logits)
        # extract guidance at the same crop location
        guidance_crop = guidance_full[:, :, zs:ze, ys:ye, xs:xe]

        # small input: concat image crop + coarse prob crop + guidance
        small_in = torch.cat([img_crop, prob_crop, guidance_crop], dim=1)

        refined_crop_logits = self.small(small_in)
        # if predicting residuals, add to coarse logits crop
        if self.predict_residual:
            refined_crop_logits = coarse_logits_crop + refined_crop_logits

        # place refined crop back into full-volume tensor at original location
        full_refined = torch.zeros_like(coarse_logits)
        full_refined[:, :, zs:ze, ys:ye, xs:xe] = refined_crop_logits

        # learnable fusion: voxel-wise gate
        # include guidance in gate input so fusion can use feature-level cues
        gate_in = torch.cat([coarse_logits, full_refined, guidance_full], dim=1)
        gate = torch.sigmoid(self.fusion_gate(gate_in))
        out = gate * full_refined + (1.0 - gate) * coarse_logits
        
        return out


def get_model():
    return LSNet(in_channels=4, out_channels=3, crop_size_fraction=0.5)
