import argparse
import sys
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    Activations,
    AsDiscrete,
)

from monai.inferers import sliding_window_inference

# allow importing local `models` when running from project root
sys.path.append(os.path.dirname(__file__))
from models import get_model as get_model_pkg


def find_case_images(case_path: Path):
    # case_path can be a folder containing the 4 modalities
    files = [p for p in case_path.iterdir() if p.is_file()]

    def find_mod(name):
        for p in files:
            ln = p.name.lower()
            if name in ln and (ln.endswith('.nii') or ln.endswith('.nii.gz')):
                return str(p)
        return None

    fl = find_mod('_flair')
    t1 = find_mod('_t1ce') or find_mod('_t1') if False else None
    # prefer exact modalities: t1, t1ce, t2
    t1_exact = find_mod('_t1')
    t1ce_exact = find_mod('_t1ce')
    t2 = find_mod('_t2')

    # Use exact finds
    t1 = t1_exact
    t1ce = t1ce_exact

    if not (fl and t1 and t1ce and t2):
        raise FileNotFoundError(f"Could not find all modalities in {case_path}")

    return [fl, t1, t1ce, t2]


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained BraTS model")
    parser.add_argument("--case", type=str, default="data/BraTS2020/TrainingData/BraTS20_Training_001",
                        help="Path to case folder containing the 4 modality files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (default: results/{model}_best.pth)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use, e.g. cuda or cpu (default auto)")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "lsnet"],
                        help="Model to use: baseline or lsnet")
    parser.add_argument("--save-pred", action="store_true", help="Save prediction numpy to results/")
    args = parser.parse_args()

    # Set checkpoint path based on model if not explicitly provided
    if args.checkpoint is None:
        args.checkpoint = f"results/{args.model}_best.pth"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = get_model_pkg(name=args.model).to(device)
    print(f"Using model: {args.model}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    print(f"✅ Loaded checkpoint: {ckpt_path}")

    # -----------------------------
    # Inference transforms
    # -----------------------------
    infer_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1, 1, 1),
            mode="bilinear",
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        EnsureTyped(keys=["image"]),
    ])

    case_path = Path(args.case)
    if case_path.is_dir():
        image_files = find_case_images(case_path)
    else:
        raise FileNotFoundError(f"Case path not found or not a directory: {case_path}")

    sample = {"image": image_files}

    data = infer_transforms(sample)
    image = data["image"].unsqueeze(0).to(device)  # (1, 4, H, W, D)

    # -----------------------------
    # Sliding window inference
    # -----------------------------
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
        )

    # -----------------------------
    # Post-processing
    # -----------------------------
    post = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ])

    # also get raw sigmoid for debugging
    post_sigmoid_only = Compose([Activations(sigmoid=True)])
    raw_prob = post_sigmoid_only(logits)[0].cpu()  # (C, H, W, D)

    pred = post(logits)[0].cpu()  # (C, H, W, D)

    # -----------------------------
    # Debug: check raw probabilities
    print(f"\n📊 Raw sigmoid probabilities (before threshold=0.5):")
    for i, name in enumerate(["TC", "WT", "ET"]):
        print(f"{name} prob | min: {raw_prob[i].min():.4f}, max: {raw_prob[i].max():.4f}, mean: {raw_prob[i].mean():.4f}")

    # Compute Dice scores if ground truth available
    # -----------------------------
    gt_path = None
    # look for *_seg.nii or *_seg.nii.gz
    for p in case_path.iterdir():
        if p.is_file() and p.name.lower().endswith(('.nii', '.nii.gz')) and '_seg' in p.name.lower():
            gt_path = p
            break

    if gt_path is not None:
        import nibabel as nib
        import numpy as np

        gt_img = nib.load(str(gt_path)).get_fdata().astype(np.int16)
        # pred: (C, H, W, D) with channels being binary masks for TC, WT, ET
        # ground truth in BraTS uses labels {0,1,2,4}; create binary maps matching TC, WT, ET
        # WT = label>0 (1,2,4), TC = labels 1 and 4? In BraTS: TC often is (1,4) depending on convention.
        # We'll build common definitions: WT (whole tumor) -> labels 1,2,4; TC (tumor core) -> labels 1,4; ET (enhancing) -> label 4
        gt_wt = (gt_img > 0).astype(np.uint8)
        gt_tc = np.isin(gt_img, [1, 4]).astype(np.uint8)
        gt_et = (gt_img == 4).astype(np.uint8)

        gt_maps = [gt_tc, gt_wt, gt_et]

        def dice_per_class(pred_mask, gt_mask):
            pred_mask = pred_mask.astype(np.uint8)
            gt_mask = gt_mask.astype(np.uint8)
            inter = (pred_mask & gt_mask).sum()
            denom = pred_mask.sum() + gt_mask.sum()
            if denom == 0:
                return 1.0
            return 2.0 * inter / denom

        dice_scores = []
        for i in range(len(gt_maps)):
            pr = pred[i].numpy().astype(np.uint8)
            ds = dice_per_class(pr, gt_maps[i])
            dice_scores.append(ds)

        mean_dice = float(np.mean(dice_scores))

        print("\n🎯 Dice scores")
        print(f"TC Dice: {dice_scores[0]:.4f}")
        print(f"WT Dice: {dice_scores[1]:.4f}")
        print(f"ET Dice: {dice_scores[2]:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}\n")

    # -----------------------------
    # Extract tumor regions
    # -----------------------------
    tc = pred[0]
    wt = pred[1]
    et = pred[2] if pred.shape[0] > 2 else None

    print("📊 Prediction summary")
    print("TC voxels:", int(tc.sum().item()))
    print("WT voxels:", int(wt.sum().item()))
    if et is not None:
        print("ET voxels:", int(et.sum().item()))
    else:
        print("ET not predicted (no channel)")

    # Optional save
    if args.save_pred:
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        case_name = case_path.name
        out_path = out_dir / f"pred_{case_name}.npy"
        import numpy as np

        np.save(str(out_path), pred.numpy())
        print(f"Saved prediction to {out_path}")

    # -----------------------------
    # Visualization (middle slice)
    # -----------------------------
    z = tc.shape[2] // 2
    flair = image[0, 0, :, :, z].cpu()

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(flair, cmap="gray")
    plt.title("FLAIR")

    plt.subplot(1, 4, 2)
    plt.imshow(flair, cmap="gray")
    plt.imshow(tc[:, :, z], cmap="Reds", alpha=0.4)
    plt.title("Tumor Core (TC)")

    plt.subplot(1, 4, 3)
    plt.imshow(flair, cmap="gray")
    plt.imshow(wt[:, :, z], cmap="Blues", alpha=0.4)
    plt.title("Whole Tumor (WT)")

    plt.subplot(1, 4, 4)
    plt.imshow(flair, cmap="gray")
    if et is not None:
        plt.imshow(et[:, :, z], cmap="Greens", alpha=0.4)
    plt.title("Enhancing Tumor (ET)")

    plt.show()


if __name__ == "__main__":
    main()
