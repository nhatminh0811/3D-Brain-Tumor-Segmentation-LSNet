import argparse
import torch
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from dataset import get_brats2020_datalist
from transforms import get_train_transforms, get_val_transforms
from models import get_model as get_model_pkg
from losses import get_loss_fn
from metrics import ComprehensiveMetrics
from schedulers import get_scheduler, TRAINING_CONFIGS


def main():
    set_determinism(seed=42)

    # create results directory for checkpoints
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # ✅ ROOT DIR (KHÔNG phải TrainingData)
    data_dir = "data/BraTS2020"

    data = get_brats2020_datalist(data_dir, section="training")

    if len(data) == 0:
        raise RuntimeError(
            "No valid subjects found in data directory. "
            "Check 'data/BraTS2020/TrainingData' and file naming."
        )

    train_files, val_files = train_test_split(
        data, test_size=0.2, random_state=42
    )

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "lsnet"],
                        help="Model to use: baseline or lsnet")
    args = parser.parse_args()

    # Get training config based on model
    config_key = f"{args.model}_brats"
    config = TRAINING_CONFIGS.get(config_key, TRAINING_CONFIGS['baseline_brats'])
    
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_epochs = config['max_epochs']
    warmup_epochs = config['warmup_epochs']
    weight_decay = config['weight_decay']
    scheduler_type = config['scheduler_type']

    print(f"\n{'='*60}")
    print(f"Training Configuration for {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Warmup Epochs: {warmup_epochs}")
    print(f"Scheduler: {scheduler_type}")
    print(f"{'='*60}\n")

    train_ds = Dataset(
        data=train_files,
        transform=get_train_transforms()
    )

    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_pkg(name=args.model).to(device)
    print(f"Using model: {args.model}")
    print(f"Device: {device}")

    # Use optimized loss function
    loss_fn = get_loss_fn('combined', alpha=0.7)
    print(f"Loss: Combined Dice + Focal Loss (alpha=0.7)\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Initialize learning rate scheduler
    scheduler = get_scheduler(
        optimizer,
        max_epochs,
        schedule_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        min_lr=1e-6
    )

    # Comprehensive metrics
    metrics = ComprehensiveMetrics()
    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5)
    ])

    # use mixed precision to reduce GPU memory usage
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda' if use_amp else 'cpu', enabled=use_amp)

    best_dice = -1.0
    best_state = None
    best_epoch_num = 0
    
    # Training loop
    for epoch in range(max_epochs):
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch + 1}/{max_epochs} | LR: {current_lr:.2e}")
        
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        print(f"Train Loss: {epoch_loss:.4f}")

        # ---------- Validation ----------
        model.eval()
        metrics.reset()
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_images = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)

                val_outputs = model(val_images)
                
                # Compute metrics
                metrics(val_outputs, val_labels)

        # Aggregate and print metrics
        agg_metrics = metrics.aggregate()
        
        print("\nValidation Metrics:")
        print(f"  Dice TC: {agg_metrics['dice_tc']:.4f}")
        print(f"  Dice WT: {agg_metrics['dice_wt']:.4f}")
        print(f"  Dice ET: {agg_metrics['dice_et']:.4f}")
        print(f"  Dice Mean: {agg_metrics['dice_mean']:.4f}")
        print(f"  HD95: {agg_metrics['hd95']:.4f}")
        print(f"  Sensitivity: {agg_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {agg_metrics['specificity']:.4f}")

        # Track best model based on mean Dice
        mean_dice = agg_metrics['dice_mean']
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_epoch_num = epoch + 1
            best_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict().copy(),
                'metrics': agg_metrics,
                'learning_rate': current_lr,
            }
            print(f"✅ New best model! Dice: {best_dice:.4f}")

        # Update learning rate
        scheduler.step()

    # Save best model
    if best_state is not None:
        best_path = os.path.join(results_dir, f"{args.model}_best.pth")
        torch.save(best_state, best_path)
        print(f"\n{'='*60}")
        print(f"Best model saved: {best_path}")
        print(f"Epoch: {best_epoch_num}")
        print(f"Dice: {best_state['metrics']['dice_mean']:.4f}")
        print(f"HD95: {best_state['metrics']['hd95']:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
