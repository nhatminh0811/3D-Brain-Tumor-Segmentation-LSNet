import sys
import torch
from tqdm import tqdm

sys.path.append('src')
from dataset import get_brats2020_datalist
from transforms import get_train_transforms, get_val_transforms
from model import get_model
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.utils import set_determinism


def run_once():
    set_determinism(seed=42)
    data_dir = 'data/BraTS2020'
    datalist = get_brats2020_datalist(data_dir, section='training')
    if len(datalist) == 0:
        raise RuntimeError('No valid subjects found. Check data folder.')

    # use a tiny subset to test pipeline
    subset = datalist[:4]

    train_files = subset[:3]
    val_files = subset[3:4]

    train_ds = Dataset(data=train_files, transform=get_train_transforms())
    val_ds = Dataset(data=val_files, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)

    loss_fn = DiceLoss(sigmoid=True, to_onehot_y=False, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dice_metric = DiceMetric(include_background=False, reduction='mean_batch')

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # single epoch
    model.train()
    print('Training loop (1 epoch) on small subset')
    epoch_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader)):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if i >= 10:
            break

    if len(train_loader) > 0:
        epoch_loss /= len(train_loader)
    print(f'Train Loss (one-epoch test): {epoch_loss:.4f}')

    # validation
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch['image'].to(device)
            val_labels = val_batch['label'].to(device)
            val_outputs = model(val_images)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            dice_metric(y_pred=val_outputs, y=val_labels)

        metric = dice_metric.aggregate()
        dice_metric.reset()
        print('Val Dice (per-channel):', metric)


if __name__ == '__main__':
    run_once()
