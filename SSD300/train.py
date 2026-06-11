import os
import csv
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import SSD300, MultiBoxLoss
from datasets import DIORDatasetSSD
from utils import *

# ── Split to train on — change this one line to switch splits ──────────
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
# ────────────────────────────────────────────────────────────────────────

# ── Training hyperparameters ───────────────────────────────────────────
MAX_EPOCHS   = 500    # safety ceiling — early stopping will trigger well before this
WARMUP_ITERS = 500    # linearly ramp lr from ~0 to lr over first 500 iterations
PATIENCE     = 10     # stop training if val loss doesn't improve for this many epochs
MIN_DELTA    = 0.01   # minimum improvement in val loss to count as progress (SSD loss is in range 1–10; 0.001 was below noise floor)
LR_PATIENCE  = 5      # reduce lr if val loss doesn't improve for this many epochs
# ──────────────────────────────────────────────────────────────────────

# Data parameters
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..')
SPLIT_FILE = os.path.join(os.path.dirname(__file__), '..', 'Python_datareader', f'{SPLIT}_indices.npy')

# generate_splits.py built indices over the full trainval pool (11,725 IDs).
# Roughly half of those indices point to val.txt images — using them for
# training causes data leakage.  We keep only indices < TRAIN_SIZE so the
# training set draws exclusively from train.txt.
TRAIN_SIZE = 5862   # number of lines in Main/train.txt
_raw_indices   = np.load(SPLIT_FILE)
TRAIN_INDICES  = _raw_indices[_raw_indices < TRAIN_SIZE]   # numpy array, passed directly to DIORDatasetSSD

# Model parameters
n_classes = len(label_map)  # 21 (20 DIOR classes + background)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint   = None   # path to checkpoint to resume from, None to start fresh
batch_size   = 8      # batch size
workers      = 4      # number of workers for loading data in the DataLoader
lr           = 1e-3   # learning rate
momentum     = 0.9    # momentum
weight_decay = 5e-4   # weight decay
grad_clip    = 4.0    # clip gradients — protects against explosion from randomly-init heads early in training

cudnn.benchmark = True


def _set_bn_eval(module):
    """Freeze a BatchNorm2d layer: use stored running stats, stop updating them."""
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()


def main():
    """
    Training with early stopping and ReduceLROnPlateau scheduler.
    """
    global start_epoch, label_map, epoch, checkpoint

    # best_val_loss tracks the lowest val loss seen — used for early stopping
    # and to decide when to save checkpoint_best
    best_val_loss    = float('inf')
    epochs_no_improve = 0  # counter — resets to 0 whenever val loss improves

    # ── Initialize model or load checkpoint ───────────────────────────
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)

        # Differential LR: pretrained backbone gets 10x lower LR than the
        # randomly-initialised auxiliary and prediction heads.  Biases always
        # get 2x their group's base LR (standard SSD practice).
        bb_w, bb_b, head_w, head_b = [], [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_bias     = name.endswith('.bias')
            is_backbone = name.startswith('base.')
            if is_backbone:
                (bb_b   if is_bias else bb_w).append(param)
            else:
                (head_b if is_bias else head_w).append(param)

        optimizer = torch.optim.SGD(
            params=[
                {'params': bb_w,   'lr': lr * 0.1},        # backbone weights
                {'params': bb_b,   'lr': lr * 0.1 * 2},    # backbone biases
                {'params': head_w, 'lr': lr},               # head weights
                {'params': head_b, 'lr': lr * 2},           # head biases
            ],
            momentum=momentum, weight_decay=weight_decay
        )

    else:
        checkpoint    = torch.load(checkpoint, map_location=device)
        start_epoch   = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model         = checkpoint['model']
        optimizer     = checkpoint['optimizer']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Move to device
    model     = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # ── ReduceLROnPlateau scheduler ────────────────────────────────────
    # Monitors val loss every epoch. If it doesn't improve for LR_PATIENCE
    # epochs, lr is multiplied by factor=0.1.
    # This replaces the old hardcoded decay_lr_at [80k, 100k] iterations —
    # the lr now decays automatically whenever training plateaus, regardless
    # of how many iterations have passed.
    # min_lr=1e-6 prevents lr from shrinking to zero.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # we want val loss to go DOWN
        factor=0.1,        # multiply lr by 0.1 on plateau
        patience=LR_PATIENCE,
        min_lr=1e-6
    )
    # ──────────────────────────────────────────────────────────────────

    # ── DataLoaders ────────────────────────────────────────────────────
    train_dataset = DIORDatasetSSD(DATA_DIR, split='trainval', split_file=TRAIN_INDICES, split_type='TRAIN')
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True
    )

    val_dataset = DIORDatasetSSD(DATA_DIR, split='val', split_file=None, split_type='TEST')
    val_loader  = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=val_dataset.collate_fn, num_workers=workers, pin_memory=True
    )
    # ──────────────────────────────────────────────────────────────────

    # ── CSV loss log ───────────────────────────────────────────────────
    log_dir  = os.path.join('checkpoints', SPLIT)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'loss_log.csv')

    csv_mode = 'a' if start_epoch > 0 else 'w'
    log_file = open(log_path, csv_mode, newline='')
    csv_writer = csv.writer(log_file)
    if start_epoch == 0:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
    # ──────────────────────────────────────────────────────────────────

    # ── Epoch loop ─────────────────────────────────────────────────────
    epoch_pbar = tqdm(range(start_epoch, MAX_EPOCHS),
                      desc=f'Training [{SPLIT}]',
                      unit='epoch')

    for epoch in epoch_pbar:

        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss   = validate(val_loader, model, criterion, epoch)

        # Step the scheduler with the current val loss.
        # If val loss hasn't improved for LR_PATIENCE epochs, lr is reduced.
        scheduler.step(val_loss)

        # Current lr after scheduler step (for logging) — group 2 = head weights (base lr)
        current_lr = optimizer.param_groups[2]['lr']

        # Update epoch bar postfix
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val':   f'{val_loss:.4f}',
            'best':  f'{best_val_loss:.4f}',
            'lr':    f'{current_lr:.6f}'
        })

        # Write to CSV
        csv_writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{current_lr:.6f}'])
        log_file.flush()

        # ── Checkpoint saving ──────────────────────────────────────────
        # Always save latest for crash recovery
        save_checkpoint(epoch, model, optimizer, SPLIT, 'checkpoint_latest.pth.tar', best_val_loss)

        # Save periodic snapshot every 50 epochs for convergence study
        if epoch % 50 == 0:
            save_checkpoint(epoch, model, optimizer, SPLIT, f'checkpoint_epoch_{epoch:04d}.pth.tar', best_val_loss)
        # ──────────────────────────────────────────────────────────────

        # ── Early stopping ─────────────────────────────────────────────
        # Check if val loss improved by at least MIN_DELTA
        if val_loss < best_val_loss - MIN_DELTA:
            # Improvement — reset counter, update best, save best checkpoint
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(epoch, model, optimizer, SPLIT, 'checkpoint_best.pth.tar', best_val_loss)
        else:
            # No improvement — increment counter
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            tqdm.write(f'\nEarly stopping triggered at epoch {epoch}. '
                       f'No improvement for {PATIENCE} epochs. '
                       f'Best val loss: {best_val_loss:.4f}')
            break
        # ──────────────────────────────────────────────────────────────

    log_file.close()


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    :return: average training loss for this epoch
    """
    model.train()
    # Freeze backbone BatchNorm layers: with batch_size=8 the mini-batch
    # statistics are too noisy to keep BN in train mode — pretrained running
    # stats are more reliable than per-batch estimates at this batch size.
    model.base.apply(_set_bn_eval)

    losses = AverageMeter()

    batch_pbar = tqdm(train_loader,
                      desc=f'Epoch [{epoch}] Train',
                      leave=False,
                      unit='batch')

    # Base LRs for the four param groups (used by warmup below)
    _base_lrs = [lr * 0.1, lr * 0.1 * 2, lr, lr * 2]

    for i, (images, boxes, labels, _) in enumerate(batch_pbar):

        # Linear LR warmup — only on fresh training (start_epoch == 0).
        # Skipped on resume to avoid overriding the scheduler-reduced LR that
        # was saved in the checkpoint with the full initial lr.
        global_iter = epoch * len(train_loader) + i
        if start_epoch == 0 and global_iter < WARMUP_ITERS:
            scale = (global_iter + 1) / WARMUP_ITERS
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = _base_lrs[idx] * scale

        images = images.to(device)
        boxes  = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), images.size(0))

        batch_pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr':   f'{optimizer.param_groups[2]["lr"]:.6f}'
        })

    del predicted_locs, predicted_scores, images, boxes, labels

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    """
    One epoch's validation.

    During validation we do NOT update model weights — forward pass only.
    model.eval() disables dropout and uses stable BatchNorm statistics.
    torch.no_grad() skips building the computation graph to save memory.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :param epoch: epoch number
    :return: average validation loss for this epoch
    """
    model.eval()

    losses = AverageMeter()

    val_pbar = tqdm(val_loader,
                    desc=f'Epoch [{epoch}] Val  ',
                    leave=False,
                    unit='batch')

    with torch.no_grad():
        for _, (images, boxes, labels, _) in enumerate(val_pbar):

            images = images.to(device)
            boxes  = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_scores = model(images)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
            val_pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    return losses.avg


if __name__ == '__main__':
    main()
