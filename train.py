import os
import csv
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from model import SSD300, MultiBoxLoss
from datasets import DIORDataset
from utils import *

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.expanduser('~/Arman/data')
SPLIT_DIR      = os.path.expanduser('~/Arman/data/Python_datareader')
CHECKPOINT_DIR = './checkpoints'   # one sub-folder per split

# ── model ─────────────────────────────────────────────────────────────────────
n_classes = len(label_map)         # 21  (20 DIOR classes + background)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE        = 8
MAX_EPOCHS        = 200
LR                = 1e-3
BACKBONE_LR_SCALE = 0.1   # pretrained ResNet50 backbone gets 10× lower LR than head
MOMENTUM          = 0.9
WEIGHT_DECAY      = 5e-4
DECAY_LR_AT       = [int(0.75 * MAX_EPOCHS), int(0.90 * MAX_EPOCHS)]   # [150, 180]
DECAY_LR_TO       = 0.1
GRAD_CLIP         = None
WORKERS           = 4
PRINT_FREQ        = 200

# Early stopping based on conf_loss plateau.
# Stop if conf_loss does not improve by more than MIN_DELTA for PATIENCE consecutive epochs.
PATIENCE  = 20
MIN_DELTA = 0.001

SPLITS = ['split_005', 'split_025', 'split_050', 'split_075', 'split_100']

cudnn.benchmark = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(model):
    """Build the 4-group SGD optimizer with differential LR for backbone vs head."""
    backbone_params, backbone_biases = [], []
    head_params,     head_biases     = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = name.startswith('base.')
        is_bias     = name.endswith('.bias')
        if   is_backbone and is_bias:  backbone_biases.append(param)
        elif is_backbone:              backbone_params.append(param)
        elif is_bias:                  head_biases.append(param)
        else:                          head_params.append(param)
    return torch.optim.SGD([
        {'params': backbone_biases, 'lr': 2 * LR * BACKBONE_LR_SCALE},
        {'params': backbone_params, 'lr':     LR * BACKBONE_LR_SCALE},
        {'params': head_biases,     'lr': 2 * LR},
        {'params': head_params,     'lr':     LR},
    ], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


def _save(path, epoch, model, optimizer, best_conf_loss, no_improve):
    torch.save({
        'epoch'         : epoch,
        'model'         : model.state_dict(),
        'optimizer'     : optimizer.state_dict(),
        'best_conf_loss': best_conf_loss,
        'no_improve'    : no_improve,
    }, path)


def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()

    batch_time   = AverageMeter()
    total_losses = AverageMeter()
    loc_losses   = AverageMeter()
    conf_losses  = AverageMeter()

    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        images = images.to(device)
        boxes  = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)
        loss, loc_loss, conf_loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        optimizer.zero_grad()
        loss.backward()

        if GRAD_CLIP is not None:
            clip_gradient(optimizer, GRAD_CLIP)

        optimizer.step()

        n = images.size(0)
        total_losses.update(loss.item(), n)
        loc_losses.update(loc_loss.item(), n)
        conf_losses.update(conf_loss.item(), n)
        batch_time.update(time.time() - start)
        start = time.time()

        if i % PRINT_FREQ == 0:
            print(
                f'  Epoch [{epoch}][{i}/{len(train_loader)}]'
                f'  Loss {total_losses.val:.4f} ({total_losses.avg:.4f})'
                f'  Loc {loc_losses.val:.4f}'
                f'  Conf {conf_losses.val:.4f}'
                f'  Batch {batch_time.val:.2f}s'
            )

    del predicted_locs, predicted_scores, images, boxes, labels
    return total_losses.avg, loc_losses.avg, conf_losses.avg


# ─────────────────────────────────────────────────────────────────────────────
# Per-split training
# ─────────────────────────────────────────────────────────────────────────────

def train_split(split_name):
    print(f'\n{"#"*64}')
    print(f'#  Training : {split_name}')
    print(f'{"#"*64}')

    split_file = os.path.join(SPLIT_DIR, f'{split_name}_indices.npy')
    out_dir    = os.path.join(CHECKPOINT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── dataset & loader ──────────────────────────────────────────────────────
    train_dataset = DIORDataset(
        data_dir   = DATA_DIR,
        split      = 'train',
        split_file = split_file,
        split_name = 'TRAIN',
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=DIORDataset.collate_fn, num_workers=WORKERS, pin_memory=True,
    )
    print(f'  Train images : {len(train_dataset)}')

    # ── model & optimiser (resume if checkpoint exists) ───────────────────────
    latest_path = os.path.join(out_dir, 'checkpoint_ssd300.pth.tar')
    best_path   = os.path.join(out_dir, 'best_ssd300.pth.tar')

    if os.path.exists(latest_path):
        print(f'  Resuming from {latest_path}')
        ckpt           = torch.load(latest_path, map_location=device, weights_only=True)
        start_epoch    = ckpt['epoch'] + 1
        best_conf_loss = ckpt.get('best_conf_loss', float('inf'))
        no_improve     = ckpt.get('no_improve', 0)
        model          = SSD300(n_classes=n_classes)
        model.load_state_dict(ckpt['model'])
        optimizer      = _build_optimizer(model)
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch    = 0
        best_conf_loss = float('inf')
        no_improve     = 0
        model          = SSD300(n_classes=n_classes)
        optimizer      = _build_optimizer(model)

    model     = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # ── CSV log ───────────────────────────────────────────────────────────────
    csv_path     = os.path.join(out_dir, 'results.csv')
    write_header = (start_epoch == 0)

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MAX_EPOCHS):

        if epoch in DECAY_LR_AT:
            adjust_learning_rate(optimizer, DECAY_LR_TO)

        total_loss, loc_loss, conf_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch
        )

        current_lr = optimizer.param_groups[3]['lr']

        # ── early stopping: conf_loss plateau ────────────────────────────────
        if conf_loss < best_conf_loss - MIN_DELTA:
            best_conf_loss = conf_loss
            no_improve     = 0
            _save(best_path, epoch, model, optimizer, best_conf_loss, no_improve)
            print(f'  *** New best conf_loss {best_conf_loss:.4f} — saved to {best_path}')
        else:
            no_improve += 1
            print(f'  No conf_loss improvement for {no_improve}/{PATIENCE} epochs.')

        # ── CSV row ───────────────────────────────────────────────────────────
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train/total_loss', 'train/loc_loss',
                                 'train/conf_loss', 'lr'])
                write_header = False
            writer.writerow([
                epoch,
                round(total_loss, 5),
                round(loc_loss,   5),
                round(conf_loss,  5),
                round(current_lr, 8),
            ])

        # ── save latest checkpoint (for resuming) ─────────────────────────────
        _save(latest_path, epoch, model, optimizer, best_conf_loss, no_improve)

        # ── stop if plateau ───────────────────────────────────────────────────
        if no_improve >= PATIENCE:
            print(f'\n  Early stopping: conf_loss did not improve by >{MIN_DELTA} '
                  f'for {PATIENCE} consecutive epochs.')
            break

    print(f'\n  Done. Results CSV  : {csv_path}')
    print(f'  Best checkpoint   : {best_path}')
    print(f'  Latest checkpoint : {latest_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — trains all 5 splits sequentially
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=None, choices=SPLITS,
                        help='Train a single split (default: train all 5 sequentially)')
    args = parser.parse_args()

    splits_to_run = [args.split] if args.split else SPLITS
    for split in splits_to_run:
        train_split(split)


if __name__ == '__main__':
    main()
