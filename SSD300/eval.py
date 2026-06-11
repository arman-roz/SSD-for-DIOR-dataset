import os
import torch
import torch.utils.data
from tqdm import tqdm
from pprint import PrettyPrinter

from model import SSD300
from datasets import DIORDatasetSSD
from utils import calculate_mAP, rev_label_map

pp = PrettyPrinter()

# ── Split and epoch to evaluate — change these two lines to switch ─────
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
EPOCH = None         # None = best checkpoint; integer = periodic snapshot (e.g. 100)
# ────────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
batch_size = 64
workers    = 4


def resolve_checkpoint():
    if EPOCH is None:
        return os.path.join('checkpoints', SPLIT, 'checkpoint_best.pth.tar')
    return os.path.join('checkpoints', SPLIT, f'checkpoint_epoch_{EPOCH:04d}.pth.tar')


def evaluate(val_loader, model, device):
    """
    Evaluate model on the full validation set and compute mAP.

    :param val_loader: DataLoader for validation data
    :param model: model in eval mode
    :param device: torch device
    """
    model.eval()

    det_boxes, det_labels, det_scores = [], [], []
    true_boxes, true_labels, true_difficulties = [], [], []

    with torch.no_grad():
        for images, boxes, labels, difficulties in tqdm(val_loader, desc='Evaluating'):
            images = images.to(device)

            predicted_locs, predicted_scores = model(images)

            # NMS + confidence filter
            #   min_score=0.01  : keep low-confidence detections (mAP penalises missed ones)
            #   max_overlap=0.45: NMS IoU threshold
            #   top_k=200       : max detections per image
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.01, max_overlap=0.45, top_k=200
            )

            boxes       = [b.to(device) for b in boxes]
            labels      = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

    APs, mAP = calculate_mAP(
        det_boxes, det_labels, det_scores,
        true_boxes, true_labels, true_difficulties
    )

    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\n"
                                f"Run train.py first, or set EPOCH to an available snapshot.")

    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    model = SSD300(n_classes=ckpt['n_classes'])
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}.\n')

    # split_type='TEST' disables augmentation — resize to 300×300 only
    val_dataset = DIORDatasetSSD(DATA_DIR, split='val', split_file=None, split_type='TEST')
    val_loader  = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=val_dataset.collate_fn, num_workers=4, pin_memory=True
    )

    evaluate(val_loader, model, device)


if __name__ == '__main__':
    main()
