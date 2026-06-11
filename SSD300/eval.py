import os
from utils import *
from datasets import DIORDatasetSSD
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# ── Split and epoch to evaluate — change these two lines to switch ─────
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
EPOCH = None         # None = best checkpoint; set to epoch number (e.g. 100) for a periodic snapshot
# ────────────────────────────────────────────────────────────────────────

# Parameters
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..')
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint path:
#   EPOCH=None  → checkpoint_best.pth.tar    (lowest val loss, for final mAP)
#   EPOCH=100   → checkpoint_epoch_0100.pth.tar  (periodic snapshot, for convergence study)
if EPOCH is None:
    checkpoint_path = os.path.join('checkpoints', SPLIT, 'checkpoint_best.pth.tar')
else:
    checkpoint_path = os.path.join('checkpoints', SPLIT, f'checkpoint_epoch_{EPOCH:04d}.pth.tar')

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint_path, map_location=device)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load validation data
# split_type='TEST' disables augmentation — resize to 300×300 only
# DIOR has no 'difficult' flag — difficulties tensor is all zeros
val_dataset = DIORDatasetSSD(DATA_DIR, split='val', split_file=None, split_type='TEST')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=val_dataset.collate_fn, num_workers=workers,
                                         pin_memory=True)


def evaluate(val_loader, model):
    """
    Evaluate the model on the validation set and compute mAP.

    :param val_loader: DataLoader for validation data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # We collect ALL detections and ALL ground truths across the entire val set
    # before computing mAP — calculate_mAP() in utils.py needs the full dataset at once
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(val_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output.
            # detect_objects() applies NMS and filters by confidence score,
            # returning the final set of bounding boxes per image.
            # These thresholds are standard SSD evaluation settings:
            #   min_score=0.01  : keep even low-confidence detections (mAP penalises missed ones)
            #   max_overlap=0.45: NMS overlap threshold to suppress duplicate boxes
            #   top_k=200       : maximum detections per image
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.01, max_overlap=0.45, top_k=200
            )

            # Store ground truth for this batch
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP across all 20 DIOR classes
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(val_loader, model)
