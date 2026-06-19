import os
import csv
import torch
import torch.utils.data
from tqdm import tqdm
from pprint import PrettyPrinter

from utils import calculate_mAP, label_map, rev_label_map
from datasets import DIORDataset
from model import SSD300

pp = PrettyPrinter()

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.expanduser('~/Arman/data')
CHECKPOINT_DIR = './checkpoints'      # train.py saves one sub-folder per split
RESULTS_CSV    = './eval_results.csv' # summary table written after all splits

SPLITS = ['split_005', 'split_025', 'split_050', 'split_075', 'split_100']

# ── eval settings (standard VOC protocol — same as YOLO eval) ─────────────────
MIN_SCORE   = 0.01
MAX_OVERLAP = 0.45
TOP_K       = 200
BATCH_SIZE  = 8
WORKERS     = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── test dataset — 11,738 fully annotated images ──────────────────────────────
# The DIOR test set (JPEGImages-test/, IDs 11726–23463) has annotation XML
# files in the same Annotations/Horizontal Bounding Boxes/ folder as trainval.
# This is the proper held-out evaluation set.
#
# Note: YOLO reported metrics on the val split during training.
# For the final comparison table we evaluate SSD on the full test set which
# is a stronger and fairer holdout. Val evaluation is also supported — see
# the comment in run_all() below.
test_dataset = DIORDataset(
    data_dir   = DATA_DIR,
    split      = 'test',    # reads test.txt, images in JPEGImages-test/
    split_file = None,      # always the full 11,738-image test set
    split_name = 'TEST',    # no augmentation
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    collate_fn  = DIORDataset.collate_fn,
    num_workers = WORKERS,
    pin_memory  = True,
)


def evaluate_split(split_name, loader):
    """
    Load the checkpoint for one split and evaluate on the provided loader.

    Returns:
        APs  : dict  {class_name: AP_value}
        mAP  : float
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, split_name, 'best_ssd300.pth.tar')

    if not os.path.exists(checkpoint_path):
        print(f'[SKIP] No checkpoint found for {split_name} at {checkpoint_path}')
        return None, None

    print(f'\n{"="*60}')
    print(f'Evaluating : {split_name}')
    print(f'Checkpoint : {checkpoint_path}')
    print(f'{"="*60}')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model      = SSD300(n_classes=len(label_map))
    model.load_state_dict(checkpoint['model'])
    model      = model.to(device)
    model.eval()

    det_boxes       = []
    det_labels      = []
    det_scores      = []
    true_boxes      = []
    true_labels     = []
    true_difficulties = []

    with torch.no_grad():
        for images, boxes, labels, difficulties in tqdm(loader, desc='  Inferring'):
            images = images.to(device)

            predicted_locs, predicted_scores = model(images)

            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=MIN_SCORE, max_overlap=MAX_OVERLAP, top_k=TOP_K,
            )

            boxes        = [b.to(device) for b in boxes]
            labels       = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

    APs, mAP = calculate_mAP(
        det_boxes, det_labels, det_scores,
        true_boxes, true_labels, true_difficulties,
    )

    print(f'\n  Per-class AP:')
    pp.pprint(APs)
    print(f'\n  mAP@0.50 : {mAP:.4f}')

    return APs, mAP


def run_all():
    """
    Evaluate every split that has a saved checkpoint and write a CSV summary.

    To evaluate on the val split instead of the test split (e.g. to compare
    directly with YOLO's per-epoch val metrics), swap test_loader for:

        val_dataset = DIORDataset(DATA_DIR, split='val', split_file=None, split_name='TEST')
        val_loader  = torch.utils.data.DataLoader(val_dataset, ...)

    and pass val_loader to evaluate_split().
    """
    class_names = [rev_label_map[i] for i in range(1, len(label_map))]

    # Actual training-set sizes after the leakage fix (train.txt only, 5,862 images).
    # Old values (586 / 2931 / 5862 / 8793 / 11725) were based on the 11,725-image
    # train+val pool and are now wrong.
    split_image_counts = {
        'split_005':  293,   # int(0.05 * 5862)
        'split_025': 1465,   # int(0.25 * 5862)
        'split_050': 2931,   # int(0.50 * 5862)
        'split_075': 4396,   # int(0.75 * 5862)
        'split_100': 5862,   # 100% of train.txt
    }

    rows = []
    for split in SPLITS:
        APs, mAP = evaluate_split(split, test_loader)
        if APs is None:
            continue
        row = {
            'split'   : split,
            'n_images': split_image_counts[split],
            'mAP'     : round(mAP, 4),
        }
        for cls in class_names:
            row[cls] = round(APs.get(cls, 0.0), 4)
        rows.append(row)

    if not rows:
        print('\nNo checkpoints found. Train at least one split first.')
        return

    # Write CSV summary
    fieldnames = ['split', 'n_images', 'mAP'] + class_names
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nResults saved to {RESULTS_CSV}')

    # Print summary table
    print(f'\n{"split":<12} {"images":>8} {"mAP":>8}')
    print('-' * 32)
    for row in rows:
        print(f'{row["split"]:<12} {row["n_images"]:>8} {row["mAP"]:>8.4f}')


if __name__ == '__main__':
    run_all()
