import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from xml.etree import ElementTree as ET
from utils import transform, dior_labels

# Single source of truth: import class names from utils.py where label_map is defined.
# Previously DIOR_CLASSES was a duplicate list defined here — two independent definitions
# that could silently drift out of sync. Now there is only one definition.
DIOR_CLASSES    = list(dior_labels)
DIOR_CLASS_TO_IDX = {c: i for i, c in enumerate(DIOR_CLASSES)}


def _parse_annotation(xml_path):
    """
    Parse a DIOR XML annotation file.

    Returns:
        boxes : (N, 4) float32  [xmin, ymin, xmax, ymax] in absolute pixels
        labels: (N,)   int64    class indices 0-19 (will be shifted to 1-20 in __getitem__)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in DIOR_CLASS_TO_IDX:
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(DIOR_CLASS_TO_IDX[name])
    if boxes:
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


class DIORDataset(Dataset):
    """
    DIOR dataset for SSD300 training and evaluation.

    Replaces PascalVOCDataset. Reads images and XML annotations directly
    from the DIOR folder structure without any JSON preprocessing step.

    Args:
        data_dir:   path to 'Project new/' — the DIOR dataset root
        split:      'trainval' or 'test'
        split_file: path to a .npy index file produced by generate_splits.py
                    (None = use the full split)
        split_name: 'TRAIN' or 'TEST' — controls augmentation inside transform()
    """

    def __init__(self, data_dir, split='trainval', split_file=None, split_name='TRAIN'):
        assert split in ('train', 'val', 'test'), "split must be 'train', 'val', or 'test'"
        assert split_name in ('TRAIN', 'TEST'), "split_name must be 'TRAIN' or 'TEST'"

        self.split_name = split_name

        # 'train' and 'val' images both live in JPEGImages-trainval/
        img_folder = 'JPEGImages-trainval' if split in ('train', 'val') else 'JPEGImages-test'
        img_dir  = os.path.join(data_dir, img_folder)
        ann_dir  = os.path.join(data_dir, 'Annotations', 'Horizontal Bounding Boxes')
        main_dir = os.path.join(data_dir, 'Main')

        # Read image IDs for this split
        if split == 'train':
            id_files = ['train.txt']        # training pool — subset mask applied below
        elif split == 'val':
            id_files = ['val.txt']          # held-out validation — never subsetted
        else:
            id_files = ['test.txt']         # final evaluation set
        all_ids = []
        for fname in id_files:
            with open(os.path.join(main_dir, fname)) as f:
                all_ids.extend(line.strip() for line in f if line.strip())

        # Apply the data-efficiency subset mask if provided
        if split_file is not None:
            indices = np.load(split_file)
            all_ids = [all_ids[i] for i in indices]

        # Build per-sample (image_path, annotation_path) records.
        # Images with zero annotated objects are skipped: an empty boxes tensor
        # causes RuntimeError in MultiBoxLoss.forward (overlap.max on a 0-row
        # tensor) and in random_crop (overlap.max on a 0-element tensor).
        self.samples = []
        skipped = 0
        for img_id in all_ids:
            img_path = os.path.join(img_dir, f'{img_id}.jpg')
            xml_path = os.path.join(ann_dir, f'{img_id}.xml')
            if not os.path.exists(img_path):
                continue
            if not os.path.exists(xml_path):
                skipped += 1
                continue
            boxes, _ = _parse_annotation(xml_path)
            if len(boxes) == 0:
                skipped += 1
                continue
            self.samples.append((img_path, xml_path))
        if skipped:
            print(f'  [{split}] Skipped {skipped} images with no annotated objects.')

    def __getitem__(self, i):
        img_path, ann_path = self.samples[i]
        image = Image.open(img_path).convert('RGB')

        # ann_path is always a valid XML path — empty-annotation images were
        # filtered out during __init__, so no None-check needed here.
        boxes, labels = _parse_annotation(ann_path)
        boxes  = torch.FloatTensor(boxes)
        # Shift labels from 0-19 (DIOR) → 1-20 (SSD, where 0 = background)
        labels = torch.LongTensor(labels) + 1
        # DIOR has no difficulty annotations — use zeros so MultiBoxLoss works unchanged
        difficulties = torch.zeros(labels.size(0), dtype=torch.uint8)

        # Resize to 300×300, normalize, and apply augmentation (only for TRAIN)
        image, boxes, labels, difficulties = transform(
            image, boxes, labels, difficulties, split=self.split_name
        )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        """
        DataLoader collate function.
        Images are stacked into a single tensor; boxes, labels, and difficulties
        remain as lists because each image has a variable number of objects.
        """
        images, boxes, labels, difficulties = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(boxes), list(labels), list(difficulties)
