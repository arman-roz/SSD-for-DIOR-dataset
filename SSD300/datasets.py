import os
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils import transform

# Allow PIL to load slightly truncated JPEGs (DIOR images are fine, but just in case).
ImageFile.LOAD_TRUNCATED_IMAGES = True

# DIOR 20 classes — order must match dior_dataset.py and utils.py dior_labels.
# Index 0 here → SSD label 1 (background occupies label 0 in SSD).
_DIOR_CLASSES = [
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
]
_CLASS_TO_IDX = {name: idx for idx, name in enumerate(_DIOR_CLASSES)}


def _parse_xml(xml_path):
    """Parse a Pascal VOC XML annotation file.

    Returns:
        boxes  : list of [xmin, ymin, xmax, ymax] in absolute pixels (float)
        labels : list of 1-indexed SSD label IDs (int)  — background=0 is reserved
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in _CLASS_TO_IDX:
            continue
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)
        # Skip degenerate boxes — zero width/height causes log(0)=-inf in
        # cxcy_to_gcxgcy, which makes SmoothL1Loss return inf for that prior,
        # collapsing the entire batch loss to inf.
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(_CLASS_TO_IDX[name] + 1)   # 0-indexed → 1-indexed for SSD
    return boxes, labels


class DIORDatasetSSD(Dataset):
    """
    DIOR dataset adapter for SSD300.

    Calls SSD's transform() (from utils.py) which jointly augments the image
    AND the bounding boxes, resizes to 300×300, and converts absolute pixel
    coordinates to fractional coordinates (0–1).

    Args:
        data_dir   : project root — the folder that contains JPEGImages-trainval/,
                     Annotations/, and Main/.
        split      : 'trainval' (uses train.txt + val.txt) or 'test' (uses test.txt).
        split_file : path to a .npy index file from Python_datareader/
                     (e.g. 'split_005_indices.npy').  None = use the full split.
        split_type : 'TRAIN' or 'TEST' — controls whether augmentation is applied
                     inside transform().  Use 'TRAIN' for training, 'TEST' for
                     validation / evaluation.

    __getitem__ returns:
        image       : FloatTensor (3, 300, 300)  — normalised, ready for the model
        boxes       : FloatTensor (N, 4)          — fractional coords [xmin,ymin,xmax,ymax]
        labels      : LongTensor  (N,)            — 1-indexed class IDs (1–20)
        difficulties: ByteTensor  (N,)            — all zeros (DIOR has no 'difficult' flag)
    """

    def __init__(self, data_dir, split='trainval', split_file=None, split_type='TRAIN'):
        assert split in ('train', 'val', 'trainval', 'test'), \
            "split must be 'train', 'val', 'trainval', or 'test'"
        assert split_type in ('TRAIN', 'TEST'), "split_type must be 'TRAIN' or 'TEST'"

        self.split_type = split_type

        # train / val / trainval all live in JPEGImages-trainval/; test has its own folder
        img_dir  = os.path.join(data_dir, 'JPEGImages-trainval' if split != 'test' else 'JPEGImages-test')
        ann_dir  = os.path.join(data_dir, 'Annotations', 'Horizontal Bounding Boxes')
        main_dir = os.path.join(data_dir, 'Main')

        # Collect image IDs for this split
        id_file_map = {
            'train':    ['train.txt'],
            'val':      ['val.txt'],
            'trainval': ['train.txt', 'val.txt'],
            'test':     ['test.txt'],
        }
        id_files = id_file_map[split]
        all_ids = []
        for fname in id_files:
            with open(os.path.join(main_dir, fname)) as f:
                all_ids.extend(line.strip() for line in f if line.strip())

        # Apply .npy subset index if provided.
        # split_file may be a file path (str/Path) or a pre-loaded numpy array.
        if split_file is not None:
            indices = np.load(split_file) if isinstance(split_file, (str, os.PathLike)) else np.asarray(split_file)
            all_ids = [all_ids[i] for i in indices]

        # Build (img_path, xml_path) pairs — skip missing images
        self.samples = []
        for img_id in all_ids:
            img_path = os.path.join(img_dir, f'{img_id}.jpg')
            xml_path = os.path.join(ann_dir, f'{img_id}.xml')
            if not os.path.exists(img_path):
                continue
            # Test images have no XML; store None so __getitem__ can handle them
            ann_path = xml_path if os.path.exists(xml_path) else None
            self.samples.append((img_path, ann_path))

    def __getitem__(self, i):
        img_path, ann_path = self.samples[i]
        image = Image.open(img_path).convert('RGB')

        if ann_path is not None:
            boxes, labels = _parse_xml(ann_path)
        else:
            boxes, labels = [], []

        # Use explicit shape (N,4) so an empty list gives (0,4) not (0,) —
        # a 1-D empty tensor would crash find_jaccard_overlap in MultiBoxLoss.
        boxes  = torch.FloatTensor(boxes).reshape(-1, 4)   # (N, 4) absolute pixel coords
        labels = torch.LongTensor(labels)                   # (N,)   1-indexed
        difficulties = torch.zeros(len(labels), dtype=torch.uint8)  # (N,) all 0

        # transform() from utils.py:
        #   TRAIN — photometric distort, expand, random crop, flip, resize to 300×300
        #   TEST  — resize to 300×300 only
        # In both cases boxes are converted from absolute pixels → fractional (0–1).
        image, boxes, labels, difficulties = transform(
            image, boxes, labels, difficulties, split=self.split_type
        )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        """For use as DataLoader(dataset, collate_fn=DIORDatasetSSD.collate_fn).

        Images are stacked into a single (B, 3, 300, 300) tensor.
        Boxes, labels, and difficulties stay as lists because each image has a
        variable number of objects.
        """
        images, boxes, labels, difficulties = [], [], [], []
        for img, bx, lb, df in batch:
            images.append(img)
            boxes.append(bx)
            labels.append(lb)
            difficulties.append(df)
        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties
