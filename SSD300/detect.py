import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from model import SSD300
from utils import rev_label_map, label_color_map

# ── Split and checkpoint to use for detection ──────────────────────────
SPLIT = 'split_005'
EPOCH = None  # None = best checkpoint; integer = periodic snapshot (e.g. 100)
# ────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint():
    if EPOCH is None:
        return os.path.join('checkpoints', SPLIT, 'checkpoint_best.pth.tar')
    return os.path.join('checkpoints', SPLIT, f'checkpoint_epoch_{EPOCH:04d}.pth.tar')


def detect(model, original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300 and return an annotated PIL image.

    :param model: trained SSD300 in eval mode
    :param original_image: PIL Image
    :param min_score: minimum confidence threshold to keep a detection
    :param max_overlap: NMS IoU threshold
    :param top_k: maximum detections to keep per image
    :param suppress: list of class names to ignore in the output
    :return: annotated PIL Image
    """
    resize    = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image = normalize(to_tensor(resize(original_image))).to(device)

    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores,
        min_score=min_score, max_overlap=max_overlap, top_k=top_k
    )

    det_boxes  = det_boxes[0].to('cpu')
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        return original_image

    # Scale fractional box coords back to original image pixels
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height,
         original_image.width, original_image.height]
    ).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    for i in range(det_boxes.size(0)):
        if suppress is not None and det_labels[i] in suppress:
            continue

        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        # Second rectangle offset by 1 pixel to thicken the border
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])

        # Text label — use getbbox() (Pillow 10+ compatible)
        bbox      = font.getbbox(det_labels[i].upper())
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        text_location   = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1],
                            box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)

    del draw
    return annotated_image


def main():
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

    img_path = os.path.join('..', 'data', 'JPEGImages-test', '00001.jpg')
    original_image = Image.open(img_path).convert('RGB')

    result = detect(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200)
    result.show()


if __name__ == '__main__':
    main()
