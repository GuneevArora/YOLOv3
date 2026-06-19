"""Run YOLOv3 on images and save annotated copies.

    python detect.py --weights yolov3.weights --source data/dog.jpg
"""
import argparse
import colorsys
import os

import torch
from PIL import Image, ImageDraw

from yolov3 import COCO_CLASSES, YOLOv3, load_darknet_weights, postprocess
from yolov3.data import letterbox, to_tensor
from yolov3.utils import unletterbox

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_model(weights, num_classes=80, device="cpu"):
    model = YOLOv3(num_classes)
    if weights.endswith(".weights"):
        load_darknet_weights(model, weights)
    else:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model", ckpt))
    return model.to(device).eval()


@torch.no_grad()
def detect_image(model, img, size=416, conf=0.5, nms=0.45):
    """Returns (n, 6) [x1, y1, x2, y2, score, class] in original image pixels."""
    x = to_tensor(letterbox(img, size))[None].to(next(model.parameters()).device)
    det = postprocess(model(x), model, conf, nms)[0].cpu()
    det[:, :4] = unletterbox(det[:, :4], img.width, img.height, size)
    return det


def draw(img, det, names):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, score, cls in det.tolist():
        rgb = colorsys.hsv_to_rgb((int(cls) * 0.618) % 1, 0.9, 0.95)
        color = tuple(int(255 * c) for c in rgb)
        label = f"{names[int(cls)]} {score:.2f}"
        d.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 300))
        tw = d.textlength(label)
        d.rectangle([x1, max(y1 - 12, 0), x1 + tw + 4, max(y1 - 12, 0) + 12], fill=color)
        d.text((x1 + 2, max(y1 - 12, 0)), label, fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Darknet .weights file or a train.py checkpoint")
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--out", default="output")
    ap.add_argument("--img-size", type=int, default=416, help="network input size (multiple of 32)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--names", help="text file with one class name per line (default: COCO)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    names = COCO_CLASSES
    if args.names:
        with open(args.names) as f:
            names = [l.strip() for l in f if l.strip()]
    model = load_model(args.weights, len(names), args.device)

    paths = [args.source]
    if os.path.isdir(args.source):
        paths = sorted(os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(IMG_EXT))
    os.makedirs(args.out, exist_ok=True)
    for p in paths:
        img = Image.open(p).convert("RGB")
        det = detect_image(model, img, args.img_size, args.conf, args.nms)
        print(p)
        for x1, y1, x2, y2, score, cls in det.tolist():
            print(f"  {names[int(cls)]:<15} {score:.2f}  ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        draw(img, det, names).save(os.path.join(args.out, os.path.basename(p)))


if __name__ == "__main__":
    main()
