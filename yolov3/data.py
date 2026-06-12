"""Datasets, Darknet-style augmentation and multi-scale batching."""
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .utils import letterbox_params

GRAY = (128, 128, 128)


def load_coco(ann_file, img_dir):
    """Parse a COCO instances json.

    Returns (samples, category_ids, class_names) where samples is a list of
    (image path, image id, float32 array of [class, cx, cy, w, h] normalised to 0-1) and
    category_ids maps the contiguous class index back to the COCO category id.
    Crowd regions are dropped.
    """
    with open(ann_file) as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_ids = [c["id"] for c in cats]
    index = {cid: i for i, cid in enumerate(cat_ids)}
    labels = {im["id"]: [] for im in coco["images"]}
    info = {im["id"]: im for im in coco["images"]}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        im = info[a["image_id"]]
        x, y, w, h = a["bbox"]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, im["width"]), min(y + h, im["height"])
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        labels[a["image_id"]].append([
            index[a["category_id"]],
            (x1 + x2) / 2 / im["width"], (y1 + y2) / 2 / im["height"],
            (x2 - x1) / im["width"], (y2 - y1) / im["height"],
        ])
    samples = [
        (os.path.join(img_dir, info[i]["file_name"]), i, np.array(l, dtype=np.float32).reshape(-1, 5))
        for i, l in sorted(labels.items())
    ]
    return samples, cat_ids, [c["name"] for c in cats]


def load_darknet_list(list_file):
    """Parse a Darknet image list: one image path per line, labels in the matching
    ``labels/<name>.txt`` with rows of ``class cx cy w h`` (normalised)."""
    samples = []
    base = os.path.dirname(os.path.abspath(list_file))
    with open(list_file) as f:
        paths = [l.strip() for l in f if l.strip()]
    for i, p in enumerate(paths):
        p = p if os.path.isabs(p) else os.path.join(base, p)
        sep = os.sep
        label = os.path.splitext(p.replace(f"{sep}images{sep}", f"{sep}labels{sep}"))[0] + ".txt"
        boxes = np.zeros((0, 5), dtype=np.float32)
        if os.path.exists(label) and os.path.getsize(label):
            boxes = np.loadtxt(label, dtype=np.float32, ndmin=2).reshape(-1, 5)
        samples.append((p, i, boxes))
    return samples


def _rand_scale(s):
    scale = random.uniform(1, s)
    return scale if random.random() < 0.5 else 1 / scale


def distort_hsv(img, hue, saturation, exposure):
    """Random hue shift and saturation / exposure scaling, as in Darknet."""
    dhue = random.uniform(-hue, hue)
    dsat, dexp = _rand_scale(saturation), _rand_scale(exposure)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hsv[..., 0] = (hsv[..., 0] + dhue * 255) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * dsat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * dexp, 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def augment(img, boxes, size, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5):
    """Darknet detection augmentation: aspect-ratio jitter, random placement on a gray
    canvas, colour distortion and horizontal flip. Boxes are [class, cx, cy, w, h]."""
    ow, oh = img.size
    dw, dh = jitter * ow, jitter * oh
    new_ar = (ow + random.uniform(-dw, dw)) / (oh + random.uniform(-dh, dh))
    if new_ar < 1:
        nh = size
        nw = nh * new_ar
    else:
        nw = size
        nh = nw / new_ar
    nw, nh = max(int(round(nw)), 1), max(int(round(nh)), 1)
    dx, dy = random.randint(0, size - nw), random.randint(0, size - nh)

    canvas = Image.new("RGB", (size, size), GRAY)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    canvas = distort_hsv(canvas, hue, saturation, exposure)
    flip = random.random() < 0.5
    if flip:
        canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)

    boxes = boxes.copy()
    if len(boxes):
        x1 = ((boxes[:, 1] - boxes[:, 3] / 2) * nw + dx) / size
        x2 = ((boxes[:, 1] + boxes[:, 3] / 2) * nw + dx) / size
        y1 = ((boxes[:, 2] - boxes[:, 4] / 2) * nh + dy) / size
        y2 = ((boxes[:, 2] + boxes[:, 4] / 2) * nh + dy) / size
        if flip:
            x1, x2 = 1 - x2, 1 - x1
        x1, y1, x2, y2 = (np.clip(v, 0, 1) for v in (x1, y1, x2, y2))
        boxes[:, 1], boxes[:, 2] = (x1 + x2) / 2, (y1 + y2) / 2
        boxes[:, 3], boxes[:, 4] = x2 - x1, y2 - y1
        boxes = boxes[(boxes[:, 3] > 1e-3) & (boxes[:, 4] > 1e-3)]
    return canvas, boxes


def letterbox(img, size, boxes=None):
    """Resize keeping aspect ratio and pad to size x size with gray (Darknet test-time)."""
    w, h = img.size
    _, dx, dy, nw, nh = letterbox_params(w, h, size)
    canvas = Image.new("RGB", (size, size), GRAY)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    if boxes is None:
        return canvas
    boxes = boxes.copy()
    if len(boxes):
        boxes[:, 1] = (boxes[:, 1] * nw + dx) / size
        boxes[:, 2] = (boxes[:, 2] * nh + dy) / size
        boxes[:, 3] *= nw / size
        boxes[:, 4] *= nh / size
    return canvas, boxes


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1).contiguous()


class DetectionDataset(Dataset):
    """Yields (image tensor, labels (n, 5), meta). Index may be ``(i, size)`` so a batch
    sampler can pick the resolution for multi-scale training."""

    def __init__(self, samples, img_size=416, train=False, **aug):
        self.samples, self.img_size, self.train, self.aug = samples, img_size, train, aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index, size = index if isinstance(index, tuple) else (index, self.img_size)
        path, image_id, boxes = self.samples[index]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if self.train:
            img, boxes = augment(img, boxes, size, **self.aug)
        else:
            img, boxes = letterbox(img, size, boxes)
        return to_tensor(img), torch.from_numpy(boxes), (image_id, w, h)


def collate(batch):
    """Stack images; labels become (N, 6) rows of [image index in batch, class, cx, cy, w, h]."""
    imgs, labels, metas = zip(*batch)
    targets = [torch.cat([torch.full((len(l), 1), i, dtype=torch.float32), l.float()], 1) for i, l in enumerate(labels)]
    return torch.stack(imgs), torch.cat(targets), list(metas)


class MultiScaleBatchSampler(Sampler):
    """Shuffled batches whose resolution changes every ``interval`` batches to a random
    multiple of 32 in [min_size, max_size] ("random=1" in yolov3.cfg, 320 to 608)."""

    def __init__(self, n, batch_size, min_size=320, max_size=608, interval=10, drop_last=True):
        self.n, self.batch_size, self.interval, self.drop_last = n, batch_size, interval, drop_last
        self.sizes = list(range(min_size, max_size + 1, 32))

    def __len__(self):
        return self.n // self.batch_size if self.drop_last else -(-self.n // self.batch_size)

    def __iter__(self):
        order = torch.randperm(self.n).tolist()
        size = random.choice(self.sizes)
        for b in range(len(self)):
            if b % self.interval == 0:
                size = random.choice(self.sizes)
            yield [(i, size) for i in order[b * self.batch_size:(b + 1) * self.batch_size]]
