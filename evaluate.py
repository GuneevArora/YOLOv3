"""Evaluate YOLOv3 detection accuracy.

COCO (official pycocotools numbers: AP, AP50, AP75, APs, APm, APl):
    python evaluate.py --weights yolov3.weights --ann instances_val2014.json --images val2014/

Darknet-format list (built-in AP implementation):
    python evaluate.py --weights yolov3.weights --list coco128/val.txt
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from detect import load_model
from yolov3 import postprocess
from yolov3.data import DetectionDataset, collate, load_coco, load_darknet_list
from yolov3.metrics import average_precision
from yolov3.utils import unletterbox


@torch.no_grad()
def predict(model, samples, img_size=416, batch_size=8, conf=0.005, nms=0.45, workers=4, log=None):
    """Returns {image id: (n, 6) array [x1, y1, x2, y2, score, class]} in original pixels."""
    device = next(model.parameters()).device
    loader = DataLoader(DetectionDataset(samples, img_size), batch_size, collate_fn=collate, num_workers=workers)
    was_training = model.training
    model.eval()
    out = {}
    for i, (imgs, _, metas) in enumerate(loader):
        for det, (image_id, w, h) in zip(postprocess(model(imgs.to(device)), model, conf, nms), metas):
            det = det.cpu()
            det[:, :4] = unletterbox(det[:, :4], w, h, img_size)
            out[image_id] = det.numpy()
        if log and (i + 1) % log == 0:
            print(f"  {i + 1}/{len(loader)} batches", flush=True)
    model.train(was_training)
    return out


def ground_truth(samples):
    """{image id: (m, 5) [x1, y1, x2, y2, class]} in original pixels, from normalised labels."""
    from PIL import Image
    gt = {}
    for path, image_id, boxes in samples:
        with Image.open(path) as im:
            w, h = im.size
        xy, wh = boxes[:, 1:3] * (w, h), boxes[:, 3:5] * (w, h)
        gt[image_id] = np.concatenate([xy - wh / 2, xy + wh / 2, boxes[:, :1]], 1)
    return gt


def evaluate(model, samples, **kw):
    return average_precision(predict(model, samples, **kw), ground_truth(samples))


def coco_evaluate(dets, ann_file, cat_ids, results_file=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    results = [
        {"image_id": i, "category_id": cat_ids[int(c)], "score": float(s),
         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
        for i, d in dets.items() for x1, y1, x2, y2, s, c in d
    ]
    if results_file:
        with open(results_file, "w") as f:
            json.dump(results, f)
    gt = COCO(ann_file)
    ev = COCOeval(gt, gt.loadRes(results), "bbox")
    ev.params.imgIds = list(dets)
    ev.evaluate(), ev.accumulate(), ev.summarize()
    return ev.stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ann", help="COCO instances json")
    ap.add_argument("--images", help="image directory for --ann")
    ap.add_argument("--list", help="Darknet-format image list")
    ap.add_argument("--num-classes", type=int, default=80)
    ap.add_argument("--img-size", type=int, default=416, help="320, 416 and 608 are the paper's settings")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results", help="write COCO-format detections json here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if bool(args.ann) == bool(args.list):
        ap.error("give exactly one of --ann or --list")

    model = load_model(args.weights, args.num_classes, args.device)
    kw = dict(img_size=args.img_size, batch_size=args.batch_size, conf=args.conf, nms=args.nms,
              workers=args.workers, log=20)
    if args.ann:
        samples, cat_ids, _ = load_coco(args.ann, args.images)
        coco_evaluate(predict(model, samples, **kw), args.ann, cat_ids, args.results)
    else:
        samples = load_darknet_list(args.list)
        r = average_precision(predict(model, samples, **kw), ground_truth(samples))
        print(f"AP@[.5:.95] {r['mAP']:.3f}   AP50 {r['AP50']:.3f}   AP75 {r['AP75']:.3f}")


if __name__ == "__main__":
    main()
