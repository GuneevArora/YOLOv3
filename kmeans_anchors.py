"""Choose anchor priors by k-means over training-set box sizes (Section 2.3).

    python kmeans_anchors.py --ann instances_train.json --images train/ --img-size 416
"""
import argparse

import numpy as np

from yolov3.data import load_coco, load_darknet_list
from yolov3.kmeans import kmeans_anchors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann")
    ap.add_argument("--images", default="")
    ap.add_argument("--list")
    ap.add_argument("-k", type=int, default=9)
    ap.add_argument("--img-size", type=int, default=416)
    args = ap.parse_args()
    samples = load_coco(args.ann, args.images)[0] if args.ann else load_darknet_list(args.list)
    wh = np.concatenate([s[2][:, 3:5] for s in samples]) * args.img_size
    anchors, iou = kmeans_anchors(wh, args.k)
    print("anchors:", ", ".join(f"({w:.0f}, {h:.0f})" for w, h in anchors))
    print(f"mean IoU with closest anchor: {iou:.3f}")


if __name__ == "__main__":
    main()
