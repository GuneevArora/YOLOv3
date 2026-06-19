"""Train Darknet-53 as an ImageNet classifier (Section 2.4, Table 2).

    python train_classifier.py --train imagenet/train --val imagenet/val

Expects torchvision ImageFolder layouts. Defaults follow darknet53.cfg: 256x256 inputs,
batch 128, SGD momentum 0.9, weight decay 5e-4, lr 0.1 with polynomial (power 4) decay
over 800,000 iterations. Save the backbone with --export to initialise the detector.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import make_optimizer
from yolov3 import Darknet53Classifier


def backbone_modules(model):
    """Darknet-53's 52 convolutions in serialisation order (for darknet53.conv.74)."""
    mods = [model.backbone.stem]
    for stage in model.backbone.stages:
        for layer in stage:
            mods += [layer.conv1, layer.conv2] if hasattr(layer, "conv1") else [layer]
    return mods


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    top1 = top5 = n = 0
    for x, y in loader:
        top = model(x.to(device)).topk(min(5, model.fc.out_features), 1).indices.cpu()
        top1 += (top[:, 0] == y).sum().item()
        top5 += (top == y[:, None]).any(1).sum().item()
        n += len(y)
    model.train()
    return top1 / n, top5 / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", default="runs/darknet53")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--max-batches", type=int, default=800000)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--export", action="store_true", help="also write darknet53.conv.74")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    # darknet53.cfg: aspect 0.75, angle 7, hue 0.1, saturation / exposure 0.75.
    train_tf = transforms.Compose([
        transforms.RandomRotation(7),
        transforms.RandomResizedCrop(args.img_size, scale=(0.25, 1), ratio=(0.75, 1 / 0.75)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, saturation=0.25, hue=0.1),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor()])
    train = datasets.ImageFolder(args.train, train_tf)
    loader = DataLoader(train, args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    val = DataLoader(datasets.ImageFolder(args.val, val_tf), args.batch_size, num_workers=args.workers) if args.val else None

    model = Darknet53Classifier(len(train.classes)).to(device)
    opt = make_optimizer(model, args.lr, 0.9, 5e-4)
    it = 0
    while it < args.max_batches:
        for x, y in loader:
            for g in opt.param_groups:
                g["lr"] = args.lr * (1 - it / args.max_batches) ** 4
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % 10 == 0 or it == 1:
                print(f"{it:>7}  loss {loss.item():.3f}", flush=True)
            if it % args.eval_every == 0 or it == args.max_batches:
                torch.save({"model": model.state_dict(), "iteration": it}, os.path.join(args.out, "last.pt"))
                if val:
                    top1, top5 = accuracy(model, val, device)
                    print(f"{it:>7}  top-1 {top1:.3f}  top-5 {top5:.3f}", flush=True)
            if it >= args.max_batches:
                break
    if args.export:
        from yolov3 import save_darknet_weights
        model.darknet_modules = lambda: backbone_modules(model)
        save_darknet_weights(model, os.path.join(args.out, "darknet53.conv.74"))


if __name__ == "__main__":
    main()
