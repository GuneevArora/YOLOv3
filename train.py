"""Train YOLOv3 with the schedule from the paper / yolov3.cfg.

COCO, initialised from the ImageNet-pretrained Darknet-53 backbone:
    python train.py --ann instances_train.json --images train/ --weights darknet53.conv.74

Darknet-format list:
    python train.py --list train.txt --num-classes 20 --weights darknet53.conv.74

Defaults reproduce yolov3.cfg: batch 64 (16 subdivisions of 4 images), SGD momentum 0.9,
weight decay 5e-4, lr 1e-3 with a 1000-iteration burn-in and x0.1 at 400k and 450k of
500,200 iterations, multi-scale 320-608 changing every 10 batches.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolov3 import TRAIN, YOLOv3, YOLOv3Loss, load_darknet_weights
from yolov3.data import DetectionDataset, MultiScaleBatchSampler, collate, load_coco, load_darknet_list

BACKBONE_LAYERS = 52  # weighted layers in darknet53.conv.74


def learning_rate(it, base, burn_in, steps, scales):
    """Darknet "steps" policy with polynomial (power 4) burn-in."""
    if it < burn_in:
        return base * ((it + 1) / burn_in) ** 4
    for step, scale in zip(steps, scales):
        if it >= step:
            base *= scale
    return base


def make_optimizer(model, lr, momentum, weight_decay):
    """SGD; as in Darknet, weight decay applies to convolution weights only."""
    decay = [p for n, p in model.named_parameters() if p.ndim == 4]
    no_decay = [p for n, p in model.named_parameters() if p.ndim != 4]
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum,
    )


def main():
    T = TRAIN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", help="COCO instances json for training")
    ap.add_argument("--images", help="image directory for --ann")
    ap.add_argument("--list", help="Darknet-format training image list")
    ap.add_argument("--val-list", help="Darknet-format list evaluated at every checkpoint")
    ap.add_argument("--num-classes", type=int, default=80)
    ap.add_argument("--weights", help="darknet53.conv.74, a full .weights file, or a checkpoint to resume")
    ap.add_argument("--out", default="runs/yolov3")
    ap.add_argument("--batch-size", type=int, default=T["batch_size"])
    ap.add_argument("--subdivisions", type=int, default=T["subdivisions"])
    ap.add_argument("--lr", type=float, default=T["learning_rate"])
    ap.add_argument("--momentum", type=float, default=T["momentum"])
    ap.add_argument("--weight-decay", type=float, default=T["weight_decay"])
    ap.add_argument("--burn-in", type=int, default=T["burn_in"])
    ap.add_argument("--max-batches", type=int, default=T["max_batches"])
    ap.add_argument("--steps", type=int, nargs="*", default=list(T["steps"]))
    ap.add_argument("--img-size", type=int, default=416, help="training size when --no-multiscale")
    ap.add_argument("--no-multiscale", action="store_true")
    ap.add_argument("--ignore-thresh", type=float, default=0.5, help="paper: 0.5 (yolov3.cfg uses 0.7)")
    ap.add_argument("--xy-loss", choices=("mse", "bce"), default="mse")
    ap.add_argument("--bn-momentum", type=float, default=0.01)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if bool(args.ann) == bool(args.list):
        ap.error("give exactly one of --ann or --list")
    if args.batch_size % args.subdivisions:
        ap.error("--batch-size must be divisible by --subdivisions")

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    samples = load_coco(args.ann, args.images)[0] if args.ann else load_darknet_list(args.list)
    mini = args.batch_size // args.subdivisions
    lo, hi = (args.img_size, args.img_size) if args.no_multiscale else T["multiscale"]
    dataset = DetectionDataset(samples, train=True, jitter=T["jitter"], hue=T["hue"],
                               saturation=T["saturation"], exposure=T["exposure"])
    sampler = MultiScaleBatchSampler(len(samples), mini, lo, hi, T["multiscale_interval"] * args.subdivisions)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)

    model = YOLOv3(args.num_classes, bn_momentum=args.bn_momentum).to(device)
    optimizer = make_optimizer(model, args.lr, args.momentum, args.weight_decay)
    it = 0
    if args.weights:
        if args.weights.endswith(".74"):
            load_darknet_weights(model, args.weights, cutoff=BACKBONE_LAYERS)
        elif args.weights.endswith(".weights"):
            load_darknet_weights(model, args.weights)
        else:
            ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            it = ckpt["iteration"]
    criterion = YOLOv3Loss(model, args.ignore_thresh, args.xy_loss)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    val = load_darknet_list(args.val_list) if args.val_list else None

    def save(name):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": it},
                   os.path.join(args.out, name))

    print(f"{len(samples)} images, {args.batch_size} per batch in {args.subdivisions} subdivisions, "
          f"iterations {it} -> {args.max_batches}, device {device}", flush=True)
    model.train()
    optimizer.zero_grad()
    sub, t0 = 0, time.time()
    running = {}
    while it < args.max_batches:
        for imgs, targets, _ in loader:
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                outputs = model(imgs)
            loss, parts = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at iteration {it}: {parts}")
            scaler.scale(loss / args.subdivisions).backward()
            for k, v in {"loss": loss.detach(), **parts}.items():
                running[k] = running.get(k, 0.0) + v.item() / args.subdivisions
            sub += 1
            if sub < args.subdivisions:
                continue

            lr = learning_rate(it, args.lr, args.burn_in, args.steps, T["scales"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            sub, it = 0, it + 1

            if it % args.log_every == 0 or it == 1:
                msg = "  ".join(f"{k} {v:.3f}" for k, v in running.items())
                print(f"{it:>7}  {msg}  lr {lr:.2e}  size {imgs.shape[-1]}  {time.time() - t0:.0f}s", flush=True)
            running = {}
            if it % args.save_every == 0 or it == args.max_batches:
                save("last.pt")
                if val:
                    from evaluate import evaluate
                    r = evaluate(model, val, workers=args.workers)
                    print(f"{it:>7}  val AP@[.5:.95] {r['mAP']:.3f}  AP50 {r['AP50']:.3f}", flush=True)
            if it >= args.max_batches:
                break
    save("last.pt")


if __name__ == "__main__":
    main()
