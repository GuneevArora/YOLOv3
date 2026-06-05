"""YOLOv3 training loss (Sections 2.1 - 2.2 of the paper)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import box_iou, wh_iou, xywh2xyxy


class YOLOv3Loss(nn.Module):
    """Loss over the three output scales.

    * Each ground-truth box is assigned to exactly one prior: the one (of all nine) whose
      shape overlaps it the most. That prior, at the grid cell containing the box centre,
      gets objectness target 1 plus coordinate and class losses.
    * Priors that are not assigned but whose predicted box overlaps some ground truth by
      more than ``ignore_thresh`` (0.5 in the paper) incur no loss at all.
    * All other priors only incur objectness loss, with target 0.
    * Coordinates use sum of squared error on (sigmoid(tx), sigmoid(ty), tw, th);
      objectness and classes use binary cross-entropy (no softmax).

    ``xy_loss="bce"`` and ``box_scale=True`` reproduce two details of the Darknet
    implementation that the paper does not spell out: Darknet's gradient for x/y is that
    of a cross-entropy, and coordinate errors are weighted by (2 - w*h) to favour small
    boxes. ``box_scale`` is on by default, ``xy_loss`` defaults to the paper's "mse".

    targets: (N, 6) rows of [image index, class, cx, cy, w, h], boxes normalised to 0-1.
    Returns (loss, dict of components). Everything is summed and divided by batch size.
    """

    def __init__(self, model, ignore_thresh=0.5, xy_loss="mse", box_scale=True):
        super().__init__()
        assert xy_loss in ("mse", "bce")
        self.anchors = model.anchors
        self.masks = model.masks
        self.strides = model.strides
        self.num_classes = model.num_classes
        self.ignore_thresh = ignore_thresh
        self.xy_loss = xy_loss
        self.box_scale = box_scale

    def forward(self, outputs, targets):
        batch = outputs[0].shape[0]
        img_h = outputs[0].shape[2] * self.strides[0]
        img_w = outputs[0].shape[3] * self.strides[0]
        img_wh = torch.tensor([img_w, img_h], device=targets.device, dtype=torch.float32)
        anchors = self.anchors.to(targets.device)

        # Best prior for every ground truth, compared by shape only.
        gt_wh_px = targets[:, 4:6] * img_wh
        best = wh_iou(gt_wh_px, anchors).argmax(1) if len(targets) else targets.new_zeros(0, dtype=torch.long)

        zero = outputs[0].sum() * 0
        l_xy, l_wh, l_obj, l_cls = zero, zero, zero, zero
        for raw, mask in zip(outputs, self.masks):
            raw = raw.float()
            _, n_a, h, w, _ = raw.shape
            a = anchors[list(mask)]
            obj_target = torch.zeros(batch, n_a, h, w, device=raw.device)
            obj_weight = torch.ones_like(obj_target)

            if len(targets):
                self._apply_ignore(raw, a, targets, img_wh, obj_weight)

                mask_t = torch.tensor(mask, device=raw.device)
                hit = best[:, None] == mask_t[None]
                sel = hit.any(1)
                t, ai = targets[sel], hit[sel].float().argmax(1)
                if len(t):
                    b, cls = t[:, 0].long(), t[:, 1].long()
                    gx, gy = t[:, 2] * w, t[:, 3] * h
                    gi, gj = gx.long().clamp(0, w - 1), gy.long().clamp(0, h - 1)
                    obj_target[b, ai, gj, gi] = 1
                    obj_weight[b, ai, gj, gi] = 1

                    p = raw[b, ai, gj, gi]
                    t_xy = torch.stack([gx - gi, gy - gj], 1)
                    t_wh = torch.log(t[:, 4:6] * img_wh / a[ai] + 1e-16)
                    scale = (2 - t[:, 4] * t[:, 5])[:, None] if self.box_scale else 1.0
                    if self.xy_loss == "mse":
                        l_xy = l_xy + (scale * (p[:, :2].sigmoid() - t_xy) ** 2).sum()
                    else:
                        l_xy = l_xy + (scale * F.binary_cross_entropy_with_logits(p[:, :2], t_xy, reduction="none")).sum()
                    l_wh = l_wh + (scale * (p[:, 2:4] - t_wh) ** 2).sum()
                    t_cls = F.one_hot(cls, self.num_classes).float()
                    l_cls = l_cls + F.binary_cross_entropy_with_logits(p[:, 5:], t_cls, reduction="sum")

            l_obj = l_obj + (obj_weight * F.binary_cross_entropy_with_logits(raw[..., 4], obj_target, reduction="none")).sum()

        parts = {"xy": l_xy / batch, "wh": l_wh / batch, "obj": l_obj / batch, "cls": l_cls / batch}
        return sum(parts.values()), {k: v.detach() for k, v in parts.items()}

    @torch.no_grad()
    def _apply_ignore(self, raw, a, targets, img_wh, obj_weight):
        """Zero the objectness weight where a prediction already overlaps a ground truth."""
        _, _, h, w, _ = raw.shape
        gy, gx = torch.meshgrid(
            torch.arange(h, device=raw.device, dtype=raw.dtype),
            torch.arange(w, device=raw.device, dtype=raw.dtype),
            indexing="ij",
        )
        grid_wh = torch.tensor([w, h], device=raw.device, dtype=raw.dtype)
        xy = (raw[..., :2].sigmoid() + torch.stack([gx, gy], -1)) / grid_wh
        wh = raw[..., 2:4].clamp(max=math.log(1e4)).exp() * (a / img_wh)[None, :, None, None, :]
        pred = xywh2xyxy(torch.cat([xy, wh], -1))
        for b in targets[:, 0].unique().long():
            gt = xywh2xyxy(targets[targets[:, 0] == b, 2:6])
            iou = box_iou(pred[b].reshape(-1, 4), gt).max(1).values
            obj_weight[b].view(-1)[iou > self.ignore_thresh] = 0
