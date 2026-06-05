"""Box utilities, prediction decoding and non-maximum suppression."""
import torch


def xywh2xyxy(b):
    xy, wh = b[..., :2], b[..., 2:4]
    return torch.cat([xy - wh / 2, xy + wh / 2], -1)


def xyxy2xywh(b):
    return torch.cat([(b[..., :2] + b[..., 2:4]) / 2, b[..., 2:4] - b[..., :2]], -1)


def box_iou(a, b):
    """Pairwise IoU between xyxy boxes a (N, 4) and b (M, 4) -> (N, M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def wh_iou(a, b):
    """IoU of boxes sharing a centre, given only (w, h): (N, 2) x (M, 2) -> (N, M)."""
    inter = torch.min(a[:, None], b[None]).prod(-1)
    return inter / (a.prod(-1)[:, None] + b.prod(-1)[None] - inter + 1e-9)


def decode(raw, anchors, stride):
    """Apply the box equations of Section 2.1 to one raw output scale.

        bx = sigmoid(tx) + cx      bw = pw * exp(tw)
        by = sigmoid(ty) + cy      bh = ph * exp(th)

    raw: (B, A, H, W, 5 + C); anchors: (A, 2) in input pixels.
    Returns (B, A, H, W, 5 + C) with xywh in input pixels and sigmoid-ed scores.
    """
    _, _, h, w, _ = raw.shape
    gy, gx = torch.meshgrid(
        torch.arange(h, device=raw.device, dtype=raw.dtype),
        torch.arange(w, device=raw.device, dtype=raw.dtype),
        indexing="ij",
    )
    xy = (raw[..., :2].sigmoid() + torch.stack([gx, gy], -1)) * stride
    wh = raw[..., 2:4].clamp(max=20).exp() * anchors.to(raw)[None, :, None, None, :]
    return torch.cat([xy, wh, raw[..., 4:].sigmoid()], -1)


def nms(boxes, scores, iou_thresh):
    """Greedy NMS on xyxy boxes. Returns kept indices sorted by descending score."""
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    iou = box_iou(boxes, boxes).triu_(1)
    keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
    for i in range(len(boxes)):
        if keep[i]:
            keep &= ~(iou[i] > iou_thresh)
    return order[keep]


@torch.no_grad()
def postprocess(outputs, model, conf_thresh=0.5, nms_thresh=0.45, max_det=300, max_candidates=5000):
    """Raw model outputs -> list (per image) of (N, 6) tensors [x1, y1, x2, y2, score, class].

    Scores are objectness * class probability; because classes are independent logistic
    outputs (Section 2.2) a box can be reported for more than one class. NMS is per class.
    """
    preds = []
    for raw, mask, stride in zip(outputs, model.masks, model.strides):
        d = decode(raw.float(), model.anchors[list(mask)], stride)
        preds.append(d.flatten(1, 3))
    preds = torch.cat(preds, 1)

    results = []
    for p in preds:
        p = p[p[:, 4] > conf_thresh]
        scores = p[:, 4:5] * p[:, 5:]
        idx, cls = (scores > conf_thresh).nonzero(as_tuple=True)
        boxes, scores = xywh2xyxy(p[idx, :4]), scores[idx, cls]
        if len(scores) > max_candidates:
            top = scores.topk(max_candidates).indices
            boxes, scores, cls = boxes[top], scores[top], cls[top]
        # Offset boxes by class so one NMS pass never suppresses across classes.
        offset = cls.to(boxes)[:, None] * (boxes.max() + 1 if len(boxes) else 1)
        keep = nms(boxes + offset, scores, nms_thresh)[:max_det]
        results.append(torch.cat([boxes[keep], scores[keep, None], cls[keep, None].to(boxes)], 1))
    return results


def letterbox_params(w, h, size):
    """Scale and padding that fit a (w, h) image inside a size x size square."""
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    return r, (size - nw) // 2, (size - nh) // 2, nw, nh


def unletterbox(boxes, w, h, size):
    """Map xyxy boxes from letterboxed network coordinates back to the original image."""
    r, dx, dy, _, _ = letterbox_params(w, h, size)
    boxes = boxes.clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - dx) / r).clamp(0, w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - dy) / r).clamp(0, h)
    return boxes
