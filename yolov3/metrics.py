"""Average precision, following the COCO protocol (101-point interpolation)."""
import numpy as np


def _iou(box, boxes):
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    return inter / ((box[2:] - box[:2]).prod() + (boxes[:, 2:] - boxes[:, :2]).prod(1) - inter + 1e-9)


def average_precision(detections, ground_truth, iou_thresholds=np.arange(0.5, 0.96, 0.05)):
    """Compute detection AP.

    detections:   {image id: array (n, 6) of [x1, y1, x2, y2, score, class]}
    ground_truth: {image id: array (m, 5) of [x1, y1, x2, y2, class]}
    Returns {"mAP": mean over thresholds and classes, "AP50": ..., "AP75": ...,
             "per_class": {class: AP averaged over thresholds}}.
    """
    iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    classes = sorted({int(c) for g in ground_truth.values() for c in g[:, 4]})
    recall_grid = np.linspace(0, 1, 101)
    ap = np.zeros((len(classes), len(iou_thresholds)))

    for ci, c in enumerate(classes):
        gts = {i: g[g[:, 4] == c, :4] for i, g in ground_truth.items()}
        n_gt = sum(len(g) for g in gts.values())
        dets = [(d[4], i, d[:4]) for i, ds in detections.items() for d in ds if int(d[5]) == c]
        dets.sort(key=lambda d: -d[0])
        tp = np.zeros((len(dets), len(iou_thresholds)), dtype=bool)
        used = {i: np.zeros((len(g), len(iou_thresholds)), dtype=bool) for i, g in gts.items()}
        for di, (_, i, box) in enumerate(dets):
            g = gts.get(i)
            if g is None or not len(g):
                continue
            iou = _iou(box, g)
            for ti, t in enumerate(iou_thresholds):
                cand = np.where(used[i][:, ti], -1, iou)
                j = cand.argmax()
                if cand[j] >= t:
                    tp[di, ti] = used[i][j, ti] = True
        if not len(dets):
            continue
        ctp = tp.cumsum(0)
        recall = ctp / max(n_gt, 1)
        precision = ctp / np.arange(1, len(dets) + 1)[:, None]
        for ti in range(len(iou_thresholds)):
            p = np.maximum.accumulate(precision[::-1, ti])[::-1]  # monotone envelope
            idx = np.searchsorted(recall[:, ti], recall_grid, side="left")
            ap[ci, ti] = np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0).mean()

    def at(t):
        k = np.where(np.isclose(iou_thresholds, t))[0]
        return float(ap[:, k[0]].mean()) if len(k) and len(classes) else float("nan")

    return {
        "mAP": float(ap.mean()) if len(classes) else float("nan"),
        "AP50": at(0.5),
        "AP75": at(0.75),
        "per_class": {c: float(ap[ci].mean()) for ci, c in enumerate(classes)},
    }
