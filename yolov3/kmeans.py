"""k-means clustering of box dimensions to choose anchor priors (Section 2.3)."""
import numpy as np


def _iou(wh, centroids):
    inter = np.minimum(wh[:, None], centroids[None]).prod(-1)
    return inter / (wh.prod(-1)[:, None] + centroids.prod(-1)[None] - inter)


def kmeans_anchors(wh, k=9, iters=300, seed=0):
    """Cluster (N, 2) box sizes with distance d = 1 - IoU.

    Returns (anchors sorted by area as a (k, 2) array, mean best IoU).
    """
    wh = np.asarray(wh, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centroids = wh[rng.choice(len(wh), k, replace=False)]
    assign = None
    for _ in range(iters):
        new = _iou(wh, centroids).argmax(1)
        if assign is not None and (new == assign).all():
            break
        assign = new
        for j in range(k):
            if (assign == j).any():
                centroids[j] = wh[assign == j].mean(0)
    centroids = centroids[np.argsort(centroids.prod(-1))]
    return centroids, float(_iou(wh, centroids).max(1).mean())
