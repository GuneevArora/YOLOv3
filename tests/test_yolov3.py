import math
import os
import random

import numpy as np
import pytest
import torch
from PIL import Image

from train import learning_rate, make_optimizer
from yolov3 import (ANCHORS, Darknet53Classifier, YOLOv3, YOLOv3Loss, load_darknet_weights,
                    postprocess, save_darknet_weights)
from yolov3.data import MultiScaleBatchSampler, augment, letterbox
from yolov3.kmeans import kmeans_anchors
from yolov3.metrics import average_precision
from yolov3.utils import box_iou, nms, unletterbox, wh_iou


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return YOLOv3(80)


def test_architecture(model):
    # 61.9M parameters is the size of the reference yolov3.cfg network.
    assert sum(p.numel() for p in model.parameters()) == 61_949_149
    assert len(model.darknet_modules()) == 75  # 52 backbone + 23 head convolutions
    cls = Darknet53Classifier()
    assert sum(isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) for m in cls.modules()) == 53
    with torch.no_grad():
        out = model.eval()(torch.zeros(1, 3, 96, 160))
    assert [tuple(o.shape) for o in out] == [(1, 3, 3, 5, 85), (1, 3, 6, 10, 85), (1, 3, 12, 20, 85)]


def test_darknet_weights_roundtrip(model, tmp_path):
    path = tmp_path / "m.weights"
    save_darknet_weights(model, path, seen=123)
    assert os.path.getsize(path) == 248_007_048  # exact size of the official yolov3.weights
    other = YOLOv3(80)
    assert load_darknet_weights(other, path) == 123
    for a, b in zip(model.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError):
        load_darknet_weights(YOLOv3(20), path)


def encode(model, targets, size, logit=20.0):
    """Raw network outputs that predict exactly ``targets`` and nothing else."""
    batch = int(targets[:, 0].max()) + 1
    outs = [torch.zeros(batch, 3, size // s, size // s, 85) for s in model.strides]
    for o in outs:
        o[..., 4:] = -logit
    for b, c, cx, cy, w, h in targets.tolist():
        k = int(wh_iou(torch.tensor([[w * size, h * size]]), model.anchors).argmax())
        scale = next(i for i, m in enumerate(model.masks) if k in m)
        a, g = model.masks[scale].index(k), size // model.strides[scale]
        i, j = int(cx * g), int(cy * g)
        cell = outs[scale][int(b), a, j, i]
        cell[0], cell[1] = torch.logit(torch.tensor([cx * g - i, cy * g - j]), eps=1e-6)
        cell[2] = math.log(w * size / ANCHORS[k][0])
        cell[3] = math.log(h * size / ANCHORS[k][1])
        cell[4] = cell[5 + int(c)] = logit
    return outs


TARGETS = torch.tensor([
    [0, 16, 0.31, 0.62, 0.40, 0.55],   # large
    [0, 1, 0.70, 0.30, 0.12, 0.20],    # medium
    [1, 7, 0.52, 0.48, 0.03, 0.05],    # small
])


def test_loss_is_zero_for_perfect_predictions_and_decodes_back(model):
    size = 416
    outs = encode(model, TARGETS, size)
    loss, parts = YOLOv3Loss(model)(outs, TARGETS)
    assert loss < 1e-2 and parts["xy"] < 1e-6 and parts["wh"] < 1e-6

    dets = postprocess(outs, model, conf_thresh=0.5)
    assert [len(d) for d in dets] == [2, 1]
    for t in TARGETS:
        cx, cy, w, h = (t[2:] * size).tolist()
        box = torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]])
        d = dets[int(t[0])]
        iou, k = box_iou(box, d[:, :4]).max(1)
        assert iou > 0.999 and d[k, 5] == t[1]


def test_loss_penalises_errors_and_backpropagates(model):
    outs = [o.requires_grad_() for o in encode(model, TARGETS, 416)]
    base = YOLOv3Loss(model)(outs, TARGETS)[0]
    wrong = TARGETS.clone()
    wrong[:, 1] = 3
    loss, parts = YOLOv3Loss(model)(outs, wrong)
    assert parts["cls"] > 1 and loss > base
    loss.backward()
    assert all(o.grad is not None and torch.isfinite(o.grad).all() for o in outs)
    empty, parts = YOLOv3Loss(model)(outs, torch.zeros(0, 6))
    assert parts["xy"] == 0 and parts["cls"] == 0 and parts["obj"] > 1  # three false positives


def test_ignore_threshold(model):
    # A confident duplicate from a prior that is not the assigned one: free if it overlaps
    # the ground truth by more than the threshold, a false positive otherwise.
    t = TARGETS[:1]
    outs = encode(model, t, 416)
    src = (outs[0][0, :, :, :, 4] > 0).nonzero()[0]
    a, j, i = src.tolist()
    dup = outs[0][0, a, j, i].clone()
    b = (a + 1) % 3
    k_a, k_b = model.masks[0][a], model.masks[0][b]
    dup[2] += math.log(ANCHORS[k_a][0] / ANCHORS[k_b][0])
    dup[3] += math.log(ANCHORS[k_a][1] / ANCHORS[k_b][1])
    outs[0][0, b, j, i] = dup
    assert YOLOv3Loss(model, ignore_thresh=0.5)(outs, t)[1]["obj"] < 1e-2
    assert YOLOv3Loss(model, ignore_thresh=1.1)(outs, t)[1]["obj"] > 5


def test_one_gt_one_prior(model):
    outs = [torch.zeros(1, 3, 416 // s, 416 // s, 85) for s in model.strides]
    crit = YOLOv3Loss(model, ignore_thresh=1.1)
    # With all logits 0 every cell costs log(2), assigned or not, so obj loss is constant
    # and the class loss counts assigned priors: exactly one per ground truth.
    _, parts = crit(outs, TARGETS[:2])
    assert parts["cls"].item() == pytest.approx(2 * 80 * math.log(2), rel=1e-5)


def test_nms():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10.5]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    assert nms(boxes, scores, 0.45).tolist() == [3, 2]
    assert nms(boxes[:0], scores[:0], 0.45).tolist() == []


def test_average_precision():
    gt = {0: np.array([[0, 0, 10, 10, 0], [20, 20, 40, 40, 1.0]]), 1: np.array([[5, 5, 25, 25, 0.0]])}
    perfect = {i: np.concatenate([g[:, :4], np.full((len(g), 1), 0.9), g[:, 4:]], 1) for i, g in gt.items()}
    r = average_precision(perfect, gt)
    assert r["mAP"] == pytest.approx(1) and r["AP50"] == pytest.approx(1)

    # Class 0: a higher-scoring false positive ahead of both true positives -> precision
    # 1/2 at recall 0.5 and 2/3 at recall 1, so interpolated AP is 2/3.
    dets = {k: v.copy() for k, v in perfect.items()}
    dets[1] = np.concatenate([dets[1], [[50, 50, 60, 60, 0.99, 0]]])
    r = average_precision(dets, gt, iou_thresholds=[0.5])
    assert r["per_class"][0] == pytest.approx(2 / 3, abs=0.01) and r["per_class"][1] == pytest.approx(1)

    # Boxes shifted to IoU = 0.6 count at 0.5 but not at 0.75.
    shifted = {0: np.array([[0, 0, 10, 10, 0.9, 0], [25, 20, 45, 40, 0.9, 1]]), 1: perfect[1]}
    r = average_precision(shifted, gt)
    assert r["AP50"] == pytest.approx(1) and r["AP75"] == pytest.approx(0.5)
    assert average_precision({}, gt)["mAP"] == 0


def test_augment_and_letterbox():
    random.seed(0)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8))
    boxes = np.array([[3, 0.5, 0.5, 0.4, 0.6], [5, 0.02, 0.5, 0.04, 0.2]], dtype=np.float32)
    for _ in range(50):
        out, b = augment(img, boxes, 320)
        assert out.size == (320, 320) and b.shape[1] == 5
        x1, y1 = b[:, 1] - b[:, 3] / 2, b[:, 2] - b[:, 4] / 2
        x2, y2 = b[:, 1] + b[:, 3] / 2, b[:, 2] + b[:, 4] / 2
        assert (x1 >= -1e-6).all() and (y1 >= -1e-6).all() and (x2 <= 1 + 1e-6).all() and (y2 <= 1 + 1e-6).all()

    out, b = letterbox(img, 416, boxes)
    assert out.size == (416, 416)
    xyxy = torch.tensor([[b[0, 1] - b[0, 3] / 2, b[0, 2] - b[0, 4] / 2, b[0, 1] + b[0, 3] / 2, b[0, 2] + b[0, 4] / 2]]) * 416
    back = unletterbox(xyxy, 200, 120, 416)[0]
    assert torch.allclose(back, torch.tensor([60.0, 24.0, 140.0, 96.0]), atol=1.0)


def test_augment_tracks_object():
    """The box must still cover the object after jitter, placement and flipping."""
    random.seed(1)
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    arr[20:60, 120:180] = 255
    boxes = np.array([[0, 0.75, 0.4, 0.3, 0.4]], dtype=np.float32)
    for _ in range(20):
        out, b = augment(Image.fromarray(arr), boxes, 256, hue=0, saturation=1, exposure=1)
        ys, xs = np.nonzero(np.asarray(out).sum(2) > 600)
        got = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / 256
        want = [b[0, 1] - b[0, 3] / 2, b[0, 2] - b[0, 4] / 2, b[0, 1] + b[0, 3] / 2, b[0, 2] + b[0, 4] / 2]
        assert np.abs(got - want).max() < 0.02


def test_multiscale_sampler():
    random.seed(0)
    batches = list(MultiScaleBatchSampler(1000, 4, interval=10))
    assert len(batches) == 250 and all(len({s for _, s in b}) == 1 for b in batches)
    sizes = [b[0][1] for b in batches]
    assert all(len(set(sizes[i:i + 10])) == 1 for i in range(0, 250, 10))
    assert set(sizes) <= set(range(320, 609, 32)) and len(set(sizes)) > 3
    assert sorted(i for b in batches for i, _ in b) == list(range(1000))


def test_schedule_and_optimizer(model):
    lr = lambda it: learning_rate(it, 1e-3, 1000, (400000, 450000), (0.1, 0.1))
    assert lr(499) == pytest.approx(1e-3 * 0.5 ** 4)
    assert lr(1000) == lr(399999) == 1e-3
    assert lr(400000) == pytest.approx(1e-4) and lr(450000) == pytest.approx(1e-5)
    opt = make_optimizer(model, 1e-3, 0.9, 5e-4)
    assert sum(p.numel() for p in opt.param_groups[0]["params"]) == sum(
        m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    assert opt.param_groups[1]["weight_decay"] == 0


def test_kmeans_recovers_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[10, 14], [60, 40], [300, 320]])
    wh = np.concatenate([c * rng.uniform(0.95, 1.05, (200, 2)) for c in centres])
    anchors, iou = kmeans_anchors(wh, k=3)
    assert np.allclose(anchors, centres, rtol=0.05) and iou > 0.9


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = YOLOv3(3, bn_momentum=0.1).train()
    crit = YOLOv3Loss(model)
    opt = make_optimizer(model, 1e-3, 0.9, 5e-4)
    imgs = torch.rand(2, 3, 96, 96)
    targets = torch.tensor([[0, 1, 0.5, 0.5, 0.6, 0.5], [1, 2, 0.3, 0.6, 0.2, 0.3]])
    losses = []
    for _ in range(12):
        loss, _ = crit(model(imgs), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
