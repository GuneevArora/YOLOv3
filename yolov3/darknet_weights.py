"""Read and write Darknet ``.weights`` files (e.g. the official yolov3.weights)."""
import numpy as np
import torch
import torch.nn as nn


def _modules(model):
    if hasattr(model, "darknet_modules"):
        return model.darknet_modules()
    raise TypeError("model must define darknet_modules()")


def load_darknet_weights(model, path, cutoff=None):
    """Load Darknet weights into ``model``.

    ``cutoff`` limits loading to the first N weighted layers, e.g. 52 to initialise only
    the backbone from ``darknet53.conv.74`` (the ImageNet-pretrained Darknet-53).
    Returns the number of images Darknet had seen during training.
    """
    with open(path, "rb") as f:
        major, minor, _ = np.fromfile(f, dtype=np.int32, count=3)
        seen = np.fromfile(f, dtype=np.int64 if major * 10 + minor >= 2 else np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32)

    ptr = 0

    def take(t):
        nonlocal ptr
        n = t.numel()
        if ptr + n > len(data):
            raise ValueError(f"{path}: file ended early; it does not match this architecture")
        t.data.copy_(torch.from_numpy(data[ptr:ptr + n]).view_as(t))
        ptr += n

    modules = _modules(model)[:cutoff]
    for m in modules:
        if isinstance(m, nn.Conv2d):
            take(m.bias)
            take(m.weight)
        else:
            conv, bn = m[0], m[1]
            for t in (bn.bias, bn.weight, bn.running_mean, bn.running_var, conv.weight):
                take(t)
    if cutoff is None and ptr != len(data):
        raise ValueError(f"{path}: {len(data) - ptr} unused values; it does not match this architecture")
    return int(seen)


def save_darknet_weights(model, path, cutoff=None, seen=0):
    """Write ``model`` in Darknet format (header version 0.2.0, 64-bit ``seen``)."""
    with open(path, "wb") as f:
        np.array([0, 2, 0], dtype=np.int32).tofile(f)
        np.array([seen], dtype=np.int64).tofile(f)
        for m in _modules(model)[:cutoff]:
            if isinstance(m, nn.Conv2d):
                tensors = (m.bias, m.weight)
            else:
                conv, bn = m[0], m[1]
                tensors = (bn.bias, bn.weight, bn.running_mean, bn.running_var, conv.weight)
            for t in tensors:
                t.detach().cpu().numpy().astype(np.float32).tofile(f)
