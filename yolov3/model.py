"""Darknet-53 and the YOLOv3 detector (Redmon & Farhadi, 2018)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANCHORS, ANCHOR_MASKS, STRIDES


class ConvBN(nn.Sequential):
    """Convolution -> batch norm -> leaky ReLU(0.1), the basic Darknet block."""

    def __init__(self, c_in, c_out, k, stride=1, bn_momentum=0.01):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, stride, k // 2, bias=False),
            # Darknet keeps rolling statistics as 0.99 * old + 0.01 * new.
            nn.BatchNorm2d(c_out, momentum=bn_momentum, eps=1e-5),
            nn.LeakyReLU(0.1, inplace=True),
        )


class Residual(nn.Module):
    """1x1 bottleneck followed by 3x3, with a shortcut connection (Table 1)."""

    def __init__(self, c, bn_momentum=0.01):
        super().__init__()
        self.conv1 = ConvBN(c, c // 2, 1, bn_momentum=bn_momentum)
        self.conv2 = ConvBN(c // 2, c, 3, bn_momentum=bn_momentum)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Darknet53(nn.Module):
    """Feature extractor from Table 1: 52 convolutional layers (+1 FC as a classifier).

    forward() returns the feature maps at strides 8, 16 and 32.
    """

    BLOCKS = (1, 2, 8, 8, 4)

    def __init__(self, bn_momentum=0.01):
        super().__init__()
        self.stem = ConvBN(3, 32, 3, bn_momentum=bn_momentum)
        stages, c = [], 32
        for n in self.BLOCKS:
            layers = [ConvBN(c, c * 2, 3, stride=2, bn_momentum=bn_momentum)]
            layers += [Residual(c * 2, bn_momentum) for _ in range(n)]
            stages.append(nn.Sequential(*layers))
            c *= 2
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats[2], feats[3], feats[4]  # 256@/8, 512@/16, 1024@/32


class Darknet53Classifier(nn.Module):
    """Darknet-53 with global average pooling and a 1000-way layer (Table 1)."""

    def __init__(self, num_classes=1000, bn_momentum=0.01):
        super().__init__()
        self.backbone = Darknet53(bn_momentum)
        self.fc = nn.Linear(1024, num_classes)
        _init_weights(self)

    def forward(self, x):
        x = self.backbone(x)[-1]
        return self.fc(x.mean(dim=(2, 3)))


class DetectionBlock(nn.Module):
    """Five alternating 1x1/3x3 convs, then a 3x3 conv and the linear 1x1 predictor."""

    def __init__(self, c_in, c, n_out, bn_momentum=0.01):
        super().__init__()
        m = bn_momentum
        self.body = nn.Sequential(
            ConvBN(c_in, c, 1, bn_momentum=m),
            ConvBN(c, c * 2, 3, bn_momentum=m),
            ConvBN(c * 2, c, 1, bn_momentum=m),
            ConvBN(c, c * 2, 3, bn_momentum=m),
            ConvBN(c * 2, c, 1, bn_momentum=m),
        )
        self.conv = ConvBN(c, c * 2, 3, bn_momentum=m)
        self.pred = nn.Conv2d(c * 2, n_out, 1)

    def forward(self, x):
        route = self.body(x)
        return route, self.pred(self.conv(route))


class YOLOv3(nn.Module):
    """YOLOv3: Darknet-53 plus predictions at three scales (Section 2.3).

    forward() returns three raw tensors of shape (B, 3, H, W, 5 + num_classes) holding
    (tx, ty, tw, th, objectness logit, class logits), coarsest scale first.
    """

    def __init__(self, num_classes=80, anchors=ANCHORS, masks=ANCHOR_MASKS, bn_momentum=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        self.masks = tuple(tuple(m) for m in masks)
        self.strides = STRIDES
        n_out = len(self.masks[0]) * (5 + num_classes)
        m = bn_momentum

        self.backbone = Darknet53(m)
        self.head1 = DetectionBlock(1024, 512, n_out, m)
        self.up1 = ConvBN(512, 256, 1, bn_momentum=m)
        self.head2 = DetectionBlock(256 + 512, 256, n_out, m)
        self.up2 = ConvBN(256, 128, 1, bn_momentum=m)
        self.head3 = DetectionBlock(128 + 256, 128, n_out, m)
        _init_weights(self)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        r1, p1 = self.head1(c5)
        x = torch.cat([F.interpolate(self.up1(r1), scale_factor=2.0, mode="nearest"), c4], 1)
        r2, p2 = self.head2(x)
        x = torch.cat([F.interpolate(self.up2(r2), scale_factor=2.0, mode="nearest"), c3], 1)
        _, p3 = self.head3(x)
        return [self._reshape(p) for p in (p1, p2, p3)]

    def _reshape(self, p):
        b, _, h, w = p.shape
        return p.view(b, len(self.masks[0]), 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()

    def darknet_modules(self):
        """Weighted layers in the order Darknet serialises them (yolov3.cfg order)."""
        mods = [self.backbone.stem]
        for stage in self.backbone.stages:
            for layer in stage:
                mods += [layer.conv1, layer.conv2] if isinstance(layer, Residual) else [layer]
        for head, up in ((self.head1, self.up1), (self.head2, self.up2), (self.head3, None)):
            mods += list(head.body) + [head.conv, head.pred]
            if up is not None:
                mods.append(up)
        return mods


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
