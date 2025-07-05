import torch
import torch.nn as nn
import math
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.tal import make_anchors
from ultralytics.nn.modules.head import Detect

# ------------- Own layers -------------

class SE(nn.Module):
    """Channel-only Squeeze-Excitation with "mini" spatial pooling (4×4)."""
    def __init__(self, c, r=16, pool_hw=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_hw)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(1, c // r), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, c // r), c, 1, bias=True),
            nn.Sigmoid()
        )
        last_conv = self.fc[-2]
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        y = self.pool(x)
        y = self.fc(y)
        y = nn.functional.adaptive_avg_pool2d(y, 1)
        return x * y


class ConvSE(nn.Module):
    """Conv -> BN -> SiLU -> SE (attention)."""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, act=act)
        self.se   = SE(c2)

    def forward(self, x):
        return self.se(self.conv(x))


class Detect_HailoFriendly(Detect):
    """
    True Hailo-friendly detection head.
    - No DFL (direct 4-channel bbox regression).
    - No post-processing in the exported ONNX graph.
    - Returns separate outputs for bbox and class predictions during export.
    - Inherits from Detect to ensure stride calculation and compatibility.
    """
    
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.shape = None
        self.reg_max = 1
        self.no = nc + 4
        c2 = [max(16, c // 4) for c in ch]
        c3 = [max(c, nc) for c in ch]
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2[i], 3),
                Conv(c2[i], c2[i], 3),
                nn.Conv2d(c2[i], 4, 1)
            ) for i, x in enumerate(ch)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3[i], 3),
                Conv(c3[i], c3[i], 3),
                nn.Conv2d(c3[i], nc, 1)
            ) for i, x in enumerate(ch)
        )
        self.dfl = nn.Identity()

    def forward(self, x):
        """
        Forward pass with conditional logic for different modes.
        This solves the problem with Hailo.
        """
        if self.export:
            outputs = []
            for i in range(self.nl):
                outputs.append(self.cv2[i](x[i]))
                outputs.append(self.cv3[i](x[i]))
            return outputs
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y, x

    def _inference(self, x):
        """
        Method for inference in Python (NOT for export).
        Decodes "raw" outputs into final bounding boxes.
        This logic should be implemented on CPU when working with Hailo.
        """
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((4, self.nc), 1)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (y.transpose(0, 1) for y in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        xy = (box[:, :2].sigmoid() * 2 - 0.5 + self.anchors) * self.strides
        wh = (box[:, 2:].sigmoid() * 2) ** 2 * self.anchors
        dbox = torch.cat((xy - wh / 2, xy + wh / 2), 1)
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """
        Bias initialization. Called correctly since we inherit from Detect.
        """
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            if s == 0:
                continue
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
