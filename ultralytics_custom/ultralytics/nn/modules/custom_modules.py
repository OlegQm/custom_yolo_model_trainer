import torch
import torch.nn as nn
import math
from ultralytics.nn.modules import Conv, DFL
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
    Train/PyTorch-inference:
        Returns already decoded (boxes, scores).
    Export (ONNX to Hailo):
        Returns 8 raw tensors
        [bbox_P3, cls_P3, ..., bbox_P6, cls_P6].
        DFL, decode, and NMS are performed in Hailo SDK.
    """
    def __init__(self,
                 nc: int = 14,
                 ch: tuple = (),
                 reg_max: int = 16,
                 stride: tuple = (4, 8, 16, 32)):
        """
        Initialize Detect_HailoFriendly module.
        """
        super().__init__(nc=nc, ch=ch)
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max
        self.stride = torch.tensor(stride)
        self.bbox = nn.ModuleList(nn.Conv2d(c, 4 * reg_max, 1) for c in ch)
        self.cls = nn.ModuleList(nn.Conv2d(c, nc, 1) for c in ch)
        self.dfl = DFL(reg_max)

    def forward(self, x):
        """
        Forward pass for Detect_HailoFriendly.
        In export mode, returns raw bbox and cls tensors.
        In train/inference mode, returns decoded boxes and scores.
        """
        if getattr(self, "export", False):
            outs = []
            for i in range(self.nl):
                outs += [self.bbox[i](x[i]), self.cls[i](x[i])]
            return outs

        raw_maps, boxes, scores = [], [], []
        for i in range(self.nl):
            feat = x[i]
            rd = self.bbox[i](feat)
            rc = self.cls[i](feat)
            B, _, h, w = rd.shape
            raw_maps.append(torch.cat((rd, rc), 1))
            d = self.dfl(rd).permute(0, 2, 3, 1)
            yv, xv = torch.meshgrid(torch.arange(h, device=feat.device),
                                    torch.arange(w, device=feat.device),
                                    indexing='ij')
            grid = torch.stack((xv, yv), 2).float()
            xy = ((d[..., :2].sigmoid() * 2 - 0.5) + grid) * self.stride[i]
            wh = ((d[..., 2:4].sigmoid() * 2) ** 2) * self.stride[i]
            box = torch.cat((xy - wh / 2, xy + wh / 2), -1)
            boxes.append(box.view(B, -1, 4))
            scores.append(rc.permute(0, 2, 3, 1).reshape(B, -1, self.nc))

        if self.training:
            return raw_maps

        return torch.cat(boxes, 1), torch.cat(scores, 1).sigmoid()

    def bias_init(self):
        """
        Initialize biases for faster convergence as in Ultralytics Detect.
        """
        for b, c, s in zip(self.bbox, self.cls, self.stride):
            b.bias.data.fill_(1.0)
            c.bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
