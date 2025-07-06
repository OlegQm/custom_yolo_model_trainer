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
    ------------------------------------------------------------------------------
    * Train / PyTorch-inference
        • Returns raw tensors [B, 78, H, W] (list per level) → loss.
        • If not self.training, also decodes to (boxes, scores).

    * Export (ONNX → Hailo)
        • Outputs 8 tensors [bbox_P3, cls_P3, ..., bbox_P6, cls_P6].
        • DFL-integral, decode, and NMS are performed in Hailo SDK/Host-CPU.
    ------------------------------------------------------------------------------
    """
    def __init__(self, nc=14, ch=(), reg_max=16, stride=(4, 8, 16, 32)):
        super().__init__(nc=nc, ch=ch)
        self.nc, self.reg_max = nc, reg_max
        self.no = nc + 4 * reg_max
        self.stride = torch.tensor(stride, dtype=torch.float32)

        self.bbox = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in ch])
        self.cls  = nn.ModuleList([nn.Conv2d(c, nc,           1) for c in ch])
        self.dfl  = DFL(reg_max)

    def forward(self, x):
        if getattr(self, "export", False):
            out = []
            for i in range(self.nl):
                out.extend((self.bbox[i](x[i]), self.cls[i](x[i])))
            return out

        if self.training:
            feats = []
            for i in range(self.nl):
                rd = self.bbox[i](x[i])           # (B, 4*reg_max, H, W)
                rc = self.cls[i](x[i])            # (B, nc,        H, W)
                y  = torch.cat((rd, rc), 1)       # (B, no,        H, W)
                bs, _, h, w = y.shape
                feats.append(y.view(bs, self.no, h * w))  # (B, no, H*W)
            return feats


        boxes, scores = [], []
        for i in range(self.nl):
            rd = self.bbox[i](x[i])
            rc = self.cls[i](x[i])

            B, _, h, w = rd.shape
            d = self.dfl(rd.flatten(2)).view(B, 4, h, w).permute(0, 2, 3, 1)

            s = self.stride[i].to(rd.device)
            yv, xv = torch.meshgrid(torch.arange(h, device=rd.device),
                                   torch.arange(w, device=rd.device),
                                   indexing='ij')
            grid = torch.stack((xv, yv), 2).float()

            xy = ((d[..., :2].sigmoid() * 2 - 0.5) + grid) * s
            wh = ((d[..., 2:4].sigmoid() * 2) ** 2)       * s
            box = torch.cat((xy - wh / 2, xy + wh / 2), -1)

            boxes.append(box.reshape(B, -1, 4))
            scores.append(rc.permute(0, 2, 3, 1)
                            .reshape(B, -1, self.nc))

        return torch.cat(boxes, 1), torch.cat(scores, 1).sigmoid()

    def bias_init(self):
        """
        Initializes as in Ultralytics to speed up cls-head convergence.
        """
        for b, c, s in zip(self.bbox, self.cls, self.stride):
            b.bias.data.fill_(1.0)
            c.bias.data[:self.nc] = math.log(
                5 / self.nc / (640 / s) ** 2
            )

