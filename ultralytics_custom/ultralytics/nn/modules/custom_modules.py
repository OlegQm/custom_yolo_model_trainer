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
    Fully "Hailo-self-contained" YOLOv8 head.

    • **train / val**  
      - returns a list of 4-D maps (B, no, H, W) — compatible with ComputeLoss.

    • **eval (validator / inference on GPU/CPU)**  
      - returns `(pred, feats)`, where  
        `pred` – tensor (B, A, 5+nc) [x1,y1,x2,y2,obj,cls…]  
        `feats` – same list of maps for validator loss.

    • **export (ONNX → Hailo)**  
      - inside the graph performs  
        1) Softmax + DFL-integral (64→4)  
        2) Decode (grid, stride)  
        3) Sigmoid for cls  
      - outputs 8 tensors: 4 bbox_P*, 4 cls_P* (for `nms_postprocess`).
    """

    def __init__(
        self,
        nc: int = 14,
        ch: tuple = (),
        reg_max: int = 16,
        stride: tuple = (4, 8, 16, 32),
    ):
        super().__init__(nc=nc, ch=ch)
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max
        self.stride_f = torch.tensor(stride, dtype=torch.float32)
        self.bbox = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in ch])
        self.cls  = nn.ModuleList([nn.Conv2d(c, nc, 1) for c in ch])

    def forward(self, x):
        if getattr(self, "export", False):
            outs = []
            for i in range(self.nl):
                rd = self.bbox[i](x[i])
                rc = self.cls[i](x[i])
                B, _, H, W = rd.shape
                rd = rd.view(B, 4, self.reg_max, H, W).softmax(2)
                bins = torch.arange(
                    self.reg_max, dtype=rd.dtype, device=rd.device
                ).view(1, 1, -1, 1, 1)
                rd = (rd * bins).sum(2)
                yv, xv = torch.meshgrid(
                    torch.arange(H, device=rd.device),
                    torch.arange(W, device=rd.device),
                    indexing="ij",
                )
                grid = torch.stack((xv, yv), 0).float()
                s = self.stride_f[i].to(rd.device)
                xy = ((rd[:, :2].sigmoid() * 2 - 0.5) + grid) * s
                wh = ((rd[:, 2:4].sigmoid() * 2).square()) * s
                box = torch.cat((xy - wh / 2, xy + wh / 2), 1)
                outs.extend((box, rc.sigmoid()))
            return outs

        feats, y_parts = [], []
        for i in range(self.nl):
            rd = self.bbox[i](x[i])
            rc = self.cls[i](x[i])
            feat = torch.cat((rd, rc), 1)
            feats.append(feat)
            bs, _, h, w = feat.shape
            y_parts.append(feat.view(bs, self.no, h * w))

        if self.training:
            return feats

        boxes, scores = [], []
        A_offset = 0
        for i in range(self.nl):
            rd, rc = feats[i].split((4 * self.reg_max, self.nc), 1)
            bs, _, h, w = rd.shape
            rd = (
                rd.view(bs, 4, self.reg_max, h, w)
                .softmax(2)
                .mul(torch.arange(self.reg_max, device=rd.device)
                     .view(1, 1, -1, 1, 1))
                .sum(2)
            )
            yv, xv = torch.meshgrid(
                torch.arange(h, device=rd.device),
                torch.arange(w, device=rd.device),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 0).float()
            s = self.stride_f[i].to(rd.device)
            xy = ((rd[:, :2].sigmoid() * 2 - 0.5) + grid) * s
            wh = ((rd[:, 2:4].sigmoid() * 2).square()) * s
            box = torch.cat((xy - wh / 2, xy + wh / 2), 1)
            boxes.append(box.flatten(2).permute(0, 2, 1))
            scores.append(rc.sigmoid().flatten(2).permute(0, 2, 1))

        pred = torch.cat(
            (
                torch.cat(boxes, 1),
                torch.ones_like(boxes[0][..., :1]),
                torch.cat(scores, 1),
            ),
            2,
        )
        return pred, feats

    def bias_init(self):
        for b, c, s in zip(self.bbox, self.cls, self.stride_f):
            b.bias.data.fill_(1.0)
            c.bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
