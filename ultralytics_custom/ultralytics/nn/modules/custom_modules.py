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
    def __init__(self, nc=80, ch=(), reg_max=16, stride=(4, 8, 16, 32)):
        """
        Initialize the Detect_HailoFriendly module.

        Args:
            nc (int): Number of classes.
            ch (tuple): Input channels for each scale.
            reg_max (int): Maximum regression value.
            stride (tuple): Strides for each scale.
        """
        super().__init__(nc, ch)
        self.nc, self.reg_max, self.bins = nc, reg_max, reg_max
        self.stride = torch.tensor(stride)
        self.cv2 = nn.ModuleList(nn.Conv2d(c, 4 * self.bins, 1) for c in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c, nc, 1) for c in ch)
        self.dfl = DFL(self.bins)
        self.grids = [self._make_grid(s) for s in stride]

    @staticmethod
    def _make_grid(s, img=640):
        """
        Create a grid for a given stride.

        Args:
            s (int): Stride value.
            img (int): Image size.

        Returns:
            torch.Tensor: Grid tensor of shape (h, w, 2).
        """
        hw = img // s
        y, x = torch.meshgrid(torch.arange(hw), torch.arange(hw), indexing="ij")
        return torch.stack((x, y), 2).float()

    def forward(self, x):
        """
        Forward pass for Detect_HailoFriendly.

        Args:
            x (list): List of feature maps for each scale.

        Returns:
            list or tuple: Training mode returns list of feature maps, inference returns boxes and scores.
        """
        raw_d, raw_c, boxes, scores = [], [], [], []
        for i in range(self.nl):
            rd = self.cv2[i](x[i])                       # (B, 4*bins, h, w)
            rc = self.cv3[i](x[i])                       # (B, nc,     h, w)
            B, _, h, w = rd.shape
            raw_d.append(rd)
            raw_c.append(rc)

            d = self.dfl(rd.flatten(2))                  # (B, 4, h*w)
            d = d.view(B, 4, h * w).permute(0, 2, 1).view(B, h, w, 4)

            g  = self.grids[i][:h, :w].to(rd.device)
            xy = ((d[..., :2].sigmoid() * 2 - 0.5) + g) * float(self.stride[i])
            wh = ((d[..., 2:4].sigmoid() * 2) ** 2)      * float(self.stride[i])
            box = torch.cat((xy - wh / 2, xy + wh / 2), -1)
            boxes.append(box.view(B, -1, 4))

            cls = rc.permute(0, 2, 3, 1).contiguous().view(B, -1, self.nc)
            scores.append(cls)
        if self.training:
            return [torch.cat((d, c), 1) for d, c in zip(raw_d, raw_c)]

        boxes  = torch.cat(boxes,  1)
        scores = torch.cat(scores, 1).sigmoid()
        return boxes, scores

    def bias_init(self):
        """
        Initialize biases for the detection heads.
        """
        for rd, rc, s in zip(self.cv2, self.cv3, self.stride):
            rd.bias.data.fill_(1.0)
            rc.bias.data[:self.nc] = math.log(5 / self.nc / (640 / s)**2)
