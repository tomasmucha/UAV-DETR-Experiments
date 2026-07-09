import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from collections import OrderedDict
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, WTConvNormLayer,BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck

__all__ = [
    'DySample', 'SPDConv', 'MFFF', 'FrequencyFocusedDownSampling', 'SemanticAlignmenCalibration',
    'P3Refine', 'NRP3CBAM', 'NRP3Lite', 'NRP3DropPath', 'P2InformationEnhance', 'MSNoiseGate',
    'P2GuidedP3Enhance', 'StemDown'
]


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x



class FFM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class ImprovedFFTKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.SiLU()

        # 改进后的 SCA 部分
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, groups=dim, bias=True)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ffm = FFM(dim)

        #通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #1*1 进行通道融合
        out = self.in_conv(x)
        #公式1开始
        # 池化后1*1卷积 
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)
        #公式1结束
        
        #公式2
        x_sca1 = self.conv1x1(x_fca)
        x_sca2 = self.conv3x3(x_fca)
        x_sca3 = self.conv5x5(x_fca)
        x_sca = x_sca1 + x_sca2 + x_sca3
        #公式2结束

        # 使用通道注意力机制
        channel_weights = self.channel_attention(x_att)
        x_sca = x_sca * channel_weights

        #FF的公式
        x_sca = self.ffm(x_sca)

        # 最终融合 公式4
        out = x + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class MFFF(nn.Module): 
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = ImprovedFFTKernel(int(dim * self.e))

    def forward(self, x):
        c1 = round(x.size(1) * self.e)
        c2 = x.size(1) - c1
        ok_branch, identity = torch.split(self.cv1(x), [c1, c2], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))


class P3Refine(nn.Module):
    """P3 experiment: lightweight multi-scale refinement for small-object features."""

    def __init__(self, dim, e=0.5):
        super().__init__()
        hidden = int(dim * e)
        self.reduce = Conv(dim, hidden, 1)
        self.local = DWConv(hidden, hidden, 3)
        self.context = DWConv(hidden, hidden, 5, d=2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, max(hidden // 4, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hidden // 4, 16), hidden, 1),
            nn.Sigmoid(),
        )
        self.expand = Conv(hidden, dim, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y = self.reduce(x)
        y = self.local(y) + self.context(y)
        y = y * self.gate(y)
        return x + self.gamma * self.expand(y)


class NRP3CBAM(nn.Module):
    """Noise-robust P3 enhancement with multi-scale context and CBAM attention."""

    def __init__(self, dim, e=0.5):
        super().__init__()
        hidden = int(dim * e)
        mid = max(hidden // 4, 16)
        self.reduce = Conv(dim, hidden, 1)
        self.local = DWConv(hidden, hidden, 3)
        self.context = DWConv(hidden, hidden, 5, d=2)
        self.frequency = FFM(hidden)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, hidden, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.expand = Conv(hidden, dim, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        y = self.reduce(x)
        y = self.local(y) + self.context(y) + self.frequency(y)
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(y, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(y, 1))
        y = y * torch.sigmoid(avg_attn + max_attn)
        spatial_attn = torch.cat([torch.mean(y, dim=1, keepdim=True), torch.max(y, dim=1, keepdim=True)[0]], dim=1)
        y = y * torch.sigmoid(self.spatial(spatial_attn))
        return x + self.gamma * self.expand(y)


class NRP3Lite(nn.Module):
    """Lite P3 residual enhancement that keeps channel gating and removes spatial CBAM noise."""

    def __init__(self, dim, e=0.5, gamma=0.075):
        super().__init__()
        hidden = int(dim * e)
        mid = max(hidden // 4, 16)
        self.reduce = Conv(dim, hidden, 1)
        self.local = DWConv(hidden, hidden, 3)
        self.context = DWConv(hidden, hidden, 5, d=2)
        self.frequency = FFM(hidden)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, hidden, 1, bias=False),
        )
        self.expand = Conv(hidden, dim, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x):
        y = self.reduce(x)
        y = self.local(y) + self.context(y) + self.frequency(y)
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(y, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(y, 1))
        y = y * torch.sigmoid(avg_attn + max_attn)
        return x + self.gamma * self.expand(y)


class NRP3DropPath(nn.Module):
    """NRP3 enhancement with stochastic depth applied only to the residual delta."""

    def __init__(self, dim, e=0.5, drop_prob=0.05):
        super().__init__()
        hidden = int(dim * e)
        mid = max(hidden // 4, 16)
        self.reduce = Conv(dim, hidden, 1)
        self.local = DWConv(hidden, hidden, 3)
        self.context = DWConv(hidden, hidden, 5, d=2)
        self.frequency = FFM(hidden)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, hidden, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.expand = Conv(hidden, dim, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.drop_prob = drop_prob

    def _drop_path(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask

    def forward(self, x):
        y = self.reduce(x)
        y = self.local(y) + self.context(y) + self.frequency(y)
        avg_attn = self.channel_mlp(F.adaptive_avg_pool2d(y, 1))
        max_attn = self.channel_mlp(F.adaptive_max_pool2d(y, 1))
        y = y * torch.sigmoid(avg_attn + max_attn)
        spatial_attn = torch.cat([torch.mean(y, dim=1, keepdim=True), torch.max(y, dim=1, keepdim=True)[0]], dim=1)
        y = y * torch.sigmoid(self.spatial(spatial_attn))
        delta = self.expand(y)
        return x + self.gamma * self._drop_path(delta)


class P2InformationEnhance(nn.Module):
    """Information-guided P2 enhancement for weak tiny-object regions."""

    def __init__(self, dim, e=0.5, gamma=0.1):
        super().__init__()
        hidden = max(int(dim * e), 32)
        self.feature_proj = nn.Conv2d(dim, hidden, 1, bias=True)
        self.detail_proj = nn.Conv2d(dim, hidden, 1, bias=True)
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.info_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        self.expand = nn.Conv2d(hidden, dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(gamma))
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x):
        detail = torch.abs(x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        y = self.local(self.feature_proj(x) + self.detail_proj(detail))
        info = torch.sigmoid(self.info_head(y))
        return x + self.gamma * info * self.expand(y)


class P2GuidedP3Enhance(nn.Module):
    """P2-guided P3 enhancement for small, dense UAV targets.

    The module keeps the decoder inputs unchanged while injecting high-resolution P2 detail into the final P3 feature.
    P2 is folded to P3 resolution with a Focus-style operation, fused with semantic P3 by channel/spatial gates, and
    applied as a residual update so the original UAV-DETR feature path remains recoverable during training.
    """

    def __init__(self, c_p2, c_p3, e=0.5):
        super().__init__()
        hidden = max(int(c_p3 * e), 64)
        self.p2_proj = Conv(c_p2 * 4, hidden, 1)
        self.p2_detail = nn.Sequential(
            DWConv(hidden, hidden, 3),
            DWConv(hidden, hidden, 5, d=2),
        )
        self.p3_proj = Conv(c_p3, hidden, 1)
        self.fuse = nn.Sequential(
            Conv(hidden * 2, hidden, 1),
            DWConv(hidden, hidden, 3),
            Conv(hidden, c_p3, 1, act=False),
        )
        self.channel_gate = nn.Sequential(
            Conv(hidden * 2, hidden, 1),
            nn.Conv2d(hidden, c_p3, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.2))

    @staticmethod
    def _focus(x):
        return torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)

    def forward(self, x):
        p2, p3 = x
        p2 = self.p2_proj(self._focus(p2))
        if p2.shape[2:] != p3.shape[2:]:
            p2 = F.interpolate(p2, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.p2_detail(p2)
        p3_sem = self.p3_proj(p3)
        fused = torch.cat([p2, p3_sem], dim=1)
        delta = self.fuse(fused)
        channel = self.channel_gate(fused)
        spatial = self.spatial_gate(torch.cat([delta.mean(1, keepdim=True), delta.max(1, keepdim=True)[0]], dim=1))
        return p3 + self.gamma * channel * spatial * delta


class MSNoiseGate(nn.Module):
    """Multi-scale noise gate applied per decoder feature level."""

    def __init__(self, dim, e=0.25):
        super().__init__()
        hidden = max(int(dim * e), 16)
        self.noise_gate = nn.Sequential(
            Conv(dim, hidden, 1),
            DWConv(hidden, hidden, 3),
            Conv(hidden, dim, 1, act=False),
            nn.Sigmoid(),
        )
        self.smooth = nn.Sequential(
            DWConv(dim, dim, 3),
            Conv(dim, dim, 1, act=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        noise = torch.abs(x - low)
        gate = self.noise_gate(noise)
        refined = gate * self.smooth(x) + (1 - gate) * low
        return x + self.gamma * (refined - x)


class StemDown(nn.Module):
    """Early downsample with parallel stride-conv and MaxPool branches."""

    def __init__(self, c1, c2):
        super().__init__()
        c_pool = c2 // 2
        c_conv = c2 - c_pool
        self.conv_branch = Conv(c1, c_conv, 3, 2, 1)
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv(c1, c_pool, 1, 1),
        )

    def forward(self, x):
        return torch.cat((self.conv_branch(x), self.pool_branch(x)), dim=1)


class ADown(nn.Module): # Downsample x2分支
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class FrequencyFocusedDownSampling(nn.Module):  # Downsample x2分支 with parallel FGM
    def __init__(self, c1, c2):  
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        self.ffm = FFM(self.c)  # FGM 模块处理 x2 分支

        # 1x1 卷积用于在拼接后减少通道数
        self.conv_reduce = Conv(self.c * 2, self.c, 1, 1)

        # 新增的卷积层用于调整 fgm_out 的空间尺寸
        self.conv_resize = Conv(self.c, self.c, 3, 2, 1)
#经过池化后分成两个分支，一个分支经过 cv1 处理，另一个分支经过 fgm + maxpool cv2 处理，然后将两个分支拼接在一起，最后使用 1x1 卷积将通道数减少到预期的值。公式写一个表达一下，x1,x2用文字描述一下是什么，cv1,cv2也是呀
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)

        # 并联处理 x2 分支
        fgm_out = self.ffm(x2)  # FGM 处理的输出
        fgm_out = self.conv_resize(fgm_out)  # 调整 fgm_out 的空间尺寸
        pooled_out = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        pooled_out = self.cv2(pooled_out)

        # 将 FGM 输出和 MaxPool2d + Conv 输出拼接
        x2 = torch.cat((fgm_out, pooled_out), 1)
        
        # 使用 1x1 卷积将通道数减少到预期的值
        x2 = self.conv_reduce(x2)

        return torch.cat((x1, x2), 1)
    
    
class SemanticAlignmenCalibration(nn.Module):  # 
    def __init__(self, inc):
        super(SemanticAlignmenCalibration, self).__init__()
        hidden_channels = inc[0]

        self.groups = 2
        self.spatial_conv = Conv(inc[0], hidden_channels, 3)  # 用于处理高分辨率的空间特征
        self.semantic_conv = Conv(inc[1], hidden_channels, 3)  # 用于处理低分辨率的语义特征

        # FGM模块：用于在频域中增强特征
        self.frequency_enhancer = FFM(hidden_channels)
        # 门控卷积：结合空间和频域特征
        self.gating_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)
        
        # 用于生成偏移量的卷积序列
        self.offset_conv = nn.Sequential(
            Conv(hidden_channels * 2, 64),  # 处理拼接后的特征
            nn.Conv2d(64, self.groups * 4 + 2, kernel_size=3, padding=1, bias=False)  # 生成偏移量
        )

        self.init_weights()
        self.offset_conv[1].weight.data.zero_()  # 初始化最后一层卷积的权重为零

    def init_weights(self):
        # 初始化卷积层的权重
        for layer in self.children():
            if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        coarse_features, semantic_features = x
        batch_size, _, out_h, out_w = coarse_features.size()

        # 处理低分辨率的语义特征 (1/8 下采样)
        semantic_features = self.semantic_conv(semantic_features)
        semantic_features = F.interpolate(semantic_features, coarse_features.size()[2:], mode='bilinear', align_corners=True)

        # 频域增强特征
        enhanced_frequency = self.frequency_enhancer(semantic_features)
        
        # 门控机制融合频域和空间域的特征
        gate = torch.sigmoid(self.gating_conv(semantic_features))
        fused_features = semantic_features * (1 - gate) + enhanced_frequency * gate

        # 处理高分辨率的空间特征 (1/8 下采样)
        coarse_features = self.spatial_conv(coarse_features)

        # 拼接处理后的空间特征和融合后的特征
        conv_results = self.offset_conv(torch.cat([coarse_features, fused_features], 1))

        # 调整特征维度以适应分组
        fused_features = fused_features.reshape(batch_size * self.groups, -1, out_h, out_w)
        coarse_features = coarse_features.reshape(batch_size * self.groups, -1, out_h, out_w)

        # 获取偏移量
        offset_low = conv_results[:, 0:self.groups * 2, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)
        offset_high = conv_results[:, self.groups * 2:self.groups * 4, :, :].reshape(batch_size * self.groups, -1, out_h, out_w)

        # 生成归一化网格用于偏移校正
        normalization_factors = torch.tensor([[[[out_w, out_h]]]]).type_as(fused_features).to(fused_features.device)
        grid_w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        grid_h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        base_grid = torch.cat((grid_h.unsqueeze(2), grid_w.unsqueeze(2)), 2)
        base_grid = base_grid.repeat(batch_size * self.groups, 1, 1, 1).type_as(fused_features).to(fused_features.device)

        # 使用生成的偏移量对网格进行调整
        adjusted_grid_l = base_grid + offset_low.permute(0, 2, 3, 1) / normalization_factors
        adjusted_grid_h = base_grid + offset_high.permute(0, 2, 3, 1) / normalization_factors

        # 进行特征采样
        coarse_features = F.grid_sample(coarse_features, adjusted_grid_l, align_corners=True)
        fused_features = F.grid_sample(fused_features, adjusted_grid_h, align_corners=True)

        # 调整维度回到原始形状
        coarse_features = coarse_features.reshape(batch_size, -1, out_h, out_w)
        fused_features = fused_features.reshape(batch_size, -1, out_h, out_w)

        # 融合增强后的特征
        attention_weights = 1 + torch.tanh(conv_results[:, self.groups * 4:, :, :])
        final_features = fused_features * attention_weights[:, 0:1, :, :] + coarse_features * attention_weights[:, 1:2, :, :]

        return final_features
