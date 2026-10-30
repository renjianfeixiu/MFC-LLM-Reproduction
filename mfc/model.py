"""论文式(2)--(15)对应的可运行信号模型。

论文未公开卷积层数/通道、下采样、z_v 来源等；本文件的默认值是复现者
明确补充的工程设定，不能称作作者的精确网络。输入 q,r 均为 [B,C,L]。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    in_channels: int = 2
    num_classes: int = 10
    channels: tuple[int, ...] = (16, 32, 64)
    kernels: tuple[int, ...] = (15, 7, 3)
    strides: tuple[int, ...] = (8, 4, 2)
    attention_tokens: int = 64
    pool_bins: int = 4
    feature_dim: int = 1024
    hidden_dim: int = 128
    dropout: float = 0.1
    value_source: str = "residual"
    variant: str = "full"

    def to_dict(self):
        return asdict(self)


def standardize(x: torch.Tensor, eps: float = 1e-6):
    """每个样本、每个物理通道沿时间标准化；不使用其他样本或未来片段。"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)
    return (x - mean) / torch.sqrt(var + eps)


class ChannelAttention(nn.Module):
    """式(6)：沿时间池化，拼接 avg/max，再用两层 MLP 产生通道权重。"""
    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * channels, max(channels // 8, 4)),
                                 nn.ReLU(), nn.Linear(max(channels // 8, 4), channels), nn.Sigmoid())

    def forward(self, h):
        statistics = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        alpha = self.mlp(statistics).unsqueeze(-1)
        return h * alpha


class ConvEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers = []
        previous = cfg.in_channels
        for channels, kernel, stride in zip(cfg.channels, cfg.kernels, cfg.strides, strict=True):
            layers += [nn.Conv1d(previous, channels, kernel, stride=stride, padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(channels), nn.LeakyReLU(0.1)]
            previous = channels
        layers.append(nn.AdaptiveAvgPool1d(cfg.attention_tokens))
        self.layers = nn.Sequential(*layers)

    def forward(self, signal):
        return self.layers(signal)


class CrossAttention(nn.Module):
    """式(8)(9)。d 取通常的 sqrt(D) 缩放；单头，无额外残差支路。"""
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)

    def forward(self, query, reference, values, return_weights=False):
        q = self.q(query.transpose(1, 2))
        k = self.k(reference.transpose(1, 2))
        v = self.v(values.transpose(1, 2))
        weights = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
        result = (weights @ v).transpose(1, 2)
        return (result, weights) if return_weights else result


class SignalPHM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.variant not in {"full", "cnn", "no_channel", "no_cross", "no_multiscale"}:
            raise ValueError(cfg.variant)
        if cfg.value_source not in {"residual", "reference"}:
            raise ValueError("value_source 必须显式选择 residual 或 reference")
        self.cfg = cfg
        channels = cfg.channels[-1]
        self.encoder = ConvEncoder(cfg)  # 三种信号共享卷积权重，论文未说明是否共享。
        self.channel_attention = (ChannelAttention(channels) if cfg.variant not in {"cnn", "no_channel"}
                                  else nn.Identity())
        self.cross_attention = (CrossAttention(channels) if cfg.variant not in {"cnn", "no_cross"} else None)
        kernels = () if cfg.variant == "cnn" else (1,) if cfg.variant == "no_multiscale" else (1, 3, 5, 7)
        self.multiscale = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels if len(kernels) == 4 else 4 * channels,
                                    k, padding=k // 2), nn.LeakyReLU(0.1)) for k in kernels])
        merged_channels = channels if cfg.variant == "cnn" else 4 * channels
        self.feature = nn.Sequential(nn.AdaptiveAvgPool1d(cfg.pool_bins), nn.Flatten(),
                                     nn.Linear(merged_channels * cfg.pool_bins, cfg.feature_dim), nn.LeakyReLU(0.1))
        self.classifier = nn.Sequential(nn.Linear(cfg.feature_dim, cfg.hidden_dim), nn.LeakyReLU(0.1),
                                        nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, cfg.num_classes))
        self.regressor = nn.Sequential(nn.Linear(cfg.feature_dim, cfg.hidden_dim), nn.LeakyReLU(0.1),
                                       nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, 1))

    def forward(self, query, reference, return_features=False):
        if query.ndim != 3 or query.shape != reference.shape:
            raise ValueError("query/reference 必须具有相同的 [batch,channels,length] 形状")
        if not torch.isfinite(query).all() or not torch.isfinite(reference).all():
            raise ValueError("信号含 NaN 或 Inf")
        if self.cfg.variant == "cnn":
            fused = self.encoder(standardize(query))
        else:
            # 先用原始物理信号做差，再分别标准化；不会悄悄把残差改成归一化信号之差。
            batch = query.shape[0]
            signals = torch.cat([standardize(query), standardize(reference), standardize(query - reference)], dim=0)
            encoded = self.encoder(signals)
            if self.cfg.variant != "no_channel":
                encoded = self.channel_attention(encoded)
            zq, zr, zres = encoded.split(batch, dim=0)
            if self.cfg.variant == "no_cross":
                fused = zq
            else:
                values = zres if self.cfg.value_source == "residual" else zr
                fused = self.cross_attention(zq, zr, values)
            fused = torch.cat([conv(fused) for conv in self.multiscale], dim=1)
        features = self.feature(fused)
        result = {"logits": self.classifier(features), "rul": self.regressor(features).squeeze(-1)}
        if return_features:
            result["features"] = features
        return result


def multitask_loss(outputs, labels, rul, rul_mask, rul_weight=1.0, classification_weight=1.0,
                   class_weights=None):
    """CWRU 没有 RUL 真值：mask=False，绝不伪造寿命标签。"""
    classification = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    if rul_mask.any():
        regression = F.mse_loss(outputs["rul"][rul_mask], rul[rul_mask])
    else:
        regression = outputs["rul"].sum() * 0.0
    total = classification_weight * classification + rul_weight * regression
    return total, {"classification": classification.detach(), "regression": regression.detach()}
