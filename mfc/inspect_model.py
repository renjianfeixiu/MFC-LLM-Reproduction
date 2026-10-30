"""逐步打印真实样本在四个模块中的形状、注意力权重和最终预测。"""
import argparse

import torch

from .data import WindowDataset
from .model import ModelConfig, SignalPHM, standardize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/cwru")
    p.add_argument("--checkpoint", default="results/cwru/full_seed42/best.pt")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SignalPHM(ModelConfig(**saved["model_config"])).eval()
    model.load_state_dict(saved["state_dict"])
    if model.cfg.variant != "full":
        raise ValueError("此教学入口逐步演示完整网络，请提供full权重")
    sample = WindowDataset(args.data, "test")[args.index]
    query, reference = sample["query"][None], sample["reference"][None]
    print("当前信号", tuple(query.shape), "前8点", query[0, 0, :8].tolist())
    print("健康参考", tuple(reference.shape), "前8点", reference[0, 0, :8].tolist())
    with torch.no_grad():
        signals = torch.cat([standardize(query), standardize(reference), standardize(query - reference)], dim=0)
        h = model.encoder(signals)
        z = model.channel_attention(h)
        zq, zr, zres = z.split(1, dim=0)
        values = zres if model.cfg.value_source == "residual" else zr
        attended, weights = model.cross_attention(zq, zr, values, return_weights=True)
        scales = [conv(attended) for conv in model.multiscale]
        fused = torch.cat(scales, dim=1)
        features = model.feature(fused)
        logits = model.classifier(features)
        actual = model(query, reference)
        torch.testing.assert_close(logits, actual["logits"])
        for label, tensor in [("三路合批标准化信号", signals), ("卷积特征", h), ("通道加权", z),
                              ("Q输入", zq), ("K输入", zr), ("V输入", values), ("注意力权重", weights),
                              ("注意力输出", attended), ("多尺度拼接", fused), ("1024维表示", features),
                              ("分类logits", logits), ("分类概率", logits.softmax(-1))]:
            print(label, tuple(tensor.shape))
        indices = weights[0, 0].topk(3)
        print("第一个查询位置最关注的3个参考位置：", indices.indices.tolist(), "权重", indices.values.tolist())
        print("该行所有参考权重之和：", weights[0, 0].sum().item())
        truth = int(sample["label"])
        predicted = int(logits.argmax(-1))
        print("真实类别：", saved["class_names"][truth], "预测类别：", saved["class_names"][predicted])
        print("所有类别概率：", logits.softmax(-1)[0].tolist())
        print("RUL是否有真值：", bool(sample["rul_mask"]), "；CWRU权重的RUL输出不作寿命解释。")


if __name__ == "__main__":
    main()
