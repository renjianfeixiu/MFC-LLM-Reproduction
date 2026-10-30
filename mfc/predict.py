"""用保存的信号模型进行推理。输入为 [N,C,L] 的 .npy 波形数组。

例：python -m mfc.predict --checkpoint results/cwru/full_seed42/best.pt \
    --query my_query.npy --reference my_healthy.npy --out predictions.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .io import read_json, write_json
from .model import ModelConfig, SignalPHM


def predict(checkpoint, query, reference, device="cpu", batch_size=32):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = SignalPHM(ModelConfig(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    sidecar = Path(checkpoint).with_name("config.json")
    rul_trained = saved.get("rul_trained")
    if rul_trained is None and sidecar.exists():
        counts = read_json(sidecar).get("data_metadata", {}).get("rul_label_counts", {})
        if "train" in counts:
            rul_trained = counts["train"] > 0
    q = torch.as_tensor(query, dtype=torch.float32)
    r = torch.as_tensor(reference, dtype=torch.float32)
    if q.ndim == 2:
        q = q.unsqueeze(0)
    if r.ndim == 2:
        r = r.unsqueeze(0)
    if len(r) == 1:
        r = r.expand(len(q), -1, -1)
    if q.shape != r.shape:
        raise ValueError("query 与 reference 形状不兼容")
    results = []
    with torch.no_grad():
        for start in range(0, len(q), batch_size):
            output = model(q[start:start + batch_size].to(device), r[start:start + batch_size].to(device))
            p = output["logits"].softmax(-1).cpu().numpy()
            remaining = output["rul"].cpu().numpy()
            for probs, rul in zip(p, remaining):
                i = int(probs.argmax())
                results.append({"class_index": i, "class_name": saved["class_names"][i],
                                "class_probabilities": probs.tolist(),
                                "rul_status": "trained" if rul_trained else "untrained" if rul_trained is False else "unknown",
                                "rul_raw": float(rul) if rul_trained else None})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", default="predictions.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    results = predict(args.checkpoint, np.load(args.query, allow_pickle=False),
                      np.load(args.reference, allow_pickle=False), args.device)
    write_json(args.out, results)
    print(f"保存 {len(results)} 条预测到 {args.out}；CWRU训练权重的RUL分支未训练，不应解释为寿命估计。")


if __name__ == "__main__":
    main()
