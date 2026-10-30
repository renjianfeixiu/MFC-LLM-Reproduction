"""多随机种子、CNN基线/消融的一致训练入口，不依据测试结果调参。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import read_json, write_json
from .train import train
from .verify import audit_data, verify_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/signal.json")
    p.add_argument("--variants", nargs="+", default=["full", "cnn"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int)
    args = p.parse_args()
    audit_data(args.data)
    runs = []
    for variant in args.variants:
        for seed in args.seeds:
            folder = Path(args.out) / f"{variant}_seed{seed}"
            if (folder / "metrics.json").exists():
                result = read_json(folder / "metrics.json")
            else:
                result = train(args.data, folder, args.config, variant, seed, args.device, args.epochs)
            verification = verify_run(folder, args.data)
            write_json(folder / "verification.json", verification)
            runs.append({"folder": str(folder), **{k: v for k, v in result.items() if k != "confusion_matrix"}})
    aggregated = {}
    for variant in args.variants:
        selected = [r for r in runs if r["variant"] == variant]
        item = {"runs": len(selected), "seeds": [r["seed"] for r in selected]}
        for key in ("accuracy", "f1_macro", "rul_rmse", "rul_mae", "rul_r2"):
            values = [r[key] for r in selected if r.get(key) is not None]
            if values:
                item[key] = {"mean": float(np.mean(values)), "std_population": float(np.std(values)), "values": values}
        aggregated[variant] = item
    write_json(Path(args.out) / "summary.json", {"aggregated": aggregated, "runs": runs})
    print(aggregated, flush=True)


if __name__ == "__main__":
    main()
