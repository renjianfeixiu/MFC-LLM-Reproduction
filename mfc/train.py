"""训练、按验证集选权重、一次性测试，保存权重/逐样本预测/曲线/配置。

例：python -m mfc.train --data data/cwru --out results/cwru/full_seed42 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import WindowDataset
from .io import read_json, sha256, write_json
from .metrics import compute_metrics
from .model import ModelConfig, SignalPHM, multitask_loss


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, device, train_settings, class_weights=None):
    model.eval()
    arrays = {k: [] for k in ("label", "probabilities", "rul", "rul_pred", "rul_mask", "index")}
    all_logits = []
    for batch in loader:
        outputs = model(batch["query"].to(device), batch["reference"].to(device))
        all_logits.append(outputs["logits"].cpu())
        arrays["probabilities"].append(outputs["logits"].softmax(-1).cpu().numpy())
        arrays["rul_pred"].append(outputs["rul"].cpu().numpy())
        for key in ("label", "rul", "rul_mask", "index"):
            arrays[key].append(batch[key].numpy())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    logits = torch.cat(all_logits)
    labels = torch.from_numpy(arrays["label"])
    ce = F.cross_entropy(logits, labels, weight=class_weights.cpu() if class_weights is not None else None).item()
    mask = arrays["rul_mask"]
    mse = float(np.mean((arrays["rul_pred"][mask] - arrays["rul"][mask]) ** 2)) if mask.any() else 0.
    result = compute_metrics(arrays["label"], arrays["probabilities"], arrays["rul"], arrays["rul_pred"],
                             mask, model.cfg.num_classes)
    result.update(classification_loss=ce, regression_loss=mse,
                  loss=train_settings["classification_weight"] * ce + train_settings["rul_weight"] * mse)
    return result, arrays


def save_predictions(out, split, arrays, data_root):
    with (Path(data_root) / "windows.csv").open(encoding="utf-8") as f:
        audit = {int(row["index"]): row for row in csv.DictReader(f) if row["split"] == split}
    rows = []
    for position, i in enumerate(arrays["index"]):
        row = {**audit[int(i)], "true_label": int(arrays["label"][position]),
               "predicted_label": int(arrays["probabilities"][position].argmax()),
               "rul_true": float(arrays["rul"][position]), "rul_predicted": float(arrays["rul_pred"][position]),
               "rul_evaluable": bool(arrays["rul_mask"][position])}
        row.update({f"probability_{c}": float(v) for c, v in enumerate(arrays["probabilities"][position])})
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (Path(out) / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out, history, metrics, arrays, class_names):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x["epoch"] for x in history], [x["train_loss"] for x in history], label="train")
    ax.plot([x["epoch"] for x in history], [x["val_loss"] for x in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Joint loss")
    ax.legend(); fig.tight_layout(); fig.savefig(Path(out) / "loss.png", dpi=150); plt.close(fig)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center", color="white" if count > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(Path(out) / "confusion_matrix.png", dpi=150); plt.close(fig)
    if arrays["rul_mask"].any():
        fig, ax = plt.subplots(figsize=(8, 4))
        use = arrays["rul_mask"]
        ax.plot(arrays["rul"][use], label="true RUL", linewidth=2)
        ax.plot(arrays["rul_pred"][use], label="predicted RUL", alpha=.8)
        ax.set(xlabel="Test-window index (trajectories concatenated)", ylabel="Normalized RUL")
        ax.legend(); fig.tight_layout(); fig.savefig(Path(out) / "rul.png", dpi=150); plt.close(fig)


def train(data_root, out, config_path="configs/signal.json", variant="full", seed=42,
          device="cpu", epochs=None, value_source=None, threads=4):
    out, data_root = Path(out), Path(data_root)
    if (out / "best.pt").exists():
        raise FileExistsError(f"不覆盖已有结果：{out}；请换一个输出目录")
    out.mkdir(parents=True, exist_ok=True)
    settings = read_json(config_path)
    metadata = read_json(data_root / "metadata.json")
    cfg_values = {**settings["model"], "in_channels": len(metadata["channels"]),
                  "num_classes": len(metadata["class_names"]), "variant": variant}
    if value_source is not None:
        cfg_values["value_source"] = value_source
    cfg = ModelConfig(**cfg_values)
    train_settings = settings["training"].copy()
    if epochs is not None:
        train_settings["epochs"] = epochs
    if threads < 1:
        raise ValueError("threads >= 1")
    torch.set_num_threads(threads)
    seed_all(seed)
    device = select_device(device)
    datasets = {split: WindowDataset(data_root, split) for split in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(seed)
    loaders = {split: DataLoader(ds, batch_size=train_settings["batch_size"], shuffle=split == "train",
                                 generator=generator if split == "train" else None, num_workers=0)
               for split, ds in datasets.items()}
    model = SignalPHM(cfg).to(device)
    if train_settings.get("balanced_classes", False):
        counts = torch.bincount(datasets["train"].labels, minlength=cfg.num_classes).float()
        weights = counts.sum() / (cfg.num_classes * counts.clamp_min(1))
        weights[counts == 0] = 0
        class_weights = weights.to(device)
    else:
        class_weights = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_settings["lr"],
                                  weight_decay=train_settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                                                         patience=train_settings["lr_patience"])
    resolved = {"model": cfg.to_dict(), "training": train_settings, "seed": seed, "device": str(device),
                "threads": threads, "data_metadata": metadata, "config_source": str(config_path),
                "data_hashes": {p.name: sha256(p) for p in data_root.iterdir() if p.is_file()},
                "class_weights": class_weights.cpu().tolist() if class_weights is not None else None,
                "code_sha256": {str(p.relative_to(Path(__file__).parent.parent)): sha256(p)
                                for p in Path(__file__).parent.glob("*.py")},
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "torch": str(torch.__version__), "numpy": np.__version__},
                "parameters": sum(p.numel() for p in model.parameters())}
    write_json(out / "config.json", resolved)
    best, stale, history = float("inf"), 0, []
    started = time.monotonic()
    for epoch in range(1, train_settings["epochs"] + 1):
        model.train()
        accumulated, seen = 0., 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch["query"], batch["reference"])
            loss, _ = multitask_loss(predictions, batch["label"], batch["rul"], batch["rul_mask"],
                                     train_settings["rul_weight"], train_settings["classification_weight"], class_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch={epoch}: loss={loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings["gradient_clip"])
            optimizer.step()
            n = len(batch["label"])
            accumulated += loss.item() * n
            seen += n
        val, _ = evaluate(model, loaders["val"], device, train_settings, class_weights)
        scheduler.step(val["loss"])
        row = {"epoch": epoch, "train_loss": accumulated / seen, "val_loss": val["loss"],
               "val_accuracy": val["accuracy"], "val_rul_rmse": val.get("rul_rmse"),
               "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        with (out / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if val["loss"] < best - 1e-7:
            best, stale = val["loss"], 0
            torch.save({"state_dict": model.state_dict(), "model_config": cfg.to_dict(),
                        "epoch": epoch, "validation_loss": best, "class_names": metadata["class_names"]}, out / "best.pt")
        else:
            stale += 1
        if train_settings.get("early_stopping_patience") and stale >= train_settings["early_stopping_patience"]:
            break
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    # 测试集到此处才执行，绝不按测试分数选择 epoch。
    test, arrays = evaluate(model, loaders["test"], device, train_settings, class_weights)
    test.update(best_epoch=checkpoint["epoch"], seed=seed, variant=variant,
                parameters=resolved["parameters"], training_seconds=time.monotonic() - started,
                prediction_clipping=False, checkpoint_sha256=sha256(out / "best.pt"))
    write_json(out / "metrics.json", test)
    save_predictions(out, "test", arrays, data_root)
    plot_results(out, history, test, arrays, metadata["class_names"])
    # 仅用训练标签的常数基线，用于解释RUL模型有没有超过简单猜测。
    majority = int(torch.bincount(datasets["train"].labels, minlength=cfg.num_classes).argmax())
    constant = {"majority_class": majority, "accuracy": float(np.mean(arrays["label"] == majority))}
    mask = datasets["train"].rul_mask
    if mask.any() and arrays["rul_mask"].any():
        mean_rul = float(datasets["train"].rul[mask].mean())
        y = arrays["rul"][arrays["rul_mask"]]
        constant.update(train_mean_rul=mean_rul, rul_mae=float(np.mean(np.abs(y - mean_rul))),
                        rul_rmse=float(np.sqrt(np.mean((y - mean_rul) ** 2))))
    write_json(out / "constant_baseline.json", constant)
    print("TEST", json.dumps({k: v for k, v in test.items() if k != "confusion_matrix"}), flush=True)
    return test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/signal.json")
    parser.add_argument("--variant", choices=["full", "cnn", "no_channel", "no_cross", "no_multiscale"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--value-source", choices=["residual", "reference"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    train(args.data, args.out, args.config, args.variant, args.seed, args.device,
          args.epochs, args.value_source, args.threads)


if __name__ == "__main__":
    main()
