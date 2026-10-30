"""真实数据预处理。先按采集文件/时间划分，再切窗；保存每个窗口的来源。

分类：CWRU 0/1 HP 训练，2 HP 验证，3 HP 测试。此划分测试负载迁移，
并不代表物理轴承完全独立。寿命：支持 PHM2012 与 XJTU 原始 CSV。
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .io import read_json, sha256, write_json


def read_cwru(path):
    mat = loadmat(path)
    channels = []
    for suffix in ("_DE_time", "_FE_time"):
        keys = [k for k in mat if k.endswith(suffix)]
        # 官方99.mat还包含X098变量；先按文件编号选X099，避免误用1HP序列造成泄漏。
        expected = f"X{int(Path(path).stem):03d}{suffix}" if Path(path).stem.isdigit() else None
        if expected in keys:
            keys = [expected]
        if len(keys) != 1:
            raise ValueError(f"{path}: 无法唯一找到 {suffix}: {keys}")
        channels.append(np.asarray(mat[keys[0]], dtype=np.float32).reshape(-1))
    n = min(map(len, channels))
    result = np.stack([x[:n] for x in channels])
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: NaN/Inf")
    return result


def read_vibration_csv(path, dataset):
    """PHM2012 最后两列为振动；XJTU 前两列为振动，兼容一行表头。"""
    first = Path(path).open(encoding="utf-8-sig").readline()
    delimiter = ";" if first.count(";") > first.count(",") else ","
    first_fields = first.strip().split(delimiter)
    try:
        list(map(float, first_fields))
        skip = 0
    except ValueError:
        skip = 1
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip, dtype=np.float32, ndmin=2)
    data = data[:, -2:] if dataset == "phm2012" else data[:, :2]
    if data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"{path}: 需要两列有限数值的振动数据")
    return np.ascontiguousarray(data.T)


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(path))]


def window_starts(n, length, stride, start=0, max_windows=None):
    if n < start + length:
        return []
    starts = np.arange(start, n - length + 1, stride, dtype=int)
    if max_windows and len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows, dtype=int)]
    return starts.tolist()


def save_processed(out, splits, reference_bank, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "references.npy", np.asarray(reference_bank, dtype=np.float32), allow_pickle=False)
    audit = []
    for name, samples in splits.items():
        if not samples:
            raise ValueError(f"{name} 没有样本")
        np.savez_compressed(out / f"{name}.npz",
                            query=np.stack([x["query"] for x in samples]).astype(np.float32),
                            label=np.array([x["label"] for x in samples], dtype=np.int64),
                            rul=np.array([x.get("rul", 0.) for x in samples], dtype=np.float32),
                            rul_mask=np.array([x.get("rul_mask", False) for x in samples], dtype=bool),
                            reference_id=np.array([x.get("reference_id", 0) for x in samples], dtype=np.int64))
        for i, sample in enumerate(samples):
            audit.append({"split": name, "index": i, **{k: v for k, v in sample.items() if k != "query"}})
    fields = list(dict.fromkeys(k for row in audit for k in row))
    with (out / "windows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    metadata["split_counts"] = {name: len(samples) for name, samples in splits.items()}
    metadata["class_counts"] = {name: dict(Counter(str(x["label"]) for x in samples)) for name, samples in splits.items()}
    metadata["rul_label_counts"] = {name: sum(x.get("rul_mask", False) for x in samples) for name, samples in splits.items()}
    write_json(out / "metadata.json", metadata)
    print(metadata["split_counts"], flush=True)
    print("有效 RUL 标签数：", metadata["rul_label_counts"], flush=True)


def prepare_cwru(root, out, length=4096, max_train=60, max_eval=30):
    root = Path(root)
    catalog = read_json(root / "catalog.json")
    names = ["normal"] + sorted({x["class_name"] for x in catalog} - {"normal"})
    labels = {name: i for i, name in enumerate(names)}
    healthy = next(x for x in catalog if x["class_name"] == "normal" and x["load_hp"] == 0)
    reference = read_cwru(root / healthy["filename"])[:, :length].copy()
    splits = {name: [] for name in ("train", "val", "test")}
    source_hashes = {}
    for item in catalog:
        path = root / item["filename"]
        signal = read_cwru(path)
        source_hashes[item["filename"]] = sha256(path)
        split = "train" if item["load_hp"] in (0, 1) else "val" if item["load_hp"] == 2 else "test"
        # 健康参考来自训练文件开头；这段及一个窗口的间隔不再作为查询样本。
        start = 2 * length if item["filename"] == healthy["filename"] else 0
        starts = window_starts(signal.shape[-1], length, length // 2 if split == "train" else length,
                               start=start, max_windows=max_train if split == "train" else max_eval)
        for s in starts:
            splits[split].append({"query": signal[:, s:s + length], "label": labels[item["class_name"]],
                                  "source_file": item["filename"], "load_hp": item["load_hp"],
                                  "start": s, "end": s + length, "reference_id": 0})
    metadata = {"dataset": "CWRU", "task": "classification", "class_names": names, "window_length": length,
                "channels": ["DE", "FE"], "protocol": "load_0_1_train__load_2_val__load_3_test",
                "reference": {"source_file": healthy["filename"], "split": "train", "start": 0, "end": length},
                "normalization": "per-window per-channel population z-score in model; residual computed before z-score",
                "source_sha256": source_hashes,
                "limitations": ["跨负载测试不等于跨物理轴承测试", "非论文的随机70/15/15划分",
                                "不从CWRU推造RUL标签", "原样使用公开DE/FE序列；正常与故障记录的采样率元数据需另核"]}
    save_processed(out, splits, [reference], metadata)


def causal_average(values, width=5):
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.r_[0., np.cumsum(values)]
    return np.array([(cumulative[t + 1] - cumulative[max(0, t + 1 - width)]) /
                     min(width, t + 1) for t in range(len(values))])


def detect_fpt(health, healthy_records=20, smooth_width=5, consecutive=3):
    """论文3σ-FPT的显式实现选择：融合RMS→因果平滑→后向二阶差分。

    基线仅取最初 healthy_records 条已知健康记录；确认连续3次越界后当下报警，
    不将未来第3次的信息回填到第1次。原文未给HI融合及平滑细节，这不是唯一实现。
    """
    if len(health) <= healthy_records + consecutive + 2:
        raise ValueError("寿命序列太短，无法估计健康基线")
    smooth = causal_average(health, smooth_width)
    second = np.zeros(len(smooth))
    second[2:] = smooth[2:] - 2 * smooth[1:-1] + smooth[:-2]
    baseline = second[max(smooth_width, 2):healthy_records]
    mu, sigma = float(baseline.mean()), max(float(baseline.std()), 1e-12)
    outside = np.abs(second - mu) > 3 * sigma
    count = 0
    for t in range(healthy_records, len(outside)):
        count = count + 1 if outside[t] else 0
        if count >= consecutive:
            return t, {"mu": mu, "sigma": sigma, "healthy_records": healthy_records,
                       "smooth_width": smooth_width, "consecutive": consecutive,
                       "fpt_record_index_zero_based": t, "fpt_record_number": t + 1}
    raise ValueError("没有检测到连续3次越界的FPT；不能悄悄改成按寿命百分比标注")


def discover_bearings(root, dataset):
    pattern = "acc_*.csv" if dataset == "phm2012" else "*.csv"
    directories = sorted({x.parent for x in Path(root).rglob(pattern)}, key=natural_key)
    directories = [p for p in directories if re.fullmatch(r"Bearing\d+_\d+", p.name, flags=re.I)]
    if not directories:
        raise ValueError(f"{root} 下没有找到 BearingX_Y 原始CSV")
    return [(p.name, sorted(p.glob(pattern), key=natural_key)) for p in directories]


def prepare_life(root, out, dataset="phm2012", length=2560, healthy_records=20, max_windows=1,
                 label_scheme="fpt", protocol="chronological", train_bearings=None,
                 val_bearings=None, test_bearings=None):
    bearings = discover_bearings(root, dataset)
    splits = {name: [] for name in ("train", "val", "test")}
    references, bearing_meta = [], []
    declared = {"train": set(train_bearings or []), "val": set(val_bearings or []), "test": set(test_bearings or [])}
    if protocol == "bearing":
        if not all(declared.values()):
            raise ValueError("bearing 划分必须显式提供 train/val/test 的轴承ID")
        if any(declared[a] & declared[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
            raise ValueError("轴承ID跨集合重叠")
        available = {name for name, _ in bearings}
        if (set.union(*declared.values()) - available):
            raise ValueError("指定的轴承文件缺失")
    for bearing, files in bearings:
        if protocol == "bearing" and bearing not in set.union(*declared.values()):
            continue
        signals = [read_vibration_csv(f, dataset) for f in files]
        if any(s.shape[-1] < length for s in signals):
            raise ValueError(f"{bearing}: 有记录短于窗口 {length}；不得把间隔采样的不同记录直接拼成连续波形")
        health = np.array([np.sqrt(np.mean(s.astype(np.float64) ** 2)) for s in signals])
        # 健康参考取最初采样的第一片；不让该参考片也成为查询。
        reference_id = len(references)
        references.append(signals[0][:, :length].copy())
        if label_scheme == "fpt":
            fpt, fpt_info = detect_fpt(health, healthy_records)
        else:
            fpt, fpt_info = 0, {"note": "full_life: 非论文FPT标签，不能和论文RUL表直接比较"}
        n = len(files)
        # 训练前70%的内部再分验证：前55%训练，接着15%验证，最后30%测试。
        train_stop, val_stop = int(n * .55), int(n * .70)
        if healthy_records >= train_stop:
            raise ValueError("健康基线越过训练时段")
        for t, (file, signal) in enumerate(zip(files, signals)):
            if t == 0:
                continue
            if protocol == "bearing":
                split = next(k for k, v in declared.items() if bearing in v)
                # 独立轴承评估允许已声明的健康启动校准，不把该校准段计入测试查询。
                if t < healthy_records:
                    continue
            else:
                split = "train" if t < train_stop else "val" if t < val_stop else "test"
            remaining = min(1., (n - 1 - t) / max(n - 1 - fpt, 1))
            for start in window_starts(signal.shape[-1], length, length, max_windows=max_windows):
                splits[split].append({"query": signal[:, start:start + length], "label": int(t >= fpt),
                                      "rul": remaining, "rul_mask": bool(t >= fpt), "reference_id": reference_id,
                                      "source_file": str(file.relative_to(root)), "bearing": bearing,
                                      "record_index": t, "start": start, "end": start + length})
        bearing_meta.append({"bearing": bearing, "records": n, "fpt": fpt_info,
                             "reference_record": str(files[0].relative_to(root)),
                             "train_stop": train_stop, "val_stop": val_stop,
                             "rms": health.tolist(), "source_files_sha256": {str(f.relative_to(root)): sha256(f) for f in files}})
        print(f"{bearing}: {n} records, FPT index={fpt}", flush=True)
    if not any(x["rul_mask"] for x in splits["train"]):
        raise ValueError("当前时序划分的训练段没有FPT后的寿命标签，无法训练回归；保留此失败，不以测试数据补训练")
    metadata = {"dataset": dataset.upper(), "task": "multitask", "class_names": ["healthy", "degraded"],
                "window_length": length, "channels": ["horizontal", "vertical"], "label_scheme": label_scheme,
                "protocol": protocol, "bearing_details": bearing_meta,
                "rul_definition": "post-FPT (end_index-t)/(end_index-FPT); pre-FPT masked; true endpoint only used as label",
                "classification_label": "derived from causal 3-sigma FPT, not expert fault-type annotation",
                "normalization": "per-window per-channel z-score; raw RMS only creates labels and is not model input",
                "limitations": ["健康状态分类标签由HI/FPT派生，不能当作人工故障类型真值",
                                "chronological模式测试同一轴承后期，不能证明跨轴承泛化",
                                "每轴承初始健康记录用于参考和基线校准，属于明确的可用启动信息",
                                "融合HI、平滑、基线长度等是为缺失细节补充的实现选择"]}
    save_processed(out, splits, references, metadata)


class WindowDataset(Dataset):
    def __init__(self, root, split):
        root = Path(root)
        with np.load(root / f"{split}.npz", allow_pickle=False) as data:
            self.query = torch.from_numpy(data["query"].copy())
            self.labels = torch.from_numpy(data["label"].copy())
            self.rul = torch.from_numpy(data["rul"].copy())
            self.rul_mask = torch.from_numpy(data["rul_mask"].copy())
            self.reference_ids = torch.from_numpy(data["reference_id"].copy())
        self.references = torch.from_numpy(np.load(root / "references.npy", allow_pickle=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"query": self.query[i], "reference": self.references[self.reference_ids[i]],
                "label": self.labels[i], "rul": self.rul[i], "rul_mask": self.rul_mask[i], "index": i}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    cwru = sub.add_parser("cwru")
    cwru.add_argument("--root", default="data/raw/cwru")
    cwru.add_argument("--out", default="data/cwru")
    cwru.add_argument("--length", type=int, default=4096)
    cwru.add_argument("--max-train", type=int, default=60)
    cwru.add_argument("--max-eval", type=int, default=30)
    for name in ("phm2012", "xjtu"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--length", type=int, default=2560 if name == "phm2012" else 4096)
        p.add_argument("--healthy-records", type=int, default=20)
        p.add_argument("--max-windows", type=int, default=1)
        p.add_argument("--label-scheme", choices=["fpt", "full_life"], default="fpt")
        p.add_argument("--protocol", choices=["chronological", "bearing"], default="chronological")
        for split in ("train", "val", "test"):
            p.add_argument(f"--{split}-bearings", nargs="+")
    args = vars(parser.parse_args())
    dataset = args.pop("dataset")
    if dataset == "cwru":
        prepare_cwru(**args)
    else:
        prepare_life(dataset=dataset, **args)


if __name__ == "__main__":
    main()
