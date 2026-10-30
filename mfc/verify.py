"""验证数据划分，并从落盘预测独立重算指标；无需重新训练。"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from .io import read_json, sha256, write_json
from .metrics import compute_metrics


def audit_data(data_root):
    data_root = Path(data_root)
    metadata = read_json(data_root / "metadata.json")
    with (data_root / "windows.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sets = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    query_hashes = {}
    for split, rs in sets.items():
        assert len(rs) == metadata["split_counts"][split]
        with np.load(data_root / f"{split}.npz", allow_pickle=False) as d:
            assert np.isfinite(d["query"]).all()
            assert d["query"].shape[0] == len(rs)
            assert d["query"].shape[-1] == metadata["window_length"]
            assert d["label"].min() >= 0 and d["label"].max() < len(metadata["class_names"])
            query_hashes[split] = {hashlib.sha256(q.tobytes()).hexdigest() for q in d["query"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        files_a = {r["source_file"] for r in sets[a]}
        files_b = {r["source_file"] for r in sets[b]}
        assert not files_a & files_b, f"采集记录跨集合：{a}/{b}"
        assert not query_hashes[a] & query_hashes[b], f"不同文件名下出现相同原始窗口：{a}/{b}"
    if metadata["dataset"] == "CWRU":
        assert {int(r["load_hp"]) for r in sets["train"]} == {0, 1}
        assert {int(r["load_hp"]) for r in sets["val"]} == {2}
        assert {int(r["load_hp"]) for r in sets["test"]} == {3}
        reference = metadata["reference"]
        for r in rows:
            if r["source_file"] == reference["source_file"]:
                assert int(r["start"]) >= reference["end"]
    elif metadata["protocol"] == "chronological":
        for bearing in {r["bearing"] for r in rows}:
            indices = {s: [int(r["record_index"]) for r in rs if r["bearing"] == bearing] for s, rs in sets.items()}
            assert max(indices["train"]) < min(indices["val"])
            assert max(indices["val"]) < min(indices["test"])
    else:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not {r["bearing"] for r in sets[a]} & {r["bearing"] for r in sets[b]}
    return {"passed": True, "split_counts": metadata["split_counts"], "source_record_disjoint": True,
            "exact_query_content_disjoint": True}


def verify_run(run_root, data_root, reload_model=True):
    run_root, data_root = Path(run_root), Path(data_root)
    config, recorded = read_json(run_root / "config.json"), read_json(run_root / "metrics.json")
    for name, expected in config["data_hashes"].items():
        assert sha256(data_root / name) == expected, f"训练后数据发生变化：{name}"
    assert sha256(run_root / "best.pt") == recorded["checkpoint_sha256"]
    with (run_root / "test_predictions.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    classes = len(config["data_metadata"]["class_names"])
    probabilities = np.array([[float(r[f"probability_{i}"]) for i in range(classes)] for r in rows])
    actual = compute_metrics(np.array([int(r["true_label"]) for r in rows]), probabilities,
                             np.array([float(r["rul_true"]) for r in rows]),
                             np.array([float(r["rul_predicted"]) for r in rows]),
                             np.array([r["rul_evaluable"] == "True" for r in rows]), classes)
    for key, value in actual.items():
        if isinstance(value, list):
            assert value == recorded[key], key
        elif value is None:
            assert recorded[key] is None
        else:
            assert np.isclose(value, recorded[key], atol=1e-7), (key, value, recorded[key])
    if reload_model:
        import torch
        from .data import WindowDataset
        from .predict import predict
        torch.set_num_threads(4)
        data = WindowDataset(data_root, "test")
        q = data.query[:8].numpy()
        r = data.references[data.reference_ids[:8]].numpy()
        outputs = predict(run_root / "best.pt", q, r)
        reloaded = np.asarray([x["class_probabilities"] for x in outputs])
        assert np.allclose(reloaded, probabilities[:8], atol=2e-5), "重载模型后概率不一致"
        if config["data_metadata"]["rul_label_counts"]["train"]:
            reloaded_rul = np.asarray([x["rul_raw"] for x in outputs])
            assert np.allclose(reloaded_rul, [float(r["rul_predicted"]) for r in rows[:8]], atol=2e-5), "重载RUL预测不一致"
        else:
            assert all(x["rul_raw"] is None and x["rul_status"] == "untrained" for x in outputs)
    return {"passed": True, "predictions_recomputed": len(rows), "checkpoint_reloaded": reload_model}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--run")
    p.add_argument("--out")
    args = p.parse_args()
    result = {"data": audit_data(args.data)}
    if args.run:
        result["run"] = verify_run(args.run, args.data)
    if args.out:
        write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
