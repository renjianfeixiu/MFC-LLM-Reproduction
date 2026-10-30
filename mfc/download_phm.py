"""下载 PHM2012 公开镜像中的原始振动 CSV，固定提交并校验 Git blob SHA1。

数据集来源是 FEMTO-ST / IEEE PHM2012；这里的传输源是第三方 GitHub 镜像。
固定镜像版本及其完整性不等于已与主办方文件逐字节交叉核验。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
from pathlib import Path
import threading
import time

import requests

from .data import read_vibration_csv
from .io import read_json, sha256, write_json

REPO = "Lucky-Loek/ieee-phm-2012-data-challenge-dataset"
COMMIT = "129d43e0aa71e1da992bb4c0f69d54fd990c58cf"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return LOCAL.session


def blob_hash(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def download(root, bearings, workers=8):
    root = Path(root)
    tree_path = root / "mirror_tree.json"
    if tree_path.exists():
        tree = read_json(tree_path)
    else:
        response = requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", timeout=120)
        response.raise_for_status()
        tree = response.json()
        if tree.get("truncated"):
            raise RuntimeError("镜像目录树被截断，不能确认全量轨迹")
        write_json(tree_path, tree)
    files = [x for x in tree["tree"] if x["type"] == "blob" and
             any(x["path"].startswith(f"Learning_set/{b}/acc_") for b in bearings)]
    if not files:
        raise ValueError("没有匹配到轴承")

    def one(item):
        relative = Path(item["path"]).relative_to("Learning_set")
        path = root / relative
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item['path']}"
        for attempt in range(3):
            try:
                if not path.exists():
                    response = session().get(url, timeout=(20, 90))
                    response.raise_for_status()
                    content = response.content
                    if blob_hash(content) != item["sha"]:
                        raise ValueError("镜像内容与固定提交中的 Git blob SHA1 不一致")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    part = path.with_suffix(".csv.part")
                    part.write_bytes(content)
                    shape = read_vibration_csv(part, "phm2012").shape
                    if shape != (2, 2560):
                        raise ValueError(f"原始振动形状异常：{shape}")
                    part.replace(path)
                else:
                    if blob_hash(path.read_bytes()) != item["sha"]:
                        raise ValueError(f"已有文件校验失败：{path}")
                return {"path": str(relative), "url": url, "git_blob_sha1": item["sha"],
                        "sha256": sha256(path), "bytes": path.stat().st_size}
            except Exception:
                path.with_suffix(".csv.part").unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, record in enumerate(executor.map(one, files), 1):
            records.append(record)
            if i % 100 == 0 or i == len(files):
                print(f"PHM2012 已校验 {i}/{len(files)} 个真实振动记录", flush=True)
    write_json(root / "download_manifest.json", {"original_dataset": "IEEE PHM2012 / FEMTO-ST PRONOSTIA",
               "mirror_repository": f"https://github.com/{REPO}", "mirror_commit": COMMIT,
               "bearings": bearings, "files": records, "cross_checked_against_official_archive": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/phm2012")
    parser.add_argument("--bearings", nargs="+", default=["Bearing3_1", "Bearing3_2"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(**vars(args))


if __name__ == "__main__":
    main()
