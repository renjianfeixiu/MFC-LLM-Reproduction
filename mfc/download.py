"""下载论文所用 CWRU 数据；每个文件保留来源、大小、SHA256。

只请求公开网页和公开数据，不需要帐号，也不读取浏览器状态。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests
from scipy.io import loadmat

from .io import sha256, write_json

FAULT_PAGE = "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data"
NORMAL_PAGE = "https://engineering.case.edu/bearingdatacenter/normal-baseline-data"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.current = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current = dict(attrs).get("href")
            self.text = []

    def handle_data(self, data):
        if self.current is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.links.append(("".join(self.text).strip(), self.current))
            self.current = None


def cwru_catalog():
    catalog = []
    for page in (NORMAL_PAGE, FAULT_PAGE):
        response = requests.get(page, timeout=60)
        response.raise_for_status()
        parser = Links()
        parser.feed(response.text)
        for label, href in parser.links:
            if not re.search(r"\.mat(?:\?|$)", href, re.I):
                continue
            label = label.replace(" ", "")
            match = re.fullmatch(r"Normal_([0-3])", label)
            if match:
                group, load = "normal", int(match[1])
            else:
                match = re.fullmatch(r"(IR|B|OR)(007|014|021)(?:@6)?_([0-3])", label)
                if not match or (match[1] == "OR" and "@6" not in label):
                    continue
                group = {"IR": "inner", "B": "ball", "OR": "outer"}[match[1]] + "_" + match[2]
                load = int(match[3])
            url = urljoin(page, href)
            catalog.append({"label": label, "class_name": group, "load_hp": load,
                            "url": url, "source_page": page, "filename": url.split("/")[-1].split("?")[0]})
    unique = {x["label"]: x for x in catalog}
    result = sorted(unique.values(), key=lambda x: (x["class_name"], x["load_hp"]))
    if len(result) != 40:
        raise RuntimeError(f"官方页面结构可能变化：应有 40 个文件，实际 {len(result)}；请检查后再继续。")
    return result


def download_one(url: str, target: Path, kind="mat"):
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        part = target.with_suffix(target.suffix + ".part")
        try:
            if not target.exists():
                with requests.get(url, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    with part.open("wb") as f:
                        for chunk in response.iter_content(1024 * 1024):
                            f.write(chunk)
                if part.stat().st_size < 1000:
                    raise ValueError("数据太小，可能不是数据文件")
                if kind == "mat":
                    mat = loadmat(part, appendmat=False)
                    if not any(k.endswith("_DE_time") for k in mat):
                        raise ValueError("MAT 中没有 DE_time 振动数据")
                part.replace(target)
            if kind == "mat":
                mat = loadmat(target)
                if not any(k.endswith("_DE_time") for k in mat):
                    raise ValueError("已有 MAT 无效，请手动检查后移走它")
            return {"url": url, "path": str(target), "bytes": target.stat().st_size,
                    "sha256": sha256(target)}
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/cwru")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)
    catalog = cwru_catalog()
    write_json(root / "catalog.json", catalog)

    def work(item):
        record = download_one(item["url"], root / item["filename"])
        print(f"已校验 {item['label']}: {record['bytes']:,} bytes", flush=True)
        return {**item, **record}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(work, catalog))
    write_json(root / "download_manifest.json", records)
    print(f"完成：{len(records)} 个真实 CWRU MAT 文件。", flush=True)


if __name__ == "__main__":
    main()
