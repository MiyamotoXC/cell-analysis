#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实数据下载：LIVECell 相差显微细胞数据集
========================================
数据源（真实公开数据集，非合成）：
  Hugging Face  einarolafsson/live-cell-segmentation-dataset
    images/livecell_phase/<name>.tif   相差显微原始视野（520x704，uint8）
    masks/livecell_phase/<name>.tif    uint16 实例 mask（0=背景，1..N=细胞）
    splits.csv                         name, source, modality, group, split, n_objects

该仓库把 LIVECell（Sartorius / Nature Methods 2021，CC BY-NC 4.0）等 14 个公开
数据集统一整理为 image/mask 配对，并按"采集批次"（孔/皿/时序/Z-stack）划分
train/valid/test，保证测试集是模型从未见过的视野。

本脚本只取 livecell_phase 子集（5239 个视野，8 个细胞系：
A172 / BT474 / BV2 / Huh7 / MCF7 / SHSY5Y / SkBr3 / SKOV3），
落到 convert_livecell.py 期望的目录布局：

    cell_data/images/{train,val,test}/<name>.png
    cell_data/masks/{train,val,test}/<name>.png

用法：
  python download_livecell.py --list                 # 只看清单与规模，不下载
  python download_livecell.py                        # 每个划分下载 40 个视野
  python download_livecell.py --max_per_split 200
  python download_livecell.py --max_per_split 0      # 全量 5239 视野（约 900 MB）
"""
import argparse
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio.v2 as iio
import numpy as np
import tifffile
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "cell_data")

REPO = "einarolafsson/live-cell-segmentation-dataset"
SOURCE = "livecell_phase"
SPLITS_CSV = "splits.csv"

# splits.csv 里的 valid/test 名与目录名的对应
SPLIT_DIR = {"train": "train", "valid": "val", "test": "test"}

# 视野名里的时序信息：..._Phase_<well>_<site>_<时间>_<序号>
# 例 livecell_phase__test-A172_Phase_C7_1_00d04h00m_1 -> 位点 (...C7_1, 1)，时间 00d04h00m
TIME_RE = re.compile(
    r"^(livecell_phase__(?:train|val|test)-[A-Za-z0-9]+_Phase_[A-Za-z0-9]+_\d+)"
    r"_(\d+d\d+h\d+m)_(\d+)$")


def time_hours(tag):
    """'02d16h00m' -> 距起点的小时数，用于按时间排序。"""
    d, h, m = re.match(r"(\d+)d(\d+)h(\d+)m$", tag).groups()
    return int(d) * 24 + int(h) + int(m) / 60


def read_splits():
    """读 splits.csv，返回 {split_dir: [name, ...]}。"""
    path = hf_hub_download(REPO, SPLITS_CSV, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source"] == SOURCE]
    out = {}
    for r in rows:
        d = SPLIT_DIR.get(r["split"])
        if d:
            out.setdefault(d, []).append((r["name"], int(r["n_objects"])))
    return out


def fetch(repo_file, retries=4):
    """HF 走代理时连接会偶发被重置，这里做退避重试。"""
    last = None
    for i in range(retries):
        try:
            return hf_hub_download(REPO, repo_file, repo_type="dataset")
        except Exception as e:      # noqa: BLE001 - 网络错误类型不固定
            last = e
    raise RuntimeError(f"下载失败 {repo_file}: {last}")


def download_one(name, split_dir):
    """下载单个视野的图像与 mask，落到 cell_data 下。"""
    img = tifffile.imread(fetch(f"images/{SOURCE}/{name}.tif"))
    msk = tifffile.imread(fetch(f"masks/{SOURCE}/{name}.tif"))

    img_dir = os.path.join(OUT_DIR, "images", split_dir)
    msk_dir = os.path.join(OUT_DIR, "masks", split_dir)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    base = name + ".png"
    # 用 imageio 而非 cv2：cv2.imwrite 在中文路径下会静默失败
    iio.imwrite(os.path.join(img_dir, base), img.astype(np.uint8))
    iio.imwrite(os.path.join(msk_dir, base), msk.astype(np.uint16))
    return int(msk.max())


def download_time_series(frames, workers):
    """下载同一位点的时间序列 mask（LIVECell 每 4 小时拍一帧，可用来跑真实的增殖曲线）。"""
    groups = {}
    for d, names in read_splits().items():
        for name, _ in names:
            m = TIME_RE.match(name)
            if m:
                groups.setdefault((m.group(1), m.group(3)), []).append((m.group(2), name))
    key, seq = max(groups.items(), key=lambda kv: len(kv[1]))
    seq = sorted(seq, key=lambda x: time_hours(x[0]))[:frames]

    out = os.path.join(OUT_DIR, "timeframes")
    os.makedirs(out, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, f"masks/{SOURCE}/{name}.tif"): (i, t, name)
                for i, (t, name) in enumerate(seq)}
        for fu in as_completed(futs):
            i, t, name = futs[fu]
            msk = tifffile.imread(fu.result())
            iio.imwrite(os.path.join(out, f"frame_{i:03d}.png"), msk.astype(np.uint16))
    return key, len(seq), out


def main():
    ap = argparse.ArgumentParser(description="下载真实 LIVECell 数据集")
    ap.add_argument("--max_per_split", type=int, default=40,
                    help="每个划分下载多少个视野（0=只下时序帧，-1=全量 5239 视野）")
    ap.add_argument("--max_objects", type=int, default=0,
                    help="只采样细胞数不超过该值的视野（0=不限）。CPU 上建议设 400："
                         "高汇合视野单张就有 2000+ 实例，训练一轮要好几分钟")
    ap.add_argument("--ts_frames", type=int, default=8,
                    help="额外下载同一位点的时序 mask 帧数（0=不下，路线 B 用）")
    ap.add_argument("--workers", type=int, default=8, help="并发下载线程数")
    ap.add_argument("--list", action="store_true", help="只打印清单，不下载")
    args = ap.parse_args()

    splits = read_splits()
    total = sum(len(v) for v in splits.values())
    print(f"数据源: {REPO} / {SOURCE}")
    print(f"可用视野: {total}")
    for d in ("train", "val", "test"):
        names = splits.get(d, [])
        cells = sum(n for _, n in names)
        print(f"  {d}: {len(names)} 个视野, {cells} 个细胞")

    if args.list:
        return

    # 随机采样而不是挑细胞最多的视野，否则拿到的全是高汇合视野，分布不真实
    rng = np.random.default_rng(seed=42)
    jobs = []
    for d, names in splits.items():
        if args.max_objects > 0:
            names = [x for x in names if x[1] <= args.max_objects]
        if args.max_per_split == 0:
            names = []                       # 只要时序帧
        elif args.max_per_split > 0:
            idx = rng.choice(len(names), size=min(args.max_per_split, len(names)),
                             replace=False)
            names = [names[i] for i in sorted(idx)]
        jobs += [(n, d) for n, _ in names]

    print(f"\n准备下载 {len(jobs)} 个视野（并发 {args.workers}）")
    done = 0
    cells = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, n, d): (n, d) for n, d in jobs}
        for fu in as_completed(futs):
            try:
                cells += fu.result()
            except Exception as e:      # noqa: BLE001
                print(f"  失败 {futs[fu][0]}: {e}")
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  已下载 {done}/{len(jobs)}，累计 {cells} 个细胞")

    print(f"\n完成：{done} 个视野 / {cells} 个细胞")
    print(f"  图像 -> {os.path.join(OUT_DIR, 'images')}")
    print(f"  mask -> {os.path.join(OUT_DIR, 'masks')}")
    print("下一步：python convert_livecell.py  # mask -> YOLO-seg 标注")

    if args.ts_frames > 0:
        key, n, out = download_time_series(args.ts_frames, args.workers)
        print(f"\n时序序列：{key[0].split('__')[-1]} 位点 {key[1]}，{n} 帧（每 4 小时一帧）")
        print(f"  -> {out}（路线 B 的输入：python 时序动态/track_cells.py）")


if __name__ == "__main__":
    main()
