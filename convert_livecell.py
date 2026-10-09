#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实数据标注转换：实例 mask -> YOLO-seg 多边形标注
=================================================
输入 download_livecell.py 下载的 LIVECell 真实数据：

    cell_data/masks/{train,val,test}/<name>.png   uint16 实例 mask（0=背景，1..N=细胞）

输出 Ultralytics YOLOv8-seg 需要的标注：

    cell_data/labels/{train,val,test}/<name>.txt
    每行 = <class_id> <x1> <y1> <x2> <y2> ...   坐标按图像宽高归一化到 0~1

所有实例统一标为 cell 一类（id=0），与 LIVECell 官方任务一致。

用法：
  python convert_livecell.py                      # 转换全部划分
  python convert_livecell.py --min_area 20        # 丢弃面积过小的实例
  python convert_livecell.py --splits train val
"""
import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np
from skimage.measure import find_contours, regionprops

HERE = os.path.dirname(os.path.abspath(__file__))
MASK_ROOT = os.path.join(HERE, "cell_data", "masks")
LABEL_ROOT = os.path.join(HERE, "cell_data", "labels")

# 单类：LIVECell 的官方任务就是"细胞"这一类，160 万个细胞同标 1 类。
# 不要按细胞系分成 8 类——数据集自带的 train/valid/test 是按采集批次划分的，
# 各 split 的细胞系几乎不重叠（train 是 A172/BT474/BV2/SkBr3/SKOV3，
# valid/test 是 Huh7/MCF7/SHSY5Y），分 8 类训练时验证集的类别从没在训练集里
# 出现过，mAP 会恒等于 0。
CLASS_ID = 0

MAX_POINTS = 32          # 单个实例最多保留的多边形顶点数
MIN_POINTS = 3           # 少于 3 个点的多边形无法构成实例


def simplify(contour, max_points=MAX_POINTS):
    """轮廓抽稀：Douglas-Peucker 保形，若顶点仍过多再等距采样兜底。"""
    pts = contour[:, ::-1].astype(np.float32)          # (row, col) -> (x, y)
    approx = cv2.approxPolyDP(
        pts, epsilon=0.01 * cv2.arcLength(pts, closed=True), closed=True)
    approx = approx.reshape(-1, 2)
    if len(approx) > max_points:
        idx = np.linspace(0, len(approx) - 1, max_points).astype(int)
        approx = approx[idx]
    return approx


def mask_to_polygons(mask, min_area):
    """uint16 实例 mask -> [(label, [(x, y), ...]), ...]，坐标已归一化。"""
    h, w = mask.shape
    out = []
    for prop in regionprops(mask):
        if prop.area < min_area:
            continue
        # 只在实例的 bbox 内找轮廓，避免对整幅大图反复扫描
        sub = (mask[prop.slice] == prop.label)
        contours = find_contours(sub, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len) + np.array(
            [prop.slice[0].start, prop.slice[1].start])
        poly = simplify(contour)
        if len(poly) < MIN_POINTS:
            continue
        norm = np.empty(poly.size, dtype=np.float64)
        norm[0::2] = np.clip(poly[:, 0] / w, 0.0, 1.0)
        norm[1::2] = np.clip(poly[:, 1] / h, 0.0, 1.0)
        out.append((prop.label, norm))
    return out


def convert_split(split, min_area):
    msk_dir = os.path.join(MASK_ROOT, split)
    if not os.path.isdir(msk_dir):
        print(f"跳过 {split}：{msk_dir} 不存在（先运行 download_livecell.py）")
        return 0, 0

    out_dir = os.path.join(LABEL_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(msk_dir) if f.lower().endswith(".png"))
    total_inst = 0
    for f in files:
        name = os.path.splitext(f)[0]
        mask = iio.imread(os.path.join(msk_dir, f))
        polygons = mask_to_polygons(mask, min_area)
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as fp:
            for _, pts in polygons:
                fp.write(str(CLASS_ID) + " " + " ".join(f"{v:.6f}" for v in pts) + "\n")
        total_inst += len(polygons)
    print(f"  {split}: {len(files)} 个视野 -> {total_inst} 个实例标注")
    return len(files), total_inst


def main():
    ap = argparse.ArgumentParser(description="实例 mask -> YOLO-seg 标注")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--min_area", type=int, default=20,
                    help="丢弃面积小于该值的实例（像素）")
    args = ap.parse_args()

    print("类别: cell（单类，与 LIVECell 官方任务一致）")
    print(f"mask 目录: {MASK_ROOT}")
    files = inst = 0
    for s in args.splits:
        f, i = convert_split(s, args.min_area)
        files += f
        inst += i
    print(f"\n合计 {files} 个视野 / {inst} 个实例")
    print(f"标注 -> {LABEL_ROOT}")
    print("下一步：python 3-yolo-cell.py --strategy baseline")


if __name__ == "__main__":
    main()
