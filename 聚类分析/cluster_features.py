"""
路线A - 细胞形态特征提取
========================
从实例分割 mask（PNG, 0=背景, n=第n个细胞）提取逐细胞形态学+强度特征，
输出 features.csv，供 5-cluster-cells.py 做聚类。

特征（每细胞一行）：
  形态：area, perimeter, eccentricity, solidity, circularity, orientation
  强度：mean_intensity, std_intensity（若提供原图）

用法：
  python cluster_features.py
  python cluster_features.py --mask_dir ../../cell_data/masks_pred --image_dir ../../cell_data/images/test
"""
import os
import glob
import csv
import argparse
import numpy as np
from skimage import io, measure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def safe_div(a, b):
    return a / b if b > 0 else 0.0


def extract_from_mask(mask, image=None):
    """从单张实例 mask 提取逐细胞特征。mask: 0=背景, n=第n个细胞。"""
    rows = []
    for rp in measure.regionprops(mask, intensity_image=image):
        area = rp.area
        perim = rp.perimeter
        circ = safe_div(4 * np.pi * area, perim ** 2)  # 圆度: 1=正圆
        row = {
            "label": rp.label,
            "area": area,
            "perimeter": perim,
            "eccentricity": rp.eccentricity,
            "solidity": rp.solidity,
            "circularity": circ,
            "orientation": rp.orientation,
            "bbox_w": rp.bbox[2] - rp.bbox[0],
            "bbox_h": rp.bbox[3] - rp.bbox[1],
        }
        if image is not None:
            row["mean_intensity"] = float(rp.intensity_mean)
            row["std_intensity"] = float(np.std(rp.image_intensity[rp.image]))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description="细胞形态特征提取")
    ap.add_argument("--mask_dir", default=os.path.join(ROOT, "cell_data", "masks", "test"),
                    help="实例 mask PNG 目录（真实 LIVECell 的 GT；"
                         "推理产物在 outputs/predict/masks，合成演示在 cell_data_synth/masks）")
    ap.add_argument("--image_dir", default=os.path.join(ROOT, "cell_data", "images", "test"),
                    help="对应原图目录（可选，用于强度特征）")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "cluster", "features.csv"),
                    help="输出 CSV 路径")
    args = ap.parse_args()

    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")) +
                        glob.glob(os.path.join(args.mask_dir, "*.tif")) +
                        glob.glob(os.path.join(args.mask_dir, "*.tiff")))
    if not mask_files:
        print(f"未找到 mask 文件于 {args.mask_dir}")
        print("请先运行 python 4-yolo-cell-predict.py 导出 mask，或 python make_sample_data.py 生成合成数据")
        return

    all_rows = []
    for mp in mask_files:
        mask = np.asarray(io.imread(mp))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int32)

        image = None
        if args.image_dir:
            base = os.path.splitext(os.path.basename(mp))[0]
            for ext in (".png", ".tif", ".tiff", ".jpg"):
                ip = os.path.join(args.image_dir, base + ext)
                if os.path.exists(ip):
                    image = np.asarray(io.imread(ip), dtype=np.float64)
                    if image.ndim == 3:
                        image = image.mean(axis=2)
                    break

        rows = extract_from_mask(mask, image)
        for r in rows:
            r["source"] = os.path.basename(mp)
        all_rows.extend(rows)

    if not all_rows:
        print("未提取到任何细胞特征。")
        return

    # 字段取所有行的并集，避免个别图缺通道时丢列
    fields = []
    for r in all_rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    # --out 只给文件名时 dirname 是空串，makedirs("") 会抛 WinError 3
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"已提取 {len(all_rows)} 个细胞 × {len(fields)} 维特征 -> {args.out}")


if __name__ == "__main__":
    main()
