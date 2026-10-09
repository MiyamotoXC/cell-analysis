#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单细胞图像处理：YOLOv8-seg 推理、可视化、细胞计数与实例 mask 导出
==============================================================
推理结果除了可视化图与计数，还会导出**逐细胞实例 mask**（uint16 PNG，0=背景，
n=第 n 个细胞）——这是下游路线 A（聚类）、路线 B（时序追踪）、路线 C（蛋白质定量）
的统一输入，缺了它下游全部跑不起来。

用法：
  python 4-yolo-cell-predict.py                      # 默认读 runs/segment/<name>/weights/best.pt
  python 4-yolo-cell-predict.py --weights runs/segment/baseline/weights/best.pt
  python 4-yolo-cell-predict.py --source cell_data/images/test --conf 0.25 --tta
  python 4-yolo-cell-predict.py --limit 5            # 只跑前 5 张（快速预览）
"""
import argparse
import os

import imageio.v2 as iio
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["cell"]      # 单类，与 livecell.yaml 的 names 一致
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def list_images(d):
    return sorted(p for p in os.listdir(d) if p.lower().endswith(IMG_EXTS))


def to_original_size(masks_np, hw):
    """把掩码还原到原图尺寸（最近邻）。

    ultralytics 的 `results.masks.data` 是**网络输入尺寸**（如 576x768），不是原图
    尺寸；不还原的话导出的实例 mask 与图像对不上，下游 regionprops 会直接报
    "Label and intensity image shapes must match"。
    """
    h, w = hw
    if masks_np.shape[1:] == (h, w):
        return masks_np
    ys = (np.arange(h) * masks_np.shape[1] / h).astype(np.int32)
    xs = (np.arange(w) * masks_np.shape[2] / w).astype(np.int32)
    return masks_np[:, ys][:, :, xs]


def save_instance_mask(masks_np, shape, out_path):
    """把 n 个二值掩码叠成实例 mask（uint16）。重叠处以后出现的实例为准。"""
    inst = np.zeros(shape, dtype=np.uint16)
    for i, m in enumerate(masks_np, start=1):
        inst[m > 0.5] = i
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # imageio 而非 cv2：cv2.imwrite 在中文路径下会静默失败（返回 True 但不落盘）
    iio.imwrite(out_path, inst)
    return int(inst.max())


def main():
    ap = argparse.ArgumentParser(description="YOLOv8-seg 细胞推理 + 计数 + mask 导出")
    ap.add_argument("--weights", default=None,
                    help="模型权重，默认 runs/segment/<name>/weights/best.pt")
    ap.add_argument("--name", default="baseline",
                    help="配合默认权重路径使用的训练任务名（baseline/enhanced/large）")
    ap.add_argument("--source", default=os.path.join(HERE, "cell_data", "images", "test"),
                    help="待推理图像目录")
    ap.add_argument("--outdir", default=os.path.join(HERE, "runs", "segment", "predict"),
                    help="可视化结果输出目录")
    ap.add_argument("--mask_dir", default=os.path.join(HERE, "cell_data", "masks_pred"),
                    help="实例 mask 导出目录（下游路线 A/B/C 的输入）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tta", action="store_true", help="开启 TTA 推理")
    ap.add_argument("--limit", type=int, default=0, help="只推理前 N 张（0=全部）")
    args = ap.parse_args()

    weights = args.weights or os.path.join(
        HERE, "runs", "segment", args.name, "weights", "best.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"未找到权重 {weights}。请先训练（python 3-yolo-cell.py --strategy {args.name}）"
            f"或用 --weights 指定。")

    if not os.path.exists(args.source):
        raise FileNotFoundError(
            f"未找到图像目录 {args.source}。"
            f"请先运行 python make_sample_data.py 或 download_livecell.py + convert_livecell.py")

    model = YOLO(weights)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    files = list_images(args.source)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"目录内没有图像：{args.source}")
        return

    counts = {n: 0 for n in CLASS_NAMES}
    total = 0
    for name in files:
        r = model(os.path.join(args.source, name), conf=args.conf, augment=args.tta)[0]
        vis = r.plot()                                   # BGR ndarray
        iio.imwrite(os.path.join(args.outdir, name), vis[..., ::-1])

        n_inst = 0
        if r.masks is not None and r.boxes is not None and len(r.boxes.cls) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            masks_np = r.masks.data.cpu().numpy()        # (n, H, W)，H/W 是网络输入尺寸
            orig_hw = iio.imread(os.path.join(args.source, name)).shape[:2]
            masks_np = to_original_size(masks_np, orig_hw)
            for c in cls:
                if 0 <= c < len(CLASS_NAMES):
                    counts[CLASS_NAMES[c]] += 1
                total += 1
            n_inst = save_instance_mask(
                masks_np, orig_hw,
                os.path.join(args.mask_dir, os.path.splitext(name)[0] + ".png"))
        print(f"  {name}: {n_inst} 个细胞")

    print("\n==== 细胞计数 ====")
    for n in CLASS_NAMES:
        print(f"  {n}: {counts[n]}")
    print(f"  合计: {total}")
    print(f"  可视化 -> {args.outdir}")
    print(f"  实例 mask -> {args.mask_dir}（下游路线 A/B/C 的输入）")


if __name__ == "__main__":
    main()
