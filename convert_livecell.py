"""
单细胞图像处理：LIVECell COCO(RLE 掩码) -> YOLO-seg 多边形标注转换
==================================================================
数据集：Sartorius LIVECell (Nature Methods 2021)，相差显微镜，8 种细胞系。
LIVECell 官方标注为 COCO 格式，segmentation 为 RLE（压缩掩码）。
YOLO-seg 要求每行： cls x1 y1 x2 y2 ... xn yn （坐标全部归一化到 [0,1]）。

流程：
  1) 运行 download_livecell.py 下载并解压到 cell_data/（得到 LIVECell_dataset_2021/）
  2) python convert_livecell.py
输出：
  cell_data/images/{train,val,test}/  复制后的图像
  cell_data/labels/{train,val,test}/  每个图像一个 .txt，每行一个实例多边形

依赖：pycocotools, opencv-python, numpy
"""
import os
import json
import shutil
import cv2
import numpy as np
from pycocotools import mask as maskUtils

# ---- 配置 ----
HERE = os.path.dirname(os.path.abspath(__file__))
CELL_DATA = os.path.join(HERE, "cell_data")
RAW = os.path.join(CELL_DATA, "LIVECell_dataset_2021")   # download_livecell.py 解压产物
ANNOT_DIR = os.path.join(RAW, "annotations")             # 含 annotations_train/val/test.json
IMAGE_DIR = os.path.join(RAW, "images")                  # COCO image.file_name 相对此目录

# 类别顺序必须与 livecell.yaml 的 names 完全一致
CLASS_NAMES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]
NAME2IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

SPLITS = ["train", "val", "test"]
MIN_AREA_RATIO = 0.0001   # 实例面积 < 图像面积 0.01% 时丢弃，过滤噪声/碎掩码


def decode_mask(seg, h, w):
    """seg 可能是 RLE(dict) 或 polygon(list)。返回二值 uint8 掩码 (H, W)。"""
    if isinstance(seg, dict):
        m = maskUtils.decode(seg)          # HxWx1 uint8
        return m[:, :, 0]
    # polygon: list[list[float]] -> 转 RLE 再解码
    rles = maskUtils.frPyObjects(seg, h, w)
    m = maskUtils.decode(rles)
    if m.ndim == 3:
        m = np.sum(m, axis=2)
    return (m > 0).astype(np.uint8)


def mask_to_polygon(binary, img_w, img_h):
    """从二值掩码取最大连通域外轮廓，抽稀为多边形并归一化。返回扁平坐标列表或 None。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)            # 取面积最大轮廓
    if cv2.contourArea(c) < MIN_AREA_RATIO * img_w * img_h:
        return None
    epsilon = 0.005 * cv2.arcLength(c, True)          # 抽稀：0.5% 周长
    poly = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    norm = []
    for x, y in poly:
        x = min(max(x / img_w, 0.0), 1.0)
        y = min(max(y / img_h, 0.0), 1.0)
        norm.extend([f"{x:.6f}", f"{y:.6f}"])
    return norm


def convert_split(split):
    ann_path = os.path.join(ANNOT_DIR, f"annotations_{split}.json")
    if not os.path.exists(ann_path):
        print(f"[skip] {ann_path} 不存在，请确认 download_livecell.py 已正确解压")
        return 0, 0
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    out_img_dir = os.path.join(CELL_DATA, "images", split)
    out_lbl_dir = os.path.join(CELL_DATA, "labels", split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # 按图像聚合实例，避免重复写文件
    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    n_img = 0
    n_inst = 0
    for im_id, anns in img_anns.items():
        im = images.get(im_id)
        if im is None:
            continue
        fname = im["file_name"]
        img_w, img_h = im["width"], im["height"]

        src = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(src):
            print(f"[warn] 图像缺失 {src}")
            continue
        dst = os.path.join(out_img_dir, os.path.basename(fname))
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        n_img += 1

        lines = []
        for ann in anns:
            name = cat_name.get(ann["category_id"])
            if name is None or name not in NAME2IDX:
                continue
            cls = NAME2IDX[name]
            seg = ann.get("segmentation")
            if not seg:
                continue
            try:
                binary = decode_mask(seg, img_h, img_w)
            except Exception as e:
                print(f"[warn] 解码失败 {fname}: {e}")
                continue
            poly = mask_to_polygon(binary, img_w, img_h)
            if poly is None:
                continue
            lines.append(str(cls) + " " + " ".join(poly))
            n_inst += 1

        base = os.path.splitext(os.path.basename(fname))[0]
        with open(os.path.join(out_lbl_dir, base + ".txt"), "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + "\n")

    print(f"[{split}] 图像 {n_img} 张，实例 {n_inst} 个")
    return n_img, n_inst


def main():
    for s in SPLITS:
        convert_split(s)
    print("\nLIVECell -> YOLO-seg 转换完成。标签位于 cell_data/labels/")


if __name__ == "__main__":
    main()
