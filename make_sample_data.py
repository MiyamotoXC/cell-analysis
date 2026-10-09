# -*- coding: utf-8 -*-
"""
细胞分析 -- 合成样本数据生成器
=============================
在没有任何外部数据集的情况下，一键生成整条流水线所需的全部数据，使
「训练 -> 推理导出 mask -> 路线 A 聚类 -> 路线 B 时序追踪」可以端到端跑通。

产出（默认写在 细胞分析/cell_data_synth/ 下，与真实数据目录 cell_data/ 完全隔离，
因此合成数据和下载来的 LIVECell 不会混进同一次训练）：
  cell_data_synth/images/{train,val,test}/*.png   合成相差显微图（3 通道 PNG）
  cell_data_synth/labels/{train,val,test}/*.txt   YOLO-seg 多边形标注（坐标归一化）
  cell_data_synth/masks/*.png                     测试图对应的 GT 实例 mask（uint16，0=背景）
  cell_data_synth/timeframes/frame_000.png ...    时序实例 mask（模拟漂移/生长/分裂）

合成图刻意模仿相差显微镜的观感：细胞体偏暗、边缘一圈亮 halo、整体有噪声与轻微模糊，
因此形态特征（面积/圆度/偏心率/实心度）分布合理，聚类与追踪能拿到有意义的结果。

真实数据路径见 download_livecell.py + convert_livecell.py（LIVECell，真实显微数据集）。

用法：
  python make_sample_data.py                       # 默认小规模，秒级生成
  python make_sample_data.py --n_train 60 --n_test 20 --frames 12
"""
import argparse
import os

import cv2
import imageio.v2 as iio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 合成数据单独放一个目录：真实 LIVECell 下载到 cell_data/，两者互不污染
SYNTH_DIR = os.path.join(HERE, "cell_data_synth")
CELL_DATA = SYNTH_DIR
TIME_DIR = os.path.join(SYNTH_DIR, "timeframes")

# 单类 cell，与 livecell.yaml / convert_livecell.py 的 names 保持一致
CLASS_NAMES = ["cell"]
# 形态模式池（半轴：长轴, 短轴）。虽然全标同一类，但形态跨度要留住——
# 路线 A 的聚类就是靠这些形态差异分群，形态太单一会让聚类退化成一个大簇。
SHAPE_MODES = [
    (20, 14),   # 中等椭圆
    (18, 17),   # 近圆、偏大
    (11, 9),    # 小、近圆
    (25, 18),   # 大椭圆
    (19, 15),   # 中等近圆
    (24, 8),    # 细长（神经元样）
    (16, 13),   # 中等偏小
    (23, 16),   # 大椭圆
]


def ellipse_mask(shape, center, axes, angle):
    """画一个实心椭圆，返回布尔掩码。"""
    m = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(m, center, axes, angle, 0, 360, 1, -1)
    return m.astype(bool)


def sample_cells(shape, n_cells, rng):
    """拒绝采样放置细胞，避免大面积重叠。返回 cell 列表。"""
    h, w = shape
    placed = np.zeros(shape, dtype=bool)
    cells = []
    tries = 0
    max_tries = n_cells * 80
    while len(cells) < n_cells and tries < max_tries:
        tries += 1
        base_a, base_b = SHAPE_MODES[int(rng.integers(len(SHAPE_MODES)))]
        a = max(4, int(round(rng.normal(base_a, base_a * 0.10))))
        b = max(3, int(round(rng.normal(base_b, base_b * 0.10))))
        if b > a:
            a, b = b, a
        axes = (a, b)
        angle = float(rng.integers(0, 180))
        margin = a + 3
        if margin >= min(h, w) // 2:
            continue
        cx = int(rng.integers(margin, w - margin))
        cy = int(rng.integers(margin, h - margin))
        m = ellipse_mask(shape, (cx, cy), axes, angle)
        if (m & placed).sum() > 0.2 * m.sum():     # 与已有细胞重叠过多则重采
            continue
        placed |= m
        cells.append({"cls": 0, "center": (cx, cy), "axes": axes, "angle": angle})
    return cells


def render_phase_contrast(shape, cells, rng):
    """按相差显微镜观感渲染：细胞体偏暗、外圈亮 halo、背景噪声 + 轻微模糊。"""
    h, w = shape
    img = np.full(shape, 118.0, dtype=np.float32)
    img += rng.normal(0.0, 4.0, shape)
    for c in cells:
        (cx, cy) = c["center"]
        (a, b) = c["axes"]
        body = ellipse_mask(shape, (cx, cy), (a, b), c["angle"])
        halo = ellipse_mask(shape, (cx, cy), (a + 3, b + 3), c["angle"]) & ~body
        # 对比度要足够：细胞体明显偏暗、边缘 halo 明显偏亮，否则检测器学不到目标
        img[body] -= 60.0
        img[halo] += 50.0
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img += rng.normal(0.0, 2.0, shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([img, img, img])


def cells_to_instance_mask(shape, cells):
    """把细胞列表渲染成实例 mask（uint16，0=背景，n=第 n 个细胞）。"""
    mask = np.zeros(shape, dtype=np.uint16)
    for i, c in enumerate(cells, start=1):
        m = ellipse_mask(shape, c["center"], c["axes"], c["angle"])
        mask[m] = i
    return mask


def mask_to_yolo_polygon(binary, img_w, img_h):
    """二值掩码 -> YOLO-seg 归一化多边形（扁平坐标字符串列表）。"""
    src = binary.astype(np.uint8)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    length = cv2.arcLength(c, True)
    poly = None
    for ratio in (0.005, 0.002, 0.0005, 0.0):
        eps = ratio * length
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            poly = approx
            break
    if poly is None:
        return None
    out = []
    for x, y in poly:
        out.append(f"{min(max(x / img_w, 0.0), 1.0):.6f}")
        out.append(f"{min(max(y / img_h, 0.0), 1.0):.6f}")
    return out


def gen_split(split, n_img, shape, rng, n_cells_range, save_mask):
    """生成一个 split 的图像 + YOLO-seg 标签；test split 额外保存 GT 实例 mask。"""
    h, w = shape
    img_dir = os.path.join(CELL_DATA, "images", split)
    lbl_dir = os.path.join(CELL_DATA, "labels", split)
    msk_dir = os.path.join(CELL_DATA, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if save_mask:
        os.makedirs(msk_dir, exist_ok=True)

    n_inst = 0
    for i in range(n_img):
        n_cells = int(rng.integers(n_cells_range[0], n_cells_range[1] + 1))
        cells = sample_cells(shape, n_cells, rng)
        image = render_phase_contrast(shape, cells, rng)
        name = f"{split}_{i:04d}.png"
        # 用 imageio 而非 cv2：cv2.imwrite 在中文路径下会静默失败
        iio.imwrite(os.path.join(img_dir, name), image)

        lines = []
        for c in cells:
            m = ellipse_mask(shape, c["center"], c["axes"], c["angle"])
            poly = mask_to_yolo_polygon(m, w, h)
            if poly is None:
                continue
            lines.append(f"{c['cls']} " + " ".join(poly))
            n_inst += 1
        base = os.path.splitext(name)[0]
        with open(os.path.join(lbl_dir, base + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        if save_mask:
            mask = cells_to_instance_mask(shape, cells)
            iio.imwrite(os.path.join(msk_dir, base + ".png"), mask)

    print(f"[{split}] 图像 {n_img} 张，实例 {n_inst} 个"
          + ("，mask 已写入 cell_data_synth/masks/" if save_mask else ""))
    return n_img, n_inst


def gen_timeframes(n_frames, shape, rng, n_init, division_rate):
    """生成时序实例 mask：细胞逐帧漂移/生长，并按概率分裂（模拟增殖）。"""
    h, w = shape
    os.makedirs(TIME_DIR, exist_ok=True)
    cells = sample_cells(shape, n_init, rng)

    for fi in range(n_frames):
        mask = cells_to_instance_mask(shape, cells)
        iio.imwrite(os.path.join(TIME_DIR, f"frame_{fi:03d}.png"), mask)

        nxt = []
        for c in cells:
            cx, cy = c["center"]
            a, b = c["axes"]
            # 漂移 + 生长
            cx = int(np.clip(cx + rng.integers(-2, 3), a + 3, w - a - 3))
            cy = int(np.clip(cy + rng.integers(-2, 3), a + 3, h - a - 3))
            a = min(a + 1, 26)
            b = min(b + 1, 22)
            new_cell = {"cls": 0, "center": (cx, cy), "axes": (a, b), "angle": c["angle"]}
            if rng.random() < division_rate:
                # 分裂：母细胞消失，原地留下两个子细胞
                ra, rb = max(3, a // 2), max(3, b // 2)
                for sign in (-1, 1):
                    sx = int(np.clip(cx + sign * (ra + 1), ra + 3, w - ra - 3))
                    sy = int(np.clip(cy + sign * (rb + 1), rb + 3, h - rb - 3))
                    nxt.append({"cls": 0, "center": (sx, sy), "axes": (ra, rb),
                                "angle": c["angle"]})
            else:
                nxt.append(new_cell)
        cells = nxt

    print(f"[timeframes] {n_frames} 帧，末帧 {len(cells)} 个细胞 -> {TIME_DIR}")


def main():
    ap = argparse.ArgumentParser(description="生成合成细胞数据（零下载跑通整条流水线）")
    ap.add_argument("--imgsz", type=int, default=512, help="图像边长（正方形）")
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=8)
    ap.add_argument("--min_cells", type=int, default=18, help="每图最少细胞数")
    ap.add_argument("--max_cells", type=int, default=36, help="每图最多细胞数")
    ap.add_argument("--frames", type=int, default=10, help="时序帧数（路线 B）")
    ap.add_argument("--n_init", type=int, default=12, help="时序初始细胞数")
    ap.add_argument("--division_rate", type=float, default=0.12, help="每帧每细胞分裂概率")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    shape = (args.imgsz, args.imgsz)
    cell_range = (args.min_cells, args.max_cells)

    print(f"生成合成数据 imgsz={args.imgsz}, 类别=cell（单类）")
    gen_split("train", args.n_train, shape, rng, cell_range, save_mask=False)
    gen_split("val", args.n_val, shape, rng, cell_range, save_mask=False)
    gen_split("test", args.n_test, shape, rng, cell_range, save_mask=True)
    gen_timeframes(args.frames, shape, rng, args.n_init, args.division_rate)
    print("\n完成。下一步：python 3-yolo-cell.py --data livecell_synth.yaml --strategy baseline")


if __name__ == "__main__":
    main()
