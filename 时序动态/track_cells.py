"""
路线B - 跨帧细胞追踪（IoU 贪心匹配）
====================================
输入时序实例 mask 序列（frame_000.png, frame_001.png, ...，每帧 0=背景, n=第n个细胞），
用帧间 IoU 贪心匹配分配 track_id，输出 tracks.csv（frame, cell_label, track_id）。

LIVECell 原始为每 4h 一帧的定时拍摄，分割逐帧后即可用本脚本追踪。

用法：
  python track_cells.py --mask_dir ./timeframes --iou_thresh 0.3 --out tracks.csv
"""
import os
import glob
import csv
import argparse
import re
import numpy as np
from skimage import io, measure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(s))]


def compute_iou_matrix(mask_a, mask_b):
    """两帧实例 mask 的 IoU 矩阵 (n_a × n_b)。0 行列为背景，忽略。"""
    ids_a = [i for i in np.unique(mask_a) if i > 0]
    ids_b = [i for i in np.unique(mask_b) if i > 0]
    iou = np.zeros((len(ids_a), len(ids_b)))
    for i, ia in enumerate(ids_a):
        ma = mask_a == ia
        for j, ib in enumerate(ids_b):
            mb = mask_b == ib
            inter = np.logical_and(ma, mb).sum()
            union = np.logical_or(ma, mb).sum()
            if union > 0:
                iou[i, j] = inter / union
    return ids_a, ids_b, iou


def track(mask_files, iou_thresh=0.3):
    """贪心 IoU 追踪。返回 [(frame_idx, cell_label, track_id), ...]"""
    assignments = []
    next_track = 1
    prev_masks = None      # 上一帧 mask
    prev_map = {}          # prev cell_label -> track_id

    for fi, fp in enumerate(mask_files):
        mask = np.asarray(io.imread(fp))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int32)

        if prev_masks is None:
            for lb in np.unique(mask):
                if lb > 0:
                    assignments.append((fi, int(lb), next_track))
                    prev_map[int(lb)] = next_track
                    next_track += 1
        else:
            ids_prev, ids_cur, iou = compute_iou_matrix(prev_masks, mask)
            used_prev = set()
            used_cur = set()
            cur_map = {}
            # 按 IoU 从大到小贪心匹配
            pairs = [(iou[i, j], i, j) for i in range(len(ids_prev)) for j in range(len(ids_cur))]
            pairs.sort(reverse=True)
            for val, i, j in pairs:
                if val < iou_thresh:
                    break
                if ids_prev[i] in used_prev or ids_cur[j] in used_cur:
                    continue
                tid = prev_map.get(ids_prev[i])
                if tid is None:
                    tid = next_track
                    next_track += 1
                assignments.append((fi, int(ids_cur[j]), tid))
                cur_map[int(ids_cur[j])] = tid
                used_prev.add(ids_prev[i])
                used_cur.add(ids_cur[j])
            # 未匹配上的新细胞 -> 新 track（可能是分裂/移入）
            for lb in ids_cur:
                if lb not in used_cur:
                    assignments.append((fi, int(lb), next_track))
                    cur_map[int(lb)] = next_track
                    next_track += 1
            prev_map = cur_map

        prev_masks = mask
    return assignments


def main():
    ap = argparse.ArgumentParser(description="跨帧细胞追踪")
    ap.add_argument("--mask_dir", default=os.path.join(ROOT, "timeframes"),
                    help="时序 mask 目录（frame_000.png, frame_001.png, ...）")
    ap.add_argument("--iou_thresh", type=float, default=0.3)
    ap.add_argument("--out", default=os.path.join(HERE, "tracks.csv"))
    args = ap.parse_args()

    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")) +
                        glob.glob(os.path.join(args.mask_dir, "*.tif")),
                        key=natural_key)
    if not mask_files:
        print(f"未找到时序 mask 于 {args.mask_dir}")
        return

    assignments = track(mask_files, args.iou_thresh)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cell_label", "track_id"])
        w.writerows(assignments)

    n_tracks = len(set(t for _, _, t in assignments))
    print(f"追踪完成：{len(mask_files)} 帧，{len(assignments)} 条记录，{n_tracks} 条轨迹 -> {args.out}")


if __name__ == "__main__":
    main()
