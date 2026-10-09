"""
路线B - 增殖曲线与分裂事件检测
==============================
读入 track_cells.py 输出的 tracks.csv：
  - 每帧细胞总数 → 增殖曲线 proliferation_curve.png
  - 检测分裂事件：同一 track 在某帧消失、且该帧新增 ≥1 条新 track（近似分裂）
  - 输出 proliferation_summary.csv（每帧计数）+ division_events.csv

用法：
  python 6-proliferation.py --tracks tracks.csv --interval_h 4 --outdir .
"""
import os
import csv
import argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # 细胞分析/
OUT_DIR = os.path.join(ROOT, "outputs", "tracking")           # 产出统一收在 outputs/


def main():
    ap = argparse.ArgumentParser(description="增殖分析")
    ap.add_argument("--tracks", default=os.path.join(OUT_DIR, "tracks.csv"))
    ap.add_argument("--interval_h", type=float, default=4.0, help="帧间隔（小时）")
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.tracks):
        print(f"未找到 {args.tracks}，请先运行 track_cells.py 生成追踪结果")
        return
    os.makedirs(args.outdir, exist_ok=True)

    # 读 tracks
    frame_tracks = defaultdict(set)   # frame -> set(track_id)
    track_frames = defaultdict(set)   # track -> set(frame)
    with open(args.tracks, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fr = int(r["frame"])
            tid = int(r["track_id"])
            frame_tracks[fr].add(tid)
            track_frames[tid].add(fr)

    frames = sorted(frame_tracks.keys())
    if not frames:
        print("tracks.csv 为空")
        return

    counts = [len(frame_tracks[fr]) for fr in frames]
    hours = [fr * args.interval_h for fr in frames]

    # 增殖曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hours, counts, "o-", color="#2a7fb8")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cell count")
    ax.set_title("Proliferation curve")
    ax.grid(alpha=0.3)
    out_png = os.path.join(args.outdir, "proliferation_curve.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 每帧计数
    with open(os.path.join(args.outdir, "proliferation_summary.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "hours", "cell_count"])
        for fr, h, c in zip(frames, hours, counts):
            w.writerow([fr, h, c])

    # 分裂事件：track 在帧 t 消失，且帧 t 有新 track 出现
    all_tids = set(track_frames.keys())
    first_frame = {tid: min(fs) for tid, fs in track_frames.items()}
    last_frame = {tid: max(fs) for tid, fs in track_frames.items()}
    new_tracks = [tid for tid, fs in track_frames.items() if min(fs) > min(frames)]

    divisions = []
    for fr in frames[1:]:
        died = [t for t in all_tids if last_frame.get(t) == fr - 1 and t not in frame_tracks[fr]]
        born = [t for t in new_tracks if first_frame[t] == fr]
        if died and born:
            divisions.append({"frame": fr, "hours": fr * args.interval_h,
                              "disappeared_tracks": ";".join(map(str, died)),
                              "new_tracks": ";".join(map(str, born))})

    with open(os.path.join(args.outdir, "division_events.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "hours", "disappeared_tracks", "new_tracks"])
        w.writeheader()
        w.writerows(divisions)

    total_growth = counts[-1] / counts[0] if counts[0] > 0 else 0
    print(f"增殖分析完成：{len(frames)} 帧，初始 {counts[0]} → 末端 {counts[-1]}（×{total_growth:.2f}）")
    print(f"  增殖曲线 -> {out_png}")
    print(f"  每帧计数 -> {os.path.join(args.outdir, 'proliferation_summary.csv')}")
    print(f"  分裂事件 -> {os.path.join(args.outdir, 'division_events.csv')}（{len(divisions)} 起）")


if __name__ == "__main__":
    main()
