#!/usr/bin/env python
# coding: utf-8
"""
单细胞图像处理：YOLOv8-seg 实例分割训练（增强版）
================================================
融合工作区内 train_yolo_enhanced.py（6 种策略）/ train_and_predict.py（工程封装）的训练技术，
针对细胞分割场景调优：

  1) 设备自适应：CPU/GPU 自动调 batch/workers/epochs 默认值
  2) 三档策略：baseline（快速基线）/ enhanced（竞赛级增强）/ large（大模型+大输入+余弦退火）
  3) 分割专属调参：overlap_mask=True（拥挤细胞重叠掩码）、mask_ratio=2（小细胞保掩码细节）、
     close_mosaic=10（末 10 轮关 Mosaic 适应真实分布）
  4) cos_lr 余弦退火 + warmup_epochs=3 + AMP 混合精度
  5) --conf_search：conf 阈值网格搜索取 mask mAP50 最优
  6) --tta：TTA 推理提分（无需重训）

数据集：LIVECell（8 种细胞系，相差显微镜）
前置：先运行 download_livecell.py + convert_livecell.py 生成 cell_data/

用法：
  python 3-yolo-cell.py                          # 默认 enhanced
  python 3-yolo-cell.py --strategy baseline      # 快速基线
  python 3-yolo-cell.py --strategy large         # yolov8s-seg + imgsz=1280 + cos_lr
  python 3-yolo-cell.py --strategy enhanced --conf_search --tta
"""
import argparse
import os

import imageio.v2 as iio
import torch
import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
# 真实 LIVECell 数据集；跑合成演示数据时用 --data livecell_synth.yaml
DEFAULT_DATA = os.path.join(HERE, "livecell.yaml")
# 所有产出统一收在 outputs/ 下，与代码（*.py）和数据（cell_data*）分开
OUTPUT_DIR = os.path.join(HERE, "outputs")


def resolve_data(data_arg):
    """解析数据集配置，返回 (yaml 绝对路径, 测试集图像目录)。"""
    path = data_arg if os.path.isabs(data_arg) else os.path.join(HERE, data_arg)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    test_dir = os.path.join(HERE, cfg["path"], "images", "test")
    return path, test_dir


def best_weights(strategy):
    """ultralytics 按 name=strategy 落盘，best.pt 在 outputs/train/<strategy>/weights/。"""
    return os.path.join(OUTPUT_DIR, "train", strategy, "weights", "best.pt")


# ---------- 设备自适应（借鉴 train_and_predict.py） ----------
def detect_device():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem = props.total_memory / 1024 ** 3
        print(f"[device] GPU: {props.name} ({gpu_mem:.1f} GB)")
        return "0", gpu_mem
    print("[device] 未检测到 GPU，使用 CPU（参数自动降档）")
    return "cpu", 0


# ---------- 三档训练策略 ----------
def build_train_kwargs(strategy, device, is_cpu, data, epochs=None, batch=None,
                       workers=None):
    """返回该策略的 model.train() 参数字典。"""
    if is_cpu:
        # CPU 短训至少要 60 轮：实测 20 轮 mAP50 仅 0.20、默认阈值下检出为 0，
        # 60 轮可到 mAP50≈0.46。imgsz 512 比 640 快约 1/3，且细胞占比更大更好学。
        base = dict(epochs=60, batch=4, workers=0, patience=10, imgsz=512)
    else:
        # workers 压到 2：真实数据单张图有几百个实例，每个 worker 都要把它们
        # 栅格化成 mask（mask_ratio=2 时是 512x512 int32），worker 一多内存直接爆
        base = dict(epochs=100, batch=8, workers=2, patience=50, imgsz=640)

    kwargs = dict(
        data=data,
        task="segment",
        device=device,
        save=True,
        pretrained=True,
        optimizer="auto",
        amp=True,                 # 混合精度
        verbose=True,
        project=os.path.join(OUTPUT_DIR, "train"),
        name=strategy,
        exist_ok=True,            # 防止 runs 目录爆炸
        # ---- 分割专属调参（细胞场景关键）----
        overlap_mask=True,        # 拥挤细胞掩码重叠时保留各自边界
        # mask_ratio 保持默认 4：设成 2 时每个实例的掩码张量会大 4 倍，
        # 一张图几百个细胞直接撑爆显存（imgsz 1024 下实测 693 个实例约 3.6 GB）
        mask_ratio=4,
    )
    kwargs.update({k: v for k, v in base.items()})
    if epochs:                    # --epochs 显式覆盖
        kwargs["epochs"] = epochs
    if batch:                     # --batch 显式覆盖（小显存跑大 imgsz 时需要调小）
        kwargs["batch"] = batch
    if workers:                   # --workers 显式覆盖
        kwargs["workers"] = workers

    if strategy == "baseline":
        # 快速基线：基础增强
        kwargs.update(
            epochs=kwargs["epochs"], batch=kwargs["batch"], patience=kwargs["patience"],
            scale=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.1, fliplr=0.5,
        )
    elif strategy == "enhanced":
        # 竞赛级增强（借鉴 train_yolo_enhanced.py 策略2）+ 余弦退火
        kwargs.update(
            epochs=max(kwargs["epochs"], 150), batch=kwargs["batch"],
            patience=max(kwargs["patience"], 80),
            mosaic=1.0,        # 4 图拼 1，增小目标检测
            mixup=0.15,        # 图混合，增泛化
            copy_paste=0.15,   # 实例复制粘贴（分割任务掩码同步生效）
            scale=0.5,         # 多尺度
            fliplr=0.5, flipud=0.5,       # 细胞各向同性，上下翻转也开
            degrees=15.0, translate=0.15,
            hsv_h=0.02, hsv_s=0.3, hsv_v=0.3,   # 显微光照波动小于自然图像，s/v 保守
            erasing=0.3,       # 随机遮挡，增鲁棒
            cos_lr=True,       # 余弦退火
        )
    elif strategy == "large":
        # 大模型 + 大输入（借鉴策略3）：imgsz=1280 保留细胞细节
        kwargs.update(
            epochs=max(kwargs["epochs"], 150), batch=8, patience=max(kwargs["patience"], 80),
            imgsz=1280,
            mosaic=1.0, mixup=0.15, copy_paste=0.15, scale=0.5,
            fliplr=0.5, flipud=0.5, degrees=15.0,
            cos_lr=True,
        )
    else:
        raise ValueError(f"未知策略: {strategy}")

    # 预热与关 Mosaic 的时机必须随总轮数收敛：固定 warmup=3 在 CPU 短训时会吃掉
    # 大半训练（学习率还没升起来就结束了），导致 mAP 恒为 0
    e = kwargs["epochs"]
    kwargs["warmup_epochs"] = min(3, max(1, e // 10))
    kwargs["close_mosaic"] = min(10, max(1, e // 10))
    return kwargs


# ---------- conf 阈值网格搜索（借鉴策略6，分割用 seg 指标） ----------
def conf_search(model, data, thresholds=None):
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    best_conf, best_map = None, -1.0
    print("\n==== conf 阈值网格搜索（mask mAP50）====")
    for conf in thresholds:
        # 必须显式指定 project/name：否则 ultralytics 会另开 runs/segment/val-N 目录
        r = model.val(data=data, conf=conf, verbose=False,
                      project=os.path.join(OUTPUT_DIR, "train"), name="conf_search",
                      exist_ok=True)
        map50 = r.seg.map50            # 分割任务用 seg 指标
        print(f"  conf={conf:<5} -> mask mAP50 = {map50:.4f}")
        if map50 > best_map:
            best_map, best_conf = map50, conf
    print(f"最优 conf={best_conf}（mask mAP50={best_map:.4f}）")
    return best_conf, best_map


def main():
    ap = argparse.ArgumentParser(description="YOLOv8-seg 单细胞分割训练（增强版）")
    ap.add_argument("--strategy", default="enhanced",
                    choices=["baseline", "enhanced", "large"])
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="数据集配置：真实 livecell.yaml / 合成 livecell_synth.yaml")
    ap.add_argument("--batch", type=int, default=0,
                    help="覆盖批量大小（0=按设备自动）；8 GB 显存跑 --imgsz 1024 时建议 4~8")
    ap.add_argument("--workers", type=int, default=0,
                    help="覆盖 dataloader 进程数（0=按设备自动）。大尺寸 + 上千实例时"
                         "每个 worker 都要把多边形栅格化成 mask，跟着调小才不会撑爆内存")
    ap.add_argument("--model", default=None,
                    help="模型权重，默认按策略选：large 用 yolov8s-seg.pt，其余 yolov8n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=0, help="覆盖默认输入尺寸（CPU 512 / GPU 640）")
    ap.add_argument("--epochs", type=int, default=0, help="覆盖策略默认轮数（0=用策略默认值）")
    ap.add_argument("--conf_search", action="store_true", help="训练后 conf 网格搜索")
    ap.add_argument("--tta", action="store_true", help="推理开启 TTA")
    ap.add_argument("--preview", type=int, default=5, help="训练后预览测试集前 N 张")
    args = ap.parse_args()

    device, gpu_mem = detect_device()
    if args.model is None:
        args.model = "yolov8s-seg.pt" if args.strategy == "large" else "yolov8n-seg.pt"

    model = YOLO(args.model)
    model.info()

    data_yaml, test_dir = resolve_data(args.data)
    kwargs = build_train_kwargs(args.strategy, device, device == "cpu", data_yaml,
                                args.epochs, args.batch, args.workers)
    if args.imgsz > 0:          # 显式指定才覆盖策略默认
        kwargs["imgsz"] = args.imgsz if args.strategy != "large" else max(args.imgsz, 1280)
    print(f"\n==== 训练开始（strategy={args.strategy}, model={args.model}, "
          f"imgsz={kwargs['imgsz']}, batch={kwargs['batch']}, epochs={kwargs['epochs']}）====")
    results = model.train(**kwargs)

    # 验证集评估（mask AP：metrics/mAP50-95(M)）
    print("\n==== 验证集评估 ====")
    metrics = model.val(project=os.path.join(OUTPUT_DIR, "train"), name="val",
                        exist_ok=True)
    print(f"mask mAP50-95(M) = {metrics.seg.map:.4f}  mAP50(M) = {metrics.seg.map50:.4f}")

    # conf 网格搜索（最优阈值直接用于下面的预览推理）
    conf = None
    if args.conf_search:
        conf, _ = conf_search(model, data_yaml)

    # 测试集预览
    wpath = best_weights(args.strategy)
    best = YOLO(wpath) if os.path.exists(wpath) else model
    if os.path.exists(test_dir) and args.preview > 0:
        pred_dir = os.path.join(OUTPUT_DIR, "train", "preview")
        os.makedirs(pred_dir, exist_ok=True)
        for name in sorted(os.listdir(test_dir))[: args.preview]:
            img_path = os.path.join(test_dir, name)
            res = best(img_path, augment=args.tta, conf=conf if conf else 0.25)  # --tta 开启 TTA
            # 用 imageio 落盘：cv2/ultralytics 的 save 在中文路径下会静默失败
            vis = res[0].plot()
            iio.imwrite(os.path.join(pred_dir, name), vis[..., ::-1])
        print(f"\n预览 {args.preview} 张已保存 -> {pred_dir}"
              + ("（TTA 已开启）" if args.tta else ""))


if __name__ == "__main__":
    main()
