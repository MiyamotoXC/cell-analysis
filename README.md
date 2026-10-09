# 细胞分析（单细胞实例分割 + 三条下游分析路线）

> 基于 LIVECell 相差显微镜数据集（Nature Methods 2021，8 种细胞系，160 万+ 标注细胞）做
> **单细胞实例分割**（YOLOv8-seg），并在分割结果之上提供**三条下游分析路线**：
> 聚类表型发现（A）、时序增殖动态（B）、蛋白质荧光定量（C，见 `../蛋白质分析/`）。

## 快速开始（零下载，推荐先跑通这条）

仓库自带**合成数据生成器**，不需要下载任何数据集、不需要训练好的权重，即可把整条流水线端到端跑通：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU；GPU 用户按官网装 CUDA 版
pip install -r requirements.txt

# 0. 生成合成数据（相差显微风格图 + YOLO-seg 标签 + GT mask + 时序帧）
python make_sample_data.py

# 1. 分割底座：训练 + 推理导出实例 mask
python 3-yolo-cell.py --strategy baseline      # CPU 约 10~15 分钟（60 轮 @512）
python 4-yolo-cell-predict.py --name baseline  # 输出 cell_data/masks_pred/

# 2. 路线 A：形态表型聚类
python 聚类分析/cluster_features.py
python 聚类分析/5-cluster-cells.py

# 3. 路线 B：时序增殖
python 时序动态/track_cells.py
python 时序动态/6-proliferation.py
```

跑完的产物：

| 文件 | 内容 |
|---|---|
| `cell_data/masks_pred/*.png` | 逐细胞实例 mask（uint16，0=背景，n=第 n 个细胞）——下游统一输入 |
| `聚类分析/features.csv` | 逐细胞形态+强度特征 |
| `聚类分析/cluster_assignments.csv` / `cluster_summary.csv` / `umap_scatter.png` | 簇标签 / 每簇平均特征 / UMAP 散点 |
| `时序动态/tracks.csv` | 帧-细胞-轨迹 ID |
| `时序动态/proliferation_summary.csv` / `division_events.csv` / `proliferation_curve.png` | 每帧计数 / 分裂事件 / 增殖曲线 |

在本机（Windows + Python 3.12 + CPU，24 张合成训练图）实测结果：

- 训练：`mask mAP50 = 0.46`，`mAP50-95 = 0.31`
- 推理：8 张测试图检出 190 个细胞（GT 211）
- 路线 A：214 个细胞 × 12 维特征 → HDBSCAN 得到 2 个簇 + 17 个噪声点
- 路线 B：10 帧 → 199 条记录 / 58 条轨迹，细胞数 12 → 34（×2.83），检出 9 起分裂事件

## 使用真实数据（LIVECell）

合成数据只用于验证流程；真实效果请用 LIVECell 全量训练（GB 级，需外网）：

```bash
python download_livecell.py          # 下载并解压到 cell_data/（--list 可只看文件清单）
python convert_livecell.py           # COCO(RLE) -> YOLO-seg 多边形标注
python 3-yolo-cell.py --strategy enhanced    # GPU 上约 150 轮
python 4-yolo-cell-predict.py --name enhanced
```

## 增强训练策略

`3-yolo-cell.py` 针对不同硬件给出三档策略：

- **设备自适应**：CPU/GPU 自动检测，自动调 batch / workers / epochs / imgsz
  - CPU：`epochs=60, batch=4, imgsz=512`（短训 20 轮 mAP 只有 0.2、且默认阈值下检出为 0，60 轮才训得起来）
  - GPU：`epochs=100, batch=16, imgsz=640`
- **三档策略**：`baseline`（快速基线）/ `enhanced`（竞赛级增强+余弦退火）/ `large`（yolov8s-seg + imgsz=1280）
- **分割专属调参**：`overlap_mask=True`（拥挤细胞重叠掩码）、`mask_ratio=2`（小目标保掩码细节）
- **训练策略**：`cos_lr` 余弦退火 + warmup + AMP 混合精度；warmup 轮数与 `close_mosaic` 会随总轮数收敛
- **`--conf_search`**：conf 阈值网格搜索取 mask mAP50 最优
- **`--tta`**：TTA 推理提分（无需重训）
- **`--epochs` / `--imgsz`**：显式覆盖策略默认值

```bash
python 3-yolo-cell.py                          # 默认 baseline
python 3-yolo-cell.py --strategy enhanced --epochs 20
python 3-yolo-cell.py --strategy enhanced --conf_search --tta
```

## 目录结构

```
细胞分析/
├── livecell.yaml              # LIVECell 数据集配置（Ultralytics，8 类）
├── make_sample_data.py        # 合成数据生成器（零下载跑通全流程）
├── download_livecell.py       # 下载+解压 LIVECell（需外网）
├── convert_livecell.py        # COCO(RLE) → YOLO-seg 多边形转换
├── 3-yolo-cell.py             # YOLOv8-seg 训练
├── 4-yolo-cell-predict.py     # 推理 + 计数 + 实例 mask 导出
├── requirements.txt
│
├── 聚类分析/                  # 路线 A：形态表型聚类
│   ├── cluster_features.py    # 从 mask 提取逐细胞形态特征
│   └── 5-cluster-cells.py     # HDBSCAN/KMeans + UMAP 可视化
│
└── 时序动态/                  # 路线 B：时序增殖
    ├── track_cells.py         # IoU 跨帧细胞追踪
    └── 6-proliferation.py     # 增殖曲线 + 分裂事件检测

../蛋白质分析/                 # 路线 C：蛋白质定量/定位/共定位
```

## 数据集

| 数据集 | 体量 | 模态 | 说明 | 链接 |
|---|---|---|---|---|
| **LIVECell** | 3823 训练 + 1564 测试图像、8 种细胞系、160 万+ 标注细胞 | 相差显微镜 | Sartorius / Nature Methods 2021，label-free 活细胞分割基准，CC BY-NC 4.0 | https://sartorius-research.github.io/LIVECell/ |
| HASSL SingleCellBench | 1M–10M | 多模态 | 聚合 BCCD/CoNIC/DSB2018/LIVECell/MoNuSeg/PanNuke/TissueNet 等 | https://huggingface.co/datasets/tum-ai/HASSL-SingleCellBench |
| NeurIPS 2022 CellSeg | 1500+ 标注图 | 多模态 | 跨平台细胞分割挑战赛 | https://neurips22-cellseg.grand-challenge.org/ |

## 三条下游路线

| 路线 | 输入 | 输出 | 核心方法 |
|---|---|---|---|
| **A 形态聚类** | 逐细胞 mask（+原图） | features.csv → 簇标签 / UMAP 散点 / 簇概要 | area/perimeter/eccentricity/solidity/circularity → HDBSCAN |
| **B 时序增殖** | 时序 mask 帧 | tracks.csv → 增殖曲线 / 分裂事件 | 帧间 IoU 贪心匹配 → track ID → 增殖/分裂统计 |
| **C 蛋白质分析** | 多通道荧光 TIFF + mask | protein_features.csv | 核/质定位比、Pearson r、Manders M1 → `../蛋白质分析/` |

路线 A 若只想看预测结果（而非合成 GT），加 `--mask_dir ../cell_data/masks_pred`：

```bash
python 聚类分析/cluster_features.py --mask_dir ../cell_data/masks_pred
```

## 依赖

| 包 | 用途 |
|---|---|
| ultralytics, torch | 分割底座（YOLOv8-seg） |
| pycocotools, opencv-python, numpy, imageio | COCO 标注解析 / 图像 IO |
| scikit-image, numpy | mask 解析 / 形态特征提取 |
| scikit-learn, hdbscan, umap-learn, matplotlib, pandas | 路线 A 聚类与可视化 |
| matplotlib, pandas | 路线 B 曲线绘制 |

## 备注

- LIVECell 原始为 COCO 格式 RLE 掩码；`convert_livecell.py` 解码掩码→取最大轮廓→抽稀多边形→归一化，生成 YOLO-seg 标注。
- 所有图像落盘统一用 `imageio`：**`cv2.imwrite` 在中文路径下会返回 True 却不写文件**，本项目路径含中文，踩过这个坑。
- HDBSCAN 的 `min_cluster_size` 必须小于真实簇的细胞数，否则所有点都会被判成噪声（脚本已按样本量自动收敛，也可用 `--min_cluster_size` 手工指定）。
- 路线 B 的时序帧：LIVECell 原始为每 4h 一帧定时拍摄，逐帧分割后用 `track_cells.py` 追踪。
