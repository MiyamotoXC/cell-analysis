# 细胞分析（单细胞实例分割 + 三条下游分析路线）

> 基于 **LIVECell**（Nature Methods 2021，8 种细胞系，160 万+ 标注细胞）做
> **单细胞实例分割**（YOLOv8-seg），并在分割结果之上提供**三条下游分析路线**：
> 聚类表型发现（A）、时序增殖动态（B）、蛋白质荧光定量（C，见 `../蛋白质分析/`）。

## 数据：默认走真实公开数据集

仓库不再依赖任何"占位数据"，`download_livecell.py` 直接从 Hugging Face 拉真实 LIVECell。

| 项 | 说明 |
|---|---|
| 数据集 | `einarolafsson/live-cell-segmentation-dataset`，取其中的 `livecell_phase` 子集 |
| 原始出处 | LIVECell（Sartorius / Nature Methods 2021，CC BY-NC 4.0） |
| 规模 | **5239 个视野 / 164 万个标注细胞 / 8 个细胞系**（A172、BT474、BV2、Huh7、MCF7、SHSY5Y、SkBr3、SKOV3） |
| 每条数据 | 520×704 相差显微图（uint8）+ uint16 实例 mask（0=背景，1..N=细胞） |
| 划分 | 数据集自带 train/valid/test，按**采集批次**划分（同一孔/皿/时序不会跨集） |
| 时序 | 同一位点每 **4 小时**一帧，最多 19~25 帧，可直接跑路线 B |

```bash
python download_livecell.py --list            # 只看规模，不下载
python download_livecell.py                   # 默认每划分 40 个视野 + 8 帧时序
python download_livecell.py --max_per_split 200 --ts_frames 12
python convert_livecell.py                    # 实例 mask -> YOLO-seg 多边形标注
```

产物布局（与 `livecell.yaml` 一致）：

```
cell_data/images/{train,val,test}/<视野>.png
cell_data/labels/{train,val,test}/<视野>.txt   YOLO-seg 多边形，类别 = 视野名里的细胞系
cell_data/masks/{train,val,test}/<视野>.png    真实实例 mask（下游路线 A/B 的输入）
cell_data/timeframes/frame_000.png ...        同一位点的时序 mask 序列
```

## 快速开始（真实数据）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU；GPU 按官网装
pip install -r requirements.txt

python download_livecell.py                   # 1. 下载真实 LIVECell
python convert_livecell.py                    # 2. 生成 YOLO-seg 标注
python 3-yolo-cell.py --strategy baseline     # 3. 训练（推荐 GPU，CPU 见下方性能提示）
python 4-yolo-cell-predict.py --name baseline # 4. 推理 + 导出预测实例 mask

python 聚类分析/cluster_features.py           # 5. 路线 A：形态表型聚类
python 聚类分析/5-cluster-cells.py
python 时序动态/track_cells.py                # 6. 路线 B：时序增殖
python 时序动态/6-proliferation.py
```

## 合成演示数据（零下载）

没有网络或只想秒级验证流程时，用生成器造一份合成相差显微数据。它写到**独立目录
`cell_data_synth/`**，不会和真实数据 `cell_data/` 混进同一次训练：

```bash
python make_sample_data.py                                   # -> cell_data_synth/
python 3-yolo-cell.py --data livecell_synth.yaml --strategy baseline
python 4-yolo-cell-predict.py --name baseline --source cell_data_synth/images/test
python 聚类分析/cluster_features.py --mask_dir ../cell_data_synth/masks \
                                   --image_dir ../cell_data_synth/images/test
python 时序动态/track_cells.py --mask_dir ../cell_data_synth/timeframes
```

## 实测（本机 Windows + Python 3.12 + CPU）

| 环节 | 结果 |
|---|---|
| 下载真实数据 | 36 个视野 / **59,374 个真实细胞**（train 12 / val 12 / test 12） |
| 标注转换 | 59,374 个实例 -> YOLO-seg 多边形（8 类，类别从视野名解析） |
| 训练 | 真实数据单张视野上千个细胞、直径约 11 px，CPU 上每轮要几分钟；完整训练请用 GPU，逐轮指标见 `runs/segment/baseline/results.csv` |
| 路线 B（真实时序） | Huh7 位点 6 帧（每 4h）：458 条记录 / 295 条轨迹，细胞数 79 → 73，检出 5 起分裂 |
| 路线 C（真实荧光） | 见 `../蛋白质分析/`：3 个视野 1784 个细胞，核/质定位比 0.38~4.12 |

**CPU 上的性能提示**：LIVECell 的高汇合视野单张就有 2000+ 个细胞（中位面积约 100 px，
直径约 11 px），CPU 上训练一轮要好几分钟，且 512 尺度下这么小的目标学不到东西。
CPU 用户请这样下载（只取中等密度视野）并在 GPU/大尺度下追求指标：

```bash
python download_livecell.py --max_per_split 40 --max_objects 400   # 每视野 <=400 个细胞
python 3-yolo-cell.py --strategy baseline --epochs 60 --imgsz 1024 # 想要指标：大尺度 + 更多轮
```

要出好指标，请加大 `--max_per_split` 并在 GPU 上跑 `--strategy enhanced --imgsz 1024`。

## 三条下游路线

| 路线 | 输入 | 输出 | 核心方法 |
|---|---|---|---|
| **A 形态聚类** | 逐细胞 mask（+原图） | features.csv → 簇标签 / UMAP 散点 / 簇概要 | area/perimeter/eccentricity/solidity/circularity → HDBSCAN |
| **B 时序增殖** | 时序 mask 帧 | tracks.csv → 增殖曲线 / 分裂事件 | 帧间 IoU 贪心匹配 → track ID → 增殖/分裂统计 |
| **C 蛋白质分析** | 多通道荧光 TIFF + mask | protein_features.csv | 核/质定位比、Pearson r、Manders M1 → `../蛋白质分析/` |

路线 A 默认读真实 GT mask（`cell_data/masks/test`）；要看模型预测结果，加
`--mask_dir ../cell_data/masks_pred`。

## 增强训练策略

`3-yolo-cell.py` 针对硬件与数据给出三档策略：

- **设备自适应**：CPU/GPU 自动检测，自动调 batch / workers / epochs / imgsz
  - CPU：`epochs=60, batch=4, imgsz=512`（短训 20 轮置信度训不起来，60 轮才够）
  - GPU：`epochs=100, batch=16, imgsz=640`
- **三档策略**：`baseline`（快速基线）/ `enhanced`（竞赛级增强+余弦退火）/ `large`（yolov8s-seg + imgsz=1280）
- **分割专属调参**：`overlap_mask=True`（拥挤细胞重叠掩码）、`mask_ratio=2`（小目标保掩码细节）
- **训练策略**：`cos_lr` 余弦退火 + warmup + AMP；warmup 轮数与 `close_mosaic` 随总轮数收敛
- **`--conf_search`**：conf 阈值网格搜索取 mask mAP50 最优
- **`--tta`**：TTA 推理提分（无需重训）
- **`--data` / `--epochs` / `--imgsz`**：切换数据集、覆盖轮数与输入尺寸

```bash
python 3-yolo-cell.py                                   # 默认 enhanced + 真实数据
python 3-yolo-cell.py --strategy baseline --epochs 20
python 3-yolo-cell.py --strategy enhanced --conf_search --tta
python 3-yolo-cell.py --data livecell_synth.yaml        # 合成演示数据
```

## 目录结构

```
细胞分析/
├── livecell.yaml              # 真实数据集配置（Ultralytics，8 类）
├── livecell_synth.yaml        # 合成演示数据配置（cell_data_synth）
├── download_livecell.py       # 下载真实 LIVECell（Hugging Face）
├── convert_livecell.py        # 实例 mask -> YOLO-seg 多边形标注
├── make_sample_data.py        # 合成数据生成器 -> cell_data_synth/
├── 3-yolo-cell.py             # YOLOv8-seg 训练
├── 4-yolo-cell-predict.py     # 推理 + 计数 + 实例 mask 导出
├── requirements.txt
│
├── 聚类分析/                  # 路线 A：形态表型聚类
│   ├── cluster_features.py
│   └── 5-cluster-cells.py
│
└── 时序动态/                  # 路线 B：时序增殖
    ├── track_cells.py
    └── 6-proliferation.py

../蛋白质分析/                 # 路线 C：蛋白质定量/定位/共定位
```

## 其他公开数据集（可替换 LIVECell）

| 数据集 | 体量 | 模态 | 链接 |
|---|---|---|---|
| LIVECell | 160 万+ 标注细胞 | 相差显微 | https://sartorius-research.github.io/LIVECell/ |
| HASSL SingleCellBench | 1M–10M | 多模态聚合 | https://huggingface.co/datasets/tum-ai/HASSL-SingleCellBench |
| NeurIPS 2022 CellSeg | 1500+ 标注图 | 多模态 | https://neurips22-cellseg.grand-challenge.org/ |

## 依赖

| 包 | 用途 |
|---|---|
| ultralytics, torch | 分割底座（YOLOv8-seg） |
| huggingface_hub | 真实数据集下载 |
| opencv-python, numpy, imageio, tifffile | 轮廓抽取 / 图像 IO |
| scikit-image | mask 解析 / 形态特征提取 |
| scikit-learn, hdbscan, umap-learn, matplotlib, pandas | 路线 A 聚类与可视化 |
| pyyaml | 数据集配置（--data 切换真实/合成） |

## 备注

- **数据隔离**：真实数据 `cell_data/`，合成数据 `cell_data_synth/`，两者不会混训。
- LIVECell 标注是 uint16 实例 mask，`convert_livecell.py` 直接抽轮廓转多边形，
  不需要 COCO RLE 解析；类别由视野名里的细胞系决定。
- 所有图像落盘统一用 `imageio`：**`cv2.imwrite` 在中文路径下会返回 True 却不写文件**。
- HDBSCAN 的 `min_cluster_size` 必须小于真实簇的细胞数，否则所有点都会被判成噪声
  （脚本已按样本量自动收敛，也可用 `--min_cluster_size` 手工指定）。
- 路线 B 的时序帧直接用 LIVECell 自带的时间序列（每 4h 一帧），`--ts_frames` 控制帧数。
