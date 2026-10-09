"""
路线A - 细胞表型聚类 + UMAP 可视化
==================================
读入 cluster_features.py 输出的 features.csv，做无监督聚类：
  - HDBSCAN（自动发现簇数，容忍噪声）或 KMeans（指定 k）
  - 标准化 → UMAP/PCA 降维 → 2D 散点图（按簇上色）
  - 输出：cluster_assignments.csv（每细胞簇标签）+ cluster_summary.csv（每簇平均特征）+ umap_scatter.png

用法：
  python 5-cluster-cells.py                                   # HDBSCAN + UMAP
  python 5-cluster-cells.py --method kmeans --k 4 --reduce pca
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))

# 参与聚类的特征列（CSV 中缺哪列就自动跳过）
FEATURE_COLS = ["area", "perimeter", "eccentricity", "solidity", "circularity",
                "mean_intensity", "std_intensity"]


def load_features(path):
    df = pd.read_csv(path)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if not cols:
        raise ValueError(f"{path} 中无可聚类列，需要 {FEATURE_COLS} 之一")
    X = df[cols].fillna(0).values
    return df, X, cols


def cluster(X, method="hdbscan", k=4, min_cluster_size=30):
    if method == "hdbscan":
        # min_cluster_size 大于真实簇的大小时会把所有点判成噪声，这里按样本量收敛
        size = max(2, min(min_cluster_size, len(X) // 10))
        return hdbscan.HDBSCAN(min_cluster_size=size).fit_predict(X)
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)


def reduce_2d(X, method="umap"):
    Xs = StandardScaler().fit_transform(X)
    if method == "umap":
        return umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(Xs)
    return PCA(n_components=2, random_state=42).fit_transform(Xs)


def main():
    ap = argparse.ArgumentParser(description="细胞表型聚类")
    ap.add_argument("--features", default=os.path.join(HERE, "features.csv"))
    ap.add_argument("--method", default="hdbscan", choices=["hdbscan", "kmeans"])
    ap.add_argument("--k", type=int, default=4, help="KMeans 簇数")
    ap.add_argument("--min_cluster_size", type=int, default=15,
                    help="HDBSCAN 最小簇大小（需小于真实簇的细胞数，否则全部判为噪声）")
    ap.add_argument("--reduce", default="umap", choices=["umap", "pca"])
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    df, X, cols = load_features(args.features)
    labels = cluster(X, args.method, args.k, args.min_cluster_size)
    df["cluster"] = labels
    os.makedirs(args.outdir, exist_ok=True)

    emb = reduce_2d(X, args.reduce)

    # 散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(labels))
    cmap = plt.cm.tab10
    for i, c in enumerate(unique):
        m = labels == c
        color = "lightgray" if c == -1 else cmap(i % 10)
        ax.scatter(emb[m, 0], emb[m, 1], s=8, c=[color],
                   label=f"cluster {c}" if c != -1 else "noise")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title(f"Cell phenotype clustering ({args.method}, {len(cols)} features)")
    ax.set_xlabel(f"{args.reduce}-1")
    ax.set_ylabel(f"{args.reduce}-2")
    out_png = os.path.join(args.outdir, "umap_scatter.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 每细胞簇标签
    df.to_csv(os.path.join(args.outdir, "cluster_assignments.csv"), index=False)

    # 每簇平均特征
    summary = df.groupby("cluster")[cols].mean().round(3)
    summary["count"] = df.groupby("cluster").size()
    summary.to_csv(os.path.join(args.outdir, "cluster_summary.csv"))

    n_real = len([c for c in unique if c != -1])
    print(f"聚类完成：{n_real} 个簇（noise: {(labels == -1).sum()} 个）")
    print(f"  散点图 -> {out_png}")
    print(f"  簇标签 -> {os.path.join(args.outdir, 'cluster_assignments.csv')}")
    print(f"  簇概要 -> {os.path.join(args.outdir, 'cluster_summary.csv')}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
