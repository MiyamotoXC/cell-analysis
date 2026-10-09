"""
单细胞图像处理：LIVECell 数据集下载与解压（就绪脚本，需手动运行）
=================================================================
数据集：Sartorius LIVECell (Nature Methods 2021)，相差显微镜，8 种细胞系，
        160 万+ 标注细胞，实例分割基准，CC BY-NC 4.0。

官方数据源（任选其一）：
  - figshare 文章 14931555：LIVECell_dataset_2021.zip（含图像 + COCO 标注 json）
  - Zenodo record 10277105：按细胞系分卷（A172.zip ... SKOV3.zip）

本脚本通过 figshare API 动态获取真实下载链接（避免硬编码 URL 失效），
并按名字模糊匹配 .zip，文件名变了也能找到。数据为 GB 级，首次运行耗时较长；
已下载的文件会自动跳过，可重复执行断点续传。

若所在网络无法访问 figshare，请手工从 https://sartorius-research.github.io/LIVECell/
下载后把 zip 放进 cell_data/，再运行 convert_livecell.py。

用法：
    python download_livecell.py                 # 下载并解压全部
    python download_livecell.py --list          # 只列出可用文件（不下）
    python download_livecell.py --no_extract    # 只下载，不解压
之后运行：
    python convert_livecell.py
"""
import argparse
import os
import zipfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CELL_DATA = os.path.join(HERE, "cell_data")
FIGSHARE_ARTICLE_ID = 14931555


def fetch_figshare_files(article_id):
    """通过 figshare API 获取文章下所有文件的 (name -> download_url, size)。"""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    return {f["name"]: (f["download_url"], f.get("size", 0)) for f in files}


def pick_targets(files):
    """按名字模糊匹配数据包：名字含 livecell 且是 zip。"""
    return [n for n in files if "livecell" in n.lower() and n.lower().endswith(".zip")]


def download(url, dest, chunk=1 << 20):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"下载 {url}\n  -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for data in r.iter_content(chunk):
                f.write(data)
    print("  完成")


def extract(zip_path, out_dir):
    print(f"解压 {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print("  完成")


def main():
    ap = argparse.ArgumentParser(description="下载 LIVECell 数据集")
    ap.add_argument("--list", action="store_true", help="只列出可用文件")
    ap.add_argument("--no_extract", action="store_true", help="只下载不解压")
    args = ap.parse_args()

    os.makedirs(CELL_DATA, exist_ok=True)
    files = fetch_figshare_files(FIGSHARE_ARTICLE_ID)

    print(f"figshare 文章 {FIGSHARE_ARTICLE_ID} 可用文件：")
    for name, (_, size) in files.items():
        print(f"  {name}  ({size / 2 ** 30:.2f} GB)")
    if args.list:
        return

    targets = pick_targets(files)
    if not targets:
        print("\n未匹配到 LIVECell 数据包。请手工下载后放入 cell_data/，再运行 convert_livecell.py")
        return

    for name in targets:
        url, _ = files[name]
        zip_path = os.path.join(CELL_DATA, name)
        if not os.path.exists(zip_path):
            download(url, zip_path)
        else:
            print(f"已存在，跳过下载：{zip_path}")
        if not args.no_extract:
            extract(zip_path, CELL_DATA)
    print("\n下载/解压完成。下一步运行：python convert_livecell.py")


if __name__ == "__main__":
    main()
