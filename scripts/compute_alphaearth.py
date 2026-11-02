"""Compute AlphaEarth Foundations tile embeddings and write them to data/embeddings/<dataset>_alphaearth.csv.

    uv run python scripts/compute_alphaearth.py --dataset NewYork
    uv run python scripts/compute_alphaearth.py --dataset NewYork --limit 20

Reads each tile's window straight from the public Cloud Optimized GeoTIFFs in gs://alphaearth_foundations (no
Earth Engine account), averages the de-quantized 10 m pixel embeddings inside the tile and renormalizes to unit
length, as the bucket README does for its overviews.

Finished batches of tiles are saved under <output_dir>/aef_cache/, so rerunning the same command after a stop or a
failure continues where it left off. Delete that folder to start over.
"""
import argparse
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from tile_geometry import DATASETS, _read_tiles

BUCKET = "https://storage.googleapis.com/alphaearth_foundations/"
INDEX_URL = BUCKET + "satellite_embedding/v1/annual/aef_index.parquet"
INDEX_COLUMNS = ["path", "crs", "year", "utm_west", "utm_south", "utm_east", "utm_north"]
NODATA = -128
NUM_BANDS = 64
COLUMNS = [f"alphaearth_{i}" for i in range(NUM_BANDS)]
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    # a connection that died while the machine slept should fail fast, so it can be retried
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
}
# seconds to wait before each retry of a failed read, 90 in total, so a short sleep or network drop is survived
RETRY_WAITS = (5, 10, 15, 30, 30)

_local = threading.local()


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS))
    parser.add_argument("--output_dir", type=str, default="data/embeddings/")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N tiles (for a quick test)")
    parser.add_argument("--year", type=int, default=None, help="embedding year, default the latest in the index")
    parser.add_argument("--workers", type=int, default=16, help="tiles read in parallel")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="tiles per saved batch; a stopped run keeps every finished batch")
    return parser.parse_args()


def load_index(cache_dir):
    path = cache_dir / "aef_index.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {INDEX_URL}")
        urllib.request.urlretrieve(INDEX_URL, path)
    return pd.read_parquet(path, columns=INDEX_COLUMNS)


def dequantize(values):
    values = values.astype(np.float32)
    return (values / 127.5) ** 2 * np.sign(values)


def open_cog(url):
    """One open dataset per thread and file; datasets are not safe to share between threads."""
    datasets = _local.__dict__.setdefault("datasets", {})
    if url not in datasets:
        datasets[url] = rasterio.open(url)
    return datasets[url]


def close_cog(url):
    """Forget this thread's dataset for a file, so the next read reconnects."""
    dataset = _local.__dict__.setdefault("datasets", {}).pop(url, None)
    if dataset is not None:
        try:
            dataset.close()
        except Exception:
            pass


def read_window(file, left, bottom, right, top):
    """Raw int8 values of a tile's part inside one file, or None if it has no pixels there."""
    src = open_cog(file.url)
    transform = src.transform
    # the files are stored south-up (transform.e > 0, rows grow northward), so from_bounds can't be used
    col0, col1 = (round((x - transform.c) / transform.a) for x in (left, right))
    row0, row1 = (round((y - transform.f) / transform.e) for y in (bottom, top))
    col0, col1 = max(col0, 0), min(col1, src.width)
    row0, row1 = max(row0, 0), min(row1, src.height)
    if col0 >= col1 or row0 >= row1:
        return None
    return src.read(window=Window(col0, row0, col1 - col0, row1 - row0))


def read_piece(file, left, bottom, right, top):
    """(sum of de-quantized embeddings, valid pixel count) of the part of a tile that lies in one file."""
    left, right = max(left, file.utm_west), min(right, file.utm_east)
    bottom, top = max(bottom, file.utm_south), min(top, file.utm_north)
    if left >= right or bottom >= top:
        return np.zeros(NUM_BANDS), 0

    for wait in (*RETRY_WAITS, None):
        try:
            with rasterio.Env(**GDAL_OPTIONS):
                values = read_window(file, left, bottom, right, top)
            break
        except rasterio.errors.RasterioIOError as error:
            close_cog(file.url)
            if wait is None:
                raise
            tqdm.write(f"read failed ({error}), retrying in {wait}s")
            time.sleep(wait)
    if values is None:
        return np.zeros(NUM_BANDS), 0
    valid = (values != NODATA).all(axis=0)  # masked in one channel means masked in all
    if not valid.any():
        return np.zeros(NUM_BANDS), 0
    return dequantize(values[:, valid]).sum(axis=1, dtype=np.float64), int(valid.sum())


def tile_embedding(tile, files):
    name, _, left, bottom, right, top = tile
    total, count = np.zeros(NUM_BANDS), 0
    for file in files.itertuples():
        if file.utm_west < right and file.utm_east > left and file.utm_south < top and file.utm_north > bottom:
            piece_sum, piece_count = read_piece(file, left, bottom, right, top)
            total += piece_sum
            count += piece_count
    if count == 0:
        return name, None
    return name, total / (np.linalg.norm(total) + 1e-9)


def save_part(parts_dir, results):
    """Write a batch of (tile id, embedding) atomically; a tile with no valid pixels is a row of NaN."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    rows = [np.full(NUM_BANDS, np.nan) if embedding is None else embedding for _, embedding in results]
    part = pd.DataFrame(rows, columns=COLUMNS, index=pd.Index([f"{name}.csv" for name, _ in results], name="name"))
    path = parts_dir / f"part_{time.time_ns()}.csv"
    part.to_csv(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def load_parts(parts_dir):
    paths = sorted(parts_dir.glob("part_*.csv"))
    if not paths:
        return pd.DataFrame(columns=COLUMNS, index=pd.Index([], name="name"), dtype=float)
    parts = pd.concat(pd.read_csv(path, index_col="name") for path in paths)
    return parts[~parts.index.duplicated(keep="last")]


def main():
    args = create_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = list(_read_tiles(args.dataset, args.limit))
    epsgs = {tile[1] for tile in tiles}
    if len(epsgs) > 1:
        raise ValueError(f"tiles span multiple projected CRSs: {sorted(epsgs)}")
    epsg = epsgs.pop()
    print(f"{len(tiles)} tiles in EPSG:{epsg}")

    index = load_index(output_dir / "aef_cache")
    year = args.year or int(index["year"].max())
    files = index[(index["year"] == year) & (index["crs"] == f"EPSG:{epsg}")].copy()
    files["url"] = files["path"].str.replace("gs://", "https://storage.googleapis.com/", regex=False)
    print(f"Year {year}, {len(files)} files in EPSG:{epsg}")

    # finished batches are kept on disk, so a stopped run continues where it left off
    parts_dir = output_dir / "aef_cache" / f"{args.dataset}_{year}"
    done = load_parts(parts_dir)
    todo = [tile for tile in tiles if f"{tile[0]}.csv" not in done.index]
    print(f"{len(tiles) - len(todo)} tiles already done, {len(todo)} to go")

    with ThreadPoolExecutor(args.workers) as pool, tqdm(total=len(todo)) as progress:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            results = list(pool.map(lambda tile: tile_embedding(tile, files), batch))
            save_part(parts_dir, results)
            progress.update(len(batch))

    done = load_parts(parts_dir)
    names = [f"{name}.csv" for name, *_ in tiles]
    embeddings = done.loc[names]
    empty = embeddings.index[embeddings.isna().any(axis=1)]
    if len(empty) > 0:
        print(f"Warning: {len(empty)} tiles have no valid pixels, filling with zeros, e.g. {list(empty[:5])}")
    embeddings = embeddings.fillna(0.0).reset_index()
    path = output_dir / f"{args.dataset}_alphaearth.csv"
    embeddings.to_csv(path, index=False)
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1] - 1} embeddings to {path}")


if __name__ == "__main__":
    main()
