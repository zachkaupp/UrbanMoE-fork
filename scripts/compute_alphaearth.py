"""Compute AlphaEarth Foundations tile embeddings and write them to data/embeddings/<dataset>_alphaearth.csv.

    uv run python scripts/compute_alphaearth.py --dataset NewYork
    uv run python scripts/compute_alphaearth.py --dataset NewYork --limit 20

Uses the public Cloud Optimized GeoTIFFs in gs://alphaearth_foundations (no Earth Engine account). Reading a tile's
10 m pixels directly is far too slow (the files are band-interleaved in 1024 x 1024 blocks, so every tile window
downloads whole blocks for all 64 bands), so each file's built-in overview (--overview_factor, default 8 = 80 m
pixels) is downloaded once and cached, and every tile is averaged from that. The bucket README builds its overviews
by summing the de-quantized vectors and renormalizing, which is the same aggregation used here for the tiles.
Compared with the exact 10 m average, tile_0 has cosine similarity 0.99997 at factor 8.

Downloaded overviews are kept under <output_dir>/aef_cache/, so rerunning after a failure only fetches what is
missing. Delete that folder to start over.
"""
import argparse
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
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
    # a connection that died while the machine slept should fail, so it can be retried
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
    "GDAL_HTTP_TIMEOUT": "120",
}
# seconds to wait before each retry of a failed download, 90 in total, so a short sleep or network drop is survived
RETRY_WAITS = (5, 10, 15, 30, 30)


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS))
    parser.add_argument("--output_dir", type=str, default="data/embeddings/")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N tiles (for a quick test)")
    parser.add_argument("--year", type=int, default=None, help="embedding year, default the latest in the index")
    parser.add_argument("--overview_factor", type=int, default=8, choices=[4, 8, 16],
                        help="pixels are 10 m x this; 4 is finer but about 4x the download")
    parser.add_argument("--workers", type=int, default=3, help="files downloaded in parallel")
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


def fetch_overview(file, factor, cache_dir):
    """The overview of one file as (int8 array of shape 64 x rows x cols, [x origin, pixel width, y origin, pixel height])."""
    path = cache_dir / f"{Path(file.path).stem}_f{factor}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["data"], saved["geo"]

    for wait in (*RETRY_WAITS, None):
        try:
            with rasterio.Env(**GDAL_OPTIONS), rasterio.open(file.url) as src:
                data = src.read(out_shape=(NUM_BANDS, src.height // factor, src.width // factor))
                # the files are stored south-up (e > 0, rows grow northward), so this is used as is
                geo = np.array([src.transform.c, src.transform.a * factor, src.transform.f, src.transform.e * factor])
            break
        except rasterio.errors.RasterioIOError as error:
            if wait is None:
                raise
            tqdm.write(f"download of {Path(file.path).name} failed ({error}), retrying in {wait}s")
            time.sleep(wait)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, data=data, geo=geo)
    tmp.replace(path)
    return data, geo


def tile_embedding(tile, files, overviews):
    name, _, left, bottom, right, top = tile
    total, count = np.zeros(NUM_BANDS), 0
    for file in files.itertuples():
        l, r = max(left, file.utm_west), min(right, file.utm_east)
        b, t = max(bottom, file.utm_south), min(top, file.utm_north)
        if l >= r or b >= t:
            continue
        data, (x0, dx, y0, dy) = overviews[file.path]
        rows, cols = data.shape[1:]
        c0, c1 = (round((x - x0) / dx) for x in (l, r))
        r0, r1 = (round((y - y0) / dy) for y in (b, t))
        c1, r1 = max(c1, c0 + 1), max(r1, r0 + 1)  # a sliver of a tile still gets the pixel it falls in
        c0, c1, r0, r1 = max(c0, 0), min(c1, cols), max(r0, 0), min(r1, rows)
        if c0 >= c1 or r0 >= r1:
            continue
        values = data[:, r0:r1, c0:c1]
        valid = (values != NODATA).all(axis=0)  # masked in one channel means masked in all
        if valid.any():
            total += dequantize(values[:, valid]).sum(axis=1, dtype=np.float64)
            count += int(valid.sum())
    if count == 0:
        return None
    return total / (np.linalg.norm(total) + 1e-9)


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

    cache_dir = output_dir / "aef_cache"
    index = load_index(cache_dir)
    year = args.year or int(index["year"].max())
    files = index[(index["year"] == year) & (index["crs"] == f"EPSG:{epsg}")].copy()
    files["url"] = files["path"].str.replace("gs://", "https://storage.googleapis.com/", regex=False)

    bounds = np.array([(left, bottom, right, top) for _, _, left, bottom, right, top in tiles])
    overlaps = np.zeros(len(files), dtype=bool)
    for left, bottom, right, top in bounds:
        overlaps |= ((files["utm_west"] < right) & (files["utm_east"] > left)
                     & (files["utm_south"] < top) & (files["utm_north"] > bottom)).to_numpy()
    files = files[overlaps]
    print(f"Year {year}, {len(files)} files cover the tiles, overview factor {args.overview_factor}")

    overview_dir = cache_dir / f"overviews_{year}"
    with ThreadPoolExecutor(args.workers) as pool:
        fetched = list(tqdm(pool.map(lambda file: fetch_overview(file, args.overview_factor, overview_dir),
                                     files.itertuples()), total=len(files), desc="downloading overviews"))
    overviews = dict(zip(files["path"], fetched))

    results = [tile_embedding(tile, files, overviews) for tile in tqdm(tiles, desc="averaging tiles")]
    empty = [tile[0] for tile, embedding in zip(tiles, results) if embedding is None]
    if empty:
        print(f"Warning: {len(empty)} tiles have no valid pixels, filling with zeros, e.g. {empty[:5]}")

    rows = [np.zeros(NUM_BANDS) if embedding is None else embedding for embedding in results]
    embeddings = pd.DataFrame(rows, columns=COLUMNS)
    embeddings.insert(0, "name", [f"{tile[0]}.csv" for tile in tiles])
    path = output_dir / f"{args.dataset}_alphaearth.csv"
    embeddings.to_csv(path, index=False)
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1] - 1} embeddings to {path}")


if __name__ == "__main__":
    main()
