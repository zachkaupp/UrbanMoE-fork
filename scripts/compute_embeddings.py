"""Compute OSM-based tile embeddings with SRAI and write them to data/embeddings/<dataset>_<embedder>.csv.

    uv run python scripts/compute_embeddings.py --dataset NewYork --embedder contextual_count
    uv run python scripts/compute_embeddings.py --dataset NewYork --embedder s2vec --limit 50
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from srai.embedders import ContextualCountEmbedder, S2VecEmbedder
from srai.embedders.s2vec.s2_utils import get_patches_from_img_gdf
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.neighbourhoods import AdjacencyNeighbourhood
from srai.regionalizers import S2Regionalizer

from osm_tags import OSM_TAGS
from tile_geometry import DATASETS, load_tiles


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS))
    parser.add_argument("--embedder", type=str, required=True, choices=["contextual_count", "s2vec"])
    parser.add_argument("--output_dir", type=str, default="data/embeddings/")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N tiles (for a quick test)")
    parser.add_argument("--exclude_keys", type=str, nargs="*", default=[],
                        help="OSM keys to drop from the default filter, e.g. building")

    parser.add_argument("--neighbour_distance", type=int, default=2)

    parser.add_argument("--img_res", type=int, default=8)
    parser.add_argument("--patch_res", type=int, default=12)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--s2_buffer_km", type=float, default=40.0,
                        help="extra area around the tiles used as S2Vec training data")
    parser.add_argument("--min_feature_count", type=int, default=20,
                        help="drop OSM feature values rarer than this from the embedder input")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def union_geometry(geoseries):
    return geoseries.union_all() if hasattr(geoseries, "union_all") else geoseries.unary_union


def run_contextual_count(tiles, tags, args, loader):
    features = loader.load(tiles, tags)
    print(f"Loaded {len(features)} OSM features")
    names = flat_feature_names(features, tags, args.min_feature_count)
    print(f"{len(names)} feature values after min_feature_count={args.min_feature_count}")
    joint = IntersectionJoiner().transform(tiles, features)
    embedder = ContextualCountEmbedder(
        neighbourhood=AdjacencyNeighbourhood(tiles),
        neighbourhood_distance=args.neighbour_distance,
        expected_output_features=names,
        count_subcategories=True,
    )
    return embedder.transform(tiles, features, joint)


def flat_feature_names(features, tags, min_count):
    names = []
    for key in tags:
        if key not in features.columns:
            continue
        counts = features[key].dropna().value_counts()
        names += [f"{key}_{value}" for value, count in counts.items() if count >= min_count]
    return sorted(names)


def patches_to_tiles(tiles, patches, patch_embeddings):
    """Average the embeddings of the patches a tile overlaps, weighted by overlap area."""
    crs = tiles.estimate_utm_crs()
    tile_geoms = tiles.to_crs(crs).geometry
    patch_geoms = patches.to_crs(crs).geometry

    left = gpd.GeoDataFrame(geometry=tile_geoms.values, index=tiles.index)
    left.index.name = None
    right = gpd.GeoDataFrame(geometry=patch_geoms.values, index=patches.index)
    right.index.name = None
    pairs = gpd.sjoin(left, right, how="inner", predicate="intersects")

    weights = np.array([
        tile_geoms.loc[tile_id].intersection(patch_geoms.loc[patch_id]).area
        for tile_id, patch_id in zip(pairs.index, pairs["index_right"])
    ])
    values = patch_embeddings.loc[pairs["index_right"]].to_numpy() * weights[:, None]
    summed = pd.DataFrame(values, index=pairs.index).groupby(level=0).sum()
    totals = pd.Series(weights, index=pairs.index).groupby(level=0).sum()
    result = summed.div(totals, axis=0)
    result.index.name = tiles.index.name
    return result.reindex(tiles.index)


def run_s2vec(tiles, tags, args, loader, output_dir):
    crs = tiles.estimate_utm_crs()
    area = union_geometry(tiles.to_crs(crs).geometry).buffer(args.s2_buffer_km * 1000)
    area_gdf = gpd.GeoDataFrame(geometry=[area], crs=crs).to_crs("EPSG:4326")

    images = S2Regionalizer(resolution=args.img_res).transform(area_gdf)
    print(f"{len(images)} S2 image cells at level {args.img_res}")
    features = loader.load(images, tags)
    print(f"Loaded {len(features)} OSM features")

    names = flat_feature_names(features, tags, args.min_feature_count)
    print(f"{len(names)} feature values after min_feature_count={args.min_feature_count}")
    embedder = S2VecEmbedder(
        target_features=names,
        count_subcategories=True,
        batch_size=args.batch_size,
        img_res=args.img_res,
        patch_res=args.patch_res,
        embedding_dim=args.embedding_dim,
    )
    patch_embeddings = embedder.fit_transform(
        images, features,
        learning_rate=args.lr,
        trainer_kwargs={"max_epochs": args.epochs, "accelerator": "auto"},
    )
    embedder.save(output_dir / f"{args.dataset}_s2vec_model")

    patches, _ = get_patches_from_img_gdf(img_gdf=images, target_level=args.patch_res)
    embeddings = patches_to_tiles(tiles, patches, patch_embeddings)
    embeddings.columns = [f"s2vec_{i}" for i in range(embeddings.shape[1])]
    return embeddings


def main():
    args = create_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tags = {key: value for key, value in OSM_TAGS.items() if key not in args.exclude_keys}
    tiles = load_tiles(args.dataset, args.limit)
    print(f"{len(tiles)} tiles")
    loader = OSMPbfLoader(download_directory=str(output_dir / "osm_cache"))

    if args.embedder == "contextual_count":
        embeddings = run_contextual_count(tiles, tags, args, loader)
    else:
        embeddings = run_s2vec(tiles, tags, args, loader, output_dir)

    missing = tiles.index.difference(embeddings.index)
    if len(missing) > 0:
        print(f"Warning: {len(missing)} tiles have no embedding, filling with zeros")
    embeddings = embeddings.reindex(tiles.index).fillna(0.0)
    embeddings.insert(0, "name", [f"{tile_id}.csv" for tile_id in embeddings.index])

    path = output_dir / f"{args.dataset}_{args.embedder}.csv"
    embeddings.to_csv(path, index=False)
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1] - 1} embeddings to {path}")


if __name__ == "__main__":
    main()
