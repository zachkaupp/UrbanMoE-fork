"""Compute OSM-based tile embeddings with SRAI and write them to data/embeddings/<dataset>_<embedder>.csv.

    uv run python scripts/compute_embeddings.py --dataset NewYork --embedder contextual_count
    uv run python scripts/compute_embeddings.py --dataset NewYork --embedder s2vec --limit 50
    uv run python scripts/compute_embeddings.py --dataset NewYork --embedder s2vec --train_region northeast

S2Vec runs on the tiles' own projected grid (srai-fork), so each tile is exactly one S2Vec cell.
It is trained on the tiles plus a buffer, or on a whole region with --train_region, and then embeds the tiles.
"""
import argparse
from pathlib import Path

import geopandas as gpd
from srai.embedders import ContextualCountEmbedder
from srai.embedders.s2vec.grid_utils import get_grid_patches
from srai.joiners import IntersectionJoiner
from srai.loaders.osm_loaders import OSMPbfLoader
from srai.neighbourhoods import AdjacencyNeighbourhood

from osm_tags import OSM_TAGS
from regions import REGIONS, load_region
from s2vec_counts import (PrecountedS2VecEmbedder, count_cells, count_feature_values, feature_value_counts,
                          union_geometry)
from tile_geometry import DATASETS, load_tile_grid, load_tiles


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS))
    parser.add_argument("--embedder", type=str, required=True, choices=["contextual_count", "s2vec"])
    parser.add_argument("--output_dir", type=str, default="data/embeddings/")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N tiles (for a quick test)")
    parser.add_argument("--exclude_keys", type=str, nargs="*", default=[],
                        help="OSM keys to drop from the default filter, e.g. building")

    parser.add_argument("--neighbour_distance", type=int, default=2)

    parser.add_argument("--img_size", type=int, default=16,
                        help="tiles along each side of an S2Vec image")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--buffer_km", type=float, default=40.0,
                        help="extra area around the tiles used as S2Vec training data")
    parser.add_argument("--train_region", type=str, default=None, choices=list(REGIONS),
                        help="also train S2Vec on this whole region (the tiles are always included)")
    parser.add_argument("--chunk_images", type=int, default=16,
                        help="S2Vec images whose OSM features are loaded and counted at a time (limits memory)")
    parser.add_argument("--min_feature_count", type=int, default=20,
                        help="drop OSM feature values rarer than this from the embedder input")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


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


def frequent_names(value_counts, min_count):
    return sorted(name for name, count in value_counts.items() if count >= min_count)


def flat_feature_names(features, tags, min_count):
    return frequent_names(feature_value_counts(features, tags), min_count)


def run_s2vec(tiles, tags, args, loader, output_dir):
    epsg, cell_size, corners = load_tile_grid(args.dataset, args.limit)
    crs = f"EPSG:{epsg}"
    origin = tuple(corners.iloc[0])
    tile_area = union_geometry(tiles.to_crs(crs).geometry)
    train_area = tile_area.buffer(args.buffer_km * 1000)
    if args.train_region:
        region = load_region(args.train_region, args.dataset, output_dir / "boundaries")
        train_area = train_area.union(union_geometry(region.to_crs(crs).geometry))
    train_gdf = gpd.GeoDataFrame(geometry=[train_area], crs=crs).to_crs("EPSG:4326")
    tile_gdf = gpd.GeoDataFrame(geometry=[tile_area], crs=crs).to_crs("EPSG:4326")

    # images extend past the training area, so count OSM features for every cell the grid covers
    cells, img_cell_joint = get_grid_patches(train_gdf, crs, origin, cell_size, args.img_size)
    print(f"{img_cell_joint.index.get_level_values(0).nunique()} images, {len(cells)} cells")
    chunk_args = (cells, img_cell_joint, args.img_size, args.chunk_images, tags)
    value_counts = count_feature_values(*chunk_args, loader)
    names = frequent_names(value_counts, args.min_feature_count)
    print(f"{len(names)} feature values after min_feature_count={args.min_feature_count}")
    counts = count_cells(*chunk_args, names, loader)

    embedder = PrecountedS2VecEmbedder(
        target_features=names,
        crs=crs,
        origin=origin,
        cell_size=cell_size,
        count_subcategories=True,
        batch_size=args.batch_size,
        img_size=args.img_size,
        embedding_dim=args.embedding_dim,
    )
    embedder.set_counts(counts)
    embedder.fit(
        train_gdf, None,
        learning_rate=args.lr,
        trainer_kwargs={"max_epochs": args.epochs, "accelerator": "auto"},
    )
    suffix = f"_{args.train_region}" if args.train_region else ""
    embedder.save(output_dir / f"{args.dataset}_s2vec{suffix}_model")
    cell_embeddings = embedder.transform(tile_gdf, None)

    # cell ids are "<x>_<y>" of the cell's top-left corner, which is the tile's top-left corner
    cell_by_corner = {tuple(map(float, cell_id.split("_"))): cell_id for cell_id in cell_embeddings.index}
    cell_ids = [cell_by_corner[(x, y)] for x, y in corners.loc[tiles.index].itertuples(index=False)]
    embeddings = cell_embeddings.loc[cell_ids].set_axis(tiles.index)
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
