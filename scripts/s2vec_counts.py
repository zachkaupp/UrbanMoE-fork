"""Count OSM features per S2Vec cell a few images at a time, so the features never all sit in memory.

S2Vec only trains on the per-cell counts (one row per cell, one column per feature value). The OSM features
and the cell/feature join are only needed to produce those counts, and they are what takes the memory.
"""
from collections import Counter

import geopandas as gpd
import numpy as np
import pandas as pd
from srai.embedders import CountEmbedder, S2VecEmbedder
from srai.embedders.s2vec.dataset import S2VecDataset
from srai.embedders.s2vec.grid_utils import get_grid_patches
from srai.joiners import IntersectionJoiner
from torch.utils.data import DataLoader
from tqdm import tqdm


def union_geometry(geoseries):
    return geoseries.union_all() if hasattr(geoseries, "union_all") else geoseries.unary_union


def feature_value_counts(features, tags):
    """Number of features for each flat "<key>_<value>" name."""
    counts = Counter()
    for key in tags:
        if key in features.columns:
            counts.update({f"{key}_{value}": count for value, count in features[key].dropna().value_counts().items()})
    return counts


def _chunks(cells, img_cell_joint, img_size, chunk_images, tags, loader, desc):
    """Yield (cells, OSM features) for each group of `chunk_images` whole images.

    get_grid_patches returns the cells of each image contiguously and every image has img_size**2 cells.
    """
    cell_ids = img_cell_joint.index.get_level_values(1)
    step = chunk_images * img_size ** 2
    for start in tqdm(range(0, len(cell_ids), step), desc=desc):
        chunk_cells = cells.loc[cell_ids[start:start + step]]
        area = gpd.GeoDataFrame(geometry=[union_geometry(chunk_cells.geometry)], crs=chunk_cells.crs)
        yield chunk_cells, loader.load(area, tags)


def count_feature_values(cells, img_cell_joint, img_size, chunk_images, tags, loader):
    """feature_value_counts over the whole grid. Features that straddle two chunks are counted in both."""
    total = Counter()
    for _, features in _chunks(cells, img_cell_joint, img_size, chunk_images, tags, loader, "counting feature values"):
        total.update(feature_value_counts(features, tags))
    return total


def count_cells(cells, img_cell_joint, img_size, chunk_images, tags, names, loader):
    """Raw (unnormalised) feature counts per cell, identical to S2VecEmbedder's own counting."""
    embedder = CountEmbedder(expected_output_features=names, count_subcategories=True)
    joiner = IntersectionJoiner()
    parts = []
    for chunk_cells, features in _chunks(cells, img_cell_joint, img_size, chunk_images, tags, loader, "counting cells"):
        joint = joiner.transform(chunk_cells, features)
        counts = embedder.transform(chunk_cells, features, joint).astype(np.float32)
        parts.append(counts.reindex(index=chunk_cells.index, columns=names, fill_value=0))
    return pd.concat(parts)


class PrecountedS2VecEmbedder(S2VecEmbedder):
    """S2VecEmbedder that takes the raw per-cell counts from set_counts() instead of OSM features.

    _prepare_dataset repeats S2VecEmbedder._prepare_dataset (srai-fork) from the counts onward.
    The features argument of fit/transform is ignored (pass None).
    """

    def set_counts(self, counts):
        self._counts = counts

    def _prepare_dataset(self, regions_gdf, features_gdf, batch_size, shuffle=True, is_fitting=True):
        patches_gdf, img_patch_joint_gdf = get_grid_patches(
            area_gdf=regions_gdf,
            crs=self._crs,
            origin=self._origin,
            cell_size=self._cell_size,
            img_size=self.img_size,
        )
        missing = patches_gdf.index.difference(self._counts.index)
        if len(missing) > 0:
            raise ValueError(f"{len(missing)} cells have no counts, e.g. {missing[0]}")
        counts_df = self._counts.loc[patches_gdf.index].copy()

        if is_fitting:
            eps = 1e-8
            self._feature_means = np.mean(counts_df.values, axis=0)
            self._feature_stds = np.std(counts_df.values, axis=0)
            self._empty_features_mask = self._feature_stds < eps
            self._feature_stds[self._empty_features_mask] = 1.0

        counts_df.loc[:, self._empty_features_mask] = 0
        counts_df = (counts_df - self._feature_means) / self._feature_stds

        dataset = S2VecDataset(counts_df, img_patch_joint_gdf)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return counts_df, dataloader, dataset
