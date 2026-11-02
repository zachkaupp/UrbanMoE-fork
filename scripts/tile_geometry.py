import geopandas as gpd
import pandas as pd
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon

DATASETS = {
    "NewYork": ("data/NewYork_merged_normalized.csv", "data/NewYork_file/"),
    "Shanghai": ("data/shanghai_merged_normalized.csv", "data/shanghai_sat/"),
    "Beijing": ("data/Beijing_merged_normalized.csv", "data/beijing_sat/"),
}

TIEPOINT_TAG = 33922
PIXEL_SCALE_TAG = 33550
GEO_KEY_DIRECTORY_TAG = 34735
PROJECTED_CS_KEY = 3072


def _epsg_from_geo_keys(keys):
    # header is (version, revision, minor, key_count), then key_count entries of (id, location, count, value)
    for i in range(keys[3]):
        key_id, location, _, value = keys[4 + 4 * i:8 + 4 * i]
        if key_id == PROJECTED_CS_KEY and location == 0:
            return value
    raise ValueError("GeoTIFF has no projected CRS key")


def _read_bounds(path):
    with Image.open(path) as im:
        width, height = im.size
        tiepoint = im.tag_v2[TIEPOINT_TAG]
        scale = im.tag_v2[PIXEL_SCALE_TAG]
        epsg = _epsg_from_geo_keys(im.tag_v2[GEO_KEY_DIRECTORY_TAG])
    left, top = tiepoint[3], tiepoint[4]
    return epsg, left, top - height * scale[1], left + width * scale[0], top


def _read_tiles(dataset, limit=None):
    """(tile id, epsg, left, bottom, right, top) for each tile, in projected coordinates."""
    csv_path, image_dir = DATASETS[dataset]
    filenames = pd.read_csv(csv_path)["filename"]
    if limit is not None:
        filenames = filenames.iloc[:limit]
    for filename in filenames:
        name = filename.replace(".tif.tif", "")
        yield (name, *_read_bounds(f"{image_dir}{name}.tif"))


def load_tile_grid(dataset, limit=None):
    """The projected grid the tiles sit on: (epsg, cell size, top-left corner of each tile).

    Corners are a DataFrame with x/y columns indexed by tile id.
    """
    rows = list(_read_tiles(dataset, limit))
    epsgs = {epsg for _, epsg, *_ in rows}
    if len(epsgs) > 1:
        raise ValueError(f"tiles span multiple projected CRSs: {sorted(epsgs)}")

    corners = pd.DataFrame(
        [(left, top) for _, _, left, _, _, top in rows],
        index=pd.Index([name for name, *_ in rows], name="region_id"),
        columns=["x", "y"],
    )
    # a few tiles at the edge of the source imagery are cut short, so use the most common width
    cell_size = pd.Series([right - left for _, _, left, _, right, _ in rows]).mode()[0]
    offsets = (corners - corners.iloc[0]) % cell_size
    if not ((offsets < 1e-6) | (cell_size - offsets < 1e-6)).all().all():
        raise ValueError("tile corners are not on a regular grid")
    return epsgs.pop(), cell_size, corners


def load_tiles(dataset, limit=None):
    """Tile polygons (WGS84) indexed by tile id, e.g. 'tile_436'."""
    names, polygons, epsgs = [], [], set()
    transformers = {}
    for name, epsg, left, bottom, right, top in _read_tiles(dataset, limit):
        if epsg not in transformers:
            transformers[epsg] = Transformer.from_crs(epsg, 4326, always_xy=True)
        # corners are transformed from identical projected coordinates, so adjacent tiles share exact vertices
        xs, ys = transformers[epsg].transform([left, right, right, left], [bottom, bottom, top, top])
        names.append(name)
        polygons.append(Polygon(zip(xs, ys)))
        epsgs.add(epsg)

    if len(epsgs) > 1:
        print(f"Warning: tiles span multiple projected CRSs: {sorted(epsgs)}")
    tiles = gpd.GeoDataFrame(geometry=polygons, index=pd.Index(names, name="region_id"), crs="EPSG:4326")
    return tiles
