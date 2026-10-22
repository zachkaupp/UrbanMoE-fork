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


def load_tiles(dataset, limit=None):
    """Tile polygons (WGS84) indexed by tile id, e.g. 'tile_436'."""
    csv_path, image_dir = DATASETS[dataset]
    filenames = pd.read_csv(csv_path)["filename"]
    if limit is not None:
        filenames = filenames.iloc[:limit]

    names, polygons, epsgs = [], [], set()
    transformers = {}
    for filename in filenames:
        name = filename.replace(".tif.tif", "")
        epsg, left, bottom, right, top = _read_bounds(f"{image_dir}{name}.tif")
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
