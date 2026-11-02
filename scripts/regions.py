"""Areas larger than a dataset's tiles that S2Vec can be trained on."""
import urllib.request
from pathlib import Path

import geopandas as gpd

STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"

# the S2Vec grid is projected in the dataset's own UTM zone, so a region only suits datasets near that zone
REGIONS = {
    "northeast": {
        "datasets": ["NewYork"],  # UTM 18N: scale error stays under 0.5% across all of these states
        "states": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New Jersey",
                   "New York", "Pennsylvania", "Rhode Island", "Vermont"],
    },
}


def load_region(name, dataset, cache_dir):
    """Region polygons in WGS84, from the Census cartographic state boundaries."""
    region = REGIONS[name]
    if dataset not in region["datasets"]:
        raise ValueError(f"region {name} is only set up for {region['datasets']}, not {dataset}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(STATES_URL).name
    if not path.exists():
        urllib.request.urlretrieve(STATES_URL, path)
    states = gpd.read_file(f"zip://{path}")
    region_states = states[states["NAME"].isin(region["states"])]
    if len(region_states) != len(region["states"]):
        raise ValueError(f"expected {len(region['states'])} states, found {sorted(region_states['NAME'])}")
    return region_states.to_crs("EPSG:4326")
