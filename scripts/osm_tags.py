"""OSM tag filter for the embedding scripts. A value of True keeps every value of that key."""
from srai.loaders.osm_loaders.filters import BASE_OSM_GROUPS_FILTER, GEOFABRIK_LAYERS, merge_osm_tags_filter

# TODO(human): confirm this is an acceptable stand-in for S2Vec's features. The S2Vec paper (Choudhury et al.) uses
# Google Maps POI categories plus traffic infrastructure counts (116 features in total) and does not publish the list;
# it only says the taxonomy "has considerable overlap" with OSM. This is srai's base OSM groups (18 groups, 172
# key=value pairs, but keys set to True such as shop expand to every value) plus Geofabrik's traffic and road layers.
ROAD_LAYERS = ("traffic", "major_roads", "minor_roads", "highway_links", "very_small_roads",
               "paths_unsuitable_for_cars")

OSM_TAGS = merge_osm_tags_filter([
    BASE_OSM_GROUPS_FILTER,
    {layer: GEOFABRIK_LAYERS[layer] for layer in ROAD_LAYERS},
])
