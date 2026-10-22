"""Wide OSM tag filter for the embedding scripts. A value of True keeps every value of that key."""

# TODO: this is a deliberately wide first pass; decide the final set of keys/values to include

OSM_TAGS = {
    # points of interest
    "amenity": True,
    "shop": True,
    "office": True,
    "tourism": True,
    "historic": True,
    "leisure": True,
    "sport": True,
    "healthcare": True,
    "craft": True,
    "emergency": True,
    "military": True,
    # industry and energy (large point sources of emissions)
    "power": True,
    "man_made": True,
    "industrial": True,
    "telecom": True,
    # transport
    "aeroway": True,
    "railway": True,
    "public_transport": True,
    "highway": [
        "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
        "secondary", "secondary_link", "tertiary", "tertiary_link", "residential",
        "service", "bus_stop", "traffic_signals",
    ],
    # land cover and buildings
    "landuse": True,
    "natural": True,
    "water": True,
    "waterway": True,
    "building": True,
}
