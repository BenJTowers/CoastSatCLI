#!/usr/bin/env python3
"""
epsg_selector.py

Given an AOI file (any vector format GeoPandas supports), this module will:
1. Load and merge the AOI geometry.
2. Reproject it to EPSG:4326 if necessary.
3. Compute its centroid longitude and latitude.
4. Confirm latitude is within Canada’s range (approx. 41°N–84°N).
5. Convert that longitude to a UTM zone (7–22).
6. Map that zone to a Canadian NAD83 UTM EPSG code (3154–3761).

If the AOI centroid lies outside zones 7–22 or outside Canadian latitudes, a ValueError is raised.

Usage:
    python epsg_selector.py /path/to/aoi_file.{geojson,kml,shp}

Dependencies:
    - geopandas
    - shapely
    - pyproj
"""

import math
import sys

import geopandas as gpd
from shapely.ops import unary_union

# Mapping from UTM zone number (7..22) to Canadian NAD83 UTM EPSG code
ZONE_TO_EPSG = {
    7:  3154,  # UTM zone 7N
    8:  3155,  # UTM zone 8N
    9:  3156,  # UTM zone 9N
    10: 3157,  # UTM zone 10N
    11: 2955,  # UTM zone 11N
    12: 2956,  # UTM zone 12N
    13: 2957,  # UTM zone 13N
    14: 3158,  # UTM zone 14N
    15: 3159,  # UTM zone 15N
    16: 3160,  # UTM zone 16N
    17: 2958,  # UTM zone 17N
    18: 2959,  # UTM zone 18N
    19: 2960,  # UTM zone 19N
    20: 2961,  # UTM zone 20N
    21: 2962,  # UTM zone 21N
    22: 3761   # UTM zone 22N
}


def load_aoi_geometry(aoi_path: str):
    """
    Load the AOI file (any vector format GeoPandas supports) and return a single
    shapely geometry (merged if multiple features). Ensure the result is in EPSG:4326.

    Raises:
        ValueError: if the AOI has no CRS.
    """
    gdf = gpd.read_file(aoi_path)
    if gdf.crs is None or gdf.crs.to_epsg() is None:
        raise ValueError("AOI has no CRS defined—cannot determine centroid in lat/lon.")
    # Merge all features into one geometry
    merged = unary_union(gdf.geometry)
    # Reproject to EPSG:4326 if needed
    if gdf.crs.to_epsg() != 4326:
        merged = gpd.GeoSeries([merged], crs=gdf.crs).to_crs(epsg=4326).iloc[0]
    return merged


def get_centroid_latlon(geom) -> tuple[float, float]:
    """
    Return (longitude, latitude) of the geometry’s centroid.
    Assumes `geom` is in EPSG:4326.
    """
    centroid = geom.centroid
    return centroid.x, centroid.y


def lon_to_utm_zone(lon: float) -> int:
    """
    Given a longitude (–180..+180), compute the UTM zone number (1..60).
    Raises ValueError if the computed zone is outside 7..22.
    """
    if lon < -180 or lon > 180:
        raise ValueError(f"Longitude {lon} is outside –180..+180 range.")
    zone = int(math.floor((lon + 180) / 6) + 1)
    if zone < 7 or zone > 22:
        raise ValueError(f"Computed UTM zone {zone} is outside supported range 7–22.")
    return zone


def zone_to_epsg(zone: int) -> int:
    """
    Map a UTM zone number (7..22) to the Canadian NAD83 UTM EPSG code.
    Raises ValueError if the zone is not in the predefined mapping.
    """
    try:
        return ZONE_TO_EPSG[zone]
    except KeyError:
        raise ValueError(f"No EPSG configured for UTM zone {zone}.")


def pick_canadian_utm_epsg(aoi_path: str) -> int:
    """
    Given a path to an AOI file, determine its centroid in EPSG:4326,
    compute the UTM zone (7..22), and return the matching EPSG code.
    Also confirm that the centroid latitude falls within Canada's typical range (41°N–84°N).

    Raises ValueError if:
    - The AOI has no CRS.
    - The centroid longitude yields a zone outside 7..22.
    - No EPSG mapping exists for the identified zone.
    - The centroid latitude is outside Canada’s lat range.
    """
    geom = load_aoi_geometry(aoi_path)
    lon, lat = get_centroid_latlon(geom)

    # Confirm latitude is within Canada (approx. 38°N to 84°N)
    lat_min, lat_max = 38.0, 84.0
    if lat < lat_min or lat > lat_max:
        raise ValueError(f"AOI centroid latitude {lat:.2f}° is outside Canada’s range ({lat_min}°–{lat_max}° N).")

    zone = lon_to_utm_zone(lon)
    epsg = zone_to_epsg(zone)
    return epsg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python epsg_selector.py /path/to/aoi_file.{geojson,kml,shp}")
        sys.exit(1)

    aoi_file = sys.argv[1]
    try:
        selected_epsg = pick_canadian_utm_epsg(aoi_file)
        print(f"Auto‐selected EPSG code: {selected_epsg}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
