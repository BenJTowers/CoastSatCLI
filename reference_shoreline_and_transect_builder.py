import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge
import numpy as np


def clip_shoreline_to_aoi(shoreline_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the portion of the shoreline that intersects with the AOI."""
    if shoreline_gdf.crs != aoi_gdf.crs:
        aoi_gdf = aoi_gdf.to_crs(shoreline_gdf.crs)
    return gpd.overlay(shoreline_gdf, aoi_gdf, how='intersection')


def generate_transects_along_line(clipped_gdf: gpd.GeoDataFrame, spacing=50, length=200, offset_ratio=0.5, skip_threshold=300.0) -> gpd.GeoDataFrame:
    """Generate perpendicular transects at fixed spacing along the dissolved shoreline.

    offset_ratio: proportion of transect extending inland (e.g., 0.25 = 25% inland, 75% seaward)
    """
    transects = []
    names = []

    # Dissolve all shoreline features into connected lines
    merged = linemerge(clipped_gdf.unary_union)

    # Ensure we work with a list of LineStrings
    if merged.geom_type == "MultiLineString":
        lines = list(merged.geoms)
    elif merged.geom_type == "LineString":
        lines = [merged]
    else:
        return gpd.GeoDataFrame(geometry=transects, crs=clipped_gdf.crs)

    transect_id = 1  # Start naming from 1

    # Generate transects across all lines
    for line in lines:
        total_length = line.length

        # Skip short lines below threshold (e.g., small islands)
        if total_length < skip_threshold:
            continue

        dist = 0
        while dist <= total_length:
            point = line.interpolate(dist)
            normal = _get_normal(line, dist, window=200.0)
            if np.linalg.norm(normal) == 0:
                dist += spacing
                continue

            # Calculate points based on offset ratio
            inland = offset_ratio * length
            seaward = (1 - offset_ratio) * length
            p1 = (point.x - normal[0] * inland, point.y - normal[1] * inland)
            p2 = (point.x + normal[0] * seaward, point.y + normal[1] * seaward)
            transects.append(LineString([p2, p1]))  # origin = land side now
            names.append(f"transect_{transect_id:03d}")  # Zero-padded names
            transect_id += 1

            dist += spacing

    return gpd.GeoDataFrame({'name': names, 'geometry': transects}, crs=clipped_gdf.crs)



def _get_normal(line: LineString, distance: float, window: float = 200.0):
    """Calculate a unit normal vector averaged over a larger window to smooth out local irregularities."""
    half_window = window / 2
    start = max(distance - half_window, 0)
    end = min(distance + half_window, line.length)

    p1 = line.interpolate(start)
    p2 = line.interpolate(end)
    dx, dy = p2.x - p1.x, p2.y - p1.y
    normal = np.array([-dy, dx])
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else np.array([0, 0])
