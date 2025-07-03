import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.ops import linemerge
import numpy as np


def clip_shoreline_to_aoi(shoreline_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the portion of the shoreline that intersects with the AOI."""
    if shoreline_gdf.crs != aoi_gdf.crs:
        aoi_gdf = aoi_gdf.to_crs(shoreline_gdf.crs)
    return gpd.overlay(shoreline_gdf, aoi_gdf, how='intersection')


def closest_intersection(segment: LineString, shoreline_union, origin_point: Point):
    inter = segment.intersection(shoreline_union)
    if inter.is_empty:
        return None
    elif isinstance(inter, Point):
        dist = origin_point.distance(inter)
        return dist if dist > 1e-6 else None
    elif isinstance(inter, (MultiPoint, GeometryCollection)):
        min_dist = None
        for geom in inter.geoms:
            if isinstance(geom, Point):
                d = origin_point.distance(geom)
                if d > 1e-6:
                    if min_dist is None or d < min_dist:
                        min_dist = d
        return min_dist
    return None


def generate_transects_along_line(clipped_gdf: gpd.GeoDataFrame, spacing=50, length=200, offset_ratio=0.75, skip_threshold=300.0, min_valid_transect=50.0) -> gpd.GeoDataFrame:
    """Generate perpendicular transects at fixed spacing along the dissolved shoreline.

    offset_ratio: proportion of transect extending landward (e.g., 0.25 = 25% landward, 75% seaward)
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

    shoreline_union = clipped_gdf.unary_union
    transect_id = 1  # Start naming from 1

    # Generate transects across all lines
    for line in lines:
        total_length = line.length

        # Skip short lines below threshold (e.g., small islands)
        if total_length < skip_threshold:
            continue

        is_loop = line.is_ring

        dist = 0
        while dist <= total_length:
            point = line.interpolate(dist)
            normal = _get_normal(line, dist, window=200.0, force_loop=is_loop)
            if np.linalg.norm(normal) == 0:
                dist += spacing
                continue

            # Calculate base points based on offset ratio
            inland = offset_ratio * length
            seaward = (1 - offset_ratio) * length

            # Get actual end points of the transect
            inland_pt_actual = Point(point.x - normal[0] * inland, point.y - normal[1] * inland)
            seaward_pt_actual = Point(point.x + normal[0] * seaward, point.y + normal[1] * seaward)

            # Define initial segments
            inland_segment = LineString([point, inland_pt_actual])
            seaward_segment = LineString([point, seaward_pt_actual])

            # Get closest intersections
            inland_inter_dist = closest_intersection(inland_segment, shoreline_union, point)
            seaward_inter_dist = closest_intersection(seaward_segment, shoreline_union, point)

            # Skip transect if both directions are obstructed early
            if (seaward_inter_dist and seaward_inter_dist < min_valid_transect) or \
               (inland_inter_dist and inland_inter_dist < min_valid_transect):
                dist += spacing
                continue

            # Shorten if intersections are found
            if inland_inter_dist:
                shorten_by = inland_inter_dist * 0.5
                inland_pt_actual = Point(point.x - normal[0] * shorten_by, point.y - normal[1] * shorten_by)

            if seaward_inter_dist:
                shorten_by = seaward_inter_dist * 0.5
                seaward_pt_actual = Point(point.x + normal[0] * shorten_by, point.y + normal[1] * shorten_by)

            transects.append(LineString([seaward_pt_actual, inland_pt_actual]))
            names.append(f"transect_{transect_id:03d}")  # Zero-padded names
            transect_id += 1

            dist += spacing

    return gpd.GeoDataFrame({'name': names, 'geometry': transects}, crs=clipped_gdf.crs)



def _get_normal(line: LineString, distance: float, window: float = 200.0, force_loop: bool = False):
    """Calculate a unit normal vector averaged over a window, using full loop if applicable."""
    if force_loop:
        half_window = window / 2
        start = (distance - half_window) % line.length
        end = min(distance + half_window, line.length)
        p1 = line.interpolate(start)
        p2 = line.interpolate(end)
    else:
        half_window = window / 2
        start = max(distance - half_window, 0)
        end = min(distance + half_window, line.length)
        p1 = line.interpolate(start)
        p2 = line.interpolate(end)

    dx, dy = p2.x - p1.x, p2.y - p1.y
    normal = np.array([-dy, dx])
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else np.array([0, 0])
