import math
import typer
import geopandas as gpd
from pathlib import Path
from shapely import union_all
from shapely.geometry import LineString, Point, MultiPoint, GeometryCollection
from shapely.ops import linemerge, unary_union
import numpy as np


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

def detect_or_prompt_epsg(aoi_path: str) -> int:
    """
    Attempts to detect a suitable EPSG code from an AOI file.
    If detection fails, prompts the user to enter one manually.
    
    Returns:
        EPSG code as an integer
    """
    try:
        auto_epsg = pick_canadian_utm_epsg(aoi_path)
        typer.secho(f"  ✓ Detected EPSG = {auto_epsg} from AOI centroid", fg=typer.colors.GREEN)
        return auto_epsg
    except Exception as e:
        typer.secho(f"  [!] Could not auto-select EPSG: {e}", fg=typer.colors.RED)
        if not typer.confirm("Do you want to specify an EPSG manually?"):
            raise typer.Exit()
        epsg_input = typer.prompt("  Enter EPSG code manually", default="3156").strip()
        try:
            return int(epsg_input)
        except ValueError:
            typer.secho("Invalid EPSG → exiting", fg=typer.colors.RED)
            raise typer.Exit()
        

def load_aoi_and_shoreline(aoi_path: str, shoreline_path: str, preloaded_shoreline: gpd.GeoDataFrame = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    aoi_gdf = gpd.read_file(aoi_path, driver="KML")
    shoreline_gdf = preloaded_shoreline if preloaded_shoreline is not None else gpd.read_file(shoreline_path)
    return aoi_gdf, shoreline_gdf

def clip_shoreline_to_aoi(shoreline_gdf: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the portion of the shoreline that intersects with the AOI."""
    if shoreline_gdf.crs != aoi_gdf.crs:
        aoi_gdf = aoi_gdf.to_crs(shoreline_gdf.crs)
    return gpd.overlay(shoreline_gdf, aoi_gdf, how='intersection')

def create_and_save_reference_shoreline(
    shoreline_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    output_path: str
) -> gpd.GeoDataFrame:
    reference = clip_shoreline_to_aoi(shoreline_gdf, aoi_gdf)
    reference.to_file(output_path, driver="GeoJSON")
    return reference

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


def generate_transects_along_line(clipped_gdf: gpd.GeoDataFrame, spacing=50, length=200, offset_ratio=0.75, skip_threshold=500.0, min_valid_transect=50.0) -> gpd.GeoDataFrame:
    """Generate perpendicular transects at fixed spacing along the dissolved shoreline.

    offset_ratio: proportion of transect extending landward (e.g., 0.25 = 25% landward, 75% seaward)
    """
    transects = []
    names = []

    # 🔧 Robust dissolve / linemerge
    geom = union_all(clipped_gdf.geometry)

    if geom.is_empty:
        return gpd.GeoDataFrame(geometry=transects, crs=clipped_gdf.crs)

    if geom.geom_type == "LineString":
        lines = [geom]
    elif geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    elif geom.geom_type == "GeometryCollection":
        line_like = [g for g in geom.geoms if g.geom_type in ["LineString", "MultiLineString"]]
        if not line_like:
            return gpd.GeoDataFrame(geometry=transects, crs=clipped_gdf.crs)
        merged = linemerge(line_like)
        lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    else:
        return gpd.GeoDataFrame(geometry=transects, crs=clipped_gdf.crs)

    shoreline_union = geom
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

def generate_and_save_transects(
    reference_gdf: gpd.GeoDataFrame,
    epsg: int,
    spacing: float,
    length: float,
    offset_ratio: float,
    skip_threshold: float,
    output_path: str
) -> gpd.GeoDataFrame:
    projected = reference_gdf.to_crs(epsg=epsg)
    transects = generate_transects_along_line(
        projected,
        spacing=spacing,
        length=length,
        offset_ratio=offset_ratio,
        skip_threshold=skip_threshold
    )
    transects.to_file(output_path, driver="GeoJSON")
    return transects


def regenerate_transects_from_config(
    base_dir: Path,
    config: dict,
    transect_settings: dict
) -> gpd.GeoDataFrame:
    """
    Regenerates transects using updated settings and the existing reference shoreline.

    Parameters:
        base_dir (Path): The root project directory.
        config (dict): The loaded settings.json config.
        transect_settings (dict): Dictionary of transect parameters.

    Returns:
        gpd.GeoDataFrame: The regenerated transects GeoDataFrame.

    Raises:
        typer.Exit: If the regeneration fails.
    """
    try:
        shoreline_path = (base_dir / config["inputs"]["reference_shoreline"]).resolve()
        shoreline_gdf = gpd.read_file(shoreline_path)
        reference_projected = shoreline_gdf.to_crs(epsg=config["output_epsg"])

        transects_gdf = generate_transects_along_line(
            reference_projected,
            spacing=transect_settings.get("transect_spacing", 100),
            length=transect_settings.get("transect_length", 200),
            offset_ratio=transect_settings.get("transect_offset_ratio", 0.75),
            skip_threshold=transect_settings.get("transect_skip_threshold", 300)
        )

        transects_path = (base_dir / config["inputs"]["transects"]).resolve()
        transects_gdf.to_file(transects_path, driver="GeoJSON")
        typer.secho(f"  ✓ Transects regenerated and saved to {transects_path}", fg=typer.colors.GREEN)
        return transects_gdf

    except Exception as e:
        typer.secho(f"❌ Failed to regenerate transects: {e}", fg=typer.colors.RED)
        raise typer.Exit()