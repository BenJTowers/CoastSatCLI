import geopandas as gpd
from shapely.geometry import box, LineString
import numpy as np

# === SETTINGS ===
TARGET_AREA = 100_000_000  # 100 km^2 in square meters
OVERLAP_DIST = 250  # meters
MIN_SIDE_LENGTH = 4000  # meters

# === LOAD COASTLINE ===
coastline_gdf = gpd.read_file(r"C:\Users\btowers\Documents\Tests\AutoAOI\Test1.shp")  # Should be in EPSG:3978
assert coastline_gdf.crs.to_epsg() == 3978, "Coastline must be in EPSG:3978 (meters)"

# === INITIALIZE ===
aois = []

for line in coastline_gdf.geometry:
    if line.is_empty or not isinstance(line, LineString):
        continue

    coords = list(line.coords)
    i = 0
    while i < len(coords):
        minx = maxx = coords[i][0]
        miny = maxy = coords[i][1]
        j = i + 1

        while j < len(coords):
            x, y = coords[j]
            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)

            width = maxx - minx
            height = maxy - miny
            area = width * height

            # Respect min side length
            if width < MIN_SIDE_LENGTH or height < MIN_SIDE_LENGTH:
                j += 1
                continue

            if area >= TARGET_AREA:
                aoi = box(minx, miny, maxx, maxy)
                aois.append(aoi)

                # Move to next point with overlap
                overlap_ratio = OVERLAP_DIST / line.length
                i = int(j - (line.length * overlap_ratio))
                break

            j += 1
        else:
            # If we run out of points, stop
            break

        i = j

# === EXPORT ===
aoi_gdf = gpd.GeoDataFrame(geometry=aois, crs=coastline_gdf.crs)
aoi_gdf.to_file("generated_aois.shp")
