import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime, timedelta
import pytz
import pyfes
import os

# --- Config ---
aoi_dir = r"C:\Users\btowers\Documents\Tests\Centroid_test"  # Folder with multiple KML files
fes_yaml = r"C:\Users\btowers\Documents\Full tides\ClippedCanada\fes2022.yaml"

# Load FES handlers
print("[Info] Loading FES2022 config...")
handlers = pyfes.load_config(fes_yaml)
ocean_tide = handlers['tide']
load_tide = handlers['radial']
print("[Info] FES configuration loaded successfully.")

# Define test time range
date_range = [
    pytz.utc.localize(datetime(2020, 1, 1)),
    pytz.utc.localize(datetime(2020, 1, 2)),
]
timestep = 3600  # one hour

# -- Helpers --

def get_fes_flags(coord):
    try:
        dates = []
        date = date_range[0]
        while date <= date_range[1]:
            dates.append(date)
            date += timedelta(seconds=timestep)

        dates_np = np.array(dates, dtype='datetime64[us]')
        lons = np.full(len(dates_np), coord[0])
        lats = np.full(len(dates_np), coord[1])

        _, _, flags_ocean = pyfes.evaluate_tide(ocean_tide, dates_np, lons, lats)
        _, _, flags_load = pyfes.evaluate_tide(load_tide, dates_np, lons, lats)

        return flags_ocean, flags_load

    except Exception as e:
        print(f"[Error] Evaluation failed at {coord}: {e}")
        return None, None


def try_centroids(geom):
    centroid = list(geom.centroid.coords)[0]
    minx, miny, maxx, maxy = geom.bounds
    edge_centroids = [
        ((minx + maxx)/2, miny),
        ((minx + maxx)/2, maxy),
        (minx, (miny + maxy)/2),
        (maxx, (miny + maxy)/2),
    ]
    test_coords = [("center", centroid)] + list(zip(["bottom", "top", "left", "right"], edge_centroids))

    best_coord = None
    best_score = float('inf')

    for label, (lon_raw, lat) in test_coords:
        lon_fes = lon_raw + 360 if lon_raw < 0 else lon_raw
        coord = [lon_fes, lat]
        print(f"  [Check] {label} point: {coord}")
        flags_ocean, flags_load = get_fes_flags(coord)

        if flags_ocean is None or flags_load is None:
            continue

        if np.all(flags_ocean == 0):
            score = np.sum(flags_load != 0)
            print(f"    ✅ Ocean=0 | Load flag score: {score}")
        elif np.all(flags_ocean <= 2):
            score = 10 + np.sum(flags_load != 0)
            print(f"    ⚠️ Ocean=2 allowed | Load flag score: {score}")
        else:
            score = 100 + np.sum(flags_ocean) + np.sum(flags_load)
            print(f"    ❌ Poor flags | Score: {score}")

        if score < best_score:
            best_score = score
            best_coord = [lon_raw, lat]

    if best_coord:
        lon_adj = best_coord[0] + 360 if best_coord[0] < 0 else best_coord[0]
        return [lon_adj, best_coord[1]]

    return None


# -- Batch Run Over AOIs --
kml_files = [f for f in os.listdir(aoi_dir) if f.lower().endswith('.kml')]

for kml in kml_files:
    print(f"\n=== Processing AOI: {kml} ===")
    try:
        path = os.path.join(aoi_dir, kml)
        aoi_gdf = gpd.read_file(path)
        aoi_geom = aoi_gdf.geometry.iloc[0]

        centroid = try_centroids(aoi_geom)

        if centroid:
            print(f"✅ Working centroid for {kml}: {centroid}")
        else:
            print(f"❌ No valid centroid found for {kml}")
    except Exception as e:
        print(f"[Error] Failed to process {kml}: {e}")

