import json
from shapely.geometry import Polygon
from pyproj import Geod

# GeoJSON input
geojson_data = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              -61.81500619084096,
              45.13067192521163
            ],
            [
              -61.81500619084096,
              45.06645365216258
            ],
            [
              -61.638816669334446,
              45.06645365216258
            ],
            [
              -61.638816669334446,
              45.13067192521163
            ],
            [
              -61.81500619084096,
              45.13067192521163
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}
# Extract coordinates
coords = geojson_data["features"][0]["geometry"]["coordinates"][0]

# Separate into lons and lats for geodesic area calculation
lons, lats = zip(*coords)

# Use WGS84 ellipsoid
geod = Geod(ellps="WGS84")
area_m2, _ = geod.polygon_area_perimeter(lons, lats)

# Convert to square kilometers and output
area_km2 = abs(area_m2) / 1_000_000
print(f"Area of the rectangle: {area_km2:,.2f} square kilometers")
