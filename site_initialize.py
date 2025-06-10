import os
import json
import shutil
import subprocess
import typer
import time
import tkinter as tk
import geopandas as gpd
from tkinter import filedialog, messagebox
from pathlib import Path
from reference_shoreline_and_transect_builder import clip_shoreline_to_aoi, generate_transects_along_line
from epsg_utils import pick_canadian_utm_epsg

def initialize_single_site(
    aoi_path: str,
    sitename: str,
    shoreline_path: str,
    fes_config_path: str,
    base_dir: str,
    preloaded_shoreline: gpd.GeoDataFrame = None
) -> dict:
    """
    Creates a CoastSat project from a single AOI and a larger shoreline.

    Parameters:
    - aoi_path: path to a KML file defining the AOI
    - sitename: name of the site (e.g., 'tuk', 'site_001')
    - shoreline_path: path to a full shoreline file (GeoJSON or SHP)
    - fes_config_path: path to the FES2022 YAML config file
    - base_dir: path to the root directory where the project folder will be created
    - preloaded_shoreline: optional, a preloaded GeoDataFrame of the full shoreline to avoid reloading it repeatedly in batch mode

    Returns:
    - Dictionary with sitename, EPSG, and settings.json path
    """
    #start timer
    start = time.time()
    
    typer.echo("→ Building CoastSat site from AOI and shoreline...")
    site_dir = os.path.join(base_dir, sitename)
    input_dir = os.path.join(site_dir, "inputs")
    output_dir = os.path.join(site_dir, "outputs")
    typer.echo(f"→ Creating project folder under\n    {site_dir}")

    aoi_dest = os.path.join(input_dir, f"{sitename}_aoi.kml")
    ref_out_path = os.path.join(input_dir, f"{sitename}_ref.geojson")
    transects_out_path = os.path.join(input_dir, f"{sitename}_transects.geojson")

    os.makedirs(os.path.dirname(aoi_dest), exist_ok=True)
    os.makedirs(os.path.dirname(ref_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(transects_out_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Detect EPSG from AOI
    try:
        auto_epsg = pick_canadian_utm_epsg(aoi_path)
        typer.secho(f"  ✓ Detected EPSG = {auto_epsg} from AOI centroid", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"  [!] Could not auto-select EPSG: {e}", fg=typer.colors.RED)
        if not typer.confirm("Do you want to specify an EPSG manually?"):
            raise typer.Exit()
        epsg_input = typer.prompt("  Enter EPSG code manually", default="3156").strip()
        try:
            auto_epsg = int(epsg_input)
        except ValueError:
            typer.secho("Invalid EPSG → exiting", fg=typer.colors.RED)
            raise typer.Exit()

    # Read AOI
    typer.echo("→ Loading AOI and shoreline data…")
    aoi_gdf = gpd.read_file(aoi_path, driver="KML")

    # Load or reuse shoreline
    if preloaded_shoreline is not None:
        shoreline_gdf = preloaded_shoreline
    else:
        shoreline_gdf = gpd.read_file(shoreline_path)

    # Clip shoreline
    typer.echo("→ Creating reference shoreline…")
    reference_gdf = clip_shoreline_to_aoi(shoreline_gdf, aoi_gdf)
    reference_gdf.to_file(ref_out_path, driver="GeoJSON")
    typer.secho(f"  ✓ Reference shoreline saved to {ref_out_path}", fg=typer.colors.GREEN)

    # Generate transects
    typer.echo("→ Generating transects…")
    reference_projected = reference_gdf.to_crs(epsg=auto_epsg)
    transects_gdf = generate_transects_along_line(reference_projected, spacing=100, length=200, offset_ratio=0.75)
    transects_gdf.to_file(transects_out_path, driver="GeoJSON")
    typer.secho(f"  ✓ Transects saved to {transects_out_path}", fg=typer.colors.GREEN)

    # Copy AOI
    typer.echo("→ Copying AOI to project input folder…")
    shutil.copy2(aoi_path, aoi_dest)
    typer.secho(f"  ✓ AOI copied to {aoi_dest}", fg=typer.colors.GREEN)

    # Create settings.json
    settings = {
        "inputs": {
            "sitename": sitename,
            "aoi_path": os.path.relpath(aoi_dest, start=site_dir),
            "reference_shoreline": os.path.relpath(ref_out_path, start=site_dir),
            "transects": os.path.relpath(transects_out_path, start=site_dir),
            "fes_config": fes_config_path
        },
        "output_dir": os.path.relpath(output_dir, start=site_dir),
        "output_epsg": auto_epsg
    }

    settings_path = os.path.join(site_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    typer.secho("\n✅ Project initialized successfully!\n", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  • Sitename            : {sitename}")
    typer.echo(f"  • Output EPSG         : {auto_epsg}")
    typer.echo(f"  • AOI (rel. path)     : {settings['inputs']['aoi_path']}")
    typer.echo(f"  • Reference GeoJSON   : {settings['inputs']['reference_shoreline']}")
    typer.echo(f"  • Transects GeoJSON   : {settings['inputs']['transects']}")
    typer.echo(f"  • FES YAML (abs. path): {settings['inputs']['fes_config']}")
    typer.echo(f"  • Output directory    : {settings['output_dir']}\n")
    typer.echo(f"  ⏱ Site processed in {time.time() - start:.2f} seconds")

    return {
        "sitename": sitename,
        "settings_path": settings_path,
        "epsg": auto_epsg,
        "output_dir": output_dir
    }

