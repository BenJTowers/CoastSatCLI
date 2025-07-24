import os
import typer
import time
import json
import geopandas as gpd
from pathlib import Path
from dialogs import choose_file, choose_folder, choose_file_multiple, get_transect_settings_from_user, prompt_and_run_analysis, run_analysis_from_config, get_tide_correction_settings
from file_utils import setup_project_directories, clear_output_directory
from geo_utils import detect_or_prompt_epsg, load_aoi_and_shoreline, create_and_save_reference_shoreline, generate_and_save_transects, regenerate_transects_from_config
import shutil

app = typer.Typer(
    help="CoastSat CLI: initialize projects, run analysis, and inspect outputs.",
    add_completion=False
)

def initialize_single_site(
    aoi_path: str,
    sitename: str,
    shoreline_path: str,
    tide_config: dict,
    base_dir: str,
    preloaded_shoreline: gpd.GeoDataFrame = None,
    transect_spacing: float = 100.0,
    transect_length: float = 200.0,
    transect_offset_ratio: float = 0.75,
    transect_skip_threshold: float = 300.0
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
    paths = setup_project_directories(base_dir, sitename)
    site_dir = paths["site_dir"]
    output_dir = paths["output_dir"]
    aoi_dest = paths["aoi_dest"]
    ref_out_path = paths["ref_out_path"]
    transects_out_path = paths["transects_out_path"]
    typer.echo(f"→ Creating project folder under\n    {site_dir}")

    # Detect EPSG from AOI
    auto_epsg = detect_or_prompt_epsg(aoi_path)

    
    # Read AOI
    typer.echo("→ Loading AOI and shoreline data…")
    aoi_gdf, shoreline_gdf = load_aoi_and_shoreline(aoi_path, shoreline_path, preloaded_shoreline)

    # Clip shoreline
    typer.echo("→ Creating reference shoreline…")
    reference_gdf = create_and_save_reference_shoreline(
    shoreline_gdf=shoreline_gdf,
    aoi_gdf=aoi_gdf,
    output_path=ref_out_path)
    typer.secho(f"  ✓ Reference shoreline saved to {ref_out_path}", fg=typer.colors.GREEN)

    # Generate transects
    typer.echo("→ Generating transects…")
    transects_gdf = generate_and_save_transects(
    reference_gdf=reference_gdf,
    epsg=auto_epsg,
    spacing=transect_spacing,
    length=transect_length,
    offset_ratio=transect_offset_ratio,
    skip_threshold=transect_skip_threshold,
    output_path=transects_out_path
)
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
        },
        "output_dir": os.path.relpath(output_dir, start=site_dir),
        "output_epsg": auto_epsg
    }

    # Add tide config
    if tide_config["method"] == "fes":
        settings["inputs"]["fes_config"] = tide_config["fes_config_path"]
    elif tide_config["method"] == "csv":
        settings["inputs"].update({
            "tide_csv_path": tide_config["tide_csv_path"],
            "reference_elevation": tide_config["reference_elevation"],
            "beach_slope": tide_config["beach_slope"]
        })

    settings_path = os.path.join(site_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    typer.secho("\n✅ Project initialized successfully!\n", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  • Sitename            : {sitename}")
    typer.echo(f"  • Output EPSG         : {auto_epsg}")
    typer.echo(f"  • AOI (rel. path)     : {settings['inputs']['aoi_path']}")
    typer.echo(f"  • Reference GeoJSON   : {settings['inputs']['reference_shoreline']}")
    typer.echo(f"  • Transects GeoJSON   : {settings['inputs']['transects']}")
    if tide_config["method"] == "fes":
        typer.echo(f"  • FES YAML (abs. path): {settings['inputs']['fes_config']}")
    elif tide_config["method"] == "csv":
        typer.echo(f"  • Tide CSV (abs. path): {settings['inputs']['tide_csv_path']}")
        typer.echo(f"  • Ref. Elevation      : {settings['inputs']['reference_elevation']}")
        typer.echo(f"  • Beach Slope         : {settings['inputs']['beach_slope']}")
    typer.echo(f"  • Output directory    : {settings['output_dir']}\n")
    typer.echo(f"  ⏱ Site processed in {time.time() - start:.2f} seconds")

    return {
        "sitename": sitename,
        "settings_path": settings_path,
        "epsg": auto_epsg,
        "output_dir": output_dir
    }

@app.command(
    "init",
    help="Create a new CoastSat project (AOI, shoreline, transects, FES + output structure)."
)
def init():
    """
    Step‐by‐step project initialization:
      1) Pick a base directory.
      2) Enter a project (site) name.
      3) Auto‐select output EPSG from AOI or let user override.
      4) Prompt separately for AOI KML, reference GeoJSON, and transects GeoJSON.
      5) Copy those three inputs under a single progress bar.
      6) Prompt for FES YAML.
      7) Create the main output folder and write settings.json.
      8) Prompt to run the full analysis immediately.
    """
    typer.secho("\n=== CoastSat Project Initialization ===", fg=typer.colors.CYAN, bold=True)
    typer.echo("You will be guided through:")
    typer.echo("  • Selecting a base directory for your project")
    typer.echo("  • Naming the project)")
    typer.echo("  • Choosing AOI's")
    typer.echo("  • Creating the folder structure and writing a settings.json\n")

    # 1) Select project folder
    typer.echo("→ Step 1: Choose the base directory where your new project will live.")
    project_dir = choose_folder("Choose base project directory")

    # 2) Ask for sitename
    typer.echo("\n→ Step 2: Enter a project (site) name (e.g., tuk, patty).")
    sitename = typer.prompt("  Sitename").strip().lower()
    while not sitename:
        sitename = typer.prompt("  Sitename cannot be empty. Please enter a short name").strip().lower()
    typer.secho(f"  ✓ Project name set to '{sitename}'\n", fg=typer.colors.GREEN)

    # 3) select large coastline that includes AOI
    typer.echo("\n→ Step 3: Select your full shoreline file (GeoJSON or Shapefile).")
    shoreline_src = choose_file("Select shoreline file", filetypes=[("Shapefiles", "*.shp"), ("GeoJSON", "*.geojson")])

    # 4) load shoreline data
    typer.echo("→ Loading shoreline data (this may take a few minutes)…")
    try:
        shoreline_gdf = gpd.read_file(shoreline_src)
    except Exception as e:
        typer.secho(f"❌ Failed to read shoreline file: {e}", fg=typer.colors.RED)
        raise typer.Exit()

    # 5) Prompt for FES YAML (no copy)
    # 5) Prompt for tide correction method (CSV or FES)
    typer.echo("\n→ Step 4: Choose a tidal correction method.")
    tide_config = get_tide_correction_settings()

    is_batch = typer.confirm("Would you like to initialize multiple AOIs as a batch?", default=False)

    if is_batch:
        # 6) Prompt for multiple AOI KML files
        typer.echo("\n→ Step 5: Select your AOI KML files.")
        aoi_srcs = choose_file_multiple("Select AOI KML files")
        transect_settings = get_transect_settings_from_user()
        settings_paths = []
        for i, aoi_src in enumerate(aoi_srcs):
            full_sitename = f"{sitename}_{str(i+1).zfill(3)}"
            typer.secho(f"\nInitializing AOI {i+1} of {len(aoi_srcs)}: {full_sitename}", fg=typer.colors.CYAN, bold=True)
            site_info = initialize_single_site(
                aoi_path=aoi_src,
                sitename=full_sitename,
                shoreline_path=shoreline_src,
                tide_config=tide_config,
                base_dir=project_dir,
                preloaded_shoreline=shoreline_gdf,
                transect_spacing=transect_settings["transect_spacing"],
                transect_length=transect_settings["transect_length"],
                transect_offset_ratio=transect_settings["transect_offset_ratio"],
                transect_skip_threshold=transect_settings["transect_skip_threshold"]
            )
            settings_paths.append(site_info['settings_path'])

        
        typer.secho("\n✅ All AOIs initialized successfully!\n", fg=typer.colors.CYAN, bold=True)
        typer.secho("\n📦 Batch Initialization Summary:", fg=typer.colors.CYAN, bold=True)
        for i, settings_path in enumerate(settings_paths):
            sitename = Path(settings_path).parent.name
            output_dir = Path(settings_path).parent / "outputs"
            typer.echo(f"  • {sitename:<12} → {output_dir}")

        # 7) Run analysis immediately
        prompt_and_run_analysis(settings_paths)

    else:
        # 6) Prompt for single AOI KML file
        typer.echo("\n→ Step 5: Select your AOI KML file.")
        aoi_src = choose_file("Select AOI KML", filetypes=[("KML files", "*.kml")])
        transect_settings = get_transect_settings_from_user()
        # Initialize single site
        site_info = initialize_single_site(
            aoi_path=aoi_src,
            sitename=sitename,
            shoreline_path=shoreline_src,
            tide_config=tide_config,
            base_dir=project_dir,
            preloaded_shoreline=shoreline_gdf,
            transect_spacing=transect_settings["transect_spacing"],
            transect_length=transect_settings["transect_length"],
            transect_offset_ratio=transect_settings["transect_offset_ratio"],
            transect_skip_threshold=transect_settings["transect_skip_threshold"]
        )

        typer.secho("\n✅ Project initialized successfully!\n", fg=typer.colors.CYAN, bold=True)

        # 7) Run analysis immediately
        prompt_and_run_analysis([site_info['settings_path']])

@app.command("site-rerun", help="Rerun a previously initialized CoastSat site with updated inputs.")

def site_rerun():
    """
    Allows rerunning an existing CoastSat site with the option to override shoreline,
    transects, or transect settings. This is useful for refining results after an
    unsatisfactory run. Note: The config file remains unchanged in structure — only 
    input files are replaced or regenerated.
    """
    typer.secho("\n=== CoastSat Site Rerun ===", fg=typer.colors.CYAN, bold=True)

    # Step 1: Select settings.json
    typer.echo("→ Step 1: Select the existing project's settings.json file.")
    settings_path = choose_file("Select settings.json", filetypes=[("JSON files", "*.json")])
    base_dir = Path(settings_path).parent

    # Step 2: Load config
    try:
        with open(settings_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        typer.secho(f"❌ Failed to load settings.json: {e}", fg=typer.colors.RED)
        raise typer.Exit()

    typer.secho(f"  ✓ Loaded settings for site: {config['inputs']['sitename']}\n", fg=typer.colors.GREEN)

    # Track what needs regeneration
    regenerate_transects = False
    transect_settings = {
        "transect_spacing": 100.0,
        "transect_length": 200.0,
        "transect_offset_ratio": 0.75,
        "transect_skip_threshold": 500.0
    }

    # Step 3: Override options
    typer.echo("→ Step 2: Would you like to override any of the following inputs?")

    if typer.confirm("  • Reference shoreline file?", default=False):
        ref_path = choose_file("Select new reference shoreline", filetypes=[("GeoJSON", "*.geojson"), ("Shapefile", "*.shp")])
        ref_path = Path(ref_path).expanduser().resolve()
        target_path = Path(os.path.join(base_dir,config["inputs"]["reference_shoreline"])).resolve()
        shutil.copy2(ref_path, target_path)
        typer.secho(f"  ✓ Replaced existing reference shoreline with: {ref_path.name}", fg=typer.colors.GREEN)
        regenerate_transects = True

    if typer.confirm("  • Transects file?", default=False):
        transects_path = choose_file("Select new transects file", filetypes=[("GeoJSON", "*.geojson"), ("Shapefile", "*.shp")])
        config["inputs"]["transects"] = str(Path(transects_path).resolve())
        regenerate_transects = False  # Don't regenerate if user chose a file manually

    if typer.confirm("  • Change transect settings?", default=False):
        transect_settings = get_transect_settings_from_user()
        regenerate_transects = True

    # Step 4: Optionally clear previous outputs
    output_dir = Path(os.path.join(base_dir, "outputs")).resolve()
    clear_output_directory(output_dir, prompt=True)

    # Step 5: Regenerate transects if needed
    if regenerate_transects:
        regenerate_transects_from_config(
            base_dir=base_dir,
            config=config,
            transect_settings=transect_settings
        )

    # Step 6: Prompt to run analysis
    prompt_and_run_analysis([settings_path])

@app.command("run", help="Run the full CoastSat analysis using your settings.json.")
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to project settings.json")
):
    """
    Invoke Complete_Analysis.py with the provided settings.json.
    """

    exit_code = run_analysis_from_config(Path(config))
    if exit_code == 0:
        typer.secho("\n✅ Analysis completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n❌ Analysis failed with exit code {exit_code}.", fg=typer.colors.RED)

if __name__ == "__main__":
    app()