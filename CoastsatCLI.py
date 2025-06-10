import os
import json
import shutil
import subprocess
import typer
import tkinter as tk
import geopandas as gpd
from tkinter import filedialog, messagebox
from pathlib import Path
from reference_shoreline_and_transect_builder import clip_shoreline_to_aoi, generate_transects_along_line
from epsg_utils import pick_canadian_utm_epsg

app = typer.Typer(
    help="CoastSat CLI: initialize projects, run analysis, and inspect outputs.",
    add_completion=False
)

def choose_file(
    title: str = "Select a file",
    filetypes: tuple = (("All files", "*.*"),)
) -> str:
    """
    Open a file‐selection dialog and return a validated path.
    Shows a brief notice before launching the system's file explorer
    (which may take a moment to appear).
    """
    while True:
        typer.echo("  [Info] Opening your system’s file browser… please wait.")
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.update()
        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()

        if filepath:
            typer.secho(f"  ✓ You selected: {filepath}", fg=typer.colors.GREEN)
            return filepath

        if not typer.confirm("No file was selected. Would you like to try again?"):
            typer.secho("  ⚠️  Operation cancelled by user.", fg=typer.colors.YELLOW)
            raise typer.Exit()


def choose_folder(title: str = "Select a folder") -> str:
    """
    Open a folder‐selection dialog and return a validated path.
    Shows a brief notice before launching the system's folder browser
    (which may take a moment to appear).
    """
    while True:
        typer.echo("  [Info] Opening your system’s folder browser… please wait.")
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.update()
        folder = filedialog.askdirectory(title=title)
        root.destroy()

        if folder:
            typer.secho(f"  ✓ You selected: {folder}", fg=typer.colors.GREEN)
            return folder

        if not typer.confirm("No folder was selected. Would you like to try again?"):
            typer.secho("  ⚠️  Operation cancelled by user.", fg=typer.colors.YELLOW)
            raise typer.Exit()


def copy_and_rename(src_path: str, dest_folder: str, new_name: str) -> str:
    """
    Copy a file into dest_folder and rename it to new_name.
    Creates dest_folder if needed.
    """
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, new_name)
    shutil.copy2(src_path, dest_path)
    return dest_path


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
    typer.echo("  • Naming the project and specifying an EPSG code (auto‐detected)")
    typer.echo("  • Choosing AOI, reference shoreline, transects, and FES config files")
    typer.echo("  • Creating the folder structure and writing settings.json\n")

    # 1) Select project folder
    typer.echo("→ Step 1: Choose the base directory where your new project will live.")
    project_dir = choose_folder("Choose base project directory")

    # 2) Ask for sitename
    typer.echo("\n→ Step 2: Enter a short project (site) name (e.g., tuk, patty).")
    sitename = typer.prompt("  Sitename").strip().lower()
    while not sitename:
        sitename = typer.prompt("  Sitename cannot be empty. Please enter a short name").strip().lower()
    typer.secho(f"  ✓ Project name set to '{sitename}'\n", fg=typer.colors.GREEN)

    # Build directory paths
    site_dir   = os.path.join(project_dir, sitename)
    input_dir  = os.path.join(site_dir, "inputs")
    output_dir = os.path.join(site_dir, "outputs")
    typer.echo(f"→ Creating project folder under\n    {site_dir}")
    os.makedirs(site_dir, exist_ok=True)

    # 3) Prompt for AOI first, to auto-detect EPSG
    typer.echo("\n→ Step 3: Select your AOI KML file (for EPSG auto‐detection).")
    typer.echo("   (A file browser will open shortly.)")
    aoi_src = choose_file("Select AOI KML", filetypes=[("KML files", "*.kml")])
    # Auto‐select EPSG
    try:
        auto_epsg = pick_canadian_utm_epsg(aoi_src)
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

    # 4) select large coastline that includes AOI
    typer.echo("\n→ Step 4: Select your full shoreline file (GeoJSON or Shapefile).")
    shoreline_src = choose_file("Select shoreline file", filetypes=[("Shapefiles", "*.shp"), ("GeoJSON", "*.geojson")])

    typer.echo("→ Loading AOI and shoreline data…")
    shoreline_gdf = gpd.read_file(shoreline_src)
    aoi_gdf = gpd.read_file(aoi_src, driver="KML")

    # 5) create reference shoreline GeoJSON
    typer.echo("→ Clipping shoreline to AOI…")
    reference_gdf = clip_shoreline_to_aoi(shoreline_gdf, aoi_gdf)
    ref_out_path = os.path.join(input_dir, "reference", f"{sitename}_ref.geojson")
    os.makedirs(os.path.dirname(ref_out_path), exist_ok=True)
    reference_gdf.to_file(ref_out_path, driver="GeoJSON")
    typer.secho(f"  ✓ Reference shoreline saved to {ref_out_path}", fg=typer.colors.GREEN)

    # 6) generate transects GeoJSON
    typer.echo("→ Generating transects from clipped shoreline…")
    reference_projected = reference_gdf.to_crs(epsg=auto_epsg)
    transects_gdf = generate_transects_along_line(reference_projected, spacing=100, length=200, offset_ratio=0.25)
    transects_out_path = os.path.join(input_dir, "transects", f"{sitename}_transects.geojson")
    os.makedirs(os.path.dirname(transects_out_path), exist_ok=True)
    transects_gdf.to_file(transects_out_path, driver="GeoJSON")
    typer.secho(f"  ✓ Transects saved to {transects_out_path}", fg=typer.colors.GREEN)

    typer.echo("→ Copying AOI to project input folder…")
    aoi_dest = os.path.join(input_dir, "aoi", f"{sitename}_aoi.kml")
    os.makedirs(os.path.dirname(aoi_dest), exist_ok=True)
    shutil.copy2(aoi_src, aoi_dest)
    typer.secho(f"  ✓ AOI copied to {aoi_dest}", fg=typer.colors.GREEN)


    # 7) Prompt for FES YAML (no copy)
    typer.echo("\n→ Step 6: Select your FES2022 YAML configuration (no copy will be made).")
    typer.echo("   (A file browser will open shortly.)")
    fes_config_path = choose_file("Select FES2022 YAML file", filetypes=[("YAML files", "*.yaml;*.yml")])
    typer.secho(f"  ✓ FES2022 config set to {fes_config_path}\n", fg=typer.colors.GREEN)

    # 8) Create main output directory only
    typer.echo("→ Step 7: Creating main output directory")
    os.makedirs(output_dir, exist_ok=True)
    typer.secho("  ✓ Main output folder created\n", fg=typer.colors.GREEN)

    # 9) Build settings.json
    typer.echo("→ Step 8: Writing settings.json")
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
    typer.secho("  ✓ settings.json created\n", fg=typer.colors.GREEN)

    typer.secho("\n✅ Project initialized successfully!\n", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  • Sitename           : {sitename}")
    typer.echo(f"  • Output EPSG       : {auto_epsg}")
    typer.echo(f"  • AOI (rel. path)    : {settings['inputs']['aoi_path']}")
    typer.echo(f"  • Reference GeoJSON  : {settings['inputs']['reference_shoreline']}")
    typer.echo(f"  • Transects GeoJSON  : {settings['inputs']['transects']}")
    typer.echo(f"  • FES YAML (abs. path): {settings['inputs']['fes_config']}")
    typer.echo(f"  • Output directory   : {settings['output_dir']}\n")

    # 10) Ask user if they want to run analysis immediately
    typer.echo("→ Would you like to begin the full analysis now?")
    root_msg = tk.Tk()
    root_msg.withdraw()
    root_msg.attributes("-topmost", True)
    run_now = messagebox.askyesno(
        "Run Analysis",
        "settings.json created. Begin analysis now?"
    )
    root_msg.destroy()

    if run_now:
        cmd = [
            "python",
            "Complete_Analysis.py",
            "--config", str(settings_path)
        ]
        typer.echo(f"\n→ Running analysis: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            typer.secho("\n✅ Analysis completed successfully!", fg=typer.colors.GREEN)
        else:
            typer.secho(f"\n❌ Analysis failed with exit code {result.returncode}.", fg=typer.colors.RED)


@app.command("run", help="Run the full CoastSat analysis using your settings.json.")
def run(
    config: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to project settings.json"
    )
):
    """
    Invoke Complete_Analysis.py with the provided settings.json.
    """
    config_path = Path(config).expanduser().resolve()
    if not config_path.exists():
        typer.secho(f"ERROR: Cannot find config at {config_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    cmd = [
        "python",
        "Complete_Analysis.py",
        "--config", str(config_path)
    ]
    typer.echo(f"\n→ Running analysis: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        typer.secho("\n✅ Analysis completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n❌ Analysis failed with exit code {result.returncode}.", fg=typer.colors.RED)


@app.command("show", help="Show the directory tree under the project's output folder.")
def show(
    config: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to your project’s settings.json"
    )
):
    cfg_path = Path(config).expanduser().resolve()
    if not cfg_path.exists():
        typer.secho(f"ERROR: Cannot find config file at {cfg_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        typer.secho(f"ERROR: Failed to read settings.json: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    base_dir   = cfg_path.parent
    output_dir = (base_dir / cfg["output_dir"]).resolve()
    if not output_dir.exists():
        typer.secho(f"ERROR: Output folder does not exist: {output_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.secho(f"\nOutput directory: {output_dir}\n", fg=typer.colors.CYAN)
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), "").count(os.sep)
        indent = " " * 4 * level
        typer.echo(f"{indent}{os.path.basename(root)}/")
        for fname in files:
            typer.echo(f"{indent}    {fname}")
    typer.echo("")


if __name__ == "__main__":
    app()
