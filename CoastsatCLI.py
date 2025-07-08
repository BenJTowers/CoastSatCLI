import os
import json
import shutil
import subprocess
import typer
import tkinter as tk
import geopandas as gpd
from tkinter import filedialog, messagebox
from pathlib import Path
from datetime import datetime
from site_initialize import initialize_single_site
from reference_shoreline_and_transect_builder import generate_transects_along_line

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
        
def choose_file_multiple(title: str = "Select AOI file(s)") -> list[str]:
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.update()
        filepaths = filedialog.askopenfilenames(title=title, filetypes=[("KML files", "*.kml")])
        root.destroy()

        if filepaths:
            typer.secho("  ✓ You selected:", fg=typer.colors.GREEN)
            for fp in filepaths:
                typer.echo(f"     • {fp}")
            return list(filepaths)

        if not typer.confirm("No files were selected. Would you like to try again?"):
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

def prompt_and_run_analysis(settings_paths: list[str]):
    """
    Prompts user to run analysis on the provided list of settings.json paths.
    Handles both single and batch site cases.
    """
    typer.echo("\n→ Would you like to begin the full analysis now?")
    root_msg = tk.Tk()
    root_msg.withdraw()
    root_msg.attributes("-topmost", True)
    run_now = messagebox.askyesno(
        "Run Analysis",
        f"{len(settings_paths)} project(s) initialized. Begin analysis now?"
    )
    root_msg.destroy()

    if not run_now:
        typer.secho("  ⚠️  Analysis skipped by user.", fg=typer.colors.YELLOW)
        return

    for settings_path in settings_paths:
        sitename = Path(settings_path).parent.name
        cmd = [
            "python",
            "Complete_Analysis.py",
            "--config", str(settings_path)
        ]
        typer.echo(f"\n→ Running analysis for {sitename}: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            typer.secho(f"  ✓ {sitename} completed successfully!", fg=typer.colors.GREEN)
        else:
            typer.secho(f"  ❌ {sitename} failed with exit code {result.returncode}.", fg=typer.colors.RED)

def get_transect_settings_from_user() -> dict:
    if typer.confirm("→ Do you want to customize transect settings?", default=False):
        spacing = typer.prompt("  Enter transect spacing (m)", default=100.0)
        length = typer.prompt("  Enter transect length (m)", default=200.0)
        offset_ratio = typer.prompt("  Enter transect offset ratio (e.g., 0.75 = 75% over water and 25% over land)", default=0.75)
        skip_threshold = typer.prompt("  Enter transect skip threshold (m)", default=300.0)
    else:
        spacing = 100.0
        length = 200.0
        offset_ratio = 0.75
        skip_threshold = 300.0

    return {
        "transect_spacing": spacing,
        "transect_length": length,
        "transect_offset_ratio": offset_ratio,
        "transect_skip_threshold": skip_threshold
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
    typer.echo("\n→ Step 4: Select your FES2022 YAML configuration (no copy will be made).")
    fes_config_path = choose_file("Select FES2022 YAML file", filetypes=[("YAML files", "*.yaml;*.yml")])
    typer.secho(f"  ✓ FES2022 config set to {fes_config_path}\n", fg=typer.colors.GREEN)

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
                fes_config_path=fes_config_path,
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
            fes_config_path=fes_config_path,
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
    transect_settings = None

    # Step 3: Override options
    typer.echo("→ Step 2: Would you like to override any of the following inputs?")

    if typer.confirm("  • Reference shoreline file?", default=False):
        ref_path = choose_file("Select new reference shoreline", filetypes=[("GeoJSON", "*.geojson"), ("Shapefile", "*.shp")])
        config["inputs"]["reference_shoreline"] = str(Path(ref_path).resolve())
        regenerate_transects = True

    if typer.confirm("  • Transects file?", default=False):
        transects_path = choose_file("Select new transects file", filetypes=[("GeoJSON", "*.geojson"), ("Shapefile", "*.shp")])
        config["inputs"]["transects"] = str(Path(transects_path).resolve())
        regenerate_transects = False  # Don't regenerate if user chose a file manually

    if typer.confirm("  • Change transect settings?", default=False):
        transect_settings = get_transect_settings_from_user()
        regenerate_transects = True

    # Step 4: Optionally clear previous outputs
    output_dir = Path(os.path.join(base_dir, "outputs"))
    if output_dir.exists() and typer.confirm("\nDo you want to clear previous outputs (this will remap the shorelines)?", default=False):
        cleared = []
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
                cleared.append(item.name)
        typer.secho(f"  ✓ Cleared {len(cleared)} files from output directory.", fg=typer.colors.YELLOW)

    # Step 5: Regenerate transects if needed
    if regenerate_transects:
        typer.echo("\n→ Regenerating transects with updated settings…")
        try:
            shoreline_path = Path(os.path.join(base_dir, config["inputs"]["reference_shoreline"]))
            shoreline_gdf = gpd.read_file(shoreline_path)
            reference_projected = shoreline_gdf.to_crs(epsg=config.get("output_epsg"))
            transects_gdf = generate_transects_along_line(
                reference_projected,
                spacing=transect_settings.get("transect_spacing", 100),
                length=transect_settings.get("transect_length", 200),
                offset_ratio=transect_settings.get("transect_offset_ratio", 0.75),
                skip_threshold=transect_settings.get("transect_skip_threshold", 300)
            )
            transects_path = Path(os.path.join(base_dir, config["inputs"]["transects"]))
            transects_gdf.to_file(transects_path, driver="GeoJSON")
            typer.secho(f"  ✓ Transects regenerated and saved to {transects_path}", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"❌ Failed to regenerate transects: {e}", fg=typer.colors.RED)
            raise typer.Exit()

    # Step 6: Prompt to run analysis
    typer.echo("\n→ Would you like to run the full analysis now?")
    root_msg = tk.Tk()
    root_msg.withdraw()
    root_msg.attributes("-topmost", True)
    run_now = messagebox.askyesno("Run Analysis", f"Rerun analysis for {config['inputs']['sitename']}?")
    root_msg.destroy()

    if not run_now:
        typer.secho("  ⚠️  Rerun cancelled by user.", fg=typer.colors.YELLOW)
        return

    cmd = [
        "python",
        "Complete_Analysis.py",
        "--config", str(settings_path)
    ]
    typer.echo(f"\n→ Running analysis: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        typer.secho(f"  ✓ Rerun for {config['inputs']['sitename']} completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"  ❌ Rerun failed with exit code {result.returncode}.", fg=typer.colors.RED)




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
