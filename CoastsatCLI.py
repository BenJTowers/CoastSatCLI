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
from site_initialize import initialize_single_site

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
                preloaded_shoreline=shoreline_gdf
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
        
        # Initialize single site
        site_info = initialize_single_site(
            aoi_path=aoi_src,
            sitename=sitename,
            shoreline_path=shoreline_src,
            fes_config_path=fes_config_path,
            base_dir=project_dir,
            preloaded_shoreline=shoreline_gdf
        )

        typer.secho("\n✅ Project initialized successfully!\n", fg=typer.colors.CYAN, bold=True)

        # 7) Run analysis immediately
        prompt_and_run_analysis([site_info['settings_path']])



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
