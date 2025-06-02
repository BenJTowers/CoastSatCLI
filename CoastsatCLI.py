import os
import json
import shutil
import typer
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

app = typer.Typer(help="CoastSat CLI: initialize projects and inspect outputs.", add_completion=False)

def choose_file(
    title: str = "Select a file",
    filetypes: tuple = (("All files", "*.*"),)
) -> str:
    """
    Open a file‐selection dialog and return a validated path.
    Ensures the dialog appears above all windows on the current monitor.
    Repeats until the user picks something or exits.
    """
    while True:
        root = tk.Tk()
        root.withdraw()
        # Make the dialog topmost and focused on current monitor
        root.attributes("-topmost", True)
        root.lift()
        root.update()
        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        if filepath:
            typer.echo(f"  Selected: {filepath}")
            return filepath
        if not typer.confirm("No file chosen. Try again?"):
            typer.echo("Operation cancelled.")
            raise typer.Exit()

def choose_folder(title: str = "Select a folder") -> str:
    """
    Open a folder‐selection dialog and return a validated path.
    Ensures the dialog appears above all windows on the current monitor.
    Repeats until the user picks something or exits.
    """
    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.lift()
        root.update()
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        if folder:
            typer.echo(f"  Selected: {folder}")
            return folder
        if not typer.confirm("No folder chosen. Try again?"):
            typer.echo("Operation cancelled.")
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

@app.command("init", help="Create a new CoastSat project (AOI, shoreline, transects, FES + output structure).")
def init():
    # 1) Select project folder
    typer.echo("→ Select the base directory where your new project will live.")
    project_dir = choose_folder("Choose base project directory")

    # 2) Ask for sitename
    sitename = typer.prompt("Enter a short project name (e.g., tuk)", default="").strip().lower()
    while not sitename:
        sitename = typer.prompt("Project name cannot be empty. Please enter a short name").strip().lower()
    
    # 2a) Ask for output EPSG (default 3156)
    epsg_input = typer.prompt("Enter desired output EPSG code", default="3156").strip()
    try:
        output_epsg = int(epsg_input)
    except ValueError:
        typer.secho(f"  [!] '{epsg_input}' is not a valid integer. Using default EPSG=3156.", fg=typer.colors.YELLOW)
        output_epsg = 3156

    # 3) Build paths
    site_dir   = os.path.join(project_dir, sitename)
    input_dir  = os.path.join(site_dir, "inputs")
    output_dir = os.path.join(site_dir, "outputs")

    typer.echo(f"\n— Creating project structure under:  {site_dir}")
    os.makedirs(site_dir, exist_ok=True)

    # 4) Prompt for AOI KML
    typer.echo("\n→ Select your AOI KML file:")
    aoi_path = choose_file("Select AOI KML", filetypes=[("KML files", "*.kml")])
    aoi_final = copy_and_rename(
        aoi_path,
        os.path.join(input_dir, "aoi"),
        f"{sitename}_aoi.kml"
    )

    # 5) Prompt for reference GeoJSON
    typer.echo("\n→ Select your reference shoreline GeoJSON:")
    ref_path = choose_file("Select reference shoreline", filetypes=[("GeoJSON files", "*.geojson")])
    ref_final = copy_and_rename(
        ref_path,
        os.path.join(input_dir, "reference"),
        f"{sitename}_ref.geojson"
    )

    # 6) Prompt for transects GeoJSON
    typer.echo("\n→ Select your transects GeoJSON:")
    transects_path = choose_file("Select transects file", filetypes=[("GeoJSON files", "*.geojson")])
    transects_final = copy_and_rename(
        transects_path,
        os.path.join(input_dir, "transects"),
        f"{sitename}_transects.geojson"
    )

    # 7) Prompt for FES YAML (no copy)
    typer.echo("\n→ Select your FES2022 YAML configuration (no copy will be made):")
    fes_config_path = choose_file("Select FES2022 YAML file", filetypes=[("YAML files", "*.yaml;*.yml")])

    # 8) Create output subfolders
    typer.echo(f"\n— Creating output subfolders under:  {output_dir}")
    for sub in ("shorelines", "plots", "time_series"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # 9) Build settings.json
    settings = {
        "inputs": {
            "sitename": sitename,
            "aoi_path": os.path.relpath(aoi_final, start=site_dir),
            "reference_shoreline": os.path.relpath(ref_final, start=site_dir),
            "transects": os.path.relpath(transects_final, start=site_dir),
            "fes_config": fes_config_path
        },
        "output_dir": os.path.relpath(output_dir, start=site_dir),
        "output_epsg": output_epsg
    }

    settings_path = os.path.join(site_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    typer.secho("\n✅ Project initialized successfully!", fg=typer.colors.GREEN)
    typer.echo(f"  • Sitename           : {sitename}")
    typer.echo(f"  • AOI (rel. path)    : {settings['inputs']['aoi_path']}")
    typer.echo(f"  • Reference GeoJSON  : {settings['inputs']['reference_shoreline']}")
    typer.echo(f"  • Transects GeoJSON  : {settings['inputs']['transects']}")
    typer.echo(f"  • FES YAML (abs. path): {settings['inputs']['fes_config']}")
    typer.echo(f"  • Output directory   : {settings['output_dir']}\n")

@app.command("show", help="Show the directory tree under the project's output folder.")
def show(
    config: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to your project’s settings.json"
    )
):
    """
    Walk the 'output_dir' defined in settings.json and display subfolders/files.
    """
    from pathlib import Path

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