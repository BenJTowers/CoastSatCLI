import tkinter as tk
from tkinter import filedialog, messagebox
import typer
from pathlib import Path
import subprocess
import json

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

def get_tide_correction_settings() -> dict:
    """
    Ask the user which tide correction method to use (CSV or FES),
    and gather all required inputs for that method using GUI dialogs.
    """
    typer.echo("→ How would you like to apply tidal correction?")
    typer.echo("   • csv : Use a tide level CSV file (e.g., from tide gauge or external source)")
    typer.echo("   • fes : Use the built-in FES2022 tide model (requires YAML config)")

    method = typer.prompt("  Enter your choice [csv/fes]", default="fes").strip().lower()
    settings = {"method": method}

    if method == "csv":
        typer.echo("\n→ Tide CSV selected.")
        settings["tide_csv_path"] = choose_file("Select Tide CSV", filetypes=[("CSV files", "*.csv")])
        settings["reference_elevation"] = float(typer.prompt("Reference elevation (e.g., 0 for MSL)", default="0"))
        settings["beach_slope"] = float(typer.prompt("Beach slope (e.g., 0.1)", default="0.1"))
    elif method == "fes":
        typer.echo("\n→ FES tide model selected.")
        settings["fes_config"] = choose_file("Select FES2022 YAML config", filetypes=[("YAML files", "*.yaml;*.yml")])
    else:
        typer.secho("❌ Invalid method. Choose 'csv' or 'fes'.", fg=typer.colors.RED)
        raise typer.Exit()

    return settings
        
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
        typer.echo(f"\n→ Running analysis for {sitename}...")
        exit_code = run_analysis_from_config(Path(settings_path))
        if exit_code == 0:
            typer.secho(f"  ✓ {sitename} completed successfully!", fg=typer.colors.GREEN)
        else:
            typer.secho(f"  ❌ {sitename} failed with exit code {exit_code}.", fg=typer.colors.RED)

def run_analysis_from_config(config_path: Path) -> int:
    """
    Runs the appropriate analysis script based on the settings.json tide config.

    Parameters:
        config_path (Path): Path to the settings.json file.

    Returns:
        int: Exit code from the subprocess (0 for success).
    """
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        typer.secho(f"ERROR: Cannot find config at {config_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with open(config_path) as f:
        config = json.load(f)

    inputs = config.get("inputs", {})
    if "tide_csv_path" in inputs:
        script = "Complete_Analysis_CSV.py"
    else:
        script = "Complete_Analysis.py"

    cmd = ["python", script, "--config", str(config_path)]
    typer.echo(f"\n→ Running analysis: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode
