import typer
from pathlib import Path
from dialogs import choose_file

app = typer.Typer()

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

@app.command()
def test():
    typer.echo("\n🔍 Testing tide correction config collection...\n")
    tide_config = get_tide_correction_settings()

    typer.echo("\n✅ Collected Tide Correction Config:\n")
    for k, v in tide_config.items():
        typer.echo(f"  {k}: {v}")

if __name__ == "__main__":
    test()  # <- directly call test()
