from pathlib import Path
import os
import typer

def setup_project_directories(base_dir: str, sitename: str) -> dict:
    """
    Sets up the folder structure and file paths for a CoastSat project.

    Returns a dictionary containing paths for:
    - site_dir
    - input_dir
    - output_dir
    - aoi_dest
    - ref_out_path
    - transects_out_path
    """
    site_dir = os.path.join(base_dir, sitename)
    input_dir = os.path.join(site_dir, "inputs")
    output_dir = os.path.join(site_dir, "outputs")

    aoi_dest = os.path.join(input_dir, f"{sitename}_aoi.kml")
    ref_out_path = os.path.join(input_dir, f"{sitename}_ref.geojson")
    transects_out_path = os.path.join(input_dir, f"{sitename}_transects.geojson")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return {
        "site_dir": site_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "aoi_dest": aoi_dest,
        "ref_out_path": ref_out_path,
        "transects_out_path": transects_out_path
    }

def clear_output_directory(output_dir: Path, prompt: bool = True) -> list[str]:
    """
    Clears all files in the given output directory after confirming with the user.

    Parameters:
        output_dir (Path): The path to the output directory.
        prompt (bool): If True, prompt the user for confirmation. If False, skip confirmation.

    Returns:
        list[str]: Names of files that were deleted.
    """
    if not output_dir.exists():
        return []

    if prompt:
        confirm = typer.confirm("\nDo you want to clear previous outputs (this will remap the shorelines)?", default=False)
        if not confirm:
            return []

    cleared = []
    for item in output_dir.iterdir():
        if item.is_file():
            item.unlink()
            cleared.append(item.name)

    typer.secho(f"  ✓ Cleared {len(cleared)} files from output directory.", fg=typer.colors.YELLOW)
    return cleared