# CoastSat Project Toolkit

This repository contains two primary components:

1. **CoastSat CLI (`coastsat_cli.py`)**
   A simple, interactive command‐line interface (CLI) that helps you scaffold a new CoastSat project directory—with all input files properly organized—and inspect the generated outputs.

2. **Complete Analysis Script (`Complete_Analysis.py`)**
   The full CoastSat processing pipeline, adapted to load a `settings.json` (created by the CLI) and run shoreline detection, intersection, tide correction, and slope estimation.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Directory Structure](#directory-structure)
3. [Installing Dependencies](#installing-dependencies)
4. [Using the CoastSat CLI (`coastsat_cli.py`)](#using-the-coastsat-cli)

   * [Initialize a New Project](#initialize-a-new-project)
   * [Inspect the Output Folder](#inspect-the-output-folder)
5. [Running the Complete Analysis (`Complete_Analysis.py`)](#running-the-complete-analysis)

   * [Preparing Your Environment](#preparing-your-environment)
   * [Execution Example](#execution-example)
6. [Configuration File (`settings.json`)](#configuration-file-settingsjson)
7. [File Explanations](#file-explanations)
8. [Troubleshooting Tips](#troubleshooting-tips)

---

## Prerequisites

* **Python 3.8+**
* A Conda or virtualenv environment is highly recommended.
* GDAL/proj4 (for GeoPandas/pyproj).
* The following Python packages (see “Installing Dependencies” below):

  * `numpy`
  * `pandas`
  * `scipy`
  * `matplotlib`
  * `geopandas`
  * `shapely`
  * `pyproj`
  * `pyfes`
  * `typer`
  * `scikit‐image`
  * `joblib`
  * `coastsat` (the CoastSat Python library)

---

## Directory Structure

When you use the CLI to initialize a new project, it will create a folder hierarchy that looks like this:

```
<project_root>/
└── <sitename>/
    ├── inputs/
    │   ├── aoi/
    │   │   └── <sitename>_aoi.kml
    │   ├── reference/
    │   │   └── <sitename>_ref.geojson
    │   └── transects/
    │       └── <sitename>_transects.geojson
    ├── outputs/
    │   ├── shorelines/
    │   ├── plots/
    │   └── time_series/
    ├── settings.json
    └── coastsat_cli.py       (if you clone or copy it here)
```

* **`inputs/`**

  * `aoi/`: User’s KML file defining the Area of Interest.
  * `reference/`: Reference‐shoreline GeoJSON for QC and buffering.
  * `transects/`: Transect GeoJSON for computing cross-shore distances.

* **`outputs/`**

  * `shorelines/`: (Optional) subfolder for detected shorelines or intermediate files.
  * `plots/`: Figures generated during analysis (e.g. tide timeseries, slope spectra).
  * `time_series/`: CSVs with raw and tidally corrected transect time series.

* **`settings.json`**
  A JSON file created by the CLI that points to all inputs (KML, GeoJSONs, FES YAML) and the relative `output_dir`. Used by `Complete_Analysis.py`.

---

## Installing Dependencies

1. **Create or activate a conda environment** (recommended):

   ```bash
   conda create -n coastsat_env python=3.9
   conda activate coastsat_env
   ```

2. **Install Python packages**:

   ```bash
   pip install numpy pandas scipy matplotlib geopandas shapely pyproj pyfes typer scikit-image joblib coastsat
   ```

   Or, if you have a `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify versions** (especially for GeoPandas/pyproj/GDAL):

   ```bash
   python -c "import geopandas; print('GeoPandas', geopandas.__version__)"
   python -c "import pyproj; print('pyproj', pyproj.__version__)"
   python -c "import pyfes; print('pyfes', pyfes.__version__)"
   ```

4. **Ensure FES2022 files** (YAML, tide grids) are available somewhere on disk. The CLI will ask for a path to that YAML (it does not copy it).

---

## Using the CoastSat CLI (`coastsat_cli.py`)

The CLI helps you scaffold a new project folder with all required inputs and write a `settings.json`.

### 1. Initialize a New Project

```bash
python coastsat_cli.py init
```

**What happens:**

1. **Select Base Directory**
   A folder where your new project subfolder will be created.

2. **Enter Sitename**
   A short, lowercase name (e.g. `tuk`, `patty`). This name will be used to name subfolders and prefix copied files.

3. **Select AOI KML**
   A file dialog appears (above all other windows). Select your `.kml` file. The CLI copies it to `inputs/aoi/<sitename>_aoi.kml`.

4. **Select Reference Shoreline GeoJSON**
   A file dialog appears. Select a `.geojson` of your reference shoreline. Copied to `inputs/reference/<sitename>_ref.geojson`.

5. **Select Transects GeoJSON**
   A file dialog appears. Select a `.geojson` of your transects. Copied to `inputs/transects/<sitename>_transects.geojson`.

6. **Select FES2022 YAML**
   A file dialog appears. Choose your tide‐model YAML (no copy is made). The CLI stores its **absolute path** in `settings.json`.

7. **Project Structure is Created**

   ```text
   <project_root>/
   └── <sitename>/
       ├── inputs/
       │   ├── aoi/
       │   │   └── <sitename>_aoi.kml
       │   ├── reference/
       │   │   └── <sitename>_ref.geojson
       │   └── transects/
       │       └── <sitename>_transects.geojson
       ├── outputs/
       │   ├── shorelines/      (empty)
       │   ├── plots/           (empty)
       │   └── time_series/     (empty)
       └── settings.json
   ```

8. **Summary Printed**
   The CLI prints the final `settings.json` content, including relative paths to AOI, reference, transects, and the absolute FES YAML path.

---

### 2. Inspect the Output Folder

```bash
python coastsat_cli.py show --config /path/to/<sitename>/settings.json
```

or

```bash
python coastsat_cli.py show -c /path/to/<sitename>/settings.json
```

* Reads `settings.json`, resolves its `output_dir`, and prints a recursive tree of `outputs/`.
* Useful for a quick glance at which shoreline CSVs, plots, or animations have been generated.

**Example:**

```text
Output directory: C:\Users\…\project\tuk\outputs

outputs/
    shorelines/
        tuk_output_points.geojson
        mapped_shorelines.jpg
    plots/
        tuk_tide_timeseries.jpg
        0_timestep_distribution.jpg
        …
    time_series/
        transect_time_series.csv
        transect_time_series_tidally_corrected.csv
```

---

## Running the Complete Analysis (`Complete_Analysis.py`)

Once the project is initialized (and `settings.json` exists), you run the full CoastSat pipeline with:

```bash
python Complete_Analysis.py --config /path/to/<sitename>/settings.json
```

### Script Behavior

1. **Load `settings.json`**

   * Resolves relative paths to absolute.
   * Reads:

     ```json
     {
       "inputs": {
         "sitename": "tuk",
         "aoi_path": "inputs/aoi/tuk_aoi.kml",
         "reference_shoreline": "inputs/reference/tuk_ref.geojson",
         "transects": "inputs/transects/tuk_transects.geojson",
         "fes_config": "C:/…/fes2022.yaml"
       },
       "output_dir": "outputs"
     }
     ```

2. **`initial_settings(cfg)`**

   * Reads the KML polygon, computes its bounding‐rectangle envelope.
   * Derives a **centroid** (lon/lat) for tidal calculations.
   * Fetches satellite metadata from GEE (via `SDS_download.retrieve_images`).
   * Builds a `settings` dictionary with all shoreline‐extraction parameters.

3. **`batch_shoreline_detection(...)`**

   * Cloud‐masking, pansharpening, and object classification to extract shorelines from each satellite image.
   * Saves preprocessed JPGs, creates animations, and writes `tuk_output_points.geojson` to `outputs/shorelines/`.

4. **`shoreline_analysis(...)`**

   * Loads the reference GeoJSON and transects GeoJSON.
   * Reprojects transects to match the shoreline CRS, then computes cross‐shore distances at each acquisition date.
   * Saves raw time‐series CSV to `outputs/time_series/transect_time_series.csv`.

5. **`slope_estimation(...)`**

   * Loads FES2022 handlers (from the YAML).
   * Reprojects centroid to EPSG:4326, generates full tide series (1984–2025), and extracts tides at satellite dates.
   * Computes power spectrum to find the dominant tide frequency.
   * Computes per‐transect slopes, saving slope‐estimation plots in `outputs/plots/slope_estimation/`.

6. **`tidal_correction(...)`**

   * Applies tidal corrections to each transect’s time series.
   * Writes `outputs/time_series/transect_time_series_tidally_corrected.csv`.

7. **`calculate_and_save_trends(...)`**

   * Fits linear trends to each transect’s corrected time series.
   * Saves a GeoJSON of transects with their trend values to `outputs/shorelines/<sitename>_transects_with_trends.geojson`.

8. **`improved_transects_plot(...)` + `time_series_post_processing(...)`**

   * Generates final colored‐transect maps (plots), seasonal/monthly averages, and other QC figures, all saved under `outputs/plots/`.

### Example Run

```bash
conda activate coastsat_env
python Complete_Analysis.py --config C:/Users/…/project/tuk/settings.json
```

* The script prints progress messages as it processes each satellite and transect.
* At the end, you should have:

  ```
  outputs/
      shorelines/
          tuk_output_points.geojson
          tuk_transects_with_trends.geojson
      plots/
          tuk_tide_timeseries.jpg
          0_timestep_distribution.jpg
          1_tides_power_spectrum.jpg
          2_energy_curve_<transectID>.jpg
          3_slope_spectrum_<transectID>.jpg
          tuk_transects_colored_by_trend.jpg
          ...
      time_series/
          transect_time_series.csv
          transect_time_series_tidally_corrected.csv
  ```

---

## Configuration File (`settings.json`)

The `settings.json` created by `coastsat_cli.py init` has this structure:

```json
{
  "inputs": {
    "sitename": "tuk",
    "aoi_path": "inputs/aoi/tuk_aoi.kml",
    "reference_shoreline": "inputs/reference/tuk_ref.geojson",
    "transects": "inputs/transects/tuk_transects.geojson",
    "fes_config": "C:/full/path/to/fes2022.yaml"
  },
  "output_dir": "outputs"
}
```

* **`inputs.aoi_path`** (string)
  Relative path (from `<sitename>/`) to the AOI KML.

* **`inputs.reference_shoreline`** (string)
  Relative path to the reference‐shoreline GeoJSON.

* **`inputs.transects`** (string)
  Relative path to the transects GeoJSON.

* **`inputs.fes_config`** (string)
  Absolute path to the FES2022 YAML config. (Large file—no copy.)

* **`output_dir`** (string)
  Relative path (from `<sitename>/`) to your output folder, where all generated files will be dumped.

---

## File Explanations

* **`coastsat_cli.py`**

  * A Typer‐based CLI with two commands:

    1. `init`: interactively gathers all input file paths (KML, GeoJSONs, FES YAML), copies them into a defined folder structure, and writes `settings.json`.
    2. `show`: reads an existing `settings.json` and prints the contents of its `output_dir`.

* **`Complete_Analysis.py`**

  * The main processing pipeline. It:

    1. Loads `settings.json` via `--config`.
    2. Builds CoastSat’s `inputs`, `settings`, and `metadata`.
    3. Detects shorelines, runs transect intersections, computes tides with FES, corrects time series, and fits shoreline‐change trends.
    4. Saves intermediate files (GeoJSON, CSV) and many QC/plot images.

* **`requirements.txt`** (optional)
  Lists all required packages. Installing with `pip install -r requirements.txt` ensures you have the correct versions.

---

## Troubleshooting Tips

1. **Missing Dependencies**

   * If you get `ModuleNotFoundError: No module named 'numpy'` (or others), verify that your active Python interpreter is the same one where you installed packages.
   * In VS Code, ensure your integrated terminal is using the correct Conda environment (`coastsat_env`).

2. **File Dialog Hidden on Multi‐Monitor Setups**

   * We set `root.attributes("-topmost", True)` and `root.lift()` in `choose_file/choose_folder` to force the dialog in front.
   * If it still appears behind other windows, verify that your window manager respects `-topmost` on Tk.

3. **All Transects Skip (“empty data”)**

   * Likely a CRS mismatch between shorelines (EPSG 3156) and transects (WGS84).
   * Confirm both are in `settings['output_epsg']` (e.g. 3156) before calling `compute_intersection_QC`.
   * You can inspect CRSes at runtime:

     ```python
     import geopandas as gpd
     tran_df = gpd.read_file(settings['inputs']['transects'])
     shore_df = gpd.read_file(<path_to_shoreline_geojson>)
     print("Transects CRS:", tran_df.crs)
     print("Shorelines CRS:", shore_df.crs)
     ```

4. **All Tide Values = NaN**

   * Ensure your FES `centroid` is in **EPSG:4326 ( longitude, latitude )** and lies in open water.
   * Example debug:

     ```python
     lon, lat = settings['inputs']['centroid']
     print("Tide @TestDate (lon,lat):", ocean_tide(test_date, lon, lat))
     print("Tide @TestDate (lat,lon):", ocean_tide(test_date, lat, lon))
     ```
   * If returned values are still NaN, choose a new centroid inside the ocean (not on land or outside model coverage).

5. **Missing or Malformed `settings.json`**

   * If `show` or `Complete_Analysis.py` can’t find or parse `settings.json`, double‐check file paths and JSON formatting.

---

## Summary

1. **Use the CLI (`coastsat_cli.py init`)** to create a clean project structure and `settings.json`.
2. **Run the analysis (`Complete_Analysis.py --config settings.json`)** to detect shorelines, compute transect distances, apply tide corrections, and save results.
3. **Inspect outputs** via `coastsat_cli.py show --config settings.json` or by browsing `outputs/` directly.

This README provides an end‐to‐end roadmap for setting up, running, and inspecting your CoastSat project. If you encounter any issues not covered here, consult the debug tips or create an issue in the repository.
