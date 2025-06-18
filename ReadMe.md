# CoastSat Project Toolkit

Automate shoreline extraction and analysis using CoastSat via a simple CLI.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
   4.1 [Initialize a Project](#41-initialize-a-project)
   4.2 [Run the Complete Analysis](#42-run-the-complete-analysis)
   4.3 [Inspect Output](#43-inspect-output)
5. [CLI Commands](#cli-commands)
6. [Settings File (settings.json)](#settings-file-settingsjson)
7. [Directory Structure](#directory-structure)
8. [Contributing](#contributing)
9. [FES2022 Tide Model Setup](#fes2022-tide-model-setup)

---

## Overview

This project is part of the Canadian Coastal Change project at NRCan, where we seek to gather a nationwide dataset explaining shoreline evolution around Canada’s coast over the past 40+ years. This CLI tool enables standardized analysis of AOIs using shoreline transects that track positional change. Landsat satellite imagery and tide correction via the FES2022 model are used to build consistent, validated shoreline time series.

This CLI builds on the [CoastSat](https://github.com/kvos/coastsat) project, an open-source toolkit that uses 40+ years of Landsat and Sentinel-2 imagery to monitor shoreline change via Google Earth Engine.

---

## Prerequisites

* Python 3.8 or higher (tested on 3.11).
* [Miniforge](https://github.com/conda-forge/miniforge) — a minimal installer for Conda (with optional Mamba support).
* Recommended: [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/en/latest/) as your environment/package manager.
* Key Python libraries:

  * `typer`, `tkinter`, `geopandas`, `shapely`, `matplotlib`, `scikit-image`, `pyfes`, `pyyaml`, `imageio`, `imageio-ffmpeg`, etc.

> 💡 Miniforge is a lightweight way to install Conda or Mamba with conda-forge as the default channel. We recommend using it to avoid conflicts with the default Anaconda distribution.

Before proceeding, verify key imports:

```bash
python - <<EOF
import json, geopandas, typer, tkinter, pyfes, numpy
print("All required libraries imported successfully")
EOF
```

---

## Installation

1. **Clone this repository** and change into its folder:

   ```bash
   git clone https://github.com/BenJTowers/CoastSatCLI
   cd coastsat-cli
   ```

2. **Create and activate a Conda environment** (recommended):

   ```bash
   conda create -n coastsat python=3.11
   conda activate coastsat
   conda install geopandas gdal
   conda install earthengine-api scikit-image matplotlib astropy notebook
   pip install pyqt5 imageio-ffmpeg
   conda install pyfes pyyaml typer
   ```

   Or, using Mamba (if you’ve installed it via Miniforge):

   ```bash
   mamba create -n coastsat python=3.11
   mamba activate coastsat
   mamba install geopandas gdal earthengine-api scikit-image matplotlib astropy notebook pyfes pyyaml typer
   pip install pyqt5 imageio-ffmpeg
   ```

3. **Install CoastSat CLI package**:

   ```bash
   pip install -e .
   ```

This installs “coastsat” on your PATH, giving you access to CLI commands such as `coastsat init`, `coastsat run`, and `coastsat show`.

> ⚠️ If you encounter installation issues, try:
>
> ```bash
> conda clean --all
> conda update conda
> ```

---

## 🌍 Step 2: Set Up Google Earth Engine

This project uses the **Google Earth Engine (GEE)** API to download satellite imagery (e.g., Landsat, Sentinel-2). To enable this:

### 2.1 Sign Up for Earth Engine

👉 [https://signup.earthengine.google.com](https://signup.earthengine.google.com)

Use your **Google account** to request access.

### 2.2 Install the Google Cloud SDK

Download: [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

After installation, run:

```bash
gcloud auth application-default login
```

> ⚠️ This ensures your Earth Engine credentials persist across runs.

✅ Once authenticated, you're ready to initialize a CoastSat project.

---

## Quick Start

Below is a minimal workflow. After installation, you will:

1. Initialize a new project folder (GUI dialogs).
2. Run the CoastSat analysis pipeline.
3. Inspect results under the output directory.

### 4.1 Initialize a Project

```bash
coastsat init
```

You will be prompted to:

* Select a **base directory** (system file‐browser dialog).
* Enter a **project name** (e.g., `tuk`).
* Select a **shoreline shapefile**.
* Confirm or override the **EPSG code** (auto‐detected).
* Select one or more **AOI files** (GeoJSON, KML, or SHP).
* For each AOI, CoastSat will:

  * Clip the shoreline.
  * Generate a reference shoreline.
  * Generate transects.
  * Save all files in a new project folder.

At the end, you will be asked whether to **immediately run the analysis**. You may choose "Yes" to begin processing right away, or "No" to run it later using the CLI.

Structure:

```
<base_dir>/<sitename>/
├── inputs/
│   ├── shoreline/...
│   ├── aoi/...
│   ├── reference/...
│   └── transects/...
├── settings.json
└── outputs/
```

### 4.2 Run the Complete Analysis

```bash
coastsat run --config path/to/<sitename>/settings.json
```

This runs:

```bash
python Complete_Analysis.py --config path/to/settings.json
```

Which downloads imagery, extracts shorelines, applies FES2022 tide correction, generates slope plots, and more.

### 4.3 Inspect Output

```bash
coastsat show --config path/to/<sitename>/settings.json
```

Sample output:

```
outputs/
├─ <sitename>_output_points.geojson
├─ mapped_shorelines.jpg
├─ plots/
│   ├─ tide_timeseries.jpg
│   ├─ energy_curve_<transect>.jpg
├─ time_series/
│   ├─ transect_time_series.csv
│   └─ transect_time_series_tidally_corrected.csv
```

---

## CLI Commands

```
Usage: coastsat [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  init  Create a new CoastSat project (shoreline, AOIs, transects).
  run   Run the full CoastSat analysis using your settings.json.
  show  Show the directory tree under the project’s output folder.
```

---

## Settings File (settings.json)

```json
{
  "inputs": {
    "sitename": "tuk",
    "shoreline_path": "inputs/shoreline/CANCOAST.geojson",
    "aoi_paths": ["inputs/aoi/tuk_aoi.geojson"],
    "reference_shoreline": "inputs/reference/tuk_ref.geojson",
    "transects": "inputs/transects/tuk_transects.geojson"
  },
  "output_dir": "outputs",
  "output_epsg": 3156,
  "fes_config": "/absolute/path/to/fes2022.yaml"
}
```

---

## Directory Structure

```
<base_dir>/<sitename>/
├─ inputs/
│   ├─ shoreline/
│   ├─ aoi/
│   ├─ reference/
│   └─ transects/
├─ outputs/
│   ├─ plots/
│   └─ time_series/
└─ settings.json
```

---

## Contributing

1. Fork this repository and create a branch:

   ```bash
   git checkout -b feature/your-feature
   ```
2. Commit your changes:

   ```bash
   git commit -m "Add new feature"
   ```
3. Push to your fork:

   ```bash
   git push origin feature/your-feature
   ```
4. Open a Pull Request with a description of your changes.

Please follow the existing style:

* Use `typer` for all CLI interactions.
* Keep each function focused and concise.
* Write clear docstrings for any new helper functions.
* Add tests or manual-validation instructions for new features.

---

## FES2022 Tide Model Setup

To apply tidal correction using the FES2022 model, you must download the required tide constituent files and provide a YAML configuration file that points to them.

1. **Download FES2022 files**:
   You must acquire the official FES2022 tide model from AVISO. Registration is required:
   👉 [https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html)

2. **Create a YAML config file** with paths to your local files:

```yaml
fes:
  path: /path/to/fes2022
  grid: fes2022_ocean_tide_loading_grid.nc
  constituents:
    - M2
    - S2
    - N2
    - K1
    - O1
    - Q1
    - K2
    - M4
```

3. **Reference this config in your `settings.json`** using the `fes_config` key.

> 🧠 Tip: For efficiency, you may crop your NetCDF files to a small bounding box around your AOIs before analysis.