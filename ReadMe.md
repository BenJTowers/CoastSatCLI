# CoastSat Project Toolkit

Automate shoreline extraction and analysis using CoastSat via a simple CLI.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation & Prerequisites](#installation--prerequisites) 
3. [Quick Start](#quick-start)
   3.1 [Initialize a Project](#31-initialize-a-project)
   3.2 [Run the Complete Analysis](#32-run-the-complete-analysis)
   3.3 [Inspect Output](#33-inspect-output)
   3.4 [Rerun an Existing Project with Updated Inputs](#34-rerun-an-existing-project-with-updated-inputs)
4. [CLI Commands](#cli-commands)
5. [Settings File (settings.json)](#settings-file-settingsjson)
6. [Directory Structure](#directory-structure)
7. [Contributing](#contributing)

---

## Overview

This project is part of the Canadian Coastal Change project at NRCan, where we seek to gather a nationwide dataset explaining shoreline evolution around Canada’s coast over the past 40+ years. This CLI tool enables standardized analysis of AOIs using shoreline transects that track positional change. Landsat satellite imagery and tide correction via the FES2022 model are used to build consistent, validated shoreline time series.

---

## Installation & Prerequisites

This CLI relies on several geospatial and scientific Python libraries. **You must install and use a Conda or Mamba environment.**

### Required Tools

* **Miniforge** (recommended): A minimal installer for Conda or Mamba with conda-forge as default.
* **Conda** or **Mamba**: For environment/package management.
* **Python**: Version 3.8 or higher (tested on 3.11).

<details>
<summary>What is Miniforge and how to install it?</summary>

[Miniforge](https://github.com/conda-forge/miniforge) is a lightweight installer that enables you to manage packages through **conda** or **mamba**, with full support for the **conda-forge** channel. This improves compatibility with geospatial and scientific packages.

**Install Miniforge:**

1. Go to [Miniforge GitHub Releases](https://github.com/conda-forge/miniforge#miniforge3).
2. Download the installer for your operating system.
3. Follow the installation instructions for your platform.

Once installed, you can use `conda` or `mamba` from the terminal.

</details>

### Required Python Packages

* `geopandas`, `shapely`, `matplotlib`, `scikit-image`, `rasterio`, `gdal`, `pyfes`, `pyyaml`, `typer`, `imageio`, `imageio-ffmpeg`, `tkinter`

### Installation Steps

1. **Clone this repository** and change into its folder:

   ```bash
   git clone https://github.com/BenJTowers/CoastSatCLI
   cd coastsat-cli
   ```

2. **Create and activate a Conda environment**:

   ```bash
   conda create -n coastsat python=3.11
   conda activate coastsat
   conda install geopandas gdal rasterio
   conda install earthengine-api scikit-image matplotlib astropy notebook
   pip install pyqt5 imageio-ffmpeg
   conda install pyfes pyyaml typer
   ```

   Or, using Mamba:

   ```bash
   mamba create -n coastsat python=3.11
   mamba activate coastsat
   mamba install geopandas gdal rasterio earthengine-api scikit-image matplotlib astropy notebook pyfes pyyaml typer
   pip install pyqt5 imageio-ffmpeg
   ```

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
## 🌍 Step 3: FES2022 Tide Model setup

This project uses the **FES2022 global tide model** to correct shoreline positions based on modeled tides at your AOI location. You’ll need to install the `pyfes` package and configure the model by downloading the required data and linking it in a `fes2022.yaml` file.

### 3.1: Install `pyfes`

If you're using a clean CoastSat environment as described above, `pyfes` should already be installed.

If not, run:

```bash
conda install -c conda-forge pyfes
```

If you encounter conflicts (common with older environments), create a new Conda environment using the steps in the main installation section.

---

### 3.2: Download FES2022 NetCDF Files

To use the tide model, download the required ocean and load tide components (NetCDF files).

#### Option A: Direct Download (Recommended)

Use this SFTP link in a client like **WinSCP**, **Cyberduck**, or another SFTP tool:

```
sftp://ftp-access.aviso.altimetry.fr:2221/auxiliary/tide_model/fes2022b
```

**Download the following folders:**

* `/ocean_tide`
* `/load_tide`

You should end up with \~34 `.nc` files total across both folders. Do **not** use the `/ocean_tide_extrapolate` folder.

#### Option B: Manual Access via AVISO Website

1. Go to: [https://www.aviso.altimetry.fr](https://www.aviso.altimetry.fr)
2. Create a user account and log in.
3. Fill out the [FES model data access form](https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html) (check **FES - Oceanic Tide Heights**).
4. After approval, access your downloads via [My Products](https://www.aviso.altimetry.fr/en/my-aviso-plus/my-products.html).
5. Use the SFTP link above to download the data.

---

### 3.3: Download and Configure `fes2022.yaml`

Download the config file:

👉 [fes2022.yaml from GitHub](https://github.com/CNES/aviso-fes/tree/main/data/fes2022b)

Place it in the same directory as your downloaded `/ocean_tide` and `/load_tide` folders.

Then, open `fes2022.yaml` in a text editor and **update the paths** to point to the exact locations of the NetCDF files on your system. Example (Windows-style):

```yaml
cartesian:
  amplitude: amplitude
  latitude: lat
  longitude: lon
  paths:
    2N2: C:\Users\ben\Documents\fes2022b\load_tide\2n2_fes2022.nc
    K1:  C:\Users\ben\Documents\fes2022b\load_tide\k1_fes2022.nc
    M2:  C:\Users\ben\Documents\fes2022b\ocean_tide\m2_fes2022.nc
```

Make sure both `radial` and `cartesian` sections are fully updated. Use find-and-replace to quickly update the root path if needed.

---

### 3.4: Test the Setup (Optional)

To verify that the tide model is correctly configured:

```bash
conda activate coastsat
python
```

Then in Python:

```python
import pyfes
handlers = pyfes.load_config("C:/path/to/fes2022.yaml")
```

If no error is returned after a few minutes, your setup is complete.

😃 Your CoastSat project can now model tides globally using FES2022.


## Quick Start

Below is a minimal workflow. After installation, you will:

1. Initialize a new project folder (GUI dialogs).
2. Run the CoastSat analysis pipeline.
3. Inspect results under the output directory.

### 3.1 Initialize a Project

```bash
python cli/coastsatcli.py init
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

### 3.2 Run the Complete Analysis

```bash
python cli/coastsatcli.py run --config path/to/<sitename>/settings.json
```

This runs:

```bash
python Complete_Analysis.py --config path/to/settings.json
```

Which downloads imagery, extracts shorelines, applies FES2022 tide correction, generates slope plots, and more.

### 3.3 Rerun an Existing Project with Updated Inputs

If your initial CoastSat results were unsatisfactory (e.g., due to poor transect placement or outdated shoreline data), the `site-rerun` command allows you to reprocess a site without creating a new project from scratch.

This command lets you:
- Replace the **reference shoreline** file
- Replace or regenerate the **transects** (with optional updated settings)
- **Clear previous outputs** to remap shorelines from scratch
- Launch the full analysis again using the updated inputs

#### ✅ Example usage:
```bash
python cli/coastsatcli.py site-rerun
```

You’ll be prompted to:
1. Select an existing `settings.json` file
2. Choose whether to replace or regenerate shoreline/transect files
3. (Optionally) clear existing outputs
4. Rerun the analysis using the updated inputs

**Note:** The original `settings.json` is not modified beyond resolving input paths. Only the input files are updated or regenerated as needed.

---

## CLI Commands

```
Usage: python cli/coastsatcli.py [OPTIONS] COMMAND [ARGS]...

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
