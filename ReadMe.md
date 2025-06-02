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
9. [License](#license)

---

## Overview

The CoastSat Project Toolkit provides a step-by-step Command-Line Interface (CLI) for:

1. Creating a standardized project folder (AOI, reference shoreline, transects, FES config, output).
2. Auto-detecting the AOI’s CRS (EPSG) or falling back to a default.
3. Running the full CoastSat analysis (Complete\_Analysis.py) with a single command.
4. Browsing and visualizing resulting shorelines, time series, and plots.

This README explains how to install, initialize, run, and inspect results in a concise manner.

---

## Prerequisites

* Python 3.8 or higher (tested on 3.9/3.10).
* Conda (recommended) or another virtual-environment manager.
* Required packages (listed in `environment.yml` or `requirements.txt`), including:

  * `typer`
  * `tkinter`
  * `geopandas`
  * `shapely`
  * `pandas`, `numpy`, `matplotlib`
  * CoastSat dependencies (`pyfes`, `scikit-image`, etc.)

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
   git clone https://github.com/yourusername/coastsat-cli.git  
   cd coastsat-cli
   ```

2. **Create and activate a Conda environment** (if you don’t already have one):

   ```bash
   conda env create -f environment.yml  
   conda activate coastsat
   ```

   Or, using pip in a virtual environment:

   ```bash
   python -m venv venv  
   source venv/bin/activate   # (Linux/macOS)  
   venv\Scripts\activate     # (Windows)  
   pip install -r requirements.txt
   ```

3. **Install CoastSat CLI package**:

   ```bash
   pip install -e .
   ```

   This installs “coastsatcli” on your PATH, giving you access to `coastsat init`, `coastsat run`, and `coastsat show`.

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
* Select an **AOI KML** (copies it, then reads for CRS).
* Confirm or override the **EPSG** (auto‐detected or default = 3156).
* Select **Reference Shoreline** (GeoJSON) and **Transects** (GeoJSON).
* Select your **FES2022 YAML** (stored as absolute path).
* Create the **outputs/** folder.
* A final “Run analysis now?” popup will appear.

Upon completion, you’ll see:

```
<base_dir>/<sitename>/  
├── inputs/  
│   ├── aoi/          
│   │   └── <sitename>_aoi.kml  
│   ├── reference/    
│   │   └── <sitename>_ref.geojson  
│   ├── transects/    
│   │   └── <sitename>_transects.geojson  
│   └── fes_config.yaml  # absolute path stored in settings.json  
├── outputs/  
└── settings.json     
```

### 4.2 Run the Complete Analysis

If you chose **No** at the final prompt or want to run later, type:

```bash
coastsat run --config /path/to/<sitename>/settings.json
```

Internally, this runs:

```bash
python Complete_Analysis.py --config /path/to/<sitename>/settings.json
```

That executes the entire CoastSat pipeline (download, preprocess, shoreline detection, transect analysis, slope + tide correction, trend plotting, etc.).

### 4.3 Inspect Output

After analysis finishes, files appear under:

```
<base_dir>/<sitename>/outputs/
```

To view the folder tree, type:

```bash
coastsat show --config /path/to/<sitename>/settings.json
```

You might see:

```
outputs/  
├─ tuk_output_points.geojson  
├─ mapped_shorelines.jpg   
├─ plots/    
│   ├─ tuk_tide_timeseries.jpg  
│   ├─ 0_timestep_distribution.jpg  
│   └─ …   
└─ time_series/  
    ├─ transect_time_series.csv  
    └─ transect_time_series_tidally_corrected.csv
```

---

## CLI Commands

```
Usage: coastsat [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  init  Create a new CoastSat project (AOI, shoreline, transects, FES).  
  run   Run the full CoastSat analysis using your settings.json.  
  show  Show the directory tree under the project's output folder.
```

* **`coastsat init`**: Walks you through project setup (file-browser dialogs, progress bars, and writes `settings.json`).
* **`coastsat run --config <settings.json>`**: Executes `Complete_Analysis.py` with the given config.
* **`coastsat show --config <settings.json>`**: Prints the folder structure of `<sitename>/outputs`.

---

## Settings File (settings.json)

After initialization, `settings.json` looks like:

```json
{
  "inputs": {
    "sitename": "tuk",
    "aoi_path": "inputs/aoi/tuk_aoi.kml",
    "reference_shoreline": "inputs/reference/tuk_ref.geojson",
    "transects": "inputs/transects/tuk_transects.geojson",
    "fes_config": "C:/absolute/path/to/fes2022.yaml"
  },
  "output_dir": "outputs",
  "output_epsg": 3156
}
```

* **`inputs.aoi_path`**, **`reference_shoreline`**, **`transects`** are relative to the project root.
* **`fes_config`** is stored as an absolute path.
* **`output_dir`** is relative (usually `"outputs"`).
* **`output_epsg`** is the EPSG code used throughout CoastSat.

---

## Directory Structure

```
<base_dir>/<sitename>/  
├─ inputs/        
│   ├─ aoi/           
│   │   └─ tuk_aoi.kml  
│   ├─ reference/     
│   │   └─ tuk_ref.geojson  
│   ├─ transects/     
│   │   └─ tuk_transects.geojson  
│   └─ fes_config.yaml  
├─ outputs/       
│   ├─ tuk_output_points.geojson  
│   ├─ mapped_shorelines.jpg  
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

## License

This project is released under the MIT License. See `LICENSE` for details.

---
