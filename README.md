# AEON SLEAP Analysis

A repository for analysing pose estimation results from [SLEAP](https://sleap.ai/) models, developed for use within the lab. Currently focused on tracking a configuration card across multiple camera angles within an enclosure, with plans to expand to broader pose estimation workflows.

---

## Overview

This repo provides scripts and notebooks for loading, processing, and visualising SLEAP output files (`.slp`, `.h5`). It is intended as a shared resource for lab members working with SLEAP-tracked data.

---

## Repository Structure

```
sleap-analysis/
├── data/               # SLEAP output files (.slp, .h5) — not tracked by git
├── notebooks/          # Jupyter notebooks for exploration and visualisation
├── scripts/            # Reusable Python scripts for analysis pipelines
├── outputs/            # Generated plots, CSVs, and other results
├── requirements.txt    # Python dependencies
└── README.md
```

> **Note:** Raw data files should be placed in `data/` locally. This folder is gitignored to avoid committing large files.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sleap-analysis.git
cd sleap-analysis
```

### 2. Create and activate the conda environment

```bash
conda create -n sleap-analysis python=3.9
conda activate sleap-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Select the interpreter in VSCode

Press `Cmd+Shift+P` → **Python: Select Interpreter** → choose `sleap-analysis`.

---

## Dependencies

Key packages used in this project:

| Package | Purpose |
|---|---|
| `sleap` | Loading and working with SLEAP files |
| `h5py` | Reading `.h5` prediction outputs |
| `numpy` | Numerical operations |
| `pandas` | Tabular data handling |
| `matplotlib` / `seaborn` | Visualisation |
| `jupyter` | Interactive notebooks |

Install all at once with:

```bash
pip install sleap h5py numpy pandas matplotlib seaborn jupyter
```

---

## Usage

A basic example for loading a SLEAP labels file:

```python
import sleap

labels = sleap.load_file("data/your_file.slp")
print(labels)
```

See the `notebooks/` folder for worked examples covering:
- Visualising instance positions over time
- Comparing predictions across camera angles
- Evaluating model metrics (mOKS, PCK, average distance)

---

## Notes on Model Training

Models in this project were trained bottom-up using the SLEAP GUI. Key training considerations:

- **Sigma**: set to `5px` — found to be important for reliable peak detection at inference
- **Peak threshold**: lowered to `0.05` or below during inference if 0 instances are detected
- Multiple camera angles require careful labelling coverage across viewing distances

---

## Contributing

This repo is intended for internal lab use. If you are a lab member and want to contribute scripts or notebooks, please create a new branch and open a pull request for review.

---

## Links

- [SLEAP Documentation](https://sleap.ai/)
- [SLEAP GitHub](https://github.com/talmolab/sleap)
