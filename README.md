# AEON SLEAP Analysis

A repository for analysing pose estimation results from [SLEAP](https://sleap.ai/) models, developed for use within the lab. It is currently focused on tracking a configuration card across multiple camera angles within an enclosure, with room to expand into broader SLEAP analysis workflows.

---

## Overview

This repo provides notebooks and supporting code for loading, processing, and visualising SLEAP outputs such as `.slp` and `.h5` files.

---

## Repository Structure

```text
AEON_sleap_analysis/
├── data/                     # Local SLEAP output files (.slp, .h5); not tracked by git
├── outputs/                  # Generated plots, CSVs, and analysis outputs
├── sleap_card_configuration/ # Current notebooks and analysis work
├── requirements.txt          # Python dependencies
└── README.md
```

> Raw data should stay local and should not be committed to git.

---

## Setup

This project uses `uv`.

### Python version

Use Python `3.10` for this repo.

Why: the pinned SLEAP/TensorFlow stack in `requirements.txt` includes `tensorflow-macos==2.9.2`, which installs correctly on Python 3.8-3.10, but not on newer versions such as Python 3.13.

### Option 1: Use the existing working SLEAP interpreter in VS Code

This is the quickest path if you already installed SLEAP with `uv tool install sleap`.

The working interpreter is:

```bash
/Users/zosiasus/.local/share/uv/tools/sleap/bin/python
```

In VS Code:

1. Open this repository.
2. Run `Cmd+Shift+P`.
3. Choose `Python: Select Interpreter`.
4. Select `/Users/zosiasus/.local/share/uv/tools/sleap/bin/python`.

You should then be able to run:

```python
import sleap
```

This repo also includes `.vscode/settings.json` pointing VS Code at that interpreter by default.

### Option 2: Create a local repo virtual environment with `uv`

Use this if you want the project to have its own local `.venv`.

```bash
uv python install 3.10
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

Activate it with:

```bash
source .venv/bin/activate
```

Then test:

```bash
python -c "import sleap; print(sleap.__version__)"
```

### If installation fails because of disk space

The full SLEAP + TensorFlow stack is large. If `uv pip install` fails with `No space left on device`, free some disk space and retry.

If you need to rebuild cleanly:

```bash
rm -rf .venv .uv-cache
uv python install 3.10
UV_CACHE_DIR=.uv-cache uv venv .venv --python 3.10
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -r requirements.txt
```

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

Install the pinned project dependencies with:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

---

## Usage

A basic example for loading a SLEAP labels file:

```python
import sleap

labels = sleap.load_file("data/your_file.slp")
print(labels)
```

Current analysis work lives in `sleap_card_configuration/`.

---

## Notes on Model Training

Models in this project were trained in the SLEAP GUI. Key training notes from current experiments:

- `sigma = 5 px` was important for reliable peak detection at inference.
- The inference peak threshold sometimes needed to be lowered to `0.05` or below when no instances were detected.
- Multiple camera angles require enough labelled examples across different distances and viewpoints.

---

## Contributing

This repo is intended for internal lab use. If you are contributing notebooks or scripts, create a branch and open a pull request for review.

---

## Links

- [SLEAP Documentation](https://sleap.ai/)
- [SLEAP GitHub](https://github.com/talmolab/sleap)
- [uv Documentation](https://docs.astral.sh/uv/)
