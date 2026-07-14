# Reduced-Order Schwarz Waveform Relaxation

This repository contains the code and experiments for my master's thesis on "Reduced-Order Schwarz Waveform Relaxation". The thesis introduces a reduced-order analogue of the overlapping Schwarz waveform relaxation method for the heat equation. The proposed algorithm reduces the computational cost of solving the subproblems by computing their solutions in reduced spaces constructed using proper orthogonal decomposition. For further details, please refer to the thesis (it will be uploaded once the degree is awarded).

## Usage and Installation

### 1. Clone the repository

```bash
git clone https://github.com/abdanar/master-thesis.git
cd master-thesis
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the experiments

The experiments are organized in the `experiments` directory and consist of one-dimensional and two-dimensional examples, located in the `dim1` and `dim2` subdirectories, respectively. Each example contains a `src` folder with the Python scripts required to reproduce the numerical results presented in the *Numerical Experiments* chapter of the thesis. The scripts in the `src` folder generate the corresponding figures, which are stored in the `figures` folder, and save the computed data in the `data` folder. To reproduce the results presented in the thesis, the scripts should be executed in the following order:

- **1D experiments:** `heat.py` → `rheat.py` → `oswr.py` → `roswr.py`
- **2D experiments:** `mesh.py` → `heat.py` → `rheat.py` → `roswr.py`

For example, to reproduce the 1D ROSWR experiment, execute the following command from the project root directory:

```bash
python -m experiments.dim1.example1.src.roswr
```

## Plotting and LaTeX Rendering

This project uses Matplotlib with LaTeX support to produce high-quality figures suitable for a Master’s thesis and scientific publications. To enable LaTeX rendering in Matplotlib, ensure you have a LaTeX distribution installed on your system (e.g., TeX Live, MiKTeX). The code is configured to use LaTeX for rendering text in the plots, which allows for mathematical expressions and symbols to be displayed correctly. Matplotlib is configured to use LaTeX for all text rendering in plots:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{amssymb}
        """,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
```

If LaTeX is not installed on the system, the project will continue running normally without errors, and automatically fall back to default Matplotlib text rendering, displaying mathematical text using standard Matplotlib fonts instead of LaTeX. No code changes are required for this fallback behavior.
