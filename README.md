# Master Thesis : Reduced-Order Overlapping Schwarz Waveform Relaxation

This repository contains the code and experiments for my master's thesis on "Reduced-Order Overlapping Schwarz Waveform Relaxation". The thesis focuses on developing and analyzing a reduced-order version of the overlapping Schwarz waveform relaxation (ROSWR) method for solving time-dependent partial differential equations (PDEs). The code is organized into several modules, including the implementation of the OSWR and ROSWR method, and visualization tools for analyzing the results. The experiments include comparisons between the OSWR and ROSWR methods, as well as error analysis and convergence studies.

## ⚙️ Usage and Installation

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
You can run the experiments by executing the corresponding Python scripts in the `experiments` directory. For example, to run the 1D ROSWR experiment:

```bash
python experiments/1D/rowr_1D.py
```

## 📊 Plotting and LaTeX Rendering

This project uses **Matplotlib with LaTeX support** to produce high-quality figures suitable for a Master’s thesis and scientific publications. To enable LaTeX rendering in Matplotlib, ensure you have a LaTeX distribution installed on your system (e.g., TeX Live, MiKTeX). The code is configured to use LaTeX for rendering text in the plots, which allows for mathematical expressions and symbols to be displayed correctly.

### ✨ LaTeX Rendering (Configuration)

Matplotlib is configured to use LaTeX for all text rendering in plots:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
    """})
```

This ensures:
- Proper mathematical notation rendering
- High-quality serif fonts
- Consistent typography across all figures

### ⚙️ System Requirements

To enable LaTeX rendering, a full LaTeX distribution must be installed on the system.

#### Ubuntu / Debian

```bash id="sysreq2"
sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```
#### Windows
1. Download and install MiKTeX from [https://miktex.org/download](https://miktex.org/download).
2. During installation, ensure you select the option to install missing packages on-the-fly.
3. After installation, open the MiKTeX Console and update the package database.
4. Install the required packages (e.g., `amsmath`, `amsfonts`, `amssymb`) if they are not already included in the base installation.
5. Restart your Python environment to ensure Matplotlib can find the LaTeX installation.

#### macOS
1. Install MacTeX from [https://tug.org/mactex/](https://tug.org/mactex/).
2. Follow the installation instructions provided on the website.
3. After installation, ensure that the LaTeX binaries are in your system's PATH. You may need to add the following line to your shell configuration file (e.g., `.bashrc` or `.zshrc`):
```bash
export PATH="/Library/TeX/texbin:$PATH"
```
4. Restart your terminal and Python environment to ensure Matplotlib can find the LaTeX installation.

### ⚠️ Fallback Behavior

If LaTeX is not installed on the system, the project will continue running normally without errors, and automatically fall back to default Matplotlib text rendering, displaying mathematical text using standard Matplotlib fonts instead of LaTeX. No code changes are required for this fallback behavior.

## 📄 How to Compile the LaTeX Thesis

The thesis is written in LaTeX and can be compiled automatically using the provided build script. This ensures that all references, citations, and cross-references are resolved correctly.


Run the following command from the `thesis/` directory:

```bash
./build.sh
```

The build script uses latexmk for automatic multi-pass compilation, and all auxiliary files and logs are stored inside the output/ directory. If the script does not run, ensure it has execution permissions:
```bash
chmod +x build.sh
```

After successful compilation, the generated PDF will be located in:

```bash
output/main.pdf
```