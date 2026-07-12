import numpy as np
from pathlib import Path
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat_fom import HeatProblem
import utils.history as history
from utils.errornorms import NormType
from utils.logger import configure_logging
configure_logging(level="info")

# -------------------- Example 1: 1D Heat equation -------------------------

# Spatial domain definition - mesh size h = 0.01
vertices = np.linspace(0, 1, 101)
mesh1D = Mesh(vertices = vertices, dim = 1)

# Decompose domain into 2 subdomains with extension layers of 5, 10, and 15 (corresponding to overlaps of 0.1, 0.2, and 0.3 respectively)
extensions = [5, 10, 15]
decomposition_infos = {ext: mesh1D.decompose(n = 2, overlap = ext, version = 2) for ext in extensions}

# Time domain definitions for short and long time intervals
t0, Tshort, Tlong, ntime = 0, 0.1, 1, 101
dt_short, dt_long = (Tshort - t0)/(ntime - 1), (Tlong - t0)/(ntime - 1)
time_grid_short = np.linspace(t0, Tshort, ntime)
time_grid_long = np.linspace(t0, Tlong, ntime)

# Linear Lagrange finite element space
femspace1D = FEMSpace(mesh = mesh1D, domain = 'interval', degree = 1)

# Define the source function
def source_function(x, t):
    return (np.pi*np.sin(np.pi*x)*np.cos(np.pi * t) - 2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*t)
            + np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*t) + 4*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*t))

# Define the boundary condition (this is exact solution of the heat equation)
def boundary_condition(x, t):
    return np.sin(np.pi*x)*np.sin(np.pi*t) + np.cos(2*np.pi*x)*np.cos(2*np.pi*t)

# Define the initial condition
def initial_condition(x):
    return np.cos(2*np.pi*x)

# Define 1D Heat problems for short and long time intervals
heat_short = HeatProblem(femspace = femspace1D, t0 = t0, T = Tshort, f = source_function, g = boundary_condition, h = initial_condition)
heat_long = HeatProblem(femspace = femspace1D, t0 = t0, T = Tlong, f = source_function, g = boundary_condition, h = initial_condition)

# Define metric for tracking the error history during Schwarz iteration
metric = history.MetricSpec(name = history.MetricType.ABSOLUTE_ERROR, spatial = history.SpatialMode.SUBDOMAINS, temporal = history.TemporalMode.STATIC, norm = NormType.LINF)

# Define the directory for saving the figures and data
fig_dir = Path(__file__).resolve().parents[1]/"figures"
data_dir = Path(__file__).resolve().parents[1]/"data"