import numpy as np
from pathlib import Path
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat_fom import HeatProblem
import utils.history as history
from utils.errornorms import NormType
from utils.logger import configure_logging
configure_logging(level="info")

# -------------------- Example 1: 2D Heat equation -----------------------------

# Spatial domain definition
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh2D = Mesh(vertices = vertices, options = 'qa0.002')

# Decompose domain into 4 subdomains with 1 extension layer
decomposition_info = mesh2D.decompose(n = 4, overlap = 1, version = 2)

# Time domain definitions for short and long time intervals
t0, Tshort, Tlong, ntime = 0, 0.1, 1, 101
dt_short, dt_long = (Tshort - t0)/(ntime - 1), (Tlong - t0)/(ntime - 1)
time_grid_short = np.linspace(t0, Tshort, ntime)
time_grid_long = np.linspace(t0, Tlong, ntime)

# Linear Lagrange finite element spaces
femspace2D = FEMSpace(mesh = mesh2D)

# Define the source function
def source_function(x, y, t):
    return np.exp(-t)*(2*np.pi**2-1)*np.sin(np.pi*x)*np.sin(np.pi*y)

# Define the boundary conditions
def boundary_condition(x, y, t):
    return np.exp(-t)*(np.sin(np.pi*x)*np.sin(np.pi*y))

# Define the initial condition
def initial_condition(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)

# Define 2D Heat problems for short and long time intervals
heat_short = HeatProblem(femspace = femspace2D, t0 = t0, T = Tshort, f = source_function, g = boundary_condition, h = initial_condition)
heat_long = HeatProblem(femspace = femspace2D, t0 = t0, T = Tlong, f = source_function, g = boundary_condition, h = initial_condition)

# Define metric for tracking the error history during Schwarz iteration
metric = history.MetricSpec(name = history.MetricType.ABSOLUTE_ERROR, spatial = history.SpatialMode.SUBDOMAINS, temporal = history.TemporalMode.STATIC, norm = NormType.LINF)

# Define the directory for saving the figures and data
fig_dir = Path(__file__).resolve().parents[1]/"figures"
data_dir = Path(__file__).resolve().parents[1]/"data"