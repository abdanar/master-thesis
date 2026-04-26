import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat_fom import HeatProblem
from rom.heat_rom import ReducedHeatProblem
from rom.pod import POD
from utils.logger import configure_logging
configure_logging(level="info")

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0, T, ntime = 0.0, 1.0, 101
time_grid = np.linspace(t0, T, ntime)

# Lagrange finite element space of degree 1
femspace1D = FEMSpace(mesh1D, domain = 'interval', degree = 1)

# Define the exact solution
def exact1D(x, t):
    return np.exp(-t)*(x**2 + x)

# Define the source function
def func1D(x, t):
    return -np.exp(-t)*(x**2 + x) - 2*np.exp(-t)

# Define the initial condition function
def h1D(x):
    return np.exp(-t0)*(x**2 + x)

# Define 1D Heat problem
fom1D = HeatProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D)

# POD reduction of the 1D Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) to r = 10 modes
tparams = np.linspace(t0, T, 31) # Time parameters for snapshot generation (31 snapshots including t0 and T)
pod1D = POD(heat_problem = fom1D, time_grid = tparams, lift = 'nodal', theta = 0.5, r = 10)
V, _ = pod1D.compute_modes()
print("Projection matrix V shape:", V.shape) # Should be (nintnodes, r) where nintnodes is the number of interior nodes and r is the number of modes retained

# Reduce the 1D Heat problem using a projection matrix
rom1D = ReducedHeatProblem(heat_problem = fom1D, V = V)

# Solve the 1D Reduced Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and reconstruct the full solution
rom_solution1D = rom1D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5, reconstruct = True)

# Error analysis
error1D = np.linalg.norm(rom_solution1D - exact1D(mesh1D.vertices[:, None], time_grid[None, :]), axis = 0)
print("max error (rom vs exact):", error1D.max())