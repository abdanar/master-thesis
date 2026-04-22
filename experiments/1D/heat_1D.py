import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat_fom import HeatProblem
from utils.errornorms import ErrorNorms, NormType
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
problem1D = HeatProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D)

# Solve the 1D Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson)
heat_solution1D_nodal = problem1D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5)

# Error analysis
norms1D = ErrorNorms(femspace = femspace1D, u1 = heat_solution1D_nodal, u2 = exact1D(femspace1D.mesh.vertices[:, None], time_grid[None, :]), time = time_grid)
error1D_nodal = norms1D.compute(norm = NormType.L2)
print("L2 error (fem vs exact) with nodal lifting:", error1D_nodal)