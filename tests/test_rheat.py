import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat import HeatProblem
from rom.rheat import ReducedHeatProblem
from rom.pod import POD

# ---------------------------------
# 1D Example (Reduced Heat problem)
# ---------------------------------

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 101
time_grid = np.linspace(t0, T, ntime)

# Finite element space of degree 1
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
pod1D = POD(heat_problem = fom1D, ntime = 31, lift = 'nodal', theta = 0.5, r = 10)

print("Projection matrix V shape:", pod1D.V.shape) # Should be (n_interior, r) where n_interior is the number of interior nodes and r is the number of modes retained

# Reduce the 1D Heat problem using a projection matrix
rom1D = ReducedHeatProblem(heat_problem = fom1D, V = pod1D.V)

# Solve the 1D Reduced Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and reconstruct the full solution
rom_solution1D = rom1D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5, reconstruct = True)

# Error analysis
error1D = np.linalg.norm(rom_solution1D - exact1D(mesh1D.vertices[:, None], time_grid[None, :]), axis = 0)
print("max error (rom vs exact):", error1D.max())

# ---------------------------------
# 2D Example (Reduced Heat problem)
# ---------------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.001')

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 101
time_grid = np.linspace(t0, T, ntime)

# Finite element space of degree 1
femspace2D = FEMSpace(mesh2D, degree = 1)

# Define the exact solution
def exact2D(x, y, t):
    return np.exp(-t)*(x**2 + y**2)

# Define a source function
def func2D(x, y, t):
    return -np.exp(-t)*(x**2 + y**2) - 4*np.exp(-t)

# Define the initial condition function
def h2D(x, y):
    return np.exp(-t0)*(x**2 + y**2)

# Define 2D Heat problem
fom2D = HeatProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D)

# POD reduction of the 2D Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) to r = 10 modes
pod2D = POD(heat_problem = fom2D, ntime = 31, lift = 'nodal', theta = 0.5, r = 10)

print("Projection matrix V shape:", pod2D.V.shape) # Should be (nintnodes, r) where nintnodes is the number of interior nodes and r is the number of modes retained

# Reduce the 2D Heat problem using a projection matrix
rom2D = ReducedHeatProblem(heat_problem = fom2D, V = pod2D.V)

# Solve the 2D Reduced Heat problem using nodal lifting and theta method with theta = 0.5 (Crank-Nicolson) and reconstruct the full solution
rom_solution2D = rom2D.solve(time_grid = time_grid, lift = 'nodal', theta = 0.5, reconstruct = True)

# Error analysis
error2D = np.linalg.norm(rom_solution2D - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_grid[None, :]), axis = 0)
print("max error (rom vs exact):", error2D.max())