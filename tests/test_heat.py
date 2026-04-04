import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fom.heat import HeatProblem

# ----------------------------
# 1D Example (Heat problem)
# ----------------------------

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 101
time_steps = np.linspace(t0, T, ntime)

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
problem1D = HeatProblem(femspace = femspace1D, t0 = t0, T = T, f = func1D, g = exact1D, h = h1D)

# Solve the 1D Heat problem using nodal lifting and theta method with theta = 1 (Backward Euler)
heat_solution1D_nodal = problem1D.solve(ntime = ntime, lift = 'nodal', theta = 1)

# Solve the 1D Heat problem using harmonic lifting and theta method with theta = 1 (Backward Euler)
heat_solution1D_harmonic = problem1D.solve(ntime = ntime, lift = 'harmonic', theta = 1)

# Error analysis
error1D_nodal = np.linalg.norm(heat_solution1D_nodal - exact1D(mesh1D.vertices[:, None], time_steps[None, :]), axis = 0)
error1D_harmonic = np.linalg.norm(heat_solution1D_harmonic - exact1D(mesh1D.vertices[:, None], time_steps[None, :]), axis = 0)
print("max error (fem vs exact) with nodal lifting:", error1D_nodal.max())
print("max error (fem vs exact) with harmonic lifting:", error1D_harmonic.max())

# ----------------------------
# 2D Example (Heat problem)
# ----------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.001')

# Time domain definition (ntime = (T - t0)/dt + 1 for uniform time steps)
t0 = 0.0
T = 1.0
ntime = 101
time_steps = np.linspace(t0, T, ntime)

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
problem2D = HeatProblem(femspace = femspace2D, t0 = t0, T = T, f = func2D, g = exact2D, h = h2D)

# Solve the 2D Heat problem using nodal lifting and theta method with theta = 1 (Backward Euler)
heat_solution2D_nodal = problem2D.solve(ntime = ntime, lift = 'nodal', theta = 1)

# Solve the 2D Heat problem using harmonic lifting and theta method with theta = 1 (Backward Euler)
heat_solution2D_harmonic = problem2D.solve(ntime = ntime, lift = 'harmonic', theta = 1)

# Error analysis
error2D_nodal = np.linalg.norm(heat_solution2D_nodal - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_steps[None, :]), axis = 0)
error2D_harmonic = np.linalg.norm(heat_solution2D_harmonic - exact2D(femspace2D.mesh.vertices[:,0][:, None], femspace2D.mesh.vertices[:,1][:, None], time_steps[None, :]), axis = 0)
print("max error (fem vs exact) with nodal lifting:", error2D_nodal.max())
print("max error (fem vs exact) with harmonic lifting:", error2D_harmonic.max())