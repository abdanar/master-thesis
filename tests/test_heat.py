import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fem.linearsolver import *
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

# Solve the 1D Heat problem using nodal lifting
heat_solution1D = problem1D.solve(ntime = ntime, lift = 'nodal')

# Error analysis
# error1D = np.linalg.norm(heat_solution1D - exact1D(mesh1D.vertices, np.linspace(t0, T, ntime)))
# print("L2 nodal error:", error1D)

# ----------------------------
# 2D Example (Heat problem)
# ----------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.001')

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

# Solve the 2D Heat problem using nodal lifting
heat_solution2D = problem2D.solve(ntime = ntime, lift = 'nodal')

# Error analysis
# error2D = np.linalg.norm(heat_solution2D.flatten() - exact2D(femspace2D.mesh.vertices[:,0], femspace2D.mesh.vertices[:,1], np.linspace(t0, T, ntime)))
# print("L2 error (approx at nodes):", error2D)
