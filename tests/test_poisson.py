import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fem.linearsolver import *
from fom.poisson import PoissonProblem
from visualization.visualize import SolutionVisualizer

# ----------------------------
# 1D Example (Poisson problem)
# ----------------------------

# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Finite element space of degree 1
femspace1D = FEMSpace(mesh1D, domain = 'interval', degree = 1)

# Define the exact solution
def exact1D(x):
    return np.sin(3*np.pi*x) + x**3

# Define a source function
def func1D(x):
    return 9*np.pi**2*np.sin(3*np.pi*x) - 6*x

# Define 1D Poisson problem
problem1D = PoissonProblem(femspace = femspace1D, f = func1D, g = exact1D)

# Solve the 1D Poisson problem using nodal lifting
poisson_solution1D = problem1D.solve(lift = 'nodal')

# Error analysis
error1D = np.linalg.norm(poisson_solution1D.flatten() - exact1D(mesh1D.vertices))
print("L2 nodal error:", error1D)

# ----------------------------
# 2D Example (Poisson problem)
# ----------------------------

# Create a simple square mesh - space mesh generation
vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
segment_markers = np.array([1, 2, 3, 4])
mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.001')

# Finite element space of degree 1
femspace2D = FEMSpace(mesh2D, degree = 1)

# Define the exact solution
def exact2D(x, y):
    return np.exp(x)*np.sin(np.pi*y) + x**2 * y + np.sin(2*np.pi*x)*np.cos(3*np.pi*y)

# Define a source function
def func2D(x, y):
    return -(1 - np.pi**2)*np.exp(x)*np.sin(np.pi*y) - 2*y + 13*(np.pi**2)*np.sin(2*np.pi*x)*np.cos(3*np.pi*y)

# Define 2D Poisson problem
problem2D = PoissonProblem(femspace = femspace2D, f = func2D, g = exact2D)

# Solve the 2D Poisson problem using nodal lifting
poisson_solution2D = problem2D.solve(lift = 'nodal')

# Error analysis
error2D = np.linalg.norm(poisson_solution2D.flatten() - exact2D(femspace2D.mesh.vertices[:,0], femspace2D.mesh.vertices[:,1]))
print("L2 error (approx at nodes):", error2D)

# Visualize the solution and compare with the exact solution
visualizer2D = SolutionVisualizer(mesh2D, poisson_solution2D)


# Alternative test examples
# def exact2D(x, y):
#     return np.sin(np.pi*x)*np.sin(np.pi*y)
# def func2D(x, y):
#     return 2*(np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)
