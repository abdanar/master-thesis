import time
import numpy as np
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from fem.linearsolver import *
from fom.poisson import PoissonProblem

# Construct Dirichlet boundary conditions using an exact solution.
def build_dirichlet_bc(femspace, exact):
    vertices = femspace.mesh.vertices
    dim = femspace.mesh.dim
    dirichlet_bc = {}
    for node in femspace.mesh.boundary_vertices():
        coord = np.atleast_1d(vertices[node])
        dirichlet_bc[node] = exact(*coord[:dim])
    return dirichlet_bc

# ----------------------------
# 1D Example (Poisson problem)
# ----------------------------

# # Create a mesh - space mesh generation
# vert1D = np.linspace(0, 2, 200)
# mesh1D = mesh.Mesh(vertices = vert1D, dim = 1)

# # Finite element space of degree 1
# femspace1D = FEMSpace(mesh1D, domain = 'interval', degree = 1)

# # Define the exact solution
# def exact1D(x):
#     return np.sin(3*np.pi*x) + x**3

# # Define a source function
# def func1D(x):
#     return 9*np.pi**2*np.sin(3*np.pi*x) - 6*x

# # Define a Dirichlet boundary condition dictionary
# dirichlet1D = build_dirichlet_bc(femspace1D, exact1D)

# # Solve the 1D Poisson problem
# poisson_solver1D = PoissonProblem(femspace = femspace1D, func = func1D, dirichlet_bc = dirichlet1D)

# # Compute the solution using harmonic lifting
# poisson_solution1D = poisson_solver1D.solve(lift = 'harmonic')

# print(poisson_solution1D)

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
    return np.sin(np.pi*x)*np.sin(np.pi*y)

# Define a source function
def func2D(x, y):
    return 2*(np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)

startp = time.time()
# Solve the 2D Poisson problem
poisson_solver2D = PoissonProblem(femspace = femspace2D, func = func2D)
poisson_solution2D = poisson_solver2D.solve(lift='nodal')
endp = time.time()

# print(poisson_solution2D)
print(f"Elapsed time: {endp - startp:.4f} seconds")

coords = mesh2D.vertices
x = coords[:,0]
y = coords[:,1]

u_exact = exact2D(x, y)
error = np.linalg.norm(poisson_solution2D.flatten() - u_exact) / np.sqrt(len(u_exact))

print("Node L2 error:", error)

print(mesh2D.info())

# # exact solution
# def exact(x, y):
#     return x**2 + y**2 + np.sin(np.pi*x)*np.sin(np.pi*y)

# # right-hand side
# def func(x, y):
#     return -4 + 2*(np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)


# def exact(x, y):
#     return (
#         np.exp(x)*np.sin(np.pi*y)
#         + x**2 * y
#         + np.sin(2*np.pi*x)*np.cos(3*np.pi*y)
#     )

# def func(x, y):
#     return (
#         -(1 - np.pi**2)*np.exp(x)*np.sin(np.pi*y)
#         - 2*y
#         + 13*(np.pi**2)*np.sin(2*np.pi*x)*np.cos(3*np.pi*y)
#     )
