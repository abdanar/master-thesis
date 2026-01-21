# main.py
import numpy as np
import fem.mesh as mesh
import visualize as visualize 
from fom.poisson import PoissonProblem
from fem.femspace import FEMSpace
from fom.schwarz import Schwarz
from errornorms import ErrorNorms

# 2D Example (Schwarz method for Poisson problem)

# Create a simple square mesh - space mesh generation
vert = np.array([[0,0],[1,0],[1,1],[0,1]])
#mesh_square = mesh.Mesh(vert, options = 'qa0.01')
mesh_square = mesh.Mesh(vert, options = 'st, dx=0.1, dy=0.1') # structured mesh

mesh_vis = visualize.MeshVisualizer(meshobj = mesh_square)

mesh_vis.visualize(carray = mesh_vis.carray_decomposition(2, direction = 'horizontal'))

# Finite element space of degree 1
femspace_sq = FEMSpace(mesh_square, degree = 1)

# Visualize the square mesh
visualizer_sq = visualize.MeshVisualizer(femspace_sq.mesh)

# # Define the exact solution
# def exact(x, y):
#     return np.sin(np.pi*x)*np.sin(np.pi*y)

# # Define a source function
# def func(x, y):
#     return 2*(np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y)

def exact(x, y):
    return (np.sin(2*np.pi*x) * np.sin(np.pi*y) + np.exp(-50*((x-0.5)**2 + (y-0.5)**2)))

def func(x, y):
    r2 = (x-0.5)**2 + (y-0.5)**2
    return (((2*np.pi)**2 + np.pi**2)*np.sin(2*np.pi*x) * np.sin(np.pi*y) + (200 - 10000*r2) * np.exp(-50*r2))

# def exact(x, y):
#     return (1 - np.exp(-20*x)) * np.sin(np.pi*y)

# def func(x, y):
#     return (400*np.exp(-20*x) + np.pi**2*(1 - np.exp(-20*x))) * np.sin(np.pi*y)

# Define a Dirichlet boundary condition dictionary
vertices = femspace_sq.mesh.vertices
dirichlet_bc = dict()
for bnodes in femspace_sq.mesh.boundary_vertices():
    x, y = vertices[bnodes]
    dirichlet_bc[bnodes] = exact(x, y)

Poisson_solver = PoissonProblem(femspace = femspace_sq, func = func, dirichlet_bc = dirichlet_bc)
poisson_solution = Poisson_solver.solve()
est = ErrorNorms(femspace = femspace_sq, u1 = poisson_solution, u_exact = exact)
print(est.compute())

Schwarz_solver = Schwarz(femspace = femspace_sq,                                        
                                n = 2, 
                                overlap = 1,
                                direction = 'horizontal', 
                                func = func,
                                dirichlet_bc = dirichlet_bc,
                                method = 'RAS',
                                maxiter = 100,
                                tol = 1e-15)

schwarz_solution = Schwarz_solver.solve(history = True, uh = poisson_solution, exact = exact)
visualizer_pde = visualize.SolutionVisualizer(femspace_sq.mesh, schwarz_solution)
# visualizer_pde.visualize_3d()
# visualizer_pde.visualize_3d_compare(exact)
visualizer_pde.plot_iteration_error(error_history = Schwarz_solver.error_history, linewidth = 0.8, markersize = 3)
visualizer_pde.write_vtk(filename = "poisson_sq", exact = exact)