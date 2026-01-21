import numpy as np
import fem.mesh as mesh
import visualize as visualize 
from fom.poisson import PoissonProblem
from fem.femspace import FEMSpace
from fom.schwarz import Schwarz
from errornorms import ErrorNorms

# 2D Example (Schwarz method for Poisson problem - stress test)

# Create circular mesh - space mesh generation

N = 40  # boundary points
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
vert = np.column_stack([np.cos(theta), np.sin(theta)])
segments = [[i, i+1] for i in range(N-1)] + [[N-1, 0]]

# Create mesh
mesh_circ = mesh.Mesh(vertices = vert, segments = segments, options = 'qa0.01')

mesh_vis = visualize.MeshVisualizer(meshobj = mesh_circ)

mesh_vis.visualize(carray = mesh_vis.carray_decomposition(2))

# Finite element space of degree 1
femspace_circ = FEMSpace(mesh_circ, degree = 1)

# Visualize the square mesh
visualizer_sq = visualize.MeshVisualizer(femspace_circ.mesh)

# # Define the exact solution
def exact(x, y):
    return (1 - x**2 - y**2) * np.sin(3*np.pi*x) * np.sin(3*np.pi*y)

# # Define a source function
def func(x, y):
    pi = np.pi
    r2 = x**2 + y**2
    sinx = np.sin(3*pi*x)
    siny = np.sin(3*pi*y)
    cosx = np.cos(3*pi*x)
    cosy = np.cos(3*pi*y)
    
    u_xx = -2*sinx*siny - 6*pi*x*cosx*siny + (1 - r2)*(-9*pi**2*sinx*siny)
    u_yy = -2*sinx*siny - 6*pi*y*sinx*cosy + (1 - r2)*(-9*pi**2*sinx*siny)
    
    return -(u_xx + u_yy)

# Define a Dirichlet boundary condition dictionary
vertices = femspace_circ.mesh.vertices
dirichlet_bc = dict()
for bnodes in femspace_circ.mesh.boundary_vertices():
    x, y = vertices[bnodes]
    dirichlet_bc[bnodes] = exact(x, y)

Poisson_solver = PoissonProblem(femspace = femspace_circ, func = func, dirichlet_bc = dirichlet_bc)
poisson_solution = Poisson_solver.solve()
est = ErrorNorms(femspace = femspace_circ, u1 = poisson_solution, u_exact = exact)
print(est.l2_error())

Schwarz_solver = Schwarz(femspace = femspace_circ,                                        
                                n = 2, 
                                overlap = 1, 
                                func = func,
                                dirichlet_bc = dirichlet_bc,
                                method = 'RAS',
                                maxiter = 180,
                                tol = 1e-6)

schwarz_solution = Schwarz_solver.solve(history = True, uh = poisson_solution, exact = exact)
visualizer_pde = visualize.SolutionVisualizer(femspace_circ.mesh, schwarz_solution)
# visualizer_pde.visualize_3d()
# visualizer_pde.visualize_3d_compare(exact)
visualizer_pde.plot_iteration_error(error_history = Schwarz_solver.error_history, linewidth = 0.8, markersize = 3)
visualizer_pde.write_vtk(filename = "poisson_circ", exact = exact)