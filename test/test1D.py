# main.py
import numpy as np
import matplotlib.pyplot as plt
import fem.mesh as mesh
import visualize as visualize 
from triangle import triangulate
from fom.heat import HeatProblem
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fom.owrelaxation import WaveformRelaxation

# 1D Example (Overlapping Waveform Relaxation method for Heat problem)

# Create a mesh - space mesh generation
vert = np.linspace(0, 2, 200)
mesh1D = mesh.Mesh(vertices = vert, dim = 1)

# Finite element space of degree 1
femspace = FEMSpace(mesh1D, domain = 'interval', degree = 1)

# Create time nodes generation
t0 = 0.0
T = 3.0
dt = 0.1
ntime = int((T - t0)/dt) + 1  # total number of time nodes
time_points = np.linspace(t0, T, ntime)  # array of time points

# Define the exact solution
def exact(x, t):
    return (1 - t) * np.sin(np.pi * x) + (1 + t**2) * np.sin(2 * np.pi * x)

# Define a source function
def func(x, t):
    u_t = -1*np.sin(np.pi*x) + 2*t*np.sin(2*np.pi*x)  # time derivative
    u_xx = -np.pi**2*(1 - t)*np.sin(np.pi*x) - 4*np.pi**2*(1 + t**2)*np.sin(2*np.pi*x)  # second derivative
    return u_t - u_xx  # f = u_t - u_xx

# Define a Dirichlet boundary condition dictionary
vertices = femspace.mesh.vertices
dirichlet_bc = dict()
for bnodes in femspace.mesh.boundary_vertices():
    x = vertices[bnodes]
    dirichlet_bc[bnodes] = exact(x, time_points)

# Define an initial condition array
ndof = vertices.shape[0]
initial_cond = np.zeros(ndof)
for i, vertex in enumerate(vertices):
    x = vertex
    initial_cond[i] = exact(x, t0)

# Heat solver
# Heat_solver = HeatProblem(
#             femspace = femspace,
#             func = func, 
#             dt = dt, 
#             t0 = t0, 
#             T = T, 
#             dirichlet_bc = dirichlet_bc, 
#             icond = initial_cond, 
#             tstepper = 'Theta',
#             theta = 1)

# Heat problem solution
# heat_solution = Heat_solver.solve()

# Overlapping Waveform Relaxation solver
OWR_solver =  WaveformRelaxation(femspace = femspace,                                        
                                n = 3, 
                                overlap = 1, 
                                func = func,
                                dt = dt, 
                                t0 = t0, 
                                T = T, 
                                dirichlet_bc = dirichlet_bc,
                                icond = initial_cond,
                                tstepper = 'Theta',
                                theta = 0.5,
                                method = 'RAS',
                                maxiter = 100,
                                tol = 1e-3)

# Overlapping Waveform Relaxation solution
owr_solution = OWR_solver.solve()

# Visualize result
visualizer_pde = visualize.SolutionVisualizer(femspace.mesh, owr_solution, dt)
visualizer_pde.visualize_1d_time_compare(exact_func = exact, nx = 100)
# femspace.visualize_1d_time_compare(u = heat_solution, exact_func = exact, dt = dt)


# # vertices of the element
# vert = femspace.mesh.vertices[femspace.mesh.elements[15]].max()
    
# inds = set()
# alls = set()
# for elem_index in [15, 16]:
#     # get shape functions for this element
#     functions = femspace.get_shape_functions(elem_index)
#     # evaluate FEM solution at points
#     for g, phi_g in functions.items():
#         alls.add(g)
#         if g not in inds:
#             inds.add(g)
#         else:
#             inds.remove(g)

# sel = alls - inds

# evals = []
# for elem_index in [15, 16]:
#     # get shape functions for this element
#     functions = femspace.get_shape_functions(elem_index)
#     # evaluate FEM solution at points
#     for g, phi_g in functions.items():
#         if g in sel:
#             evals.append(phi_g(vert))

# print(sel)
# print(evals)