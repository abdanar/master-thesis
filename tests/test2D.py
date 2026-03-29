import numpy as np
import fem.mesh as mesh
import visualization.visualize as visualize 
from triangle import triangulate
from fom.heat import HeatProblem
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fom.oswrelaxation import WaveformRelaxation

# 2D Example (Overlapping Waveform Relaxation method for Heat problem)

# Create a simple square mesh - space mesh generation
vert = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh_square = mesh.Mesh(vert, options = 'qa0.001')

# Finite element space of degree 1
femspace_sq = FEMSpace(mesh_square, degree = 1)

# Visualize the square mesh
visualizer_sq = visualize.MeshVisualizer(femspace_sq.mesh)

# Create time nodes generation
t0 = 0.0
T = 3.0
dt = 0.01
ntime = int((T - t0)/dt) + 1  # total number of time nodes
time_points = np.linspace(t0, T, ntime)  # array of time points

# Define the exact solution
def exact(x, y, t):
    return (1 - t)*np.sin(np.pi*x)*np.sin(2*np.pi*y) + (1 + t**2)*np.sin(2*np.pi*x)*np.sin(np.pi*y)

# Define a source function
def func(x, y, t):
    return (-1 + 5*np.pi**2*(1 - t))*np.sin(np.pi*x)*np.sin(2*np.pi*y) + (2*t + 5*np.pi**2*(1 + t**2))*np.sin(2*np.pi*x)*np.sin(np.pi*y)

# Define a Dirichlet boundary condition dictionary
vertices = femspace_sq.mesh.vertices
dirichlet_bc = dict()
for bnodes in femspace_sq.mesh.boundary_vertices():
    x, y = vertices[bnodes]
    dirichlet_bc[bnodes] = exact(x, y, time_points)

# Define an initial condition array
ndof = vertices.shape[0]
initial_cond = np.zeros(ndof)
for i, vertex in enumerate(vertices):
    x, y = vertex
    initial_cond[i] = exact(x, y, t0)

Heat_solver = HeatProblem(
            femspace = femspace_sq,
            func = func, 
            dt = dt, 
            t0 = t0, 
            T = T, 
            dirichlet_bc = dirichlet_bc, 
            icond = initial_cond, 
            tstepper = 'Theta',
            theta = 0.5)

heat_solution = Heat_solver.solve()

# OWR_solver =  WaveformRelaxation(femspace = femspace_sq,                                        
#                                 n = 2, 
#                                 overlap = 1, 
#                                 func = func,
#                                 dt = dt, 
#                                 t0 = t0, 
#                                 T = T, 
#                                 dirichlet_bc = dirichlet_bc,
#                                 icond = initial_cond,
#                                 tstepper = 'Theta',
#                                 theta = 0.5,
#                                 method = 'RAS',
#                                 maxiter = 100,
#                                 tol = 1e-3)


# owr_solution = OWR_solver.solve()
visualizer_pde = visualize.SolutionVisualizer(femspace_sq.mesh, heat_solution, dt)
# visualizer_pde.write_vtk_time_series(exact_func = exact, folder="vtk", prefix="heat_solution")
# visualizer_pde.write_vtk_time_series(exact_func = exact, folder="vtk", prefix="heat_solution")
# visualizer_pde.visualize_3d_time()
visualizer_pde.visualize_3d_time_compare(exact_func = exact)
visualizer_pde.visualize_3d_time_error(exact_func = exact)
# femspace_sq.visualize_3d_time_compare(heat_solution, exact, dt)

# pde = HeatProblem(mesh_square, func, dt, d0, T, dirichlet_bc, icond)
# pde_solution = pde.solve()

# ndof = Assembler(mesh_square).ndof(degree = 1)


# visualizer_pde = visualize.SolutionVisualizer(mesh_square, pde_solution)
# visualizer_pde.visualize()
# visualizer_pde.visualize_3d()


# # Create a donut mesh with a hole in the center
# def donut_mesh():
#     # computes vertices and triangles for a donut mesh by determining maximum area of triangles
#     def circle(N, R):
#         i = np.arange(N)
#         theta = i * 2 * np.pi / N
#         pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
#         seg = np.stack([i, i + 1], axis=1) % N
#         return pts, seg

#     pts0, seg0 = circle(30, 1.4)
#     pts1, seg1 = circle(16, 0.6)
#     pts = np.vstack([pts0, pts1])
#     seg = np.vstack([seg0, seg1 + seg0.shape[0]])
#     return pts, seg, [[0, 0]]

# vertices, segments, holes = donut_mesh()
# mesh_donut = mesh.Mesh(vertices = vertices, segments = segments, holes = holes, options = 'qpa0.01')
# print(mesh_donut.holes)

# # Get information about the mesh
# print(mesh_donut.info()) 

# # Get information about subdomains
# submeshes, _,  membership = mesh_donut.decompose(10)
# for submesh in submeshes:
#     print(submesh.info())

# # Create visualizer and show decomposition into 10 subdomains 
# visualizer_donut = visualize.MeshVisualizer(mesh_donut)
# visualizer_donut.visualize(visualizer_donut.carray_decomposition(10))

# # Create visualizer and show decomposition into 10 subdomains 
# submeshes,_ , membership = mesh_square.decompose(20)
# for submesh in submeshes:
#     print(submesh.info())

# visualizer_sq = visualize.MeshVisualizer(mesh_square)
# visualizer_sq.visualize(visualizer_sq.carray_decomposition(20))


