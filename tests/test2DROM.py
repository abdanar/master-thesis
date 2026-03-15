import numpy as np
import fem.mesh as mesh
import visualization.visualize as visualize 
from triangle import triangulate
from fom.heat import HeatProblem
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from rom.reducedowr import ReducedWaveformRelaxation
from fom.owrelaxation import WaveformRelaxation
from utils.errornorms import ErrorNorms

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
dt = 0.1
ntime = int((T - t0)/dt) + 1  # total number of time nodes
time_points = np.linspace(t0, T, ntime)  # array of time points


## Example 1 
# def exact(x, y, t):
#     return (1 - t)*np.sin(np.pi*x)*np.sin(2*np.pi*y) + (1 + t**2)*np.sin(2*np.pi*x)*np.sin(np.pi*y)

# def func(x, y, t):
#     return (-1 + 5*np.pi**2*(1 - t))*np.sin(np.pi*x)*np.sin(2*np.pi*y) + (2*t + 5*np.pi**2*(1 + t**2))*np.sin(2*np.pi*x)*np.sin(np.pi*y)

## Example 2 
def exact(x, y, t):
    return ((1 - t)*np.sin(np.pi*x)*np.sin(2*np.pi*y)
        + (1 + t**2)*np.sin(2*np.pi*x)*np.sin(np.pi*y)
        + 1 + x + y)

def func(x, y, t):
    return ((-1 + 5*np.pi**2*(1 - t))*np.sin(np.pi*x)*np.sin(2*np.pi*y)
        + (2*t + 5*np.pi**2*(1 + t**2))*np.sin(2*np.pi*x)*np.sin(np.pi*y))

def construct_dirichlet_bc(femspace: FEMSpace, exact_func, time_points: np.ndarray) -> dict:
    """
    Construct Dirichlet boundary conditions for all time steps.

    Parameters
    ----------
    femspace : FEMSpace
        The finite element space.
    exact_func : callable
        The exact solution function of the form exact(x, y, t).
    time_points : np.ndarray
        Array of time points.

    Returns
    -------
    dirichlet_bc : dict
        Dictionary mapping global node indices to arrays of prescribed values at each time step.
    """
    vertices = femspace.mesh.vertices
    dirichlet_bc = dict()
    for bnodes in femspace.mesh.boundary_vertices():
        x, y = vertices[bnodes]
        dirichlet_bc[bnodes] = exact_func(x, y, time_points)
    return dirichlet_bc

def construct_initial_condition(femspace: FEMSpace, exact_func, t0: float) -> np.ndarray:
    """
    Construct the initial condition vector at time t0.

    Parameters
    ----------
    femspace : FEMSpace
        The finite element space.
    exact_func : callable
        The exact solution function of the form exact(x, y, t).
    t0 : float
        Initial time.

    Returns
    -------
    initial_cond : np.ndarray
        Initial condition vector at time t0.
    """
    vertices = femspace.mesh.vertices
    ndof = vertices.shape[0]
    initial_cond = np.zeros(ndof)
    for i, vertex in enumerate(vertices):
        x, y = vertex
        initial_cond[i] = exact_func(x, y, t0)
    return initial_cond

# FEM solver
Heat_solver = HeatProblem(
            femspace = femspace_sq,
            func = func, 
            dt = dt, 
            t0 = t0, 
            T = T, 
            dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points), 
            icond = construct_initial_condition(femspace_sq, exact, t0),
            tstepper = 'Theta',
            theta = 1)

# Number of subdomains
n = 2

# Reduced order
r = 10

# Number of snapshots for POD basis
nsnap = 21  

# Define solvers
ReducedOWR_solver = ReducedWaveformRelaxation(femspace = femspace_sq,                                        
                        n = 2, 
                        overlap = 1,
                        r = r, 
                        nsnap = nsnap,
                        func = func,
                        dt = dt, 
                        t0 = t0, 
                        T = T, 
                        dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points),
                        icond = construct_initial_condition(femspace_sq, exact, t0),
                        tstepper = 'Theta',
                        theta = 1,
                        method = 'RAS',
                        maxiter = 100,
                        tol = 1e-3)

OWR_solver =  WaveformRelaxation(femspace = femspace_sq,                                        
                                n = 2, 
                                overlap = 1, 
                                func = func,
                                dt = dt, 
                                t0 = t0, 
                                T = T, 
                                dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points),
                                icond = construct_initial_condition(femspace_sq, exact, t0),
                                tstepper = 'Theta',
                                theta = 1,
                                method = 'RAS',
                                maxiter = 100,
                                tol = 1e-3)

# Solve
# uh_solution = Heat_solver.solve()
# owr_solution = OWR_solver.solve()
rowr_solution = ReducedOWR_solver.solve()#True, uh = uh_solution, exact = exact)

# Mesh
mesh_vis = visualize.MeshVisualizer(meshobj = mesh_square)
mesh_vis.visualize(carray = mesh_vis.carray_decomposition(n))

# Visualize solution
visualizer_pde = visualize.SolutionVisualizer(femspace_sq.mesh, rowr_solution, dt)
visualizer_pde.visualize_3d_time_error(exact_func = exact)
visualizer_pde.visualize_3d_time_compare(exact_func = exact)
# visualizer_pde.plot_iteration_error(error_history = ReducedOWR_solver.error_history, linewidth = 0.8, markersize = 3)
visualizer_pde.write_vtk_time_series(exact_func = exact, folder="vtk", prefix="rowrheat")


