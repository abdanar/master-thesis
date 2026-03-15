import numpy as np
import fem.mesh as mesh
import visualization.visualize as visualize 
from triangle import triangulate
from rom.romheat import HeatROM
from rom.heatrom import TestHeatROM
from fom.heat import HeatProblem
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from rom.reducedowr import ReducedWaveformRelaxation
from fom.owrelaxation import WaveformRelaxation

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

# Reduced order
r = 10

# Number of snapshots for POD basis
nsnap = 21  

# Construct basis matrix for ROM (POD basis from snapshots)

## Create time nodes generation for snapshot collection
# dt_snap = (T - t0)/(nsnap - 1)
# sntime = int((T - t0)/dt_snap) + 1  # total number of time nodes
# time_points_snap = np.linspace(t0, T, sntime)  # array of time points

# Heat_solver = HeatProblem(
#             femspace = femspace_sq,
#             func = func, 
#             dt = dt_snap, 
#             t0 = t0, 
#             T = T, 
#             dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points_snap), 
#             icond = construct_initial_condition(femspace_sq, exact, t0),
#             tstepper = 'Theta',
#             theta = 0.5)

# snapshot_matrix = Heat_solver.solve()

# Perform SVD on the snapshot matrix
# U, _, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)

# Select the first r modes as the POD basis
# pod_basis = U[:, :r]

#pod_basis = np.eye(femspace_sq.mesh.nnodes())

# optROMHeat_solver = OptionalHeatROM(
#             femspace = femspace_sq,
#             basis = pod_basis,
#             func = func, 
#             dt = dt, 
#             t0 = t0, 
#             T = T, 
#             dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points),
#             icond = construct_initial_condition(femspace_sq, exact, t0),
#             tstepper = 'Theta',
#             theta = 0.5)

# Create ROM solver instance
# ROMHeat_solver = HeatROM(
#             femspace = femspace_sq,
#             basis = pod_basis,
#             func = func, 
#             dt = dt, 
#             t0 = t0, 
#             T = T, 
#             dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points),
#             icond = construct_initial_condition(femspace_sq, exact, t0),
#             tstepper = 'Theta',
#             theta = 0.5)

# Solve the ROM problem

#rheat_solution = ROMHeat_solver.solve()
# optrheat_solution = optROMHeat_solver.solve()

#print(np.max(np.abs(rheat_solution[:,1:] - optrheat_solution[:,1:])))

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


rowr_solution = ReducedOWR_solver.solve()

# OWR_solver =  WaveformRelaxation(femspace = femspace_sq,                                        
#                                 n = 2, 
#                                 overlap = 1, 
#                                 func = func,
#                                 dt = dt, 
#                                 t0 = t0, 
#                                 T = T, 
#                                 dirichlet_bc = construct_dirichlet_bc(femspace_sq, exact, time_points),
#                                 icond = construct_initial_condition(femspace_sq, exact, t0),
#                                 tstepper = 'Theta',
#                                 theta = 0.5,
#                                 method = 'RAS',
#                                 maxiter = 100,
#                                 tol = 1e-3)

# owr_solution = OWR_solver.solve()

visualizer_pde = visualize.SolutionVisualizer(femspace_sq.mesh, rowr_solution, dt)
visualizer_pde.visualize_3d_time_compare(exact_func = exact)
visualizer_pde.visualize_3d_time_error(exact_func = exact)

#visualizer_pder = visualize.SolutionVisualizer(femspace_sq.mesh, rheat_solution, dt)
# visualizer_pdeopt = visualize.SolutionVisualizer(femspace_sq.mesh, optrheat_solution, dt)
# visualizer_pde = visualize.SolutionVisualizer(femspace_sq.mesh, snapshot_matrix, dt_snap)
# visualizer_pde.write_vtk_time_series(exact_func = exact, folder="vtk", prefix="heat_solution")
# visualizer_pde.write_vtk_time_series(exact_func = exact, folder="vtk", prefix="heat_solution")
# visualizer_pde.visualize_3d_time()
# visualizer_pde.visualize_3d_time_compare(exact_func = exact)
# visualizer_pdeopt.visualize_3d_time_compare(exact_func = exact)
#visualizer_pder.visualize_3d_time_compare(exact_func = exact)
# visualizer_pde.visualize_3d_time_error(exact_func = exact)
# visualizer_pdeopt.visualize_3d_time_error(exact_func = exact)
#visualizer_pder.visualize_3d_time_error(exact_func = exact)
# femspace_sq.visualize_3d_time_compare(heat_solution, exact, dt)



