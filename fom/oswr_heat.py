from typing import Callable, Literal, Optional
import numpy as np
from fem.femspace import FEMSpace
from fem.linearsolver import DirectSolver, LinearSolver
from fom.heat_fom import HeatProblem
from utils.history import History, HistoryConfig, SpatialMode, finalize, initialize_history, record
from utils.logger import get_logger
from utils.metrics import compute_metrics
logger = get_logger(__name__)

class OSWRProblem():
    def __init__(self, femspace: FEMSpace, t0: float, T: float, f: Callable, g: Callable, h: Callable, n: int, overlap: int, version: int = 1):
        """
        Initialize an Overlapping Schwarz Waveform Relaxation (OSWR) problem for solving the heat equation.

        Parameters
        ----------
        femspace : FEMSpace
            The finite element space representing the full computational domain.
        t0 : float
            Initial time.
        T : float
            Final time.
        f : Callable
            The source function for the heat equation, defined as f(x, t) for 1D or f(x, y, t) for 2D problems.
        g : Callable
            The Dirichlet boundary condition function.
            - It should be defined as g(x, t) for 1D or g(x, y, t) for 2D problems.
        h : Callable
            The initial condition function, defined as h(x) for 1D or h(x, y) for 2D problems.
        n : int
            Number of subdomains to decompose the mesh into.
        overlap : int
            Number of layers added to a non-overlapping decomposition to create overlap.
        version : int, optional
            Version of the decomposition algorithm to use (default is 1).
        """
        self.femspace = femspace
        self.t0 = t0
        self.T = T
        self.f = f
        self.g = g
        self.h = h
        self.n = n
        self.overlap = overlap
        self.nspace = self.femspace.nnodes
        self.verts = self.femspace.mesh.vertices
        self.boundary_nodes = self.femspace.boundary_nodes
        logger.info(f"[Schwarz Waveform Relaxation] Decomposing mesh into {n} subdomains with overlap of {overlap} layers using version {version} ...")
        self.subdomains, self.ltog, self.gtol, self.maps, self.membership = self.femspace.mesh.decompose(n = n, overlap = overlap, version = version)
        logger.info(f"[Schwarz Waveform Relaxation] Mesh decomposition completed. Number of subdomains: {len(self.subdomains)}")
        if self.femspace.dim == 1:
            self.icond = self.h(self.verts)
        else:
            self.icond = self.h(self.verts[:,0], self.verts[:,1])

    def restrict(self, global_solution: np.ndarray, domainID: int) -> np.ndarray:
        """
        Restrict a global solution to a specific subdomain.

        Parameters
        ----------
        global_solution : np.ndarray
            An array containing the global solution values for all degrees of freedom in the full domain. 
            The ordering of the nodes in `global_solution` corresponds to the global node indices of the 
            full domain.
        domainID : int
            The ID of the subdomain to which the global solution should be restricted.

        Returns
        -------
        np.ndarray
            An array containing the restricted values for the degrees of freedom in the specified subdomain.
        """
        return global_solution[self.ltog[domainID], :]
    
    def construct_initial(self, domainID: int) -> np.ndarray:
        """
        Extract the initial condition corresponding to a specific subdomain.

        Parameters
        ----------
        domainID : int
            The ID of the subdomain for which to construct the initial condition.

        Returns
        -------
        np.ndarray
            An array containing the initial values for the degrees of freedom
            in the specified subdomain. The mapping from global to local indices
            is handled via `self.ltog[domainID]`.
        """
        return self.icond[self.ltog[domainID]]

    def construct_dirichlet_bc(self, subfemspace: FEMSpace, ntime: int, data: dict[int, np.ndarray], method: Literal['AS', 'RAS'], domainID: int) -> np.ndarray:
        """
        Construct Dirichlet boundary values for a specific subdomain.

        This function constructs the Dirichlet boundary values for the boundary nodes of a given subdomain
        with domain ID `domainID`. For nodes shared by multiple subdomains, the boundary value is computed 
        considering the method:

        - For 'AS' (Additive Schwarz), the value is the sum of contributions from all subdomains sharing that node.
        - For 'RAS' (Restricted Additive Schwarz), the value is the average of contributions from all subdomains sharing that node.

        However, if any of the shared entries corresponds to subdomain 0 (the original global domain), 
        the Dirichlet value from `self.dirichlet_values` is used directly.        

        Parameters
        ----------
        subfemspace : FEMSpace
            The finite element space corresponding to the subdomain for which to construct the Dirichlet boundary values. 
        ntime : int
            The number of time steps for which to construct the Dirichlet boundary values.
        data : dict[int, np.ndarray]
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(nnodes, ntime)` representing the local solution 
            for each subdomain.
        method : Literal['AS', 'RAS']
            The Schwarz method to use ('AS' or 'RAS').
        domainID : int
            The ID of the subdomain for which Dirichlet data is constructed.
        
        Returns
        -------
        dirichlet_bc : np.ndarray, shape (nbnodes, ntime)
            An array containing the Dirichlet boundary values for the boundary nodes of the specified subdomain, where
            the values for shared nodes are computed according to the chosen method. The ordering of the nodes in `dirichlet_bc` 
            corresponds to the local node indices of the specified subdomain, so that `dirichlet_bc[i, :]` gives the Dirichlet 
            values for all time steps for the index `self.subdomains[domainID].boundary_nodes()[i]`.
        """ 
        # The global Dirichlet values, shape (nspace, ntime)
        global_dirichlet = np.zeros((self.nspace, ntime))
        global_dirichlet[self.boundary_nodes] = self.dirichlet_values

        # The subdomain Dirichlet values, shape (nnodes, ntime)
        dirichlet_bc = np.zeros((subfemspace.nnodes, ntime))
        for node, trmaps in self.maps[domainID].items():
            index = next((i for d, i in trmaps if d == 0), None)
            if index is not None:
                dirichlet_bc[node, :] = global_dirichlet[index, :]
            else:
                values = np.array([data[d][i, :] for d, i in trmaps]) # shape (nmaps, ntime)
                if method == 'AS':
                    dirichlet_bc[node, :] = np.sum(values, axis = 0)
                else: 
                    dirichlet_bc[node, :] = np.mean(values, axis = 0)
        return dirichlet_bc[subfemspace.boundary_nodes, :]
    
    def combine(self, ntime: int, method: Literal['RAS', 'AS'], data: dict[int, np.ndarray]) -> np.ndarray:
        """
        Assemble a global solution from subdomain solutions.

        Global solution is constructed by mapping local subdomain solutions 
        to the global domain using `self.ltog`. The assembly process depends 
        on the chosen method:

        - For RAS (`method = 'RAS'`), contributions from all subdomains are averaged for 
          shared DOFs, resulting in a weighted average at overlaps. 

        - For AS (`method = 'AS'`), contributions from all subdomains are added together 
          for shared DOFs, resulting in a sum of contributions at overlaps. 

        An important note is that global construction is performed by combining local solutions interior 
        to the subdomains (only interior nodes of each subdomain), while the Dirichlet boundary conditions 
        are applied directly to the global solution. This means that the values at the boundary nodes of 
        the subdomains are not used in the assembly process, but rather the Dirichlet values from 
        `self.dirichlet_values` are applied to the global solution at the corresponding indices. This 
        approach ensures that the boundary conditions are correctly enforced in the global solution, 
        while the interior values from the subdomains are combined according to the chosen method.
        
        Parameters
        ----------
        ntime : int
            Total number of time steps, including the initial condition at t0, so the time 
            points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        method : Literal['RAS', 'AS']
            The Schwarz method to use for combining subdomain solutions ('RAS' or 'AS').
        data : dict[int, np.ndarray]
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(nnodes, ntime)` representing the local solution 
            for each subdomain.

        Returns
        -------
        global_solution : ndarray, shape (nspace, ntime)
            Global solution assembled from subdomain solutions.
        """
        assert method in ('RAS', 'AS'), f"Invalid method '{method}'. Must be 'RAS' or 'AS'."

        # Initialize the global solution array of shape (nspace, ntime).
        global_solution = np.zeros((self.nspace, ntime))

        # Apply Dirichlet boundary conditions
        global_solution[self.boundary_nodes, :] = self.dirichlet_values

        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog

        # Assemble global solution from subdomain solutions based on the chosen method
        if method == 'RAS':
            count = np.zeros(self.nspace)  # counts how many subdomains contribute to each global DOF
            for subdomain_id, subdomain in self.subdomains.items():
                bdindices = subdomain.boundary_nodes()
                for local_index, global_index in enumerate(local_to_global[subdomain_id]):
                    if local_index not in bdindices:
                        global_solution[global_index, :] += data[subdomain_id][local_index, :]
                        count[global_index] += 1
            for ix in range(self.nspace):
                if count[ix] > 0:
                    global_solution[ix, :] /= count[ix]
        else:  # method == 'AS'
            for subdomain_id, subdomain in self.subdomains.items():
                bdindices = subdomain.boundary_nodes()
                for local_index, global_index in enumerate(local_to_global[subdomain_id]):
                    if local_index not in bdindices:
                        global_solution[global_index, :] += data[subdomain_id][local_index, :] # sum contributions from all subdomains, to have perfect solution on overlaps remove + sign
        
        # Apply initial conditions - initial condition is given considering the ordering of global solution
        global_solution[:, 0] = self.icond

        return global_solution
    
    @staticmethod
    def max_difference(data_old: dict[int, np.ndarray], data_new: dict[int, np.ndarray]) -> float:
        """
        Compute the maximum absolute difference between two sets of subdomain solutions.

        This function iterates through each subdomain and computes the maximum absolute difference 
        between the old and new solutions for that subdomain across all local DOFs and time steps. 
        The maximum absolute difference is computed as

            max_diff = max(max_diff, np.abs(new_sol - old_sol).max())

        for each subdomain, where `new_sol` and `old_sol` are the local solutions for the current 
        and previous iterations, respectively.

        Parameters
        ----------
        data_old : dict[int, np.ndarray]
            Dictionary containing subdomain solutions from the previous iteration.
            Keys are subdomain IDs (starting from 1), and values are arrays of shape
            `(nnodes, ntime)` representing the local solution for each subdomain 
            at all time steps.
        data_new : dict[int, np.ndarray]
            Dictionary containing subdomain solutions from the current iteration.
            Same format as `data_old`.

        Returns
        -------
        float
            The maximum absolute difference between `data_new` and `data_old` across all subdomains, local DOFs, and time steps.
        """
        max_diff = 0.0
        for subdomain_id in data_old:
            old_sol = data_old[subdomain_id]
            new_sol = data_new[subdomain_id]
            diff = np.abs(new_sol - old_sol).max()
            if diff > max_diff:
                max_diff = diff
        return max_diff
    
    def initial_data(self, ntime: int) -> dict[int, np.ndarray]:
        """
        Initialize the local solution data for each subdomain.

        This function constructs the initial solution data for each subdomain, which includes setting the initial condition at t0 
        for each subdomain and applying the Dirichlet boundary values for the boundary nodes of each subdomain across all time steps. 
        At other nodes (interior nodes of the subdomains), the data is initialized to zero for all time steps except for the 
        initial condition at t0. 

        Parameters
        ----------
        ntime : int
            Total number of time steps, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.

        Returns
        -------
        idata : dict[int, np.ndarray]
            A dictionary mapping subdomain IDs to their initialized local solution arrays.
        """
        idata = {}
        local_to_global = self.ltog
        global_to_boundary_nodes = self.femspace.gtobd # for whole domain, mapping from global node to boundary node (or -1 if not a boundary node)
        for subdomain_id, subdomain in self.subdomains.items():
            data = np.zeros((subdomain.nnodes(), ntime))
            
            # Set initial condition at t0 for each subdomain and set Dirichlet values for boundary nodes of each subdomain for all time steps
            data[:, 0] = self.construct_initial(subdomain_id) # set initial condition at t0 for each subdomain

            # Set Dirichlet values for boundary nodes of each subdomain for all time steps
            subbdnodes = subdomain.boundary_nodes() # local subdomain boundary nodes
            subbdindices = local_to_global[subdomain_id][subbdnodes] # global indices of the local boundary nodes of the subdomain

            # Only keep the nodes that are actually global boundary nodes
            mask = global_to_boundary_nodes[subbdindices] != -1 # mask to identify which local boundary nodes of the subdomain are actually global boundary nodes
            subbdnodes = subbdnodes[mask] # local boundary nodes of the subdomain that are also global boundary nodes
            subbdindices = subbdindices[mask] # global indices of the local boundary nodes of the subdomain that are also global boundary nodes
            global_bd_indices = global_to_boundary_nodes[subbdindices]  # indices into dirichlet_values
            data[subbdnodes, :] = self.dirichlet_values[global_bd_indices, :] # set Dirichlet values for boundary nodes of each subdomain for all time steps, only for those local boundary nodes that are also global boundary nodes
            idata[subdomain_id] = data
        return idata

    def solve(self, time_grid: np.ndarray, theta: float = 0.5, lift: Literal['nodal', 'harmonic', 'parabolic'] = 'nodal', method: Literal['AS', 'RAS'] = 'RAS', 
              solver: LinearSolver = DirectSolver(), maxiter: int = 100, tol: float = 1e-3, criterion: Callable[[dict[int, np.ndarray], dict[int, np.ndarray]], float] = max_difference,
              histconfig: Optional[HistoryConfig] = None) -> tuple[History, np.ndarray] | np.ndarray:
        """
        Solve the Heat problem using the Overlapping Schwarz Waveform Relaxation (OSWR) method.

        Parameters
        ----------
        time_grid : np.ndarray
            Array of time points, including the initial condition at t0, so the time points 
            are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        theta : float, default = 0.5
            Parameter for the theta time-stepping scheme.
        lift : Literal['nodal', 'harmonic', 'parabolic'], default = 'nodal'
            Type of lifting function used to solve the local problems. Available options are 
            'nodal' for nodal lifting, 'harmonic' for harmonic lifting and 'parabolic' for 
            parabolic lifting. Default is 'nodal'.
        method : Literal['AS', 'RAS'], default = 'RAS'
            Schwarz method, either 'RAS' (Restricted Additive Schwarz) or 'AS' (Additive Schwarz).
        solver : LinearSolver, optional
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        maxiter : int, default = 100
            Maximum number of Schwarz iterations.
        tol : float, default = 1e-3
            Convergence tolerance for the stopping criterion.
        criterion : Callable, default = `max_difference`
            Convergence criterion function. It should accept the old and new subdomain solutions 
            and return a float representing the error.
        histconfig : HistoryConfig, optional
            Configuration for tracking convergence history.
        
        Returns
        -------
        np.ndarray, shape (nspace, ntime)
            The computed global solution at the FEM nodes for all time steps, assembled from the subdomain solutions 
            using the specified Schwarz method. Each column corresponds to the solution at a specific time step.
        """
        assert theta >= 0 and theta <= 1, f"Invalid theta value {theta}. Must be in [0, 1]."
        assert lift in ('nodal', 'harmonic', 'parabolic'), f"Invalid lift type '{lift}'. Must be 'nodal', 'harmonic' or 'parabolic'."
        assert method in ('AS', 'RAS'), f"Invalid method '{method}'. Must be 'AS' or 'RAS'."

        logger.info("="*80)
        logger.info("[Schwarz Waveform Relaxation] Starting solver")
        logger.info(
            f"dim: {self.femspace.dim}D | "
            f"subdomains: {len(self.subdomains)} | "
            f"method: {method} | "
            f"maxiter: {maxiter} | "
            f"tol: {tol:.2e}")
        logger.info("="*80)

        # Initialize error history if `histconfig` is provided
        history = initialize_history(histconfig) if histconfig is not None else None

        # Precompute Dirichlet boundary values for all time steps at the boundary nodes
        if self.femspace.dim == 1:
            self.dirichlet_values = self.g(self.verts[self.boundary_nodes][:, None], time_grid[None, :])
        elif self.femspace.dim == 2:
            self.dirichlet_values = self.g(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_grid[None, :])
        else:
            raise ValueError(f"Unsupported dimension {self.femspace.dim}. Only 1D and 2D are supported.")

        # Total number of time steps, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        ntime = len(time_grid)
 
        # Initialize local solution data for each subdomain, which will be updated iteratively.
        data = self.initial_data(ntime)

        # Create Heat problems for each subdomain
        subfems, subproblems = {}, {}
        domain = self.femspace.domain
        space = self.femspace.space
        degree = self.femspace.degree
        for subdomain_id, subdomain in self.subdomains.items():
            subfem = FEMSpace(mesh = subdomain, domain = domain, space = space, degree = degree)
            g_local = self.construct_dirichlet_bc(subfemspace = subfem, ntime = ntime, data = data, method = method, domainID = subdomain_id)
            subfems[subdomain_id] = subfem
            subproblems[subdomain_id] = HeatProblem(femspace = subfem, t0 = self.t0, T = self.T, f = self.f, g = g_local, h = self.h)
    
        # Overlapping Schwarz Waveform Relaxation iterations
        error: float = float("inf")
        for iter in range(maxiter):
            new_data = {}
            for subdomain_id, subdomain in self.subdomains.items():
                dirichlet_bc = self.construct_dirichlet_bc(subfemspace = subproblems[subdomain_id].femspace, ntime = ntime, data = data, method = method, domainID = subdomain_id) if iter > 0 else None
                new_data[subdomain_id] = subproblems[subdomain_id].solve(time_grid = time_grid, theta = theta, lift = lift, solver = solver, reuse_load = True, g_new = dirichlet_bc)

            # Compute error using the provided criterion function, which compares the new subdomain solutions with the previous ones to determine convergence. 
            error = criterion(data, new_data)
            logger.info(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Iteration \033[92m{iter + 1}\033[0m: error = \033[91m{error:.6e}\033[0m")

            # Store error history for each subdomain and/or the global solution if `histconfig` is provided
            if histconfig is not None:
                logger.info(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Computing error metrics for iteration \033[92m{iter + 1}\033[0m ...")
                needs_global = any(spec.spatial in (SpatialMode.GLOBAL, SpatialMode.BOTH) for spec in histconfig.metrics)
                values = compute_metrics(config = histconfig, time_grid = time_grid, ltog = self.ltog, gfemspace = self.femspace, lfemspace = subfems,
                                        current_ldata = new_data, prev_ldata = data, 
                                        current_gdata = self.combine(ntime = ntime, method = method, data = new_data) if needs_global else None,
                                        prev_gdata = self.combine(ntime = ntime, method = method, data = data) if needs_global else None, mode = histconfig.mode)
                assert history is not None
                for metric_spec in histconfig.metrics:
                    metric_values = values.get(metric_spec.name)
                    if metric_values is not None:
                        record(history, metric_spec, metric_values)

            # Update the data for the next iteration
            data = new_data

            if error < tol:
                logger.info(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Converged after \033[92m{iter + 1}\033[0m iterations with error = \033[91m{error:.6e}\033[0m")
                logger.info("="*80)
                break
        else:
            logger.warning(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Reached max iterations ({maxiter}) with error = \033[91m{error:.6e}\033[0m")
        
        if history is not None:
            history = finalize(history)
        
        logger.info("\033[92m[Schwarz Waveform Relaxation]\033[0m Solver finished successfully.")
        logger.info("="*80)

        if history is not None:
            return history, self.combine(ntime = ntime, method = method, data = data)
        else:
            return self.combine(ntime = ntime, method = method, data = data) 