from typing import Callable, Literal, Optional
import numpy as np
from fem.femspace import FEMSpace
from fem.linearsolver import DirectSolver, LinearSolver
from fom.heat_fom import HeatProblem
from rom.heat_rom import ReducedHeatProblem
from utils.history import History, HistoryConfig, SpatialMode, finalize, initialize_history, record
from utils.logger import get_logger
from utils.metrics import compute_metrics
logger = get_logger(__name__)

class ROSWRProblem():
    def __init__(self, heat_problem: HeatProblem, n: int, overlap: int, version: int = 1):
        """
        Initialize a Reduced Overlapping Schwarz Waveform Relaxation (ROSWR) problem for solving the heat equation.

        Parameters
        ----------
        heat_problem : HeatProblem
            An instance of the `HeatProblem` class that defines the full-order heat problem to be solved using the Reduced Schwarz method. 
        n : int
            Number of subdomains to decompose the mesh into.
        overlap : int
            Number of layers added to a non-overlapping decomposition to create overlap.
        version : int, optional
            Version of the decomposition algorithm to use (default is 1).
        """
        self.heat_problem = heat_problem
        self.femspace = heat_problem.femspace
        self.t0 = heat_problem.t0
        self.T = heat_problem.T
        self.f = heat_problem.f
        self.g = heat_problem.g
        self.h = heat_problem.h
        self.icond = heat_problem.icond
        self.n = n
        self.overlap = overlap
        self.version = version
        self.nspace = self.femspace.nnodes
        self.verts = self.femspace.mesh.vertices
        self.boundary_nodes = self.femspace.boundary_nodes
        logger.info(f"[Reduced Schwarz Waveform Relaxation] Decomposing mesh into {n} subdomains with overlap of {overlap} layers using version {version} ...")
        self.subdomains, self.ltog, self.gtol, self.maps, self.membership = self.femspace.mesh.decompose(n = n, overlap = overlap, version = version)
        logger.info(f"[Reduced Schwarz Waveform Relaxation] Mesh decomposition completed. Number of subdomains: {len(self.subdomains)}")
        self.iterates = {}
    
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

    def solve(self, projs: dict[int, np.ndarray], time_grid: np.ndarray, theta: float = 1.0, lift: str = 'nodal', method: Literal['AS', 'RAS'] = 'RAS', 
            solver: LinearSolver = DirectSolver(), maxiter: int = 100, tol: float = 1e-9, criterion: Callable[[dict[int, np.ndarray], dict[int, np.ndarray]], float] = max_difference,
            combine: bool = True, store_solution: Optional[list] = None, histconfig: Optional[HistoryConfig] = None) -> tuple[History, dict | np.ndarray] | dict | np.ndarray:
        """
        Solve the Reduced Overlapping Schwarz Waveform Relaxation (ROSWR) method.

        Parameters
        ----------
        projs: dict[int, np.ndarray]
            A dictionary mapping subdomain IDs to their respective projection matrices for the reduced-order model. 
        time_grid : np.ndarray
            Array of time points, including the initial condition at t0, so the time points 
            are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        theta : float, default = 1.0
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
            and return a float representing the error, i.e., input values are dictionaries with keys 
            as subdomain IDs and values as arrays of shape `(nnodes, ntime)` representing the local 
            solution for each subdomain at all time steps. The function should compute a scalar error 
            value that is used to check for convergence.
        combine : bool, default = True
            If True, the final global solution is assembled from the subdomain solutions. If False, a dictionary of subdomain solutions is returned.
        store_solution : list, optional
            If provided, for the given iteration indices, the corresponding subdomain solutions will be stored in a dictionary
            with keys as iteration indices and values as dictionary of subdomain solutions with keys as subdomain IDS and values
            as arrays of shape `(nnodes, ntime)` representing the local solution for each subdomain at all time steps.
        histconfig : HistoryConfig, optional
            Configuration for tracking convergence history.

        Returns
        -------
        If `histconfig` is provided, returns a tuple containing:
        - `history`: An instance of the `History` class that contains the recorded convergence history according to the specified `histconfig`.
        - `solution`: The final global solution assembled from the subdomain solutions at the last iteration, returned as a NumPy array of shape `(nspace, ntime)`
        if `combine` is True, or a dictionary of subdomain solutions if `combine` is False.
        """
        assert theta >= 0 and theta <= 1, f"Invalid theta value {theta}. Must be in [0, 1]."
        assert lift in ('nodal', 'harmonic', 'parabolic'), f"Invalid lift type '{lift}'. Must be 'nodal', 'harmonic' or 'parabolic'."
        assert method in ('AS', 'RAS'), f"Invalid method '{method}'. Must be 'AS' or 'RAS'."

        logger.info("="*80)
        logger.info("[Reduced Schwarz Waveform Relaxation] Starting solver")
        logger.info(
            f"dim: {self.femspace.dim}D | "
            f"subdomains: {len(self.subdomains)} | "
            f"method: {method} | "
            f"r: {projs[1].shape[1]} | "
            f"theta: {theta} | "
            f"lift: {lift} | "
            f"version: {self.version} | "
            f"maxiter: {maxiter} | "
            f"tol: {tol:.2e}")
        logger.info("="*80)

        # Total number of time steps, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        ntime = len(time_grid)

        # Initialize error history if `histconfig` is provided
        history = initialize_history(histconfig) if histconfig is not None else None

        if isinstance(self.g, np.ndarray):
            assert self.g.shape == (self.femspace.nbdnodes, ntime), f"Invalid shape for boundary values array. Expected {(self.femspace.nbdnodes, ntime)}, got {self.g.shape}."
            self.dirichlet_values = self.g
        else:
            # Precompute Dirichlet boundary values for all time steps at the boundary nodes
            if self.femspace.dim == 1:
                self.dirichlet_values = self.g(self.verts[self.boundary_nodes][:, None], time_grid[None, :])
            elif self.femspace.dim == 2:
                self.dirichlet_values = self.g(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_grid[None, :])
            else:
                raise ValueError(f"Unsupported dimension {self.femspace.dim}. Only 1D and 2D are supported.")

        # Initialize local solution data for each subdomain, which will be updated iteratively.
        data = self.initial_data(ntime)

        # Create Heat problems for each subdomain
        subfems, subroms = {}, {}
        domain = self.femspace.domain
        space = self.femspace.space
        degree = self.femspace.degree
        for subdomain_id, subdomain in self.subdomains.items():
            subfem = FEMSpace(mesh = subdomain, domain = domain, space = space, degree = degree)
            h_local = self.construct_initial(subdomain_id)
            g_local = self.construct_dirichlet_bc(subfemspace = subfem, ntime = ntime, data = data, method = method, domainID = subdomain_id)
            subfom = HeatProblem(femspace = subfem, t0 = self.t0, T = self.T, f = self.f, g = g_local, h = h_local)
            subfems[subdomain_id] = subfem
            subroms[subdomain_id] = ReducedHeatProblem(heat_problem = subfom, V = projs[subdomain_id])

        # Reduced Schwarz Waveform Relaxation iteration
        error: float = float("inf")
        for iter in range(maxiter):
            new_data = {}
            for subdomain_id, subdomain in self.subdomains.items():
                dirichlet_bc = self.construct_dirichlet_bc(subfemspace = subroms[subdomain_id].femspace, ntime = ntime, data = data, method = method, domainID = subdomain_id) if iter > 0 else None
                new_data[subdomain_id] = subroms[subdomain_id].solve(time_grid = time_grid, theta = theta, lift = lift, solver = solver, reuse_load = True, g_new = dirichlet_bc, reconstruct = True)

            # Compute convergence criterion
            error = criterion(data, new_data)
            logger.info(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Iteration \033[92m{iter + 1}\033[0m: error = \033[91m{error:.6e}\033[0m")

            # Store error history for each subdomain and/or the global solution if `histconfig` is provided
            if histconfig is not None:
                logger.info(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Computing error metrics for iteration \033[92m{iter + 1}\033[0m ...")
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

            # Store the (reconstructed) subdomain solutions for the current iteration if `store_solution` is provided and the current iteration index is in `store_solution`.
            if store_solution is not None and (iter + 1) in store_solution:
                self.iterates[iter + 1] = {subdomain_id: new_data[subdomain_id] for subdomain_id in new_data}

            # Update the data for the next iteration
            data = new_data

            if error < tol:
                logger.info(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Converged after \033[92m{iter + 1}\033[0m iterations with error = \033[91m{error:.6e}\033[0m")
                logger.info("="*80)
                break
        else:
            logger.warning(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Reached max iterations ({maxiter}) with error = \033[91m{error:.6e}\033[0m")
        
        if history is not None:
            history = finalize(history)

        logger.info("\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Solver finished successfully.")
        logger.info("="*80)

        solution = self.combine(ntime = ntime, method = method, data = data) if combine else data

        if history is not None:
            return history, solution
        else:
            return solution