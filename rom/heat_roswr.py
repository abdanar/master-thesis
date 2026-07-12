from typing import Callable, Literal, Optional
import numpy as np
from rom.pod import POD
from fem.femspace import FEMSpace
from fem.linearsolver import DirectSolver, LinearSolver
from fom.heat_fom import HeatProblem
from rom.heat_rom import ReducedHeatProblem
from utils.history import History, HistoryConfig, SpatialMode, finalize, initialize_history, record
from utils.logger import get_logger
from utils.metrics import compute_metrics
from fem.mesh import DecompositionInfo
logger = get_logger(__name__)

class ROSWRHeat():
    def __init__(self, heat_problem: HeatProblem, decinfo: DecompositionInfo):
        """
        Initialize a Reduced-Order Schwarz Waveform Relaxation (ROSWR) problem 
        for solving the `heat_problem`.

        Parameters
        ----------
        heat_problem : HeatProblem
            An instance of the `HeatProblem` class that defines the full-order heat problem 
            to be reduced using the ROSWR method.
        decinfo : DecompositionInfo
            An instance of the `DecompositionInfo` dataclass that contains the necessary 
            information about the mesh decomposition, including the local-to-global and 
            global-to-local mappings, the subdomain maps, and the membership array.
        """
        self.heat_problem = heat_problem
        self.icond = heat_problem.icond
        self.femspace = heat_problem.femspace
        self.verts = self.femspace.mesh.vertices
        self.boundary_nodes = self.femspace.boundary_nodes
        self.nsub = decinfo.nsub
        self.subids = decinfo.subdomain_ids
        self.subdomains, self.ltog, self.gtol, self.maps, self.version = decinfo.submeshes, decinfo.ltog, decinfo.gtol, decinfo.subdomain_maps, decinfo.version
        self.subfemspaces = self._construct_subfemspaces()
        self.history: Optional[History] = None
        self.iterates = {}
    
    def _compute_dirichlet_values(self, time_grid: np.ndarray) -> np.ndarray:
        """
        Compute the Dirichlet boundary values for all time steps at 
        the boundary nodes of the global domain.

        Parameters
        ----------
        time_grid : np.ndarray
            Array of time points, including the initial condition at t0, so the time 
            points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        
        Returns
        -------
        dirichlet_values : np.ndarray, shape (nbnodes, ntime)
            An array containing the Dirichlet boundary values for all time steps at 
            the boundary nodes of the global domain. The ordering of the nodes in 
            `dirichlet_values` corresponds to the ordering of `self.boundary_nodes`,
            so that `dirichlet_values[i, :]` gives the Dirichlet values for all 
            time steps for the index `self.boundary_nodes[i]`.
        """
        dirichlet = self.heat_problem.g
        if isinstance(dirichlet, np.ndarray):
            return dirichlet
        else:
            if self.femspace.dim == 1:
                return dirichlet(self.verts[self.boundary_nodes][:, None], time_grid[None, :])
            else: # self.femspace.dim == 2:
                return dirichlet(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_grid[None, :])

    def _construct_subfemspaces(self) -> dict[int, FEMSpace]:
        """
        Return the finite element spaces for each subdomain based on
        the original finite element space.

        This function iterates through each subdomain defined in `decinfo.submeshes`
        and constructs a corresponding finite element space using the same domain, 
        space, and degree as the original finite element space (`heat_problem.femspace`).
        
        Returns
        -------
        dict[int, FEMSpace]
            A dictionary mapping each subdomain ID to its corresponding finite element space.
        """
        subfemspaces = {}
        for subdomain_id, subdomain in self.subdomains.items():
            subfemspaces[subdomain_id] = FEMSpace(mesh = subdomain, domain = self.femspace.domain, space = self.femspace.space, degree = self.femspace.degree)
        return subfemspaces

    def _construct_subproblems(self, ntime: int, data: dict[int, np.ndarray], bases: dict[int, np.ndarray]) -> dict[int, ReducedHeatProblem]:
        """
        Return the Reduced-Order Heat problems for each subdomain based on the original 
        Heat problem and the provided data.

        This function constructs a `ReducedHeatProblem` instance for each subdomain by using the 
        corresponding finite element space, initial condition extracted from the global 
        initial condition using the local-to-global mapping, and the Dirichlet boundary 
        values constructed using the `construct_dirichlet_bc` method which takes into 
        account the contributions from all subdomains sharing the boundary nodes.

        Parameters
        ----------
        ntime : int
            Total number of time steps, including the initial condition at t0, 
            so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        data : dict[int, np.ndarray]
            Dictionary containing reconstructed subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(nnodes, ntime)` representing the local reconstructed solution 
            for each subdomain.
        bases : dict[int, np.ndarray]
            Dictionary containing the reduced basis for each subdomain. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(nnodes, r)` representing the reduced basis for each subdomain.
        
        Returns
        -------
        dict[int, ReducedHeatProblem]
            A dictionary mapping each subdomain ID to its corresponding `ReducedHeatProblem` instance.
        """
        subproblems = {}
        source = self.heat_problem.f
        t0, T = self.heat_problem.t0, self.heat_problem.T
        for subdomain_id in self.subids:
            subfem = self.subfemspaces[subdomain_id]
            h_local = self.icond[self.ltog[subdomain_id]]
            g_local = self.construct_dirichlet_bc(domainID = subdomain_id, ntime = ntime, data = data)
            subfom = HeatProblem(femspace = subfem, t0 = t0, T = T, f = source, g = g_local, h = h_local)
            subproblems[subdomain_id] = ReducedHeatProblem(heat_problem = subfom, V = bases[subdomain_id])
        return subproblems
    
    def construct_dirichlet_bc(self, domainID: int, ntime: int, data: dict[int, np.ndarray]) -> np.ndarray:
        """
        Return the Dirichlet boundary values for the domain with ID `domainID`.

        This function constructs the Dirichlet boundary values for the boundary 
        nodes of a given subdomain with ID `domainID`. For nodes shared by multiple 
        subdomains, the boundary value is computed as the average of contributions 
        from all subdomains sharing that node. However, if any of the shared entries 
        corresponds to subdomain 0 (the original global domain), the corresponding 
        Dirichlet value is used directly.

        Parameters
        ----------
        domainID : int
            The ID of the subdomain for which to construct the Dirichlet boundary values.
        ntime : int
            The number of time steps for which to construct the Dirichlet boundary values.
        data : dict[int, np.ndarray]
            Dictionary containing reconstructed subdomain solutions. Keys are subdomain IDs 
            (starting from 1), and values are arrays of shape `(nnodes, ntime)` representing 
            the local reconstructed reduced solution for each subdomain.
        
        Returns
        -------
        dirichlet_bc : np.ndarray, shape (nbnodes, ntime)
            An array containing the Dirichlet boundary values for the boundary nodes of the 
            specified subdomain. The ordering of the nodes in `dirichlet_bc` corresponds to 
            the local node indices of the specified subdomain, so that `dirichlet_bc[i, :]` 
            gives the Dirichlet values for all time steps for the index 
            `subdomains[domainID].boundary_nodes()[i]`.
        """ 
        # Get the finite element space for the specified subdomain
        subfemspace = self.subfemspaces[domainID]
        # The global Dirichlet values, shape (femspace.nnodes, ntime)
        global_dirichlet = np.zeros((self.femspace.nnodes, ntime))
        global_dirichlet[self.boundary_nodes] = self.dirichlet_values
        # The subdomain Dirichlet values, shape (nnodes, ntime)
        dirichlet_bc = np.zeros((subfemspace.nnodes, ntime))
        for node, trmaps in self.maps[domainID].items():
            index = next((i for d, i in trmaps if d == 0), None)
            if index is not None:
                dirichlet_bc[node, :] = global_dirichlet[index, :]
            else:
                values = np.array([data[d][i, :] for d, i in trmaps]) # shape (nmaps, ntime)
                dirichlet_bc[node, :] = np.mean(values, axis = 0)
        return dirichlet_bc[subfemspace.boundary_nodes, :]

    def _construct_bases(self, pod: POD, option: Literal['noDQ', 'DQ'] = 'noDQ') -> dict[int, np.ndarray]:
        """
        Construct the reduced basis for each subdomain using the provided data.

        This function constructs a reduced basis for each subdomain by performing 
        Proper Orthogonal Decomposition (POD) on the local solution data for each 
        subdomain. The number of modes retained in the reduced basis is determined 
        by the parameter `r`.

        Parameters
        ----------
        pod : POD
            Proper Orthogonal Decomposition (POD) object containing necessary data for local
            POD bases construction, including the snapshots, the number of modes `r`, and 
            the energy tolerance `energy_tol`.
        option : Literal['noDQ', 'DQ'], optional
            Option to specify whether to use standard POD ('noDQ') or Difference 
            Quotient POD ('DQ') for constructing the reduced basis. Default is 'noDQ'.
        
        Returns
        -------
        dict[int, np.ndarray]
            A dictionary mapping each subdomain ID to its corresponding reduced basis, where 
            each reduced basis is an array of shape `(nintnodes, r)`.
        """
        bases = {}
        self.weights = {}
        gsnapshots = np.zeros((self.femspace.nnodes, pod.snapshots.shape[1]))
        gsnapshots[self.femspace.interior_nodes, :] = pod.snapshots
        dt = (self.heat_problem.T - self.heat_problem.t0) / (gsnapshots.shape[1] - 1)
        for subdomain_id in self.subids:
            logger.info(f"[Reduced Schwarz Waveform Relaxation] Computing POD basis for subdomain {subdomain_id} ...")
            idx = self.ltog[subdomain_id]
            intnodes = self.subfemspaces[subdomain_id].interior_nodes
            weight = pod.weight[np.ix_(idx, idx)][np.ix_(intnodes, intnodes)] if pod.weight is not None else None
            pod_reductor = POD(snapshots = gsnapshots[idx][intnodes], r = pod.r, energy_tol = pod.energy_tol, weight = weight)
            if option == 'DQ':
                pod_reductor.snapshots = pod_reductor.dq_snapshots(dt) # update snapshots to include difference quotients for DQ POD
            self.weights[subdomain_id] = pod_reductor.weight
            bases[subdomain_id] = pod_reductor.basis()
            logger.info(f"[Reduced Schwarz Waveform Relaxation] POD basis for subdomain {subdomain_id} computed. Shape: {bases[subdomain_id].shape}")    
        return bases

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
            data[:, 0] = self.icond[self.ltog[subdomain_id]] # set initial condition at t0 for each subdomain

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
    
    def combine(self, data: dict[int, np.ndarray]) -> np.ndarray:
        """
        Assemble a global solution from subdomain solutions.

        Global solution is constructed by mapping local subdomain solutions 
        to the global domain using `ltog`. The contributions from all 
        subdomains are averaged for shared DOFs, resulting in a weighted 
        average at overlaps. 
        
        Parameters
        ----------
        data : dict[int, np.ndarray]
            Dictionary containing subdomain solutions. Keys are subdomain IDs 
            (starting from 1),and values are arrays of shape `(nnodes, ntime)` 
            representing the local solution for each subdomain.

        Returns
        -------
        global_solution : ndarray, shape (nnodes, ntime)
            Global solution assembled from subdomain solutions.

        Notes
        -----
        The global solution is constructed by combining the interior values 
        from the subdomains, while the Dirichlet boundary conditions are applied 
        directly to the global solution. This means that the values at the boundary 
        nodes of the subdomains are not used in the assembly process, but rather 
        the Dirichlet values from `dirichlet_values` are applied to the global 
        solution at the corresponding indices.
        """
        # Number of global nodes in the original global domain
        nnodes = self.femspace.nnodes
        # Initialize the global solution array of shape (nnodes, ntime).
        global_solution = np.zeros((nnodes, data[next(iter(data))].shape[1]))
        # Apply Dirichlet boundary conditions
        global_solution[self.boundary_nodes, :] = self.dirichlet_values
        # Assemble global solution from subdomain solutions
        count = np.zeros(nnodes)  # counts how many subdomains contribute to each global DOF
        for subdomain_id, subdomain in self.subdomains.items():
            bdindices = subdomain.boundary_nodes()
            for local_index, global_index in enumerate(self.ltog[subdomain_id]):
                if local_index not in bdindices:
                    global_solution[global_index, :] += data[subdomain_id][local_index, :]
                    count[global_index] += 1
        for ix in range(nnodes):
            if count[ix] > 0:
                global_solution[ix, :] /= count[ix]
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
            The maximum absolute difference between `data_new` and `data_old` across 
            all subdomains, local DOFs, and time steps.
        """
        max_diff = 0.0
        for subdomain_id in data_old:
            old_sol = data_old[subdomain_id]
            new_sol = data_new[subdomain_id]
            diff = np.abs(new_sol - old_sol).max()
            if diff > max_diff:
                max_diff = diff
        return max_diff
    
    def _record_history(self, iter: int, time_grid: np.ndarray, histconfig: HistoryConfig, 
                        data_old: dict[int, np.ndarray], data_new: dict[int, np.ndarray]):
        """
        Compute and record the metrics specified in `histconfig` for the current Schwarz iteration.

        This function first checks if the `history` attribute is initialized, and if not, 
        it initializes it using the provided `histconfig`. Then, it computes the metrics 
        specified in `histconfig` for the current iteration by calling the `compute_metrics`
        function, which takes into account the current and/or previous subdomain solutions, 
        as well as the global solution if needed. Finally, it records the computed metric 
        values in the history object for later analysis.

        Parameters
        ----------
        iter : int
            The current iteration number (starting from 0).
        time_grid : np.ndarray
            Array of time points, including the initial condition at t0, so the time points are
            t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        histconfig : HistoryConfig
            Configuration for tracking history.
        data_old : dict[int, np.ndarray]
            Dictionary containing subdomain reconstructed reduced solutions from the previous iteration.
            Keys are subdomain IDs (starting from 1), and values are arrays of shape
            `(nnodes, ntime)` representing the local reconstructed reduced solution for each subdomain 
            at all time steps.
        data_new : dict[int, np.ndarray]
            Dictionary containing subdomain reconstructed reduced solutions from the current iteration.
            Same format as `data_old`.
        """
        # Initialize error history if `histconfig` is provided
        if self.history is None:
            self.history = initialize_history(histconfig)
        logger.info(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Computing error metrics for iteration \033[92m{iter + 1}\033[0m ...")
        # Compute metrics specified in `histconfig` for the current iteration
        needs_global = any(spec.spatial in (SpatialMode.GLOBAL, SpatialMode.BOTH) for spec in histconfig.metrics)
        values = compute_metrics(config = histconfig, time_grid = time_grid, ltog = self.ltog, gfemspace = self.femspace, 
                                 lfemspace = self.subfemspaces, current_ldata = data_new, prev_ldata = data_old, 
                                 current_gdata = self.combine(data = data_new) if needs_global else None,
                                 prev_gdata = self.combine(data = data_old) if needs_global else None, mode = histconfig.mode)
        # Record the computed metric values in the history object
        for metric_spec in histconfig.metrics:
            metric_values = values.get(metric_spec)
            if metric_values is not None:
                record(self.history, metric_spec, metric_values)

    def reset_history(self):
        self.history = None
        self.iterates = {}

    def solve(self, pod: POD, time_grid: np.ndarray, theta: float = 1.0, lift: Literal['nodal', 'harmonic'] = 'nodal', 
              solver: LinearSolver = DirectSolver(), criterion: Callable = max_difference, maxiter: int = 100, 
              tol: float = 1e-9, option: Literal['noDQ', 'DQ'] = 'noDQ', omega: float = 1.0, combine: bool = True, 
              histconfig: Optional[HistoryConfig] = None, store_solution: Optional[list | bool] = None) -> dict[int, np.ndarray] | np.ndarray:
        """
        Solve the Heat problem using the Reduced-Order Schwarz Waveform Relaxation (ROSWR) method.

        Parameters
        ----------
        pod : POD
            Proper Orthogonal Decomposition (POD) object containing necessary data for local
            POD bases construction. This object is for global snapshots, and the local POD 
            bases will be constructed from the global snapshots using the local-to-global 
            mapping. Therefore, `pod.snapshots` should contain the global snapshots of 
            shape `(nintnodes, ntime)`, where `nintnodes` is the number of interior nodes 
            of the global mesh and `ntime` is the number of time steps.
        time_grid : np.ndarray
            Array of time points, including the initial condition at t0, so the time points 
            are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        theta : float, default = 1.0
            Parameter for the theta time-stepping scheme.
        lift : Literal['nodal', 'harmonic'], default = 'nodal'
            Type of lifting function used to solve the local problems. Available options are 
            'nodal' for nodal lifting and 'harmonic' for harmonic lifting. Default is 'nodal'.
        solver : LinearSolver, optional
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        criterion : Callable, default = `max_difference`
            Convergence criterion function. It should accept the old and new subdomain solutions 
            and return a float representing the error, i.e., input values are dictionaries with keys 
            as subdomain IDs and values as arrays of shape `(nnodes, ntime)` representing the local 
            reconstructed reduced solution for each subdomain at all time steps. The function should compute a scalar error 
            value that is used to check for convergence.
        maxiter : int, default = 100
            Maximum number of Schwarz iterations.
        tol : float, default = 1e-9
            Convergence tolerance for the stopping criterion.
        option : Literal['noDQ', 'DQ'], default = 'noDQ'
            The method to use for computing local POD bases. Available options are 
            - 'noDQ' for standard POD
            - 'DQ' for Difference Quotient POD
        omega : float, default = 1.0
            Relaxation parameter for the Schwarz iteration. Default is 1.0 (no relaxation).
        combine : bool, default = True
            If True, the final global reconstructed reduced solution is assembled from the subdomain solutions. 
            If False, a dictionary of subdomain reconstructed reduced solutions is returned.
        store_solution : list | bool, optional
            If True, the subdomain reconstructed reduced solutions for all iterations are stored. If a list of iteration 
            indices is provided, only the subdomain reconstructed reduced solutions for those iterations are stored. The 
            stored solutions are returned in a dictionary with keys as iteration indices and values as dictionaries of 
            subdomain reconstructed reduced solutions, where each subdomain reconstructed reduced solution is an array 
            of shape `(nnodes, ntime)` representing the local reconstructed reduced solution for that subdomain at all 
            time steps.
        histconfig : HistoryConfig, optional
            Configuration for tracking history.

        Returns
        -------
        solution : dict[int, np.ndarray] | np.ndarray
            If `combine` is True, returns a global reconstructed reduced solution array of shape `(nnodes, ntime)`. 
            If `combine` is False, returns a dictionary mapping subdomain IDs to their local reconstructed reduced 
            solution arrays of shape `(nnodes, ntime)`, where nnodes is the number of nodes in the subdomain.
        """
        assert omega > 0 and omega <= 1, "Relaxation parameter omega must be in the range (0, 1]."
        logger.info("-"*80)
        logger.info("[Reduced Schwarz Waveform Relaxation] Starting solver")
        logger.info(f"dim: {self.femspace.dim}D | " f"r: {pod.r} | " f"subdomains: {self.nsub} | " 
                    f"version: {self.version} | " f"theta: {theta} | " f"lift: {lift} | " f"maxiter: {maxiter} | " 
                    f"tol: {tol:.2e} | " f"omega: {omega}")
        logger.info("-"*80)

        # Total number of time steps, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        ntime = len(time_grid)

        # Precompute Dirichlet boundary values for all time steps at the boundary nodes if needed
        self.dirichlet_values = self._compute_dirichlet_values(time_grid)

        # Initialize local solution data for each subdomain, which will be updated iteratively.
        data = self.initial_data(ntime)

        # Construct the reduced basis for each subdomain
        bases = self._construct_bases(pod = pod, option = option)

        # Construct local Heat problems for each subdomain
        subproblems = self._construct_subproblems(ntime = ntime, data = data, bases = bases)
    
        # Reduced-Order Schwarz Waveform Relaxation iterations
        error: float = float("inf")
        for iter in range(maxiter):
            new_data = {}
            # Solve local Reduced-Order Heat problems 
            for subdomain_id in self.subids:
                dirichlet_bc = self.construct_dirichlet_bc(domainID = subdomain_id, ntime = ntime, data = data) if iter > 0 else None
                new_data[subdomain_id] = subproblems[subdomain_id].solve(time_grid = time_grid, theta = theta, lift = lift, solver = solver, 
                                                                         weight = self.weights[subdomain_id] if pod.weight is not None else None, 
                                                                         reuse_load = True, g_new = dirichlet_bc)
            # Apply relaxation
            if omega < 1.0:
                for subdomain_id in new_data:
                    new_data[subdomain_id] = omega * new_data[subdomain_id] + (1 - omega) * data[subdomain_id]
            # Compute error using the provided criterion function
            error = criterion(data, new_data)
            logger.info(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Iteration \033[92m{iter + 1}\033[0m: error = \033[91m{error:.6e}\033[0m")
            # Store error history for each subdomain and/or the global solution if `histconfig` is provided
            if histconfig is not None:
                self._record_history(iter = iter, time_grid = time_grid, histconfig = histconfig, data_old = data, data_new = new_data)
            # Store the subdomain solutions for the current iteration if `store_solution` is True or if the current iteration index is in the provided list of iteration indices to store
            if store_solution is True or (isinstance(store_solution, list) and iter + 1 in store_solution):
                self.iterates[iter + 1] = {subdomain_id: new_data[subdomain_id] for subdomain_id in new_data}
            # Update the data for the next iteration
            data = new_data
            if error < tol:
                logger.info(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Converged after \033[92m{iter + 1}\033[0m iterations with error = \033[91m{error:.6e}\033[0m")
                logger.info("-"*80)
                break
        else:
            logger.warning(f"\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Reached max iterations ({maxiter}) with error = \033[91m{error:.6e}\033[0m")
    
        logger.info("\033[92m[Reduced Schwarz Waveform Relaxation]\033[0m Solver finished successfully.")
        logger.info("-"*80)
        self.history = finalize(self.history) if self.history is not None else None
        solution = self.combine(data) if combine else data
        return solution