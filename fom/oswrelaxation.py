from typing import Callable, Literal, Optional
import numpy as np
from fem.femspace import FEMSpace
from fem.linearsolver import DirectSolver, LinearSolver
from fom.heat import HeatProblem
from utils.errornorms import ErrorNorms
from utils.logger import setup_logger

logger = setup_logger(__name__, level = 'info')

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
        self.subdomains, self.ltog, self.gtol, self.maps, _ = self.femspace.mesh.decompose(n = n, overlap = overlap, version = version)
        logger.info(f"[Schwarz Waveform Relaxation] Mesh decomposition completed. Number of subdomains: {len(self.subdomains)}")
        if self.femspace.dim == 1:
            self.icond = self.h(self.verts)
        else:
            self.icond = self.h(self.verts[:,0], self.verts[:,1])
        self.error_subdomains = {}
        self.error_history = []
        self.solution = []

    def construct_dirichlet_bc(self, subfemspace: FEMSpace, data: dict[int, np.ndarray], method: Literal['AS', 'RAS'], domainID: int) -> np.ndarray:
        """
        Construct Dirichlet boundary values for a specific subdomain, accounting for shared nodes.

        This function computes the Dirichlet boundary values for the nodes of a given subdomain.
        For nodes shared by multiple subdomains, the boundary value is computed considering the method:
        - For 'AS' (Additive Schwarz), the value is the sum of contributions from all subdomains sharing that node.
        - For 'RAS' (Restricted Additive Schwarz), the value is the average of contributions from all subdomains sharing that node.
        However, if any of the shared entries corresponds to subdomain 0 (the original global domain), 
        the Dirichlet value from `gdirichlet` is used directly.             

        Parameters
        ----------
        femspace : FEMSpace
            The finite element space corresponding to the subdomain for which to construct the Dirichlet boundary values. 
        data : dict[int, np.ndarray]
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(local_dofs, ntime)` representing the local solution 
            for each subdomain.
        method : Literal['AS', 'RAS']
            The Schwarz method to use ('AS' or 'RAS').
        domainID : int
            The ID of the subdomain for which Dirichlet data is constructed.

        Note to user
        ------------
        One could include the following line
            dirichlet_bc[dindex][0] = self.icond[local_to_global[dindex]]
        in the code below to explicitly set the initial condition at t0 for the boundary nodes of each subdomain, 
        which can help ensure that the initial condition is correctly enforced at the boundaries of the subdomains. 
        However, since the local solutions in `data` are initialized with the initial condition at t0, and 
        the Dirichlet values from `gdirichlet` also include the initial condition at t0, the initial condition should 
        already be correctly applied to the boundary nodes through the existing logic. Therefore, adding this line may 
        be redundant, but it can serve as an additional safeguard to ensure that the initial condition is properly 
        enforced at the boundaries of the subdomains.
        
        Returns
        -------
        dirichlet_bc : np.ndarray, shape (n_local_boundary_nodes, ntime)
            An array containing the Dirichlet boundary values for the nodes of the specified subdomain, where the values 
            for shared nodes are computed according to the chosen method. The ordering of the nodes in `dirichlet_bc` 
            corresponds to the local node indices of the specified subdomain, so that `dirichlet_bc[local_index, :]` 
            gives the Dirichlet values for all time steps for the index `self.subdomains[domainID].boundary_nodes()[local_index]`.
        """ 
        global_dirichlet = self.dirichlet_values
        global_to_boundary_nodes = self.femspace.gtobd # for whole domain, mapping from global node to boundary node (or -1 if not a boundary node)
        local_to_boundary_nodes = subfemspace.gtobd # for subdomain, mapping from local node to boundary node (or -1 if not a boundary node)
        dirichlet_bc = np.zeros((subfemspace.nbdnodes, global_dirichlet.shape[1]))
        for dindex, dlist in self.maps[domainID].items():
            bindex = local_to_boundary_nodes[dindex] # local boundary node index in the subdomain
            if bindex == -1: # This should not happen since we are only constructing Dirichlet BC for boundary nodes of the subdomain, but we include this check for safety.
                raise ValueError(f"Local DOF {dindex} in subdomain {domainID} is not a boundary node.")
            share = len(dlist) # number of subdomains that shares dindex node
            dirichlet_entry = next(((i, j) for (i, j) in dlist if i == 0), None) # Check if any entry uses Dirichlet value (can be optimized but no need for now since this is only done for boundary nodes and number of subdomains is not large)
            if dirichlet_entry is not None: # If any entry corresponds to subdomain 0 (the original global domain), use the Dirichlet value from gdirichlet directly for this node, regardless of the method.
                _, j = dirichlet_entry
                dirichlet_bc[bindex, :] = global_dirichlet[global_to_boundary_nodes[j], :]
            elif share > 1: # If multiple subdomains share this node, compute the value based on the method (AS or RAS)
                val = sum(data[i][j, :] for (i, j) in dlist) 
                if method == 'RAS':
                    val /= share
                dirichlet_bc[bindex, :] = val
            else: # If only one subdomain shares this node, use its value directly
                i, j = dlist[0]
                dirichlet_bc[bindex, :] = data[i][j, :]
        return dirichlet_bc

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
    
    def store_history(self, subproblems: dict[int, HeatProblem], ntime: int, time_grid: np.ndarray, method: Literal['AS', 'RAS'], oswr_data: dict[int, np.ndarray], uh: Optional[np.ndarray] = None, exact: Optional[Callable] = None, norm: str = 'l2'):
        """
        Store the error history for each subdomain and the global solution.

        This function computes the error norms for each subdomain and the global solution at the current iteration
        and stores them in `self.error_subdomains` and `self.error_history`, respectively. The error norms are computed
        using the `ErrorNorms` class, which takes into account the finite element space, the computed solution, the 
        reference solution (either `uh` or `exact`), and the time points.

        Parameters
        ----------
        subproblems : dict[int, HeatProblem]
            A dictionary containing the HeatProblem instances for each subdomain, keyed by their domain IDs. 
            This is used to access the finite element space for each subdomain when computing error norms.
        ntime : int
            The number of time steps in the simulation.
        time_grid : np.ndarray
            An array containing the time points for the simulation.
        method : {'AS', 'RAS'}
            The Schwarz Waveform Relaxation method used ('AS' for Additive Schwarz, 'RAS' for Restricted Additive Schwarz).
        oswr_data : dict[int, np.ndarray]
            A dictionary containing the solutions for each subdomain at the current iteration.
        uh : np.ndarray, optional
            The finite element solution obtained from solving the global problem on the full mesh. 
            Used as a reference solution for error analysis if provided.
        exact : Callable, optional
            The exact solution function for the Heat problem. Used as a reference solution for error analysis if provided.
        """
        # Compute error norms for each subdomain and store them in self.error_subdomains
        for subdomain in self.subdomains:
            if subdomain.domainID not in self.error_subdomains:
                self.error_subdomains[subdomain.domainID] = []
            subest = ErrorNorms(femspace = subproblems[subdomain.domainID].femspace, u1 = oswr_data[subdomain.domainID], u2 = self.restrict(uh, subdomain.domainID) if uh is not None else None, u_exact = exact, time = time_grid)
            suberror_norm = subest.compute(norm)
            self.error_subdomains[subdomain.domainID].append(suberror_norm)
        # Compute error norm for the global solution and store it in self.error_history
        oswr_sol = self.combine(ntime = ntime, method = method, data = oswr_data)
        est = ErrorNorms(femspace = self.femspace, u1 = oswr_sol, u2 = uh, u_exact = exact, time = time_grid)
        error_norm = est.compute(norm)
        self.error_history.append(error_norm)

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
    
    def combine(self, ntime: int, method: Literal['RAS', 'AS'], data: dict[int, np.ndarray]) -> np.ndarray:
        """
        Assemble a global solution from subdomain solutions.

        Global solution is constructed by mapping local subdomain solutions 
        to the global domain using `self.gindices`. The assembly process depends 
        on the chosen method:

        - For RAS (`method = 'RAS'`), contributions from all subdomains are averaged for 
          shared DOFs, resulting in a weighted average at overlaps. 

        - For AS (`method = 'AS'`), contributions from all subdomains are added together 
          for shared DOFs, resulting in a sum of contributions at overlaps. This can lead to a 
          more accurate solution on overlaps but may require relaxation (omega < 1) for stability 
          when multiple overlapping subdomains are present.

        An important note is that global construction is performed by combining local solutions interior 
        to the subdomains (only interior nodes of each subdomain), while the Dirichlet boundary conditions 
        are applied directly to the global solution. This means that the values at the boundary nodes of 
        the subdomains are not used in the assembly process, but rather the Dirichlet values from `self.g` 
        are applied to the global solution at the corresponding indices. This approach ensures that the 
        boundary conditions are correctly enforced in the global solution, while the interior values 
        from the subdomains are combined according to the chosen method.
        
        Parameters
        ----------
        ntime : int
            Total number of time steps, including the initial condition at t0, so the time 
            points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        method : Literal['RAS', 'AS']
            The Schwarz method to use for combining subdomain solutions ('RAS' or 'AS').
        data : dict[int, np.ndarray]
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(local_dofs, ntime)` representing the local solution 
            for each subdomain.

        Returns
        -------
        global_solution : ndarray, shape (nspace, ntime)
            Global solution assembled from subdomain solutions.

        Raises
        ------
        ValueError
            If `method` is not 'RAS' or 'AS'.
        """
        # Initialize the global solution array of shape (nspace, ntime).
        global_solution = np.zeros((self.nspace, ntime))

        # Apply Dirichlet boundary conditions
        global_solution[self.boundary_nodes, :] = self.dirichlet_values

        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog

        # Assemble global solution from subdomain solutions based on the chosen method
        if method == 'RAS':
            count = np.zeros(self.nspace)  # counts how many subdomains contribute to each global DOF
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_nodes()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index, :] += data[i][local_index, :]
                        count[global_index] += 1
            for ix in range(self.nspace):
                if count[ix] > 0:
                    global_solution[ix, :] /= count[ix]
        elif method == 'AS':
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_nodes()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index, :] += data[i][local_index, :] # sum contributions from all subdomains, to have perfect solution on overlaps remove + sign
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'RAS' or 'AS'.")
        
        # Apply initial conditions - initial condition is given considering the ordering of global solution
        global_solution[:, 0] = self.icond

        return global_solution
    
    @staticmethod
    def boundary_criterion(data_old: dict[int, np.ndarray], data_new: dict[int, np.ndarray]) -> float:
        """
        Evaluate the convergence criterion on subdomain boundaries.

        This function computes
            max_{subdomain_id} max_{local_dofs, ntime} |data_new[subdomain_id] - data_old[subdomain_id]|
        which is the maximum absolute difference between `data_new` and `data_old` across all subdomains, 
        local DOFs, and time steps. This criterion is used to assess the convergence of the Schwarz iteration 
        by checking how much the solutions on the subdomain boundaries have changed between iterations.

        Parameters
        ----------
        data_old : dict[int, np.ndarray]
            Dictionary containing subdomain solutions from the previous iteration.
            Keys are subdomain IDs (starting from 1), and values are arrays of shape
            `(local_dofs, ntime)` representing the local solution for each subdomain 
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
        for subdomain in self.subdomains:
            data = np.zeros((subdomain.nnodes(), ntime))
            
            # Set initial condition at t0 for each subdomain and set Dirichlet values for boundary nodes of each subdomain for all time steps
            data[:, 0] = self.construct_initial(subdomain.domainID) # set initial condition at t0 for each subdomain

            # Set Dirichlet values for boundary nodes of each subdomain for all time steps
            subbdnodes = subdomain.boundary_nodes() # local subdomain boundary nodes
            subbdindices = local_to_global[subdomain.domainID][subbdnodes] # global indices of the local boundary nodes of the subdomain

            # Only keep the nodes that are actually global boundary nodes
            mask = global_to_boundary_nodes[subbdindices] != -1 # mask to identify which local boundary nodes of the subdomain are actually global boundary nodes
            subbdnodes = subbdnodes[mask] # local boundary nodes of the subdomain that are also global boundary nodes
            subbdindices = subbdindices[mask] # global indices of the local boundary nodes of the subdomain that are also global boundary nodes
            global_bd_indices = global_to_boundary_nodes[subbdindices]  # indices into dirichlet_values
            data[subbdnodes, :] = self.dirichlet_values[global_bd_indices, :] # set Dirichlet values for boundary nodes of each subdomain for all time steps, only for those local boundary nodes that are also global boundary nodes
            idata[subdomain.domainID] = data
        return idata

    def solve(self, time_grid: np.ndarray, theta: float = 0.5, lift: str = 'nodal', method: Literal['AS', 'RAS'] = 'RAS', omega: float = 1.0, 
              solver: LinearSolver = DirectSolver(), maxiter: int = 100, tol: float = 1e-3, criterion: Callable[[dict[int, np.ndarray], dict[int, np.ndarray]], float] = boundary_criterion,
              store_solution: tuple = (0, 0), history: bool = False, norm: str = 'l2', uh: Optional[np.ndarray] = None, exact: Optional[Callable] = None) -> np.ndarray:
        """
        Solve the Heat problem using the Overlapping Schwarz Waveform Relaxation (OSWR) method.

        Parameters
        ----------
        time_grid : np.ndarray
            Array of time points, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        theta : float, default = 0.5
            Parameter for the theta time-stepping scheme. It determines the weighting between explicit and implicit contributions.
        lift : str
            Type of lifting function used to solve the local problems. 
            Available options are 'nodal' for nodal lifting, 'harmonic' for harmonic lifting and 'parabolic' for parabolic lifting. Default is 'nodal'.
        method : Literal['AS', 'RAS'], default = 'RAS'
            Schwarz method, either 'RAS' (Restricted Additive Schwarz) or 'AS' (Additive Schwarz).
        omega : float, optional
            Relaxation parameter for the Schwarz iteration. The global iterate is updated as
                u^{k+1} = (1 - omega) u^k + omega * u_tilde^{k+1}.
            Values 0 < omega <= 1 are allowed. Using omega < 1 stabilizes the additive Schwarz method for multiple overlapping subdomains.
            Default is 1.0 (no relaxation).
        solver : LinearSolver, optional
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        maxiter : int, default = 100
            Maximum number of Schwarz iterations.
        tol : float, default = 1e-3
            Convergence tolerance for the stopping criterion.
        criterion : Callable, default = boundary_criterion
            Convergence criterion function. It should accept the old and new subdomain solutions and return a float representing the error.
        store_solution : tuple, default = (0, 0)
            A tuple (domainID, time_step) specifying which subdomain solution and time step to store in `self.solution` for later analysis.
             - `domainID` is the ID of the subdomain whose solution to store (starting from 1).
             - `time_step` is the index of the time step to store (starting from 0, where 0 corresponds to t0). 
             If `store_solution` is (0, 0), no solution will be stored. If `store_solution` is (domainID, time_step), the solution for the specified subdomain and time step will be stored in
            `self.solution` at each iteration for later analysis.
        history : bool, default = False
            Whether to store the error history at each iteration for later analysis.
        norm : str, default = 'l2'
            Norm type to use for error analysis if `history` is True. Options include 'l2', 'linf', etc. (see `ErrorNorms` class for supported norms).
        uh : ndarray, optional
            The finite element solution obtained from solving the global problem on the full mesh. 
            Used for error analysis if `history` is True.
        exact : Callable, optional
            The exact solution function for the Heat problem. Used for error analysis if `history` is True.
        
        Returns
        -------
        np.ndarray, shape (nspace, ntime)
            The computed global solution at the FEM nodes for all time steps, assembled from the subdomain solutions 
            using the specified Schwarz method and relaxation. Each column corresponds to the solution at a specific time step.
        """
        assert theta >= 0 and theta <= 1, f"Invalid theta value {theta}. Must be in [0, 1]."
        assert lift in ('nodal', 'harmonic', 'parabolic'), f"Invalid lift type '{lift}'. Must be 'nodal', 'harmonic' or 'parabolic'."
        assert method in ('AS', 'RAS'), f"Invalid method '{method}'. Must be 'AS' or 'RAS'."
        if history and uh is None and exact is None:
            raise ValueError("Error history cannot be stored because both uh and exact are None. At least one of them must be provided for error analysis.")

        logger.info("="*80)
        logger.info("[Schwarz Waveform Relaxation] Starting solver")
        logger.info(
            f"dim: {self.femspace.dim}D | "
            f"subdomains: {len(self.subdomains)} | "
            f"method: {method} | "
            f"omega: {omega:.2f} | "
            f"maxiter: {maxiter} | "
            f"tol: {tol:.2e}")
        logger.info("="*80)

        # Precompute Dirichlet boundary values for all time steps at the boundary nodes
        if self.femspace.dim == 1:
            self.dirichlet_values = self.g(self.verts[self.boundary_nodes][:, None], time_grid[None, :])
        elif self.femspace.dim == 2:
            self.dirichlet_values = self.g(self.verts[self.boundary_nodes][:, 0][:, None], self.verts[self.boundary_nodes][:, 1][:, None], time_grid[None, :])
        else:
            raise ValueError(f"Unsupported dimension {self.femspace.dim}. Only 1D and 2D are supported.")
        
        # The number of time steps, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        ntime = len(time_grid)

        # Initialize local solution data for each subdomain, which will be updated iteratively.
        data = self.initial_data(ntime)

        # Create Heat problems for each subdomain
        subproblems = {}

        domain = self.femspace.domain
        space = self.femspace.space
        degree = self.femspace.degree

        for subdomain in self.subdomains:
            subfem = FEMSpace(mesh = subdomain, domain = domain, space = space, degree = degree)
            g_local = self.construct_dirichlet_bc(subfemspace = subfem, data = data, method = method, domainID = subdomain.domainID)
            subproblems[subdomain.domainID] = HeatProblem(femspace = subfem, t0 = self.t0, T = self.T, f = self.f, g = g_local, h = self.h)

        # Store the subproblems in the instance for potential later use (e.g., for error analysis, visualization, etc.)
        self.subproblems = subproblems
    
        # Overlapping Schwarz Waveform Relaxation iterations
        error: float = float("inf")
        for iter in range(maxiter):
            new_data = {}
            for subdomain in self.subdomains:
                domainid = subdomain.domainID
                dirichlet_bc = self.construct_dirichlet_bc(subfemspace = subproblems[domainid].femspace, data = data, method = method, domainID = domainid) if iter > 0 else None
                new_data[domainid] = subproblems[domainid].solve(time_grid = time_grid, theta = theta, lift = lift, solver = solver, reuse_load = True, g_new = dirichlet_bc)

            error = criterion(data, new_data)
            logger.info(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Iteration \033[93m{iter + 1}\033[0m: error = \033[91m{error:.6e}\033[0m")

            if omega == 1.0:
                data = new_data
            else:
                for i in data:
                    data[i] = (1 - omega) * data[i] + omega * new_data[i]

            # Store error history for each subdomain and the global solution if history is True
            if history:
                self.store_history(subproblems = subproblems, ntime = ntime, time_grid = time_grid, method = method, oswr_data = data, uh = uh, exact = exact, norm = norm)

            # Store the solution of the specified subdomain at the specified time step for visualization
            if store_solution != (0, 0):
                domainID, time_step = store_solution
                self.solution.append(data[domainID][:, time_step])

            if error < tol:
                logger.info(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Converged after \033[93m{iter + 1}\033[0m iterations with error = \033[91m{error:.6e}\033[0m")
                logger.info("="*80)
                break
        else:
            logger.warning(f"\033[92m[Schwarz Waveform Relaxation]\033[0m Reached max iterations ({maxiter}) with error = \033[91m{error:.6e}\033[0m")
        
        logger.info("\033[92m[Schwarz Waveform Relaxation]\033[0m Solver finished successfully.")
        logger.info("="*80)

        return self.combine(ntime = ntime, method = method, data = data)