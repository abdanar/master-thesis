import numpy as np
from fom.heat import HeatProblem
from fem.femspace import FEMSpace
from utils.logger import setup_logger
from typing import Callable, Optional
from fem.linearsolver import LinearSolver, DirectSolver
from utils.errornorms import ErrorNorms

logger = setup_logger(__name__, level = 'info')

class OSWRProblem():
    def __init__(self, femspace: FEMSpace, t0: float, T: float, f: Callable, g: Callable, h: Callable, n: int, overlap: int):
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
        """
        self.femspace = femspace
        self.t0 = t0
        self.T = T
        self.f = f
        self.g = g
        self.h = h
        self.n = n
        self.overlap = overlap
        logger.info(f"[Schwarz Waveform Relaxation] Decomposing mesh into {n} subdomains with overlap of {overlap} layers...")
        self.subdomains, self.ltog, self.gtol, self.maps, _ = self.femspace.mesh.decompose(n = n, overlap = overlap)
        logger.info(f"[Schwarz Waveform Relaxation] Mesh decomposition completed. Number of subdomains: {len(self.subdomains)}")
        self.nspace = self.femspace.nnodes
        verts = self.femspace.mesh.vertices
        boundary_nodes = self.femspace.mesh.boundary_nodes()
        logger.info(f"[Schwarz Waveform Relaxation] Evaluating Dirichlet boundary functions at {len(boundary_nodes)} boundary nodes and initial condition at {self.nspace} nodes...")
        # `dirichlet_func` can be optimized by using vectorized evaluation of g at all boundary nodes and time steps, rather than calling g for each node and time step separately.
        if self.femspace.dim == 1:
            self.dirichlet_func = lambda t: {j: self.g(verts[j], t) for j in boundary_nodes}
            self.icond = self.h(verts)
        else:
            self.dirichlet_func = lambda t: {j: self.g(*verts[j], t) for j in boundary_nodes}
            self.icond = self.h(verts[:,0], verts[:,1])
        logger.info(f"[Schwarz Waveform Relaxation] Dirichlet boundary functions and initial condition evaluated successfully.")
        self.error_history = [] # also possible to store subdomain-wise error history for more detailed analysis of convergence behavior on each subdomain and at the overlaps, which can provide insights into how the errors evolve in different regions of the domain during the Schwarz iterations.

    def construct_dirichlet_bc(self, gdirichlet: dict, method: str, domainID: int, data: dict) -> dict:
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
        gdirichlet : dict
            The dictionary mapping global node indices to the corresponding Dirichlet values 
            at all time steps, i.e., {global_node_index: [value_at_t0, value_at_t1, ...]}. 
        method : str
            The Schwarz method to use ('AS' or 'RAS').
        domainID : int
            The ID of the subdomain for which Dirichlet data is constructed.
        data : dict
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(local_dofs, ntime)` representing the local solution 
            for each subdomain.

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
        dirichlet_bc : dict
            A dictionary mapping local boundary node indices of `domainID` to their 
            Dirichlet boundary values at all time steps, including initial condition at t0. 
            The keys are local node indices within the subdomain, and the values are 
            arrays of shape `(ntime,)` representing the Dirichlet values at each time step for those nodes.
        """ 
        dirichlet_bc = {}
        local_to_global = self.ltog[domainID]
        for dindex, dlist in self.maps[domainID].items():
            share = len(dlist) # number of subdomains that shares dindex node
            dirichlet_entry = next(((i, j) for (i, j) in dlist if i == 0), None) # Check if any entry uses Dirichlet value
            if dirichlet_entry is not None: # If any entry corresponds to subdomain 0 (the original global domain), use the Dirichlet value from gdirichlet directly for this node, regardless of the method. This ensures that the boundary conditions are correctly enforced at the global level, while the contributions from the subdomains are used for the interior nodes and overlaps.
                _, j = dirichlet_entry
                dirichlet_bc[dindex] = gdirichlet[j]
            elif share > 1: # If multiple subdomains share this node, compute the value based on the method (AS or RAS)
                val = sum(data[i][j, :] for (i, j) in dlist) 
                if method == 'AS':
                    dirichlet_bc[dindex] = val
                elif method == 'RAS':
                    dirichlet_bc[dindex] = val/share
                else:
                    raise ValueError(f"Invalid method '{method}'. Must be 'RAS' or 'AS'.")
            else: # If only one subdomain shares this node, use its value directly
                i, j = dlist[0]
                dirichlet_bc[dindex] = data[i][j, :]
        return dirichlet_bc
    
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
    
    def combine(self, gdirichlet: dict, ntime: int, method: str, data: dict) -> np.ndarray:
        """
        Assemble a global solution from subdomain solutions.

        Global solution is constructed by mapping local subdomain solutions 
        to the global domain using `self.ltog`. The assembly process depends 
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
        gdirichlet : dict
            The dictionary mapping global node indices to the corresponding Dirichlet values 
            at all time steps, i.e., {global_node_index: [value_at_t0, value_at_t1, ...]}. 
        ntime : int
            Total number of time steps, including the initial condition at t0, so the time 
            points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        method : str
            The Schwarz method to use for combining subdomain solutions ('RAS' or 'AS').
        data : dict
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
        for idx, values in gdirichlet.items():
            global_solution[idx, :] = values

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
    def boundary_criterion(data_old: dict, data_new: dict) -> float:
        """
        Evaluate the convergence criterion on subdomain boundaries.

        Parameters
        ----------
        data_old : dict
            Dictionary containing subdomain solutions from the previous iteration.
            Keys are subdomain IDs (starting from 1), and values are arrays of shape
            `(local_dofs, ntime)` representing the local solution for each subdomain 
            at all time steps.
        data_new : dict
            Dictionary containing subdomain solutions from the current iteration.
            Same format as `data_old`.

        Returns
        -------
        float
            Maximum difference on boundary nodes between the old and new solutions
            across all subdomains and all time steps.
        """
        max_diff = 0.0
        for subdomain_id in data_old:
            old_sol = data_old[subdomain_id]
            new_sol = data_new[subdomain_id]
            diff = np.abs(new_sol - old_sol).max()
            if diff > max_diff:
                max_diff = diff
        return max_diff
    
    def initial_data(self, gdirichlet: dict, ntime: int) -> dict:
        """
        Initialize the local solution data for each subdomain.

        This function creates a dictionary mapping subdomain IDs to their initialized local 
        solution arrays. The initialization depends on the type of `self.g`:
        - If `self.g` is a dictionary, the local solution arrays are initialized to zero, and the 
          Dirichlet values from `self.g` are applied to the corresponding indices.
        - If `self.g` is a function, the local solution arrays are initialized to zero, and the 
          Dirichlet values are evaluated at the boundary nodes of each subdomain and applied accordingly.     

        Parameters
        ----------
        gdirichlet : dict
            The dictionary mapping global node indices to the corresponding Dirichlet values 
            at all time steps, i.e., {global_node_index: [value_at_t0, value_at_t1, ...]}.

        Returns
        -------
        idata : dict
            A dictionary mapping subdomain IDs to their initialized local solution arrays.
        """
        # The dictionary that contains dictionary for global dof to local dof mappings for each subdomains with keys to be domainID
        global_to_local = self.gtol

        # Construct initial data for each subdomain
        idata = {}
        for subdomain in self.subdomains:
            global_indices = global_to_local[subdomain.domainID]
            data = np.zeros((subdomain.nnodes(), ntime))
            for idx, values in gdirichlet.items():
                if idx in global_indices:
                    data[global_indices[idx]] = values
            idata[subdomain.domainID] = data
        return idata

    def solve(self, ntime: int, theta: float = 0.5, lift: str = 'nodal', method: str = 'RAS', omega: float = 1.0, 
              solver: LinearSolver = DirectSolver(), maxiter: int = 100, tol: float = 1e-3, criterion: Callable = boundary_criterion,
              history: bool = False, norm: str = 'l2', uh: Optional[np.ndarray] = None, exact: Optional[Callable] = None) -> np.ndarray:
        """
        Solve the Heat problem using the Overlapping Schwarz Waveform Relaxation (OSWR) method.

        Parameters
        ----------
        ntime : int
            Total number of time steps, including the initial condition at t0, so the time points are t0, t1, ..., t_{ntime-1} with t_{ntime-1} = T.
        theta : float, default = 0.5
            Parameter for the theta time-stepping scheme. It determines the weighting between explicit and implicit contributions.
        lift : str
            Type of lifting function used to solve the local problems. 
            Available options are 'nodal' for nodal lifting, 'harmonic' for harmonic lifting and 'parabolic' for parabolic lifting. Default is 'nodal'.
        method : str, default = 'RAS'
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

        # Precompute Dirichlet boundary values for all time steps at the global level, which can be reused across iterations and subdomains.
        dirichlet = self.dirichlet_func(np.linspace(self.t0, self.T, ntime)) 

        error: float = float("inf")

        # Initialize local solution data for each subdomain, which will be updated iteratively.
        data = self.initial_data(ntime = ntime, gdirichlet = dirichlet)

        # Create Heat problems for each subdomain
        subproblems = {subdomain.domainID: HeatProblem(femspace = FEMSpace(mesh = subdomain, domain = self.femspace.domain, space = self.femspace.space, degree = self.femspace.degree),
            t0 = self.t0, T = self.T, f = self.f, g = self.construct_dirichlet_bc(gdirichlet = dirichlet, method = method, domainID = subdomain.domainID, data = data), h = self.h) for subdomain in self.subdomains} 

        for iter in range(maxiter):
            new_data = {}
            for subdomain in self.subdomains:
                domainid = subdomain.domainID
                dirichlet_bc = self.construct_dirichlet_bc(gdirichlet = dirichlet, method = method, domainID = domainid, data = data) if iter > 0 else None
                new_data[domainid] = subproblems[domainid].solve(ntime = ntime, theta = theta, lift = lift, solver = solver, g_new = dirichlet_bc)

            error = criterion(data, new_data)
            logger.info(f"[Schwarz Waveform Relaxation] Iteration {iter + 1}: error = {error:.6e}")

            if error < tol:
                logger.info(f"[Schwarz Waveform Relaxation] Converged after {iter + 1} iterations with error = {error:.6e}")
                logger.info("="*80)
                break
            else:
                if omega == 1.0:
                    data = new_data
                else:
                    for i in data:
                        data[i] = (1 - omega) * data[i] + omega * new_data[i]

            if history: # store error history
                oswr_sol = self.combine(gdirichlet = dirichlet, ntime = ntime, method = method, data = data)
                est = ErrorNorms(femspace = self.femspace, u1 = oswr_sol, u2 = uh, u_exact = exact, time = np.linspace(self.t0, self.T, ntime))
                error_norm = est.compute(norm)
                logger.info(f"[Schwarz Waveform Relaxation] Iteration {iter + 1}: error norm ({norm}) = {error_norm:.6e}")
                self.error_history.append(error_norm)
        else:
            logger.warning(f"[Schwarz Waveform Relaxation] Reached max iterations ({maxiter}) with error = {error:.6e}")
        
        logger.info("[Schwarz Waveform Relaxation] Solver finished successfully")
        logger.info("="*80)

        return self.combine(gdirichlet = dirichlet, ntime = ntime, method = method, data = data)