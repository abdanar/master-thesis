import numpy as np
from heat import HeatProblem
from mesh import Mesh
from logger import setup_logger

logger = setup_logger(__name__, level = 'info')

class WaveformRelaxation():

    def __init__(self, mesh: Mesh, n: int, overlap: int, func, dt: float, t0: float, T: float, dirichlet_bc: dict, icond: np.ndarray, degree: int = 1, tstepper: str = 'BackwardEuler', theta: float = 0.5, method: str = 'RAS', maxiter: int = 100, tol: float = 1e-3, criterion: str = 'boundary'):
        
        """
        Initialize a Waveform Relaxation solver for time-dependent PDEs over a decomposed domain.

        Parameters
        ----------
        mesh : Mesh
            The global mesh representing the full computational domain.
        n : int
            Number of subdomains to decompose the mesh into.
        overlap : int
            Number of overlapping layers between subdomains (0 for non-overlapping decomposition).
        func : callable
            Source function of the PDE, e.g., f(x, t).
        dt : float
            Time step size.
        t0 : float
            Initial time.
        T : float
            Final time.
        dirichlet_bc : dict
            Dirichlet boundary conditions for the whole domain.
            Keys are global node indices, values are the corresponding Dirichlet values.
        icond : np.ndarray
            Initial condition vector for the whole domain at time t0.
        degree : int, default=1
            Polynomial degree for finite element discretization.
        tstepper : str, default='BackwardEuler'
            Time-stepping method to use, e.g., 'BackwardEuler', 'CrankNicolson', 'Theta'.
        theta : float, default=0.5
            Parameter for the θ-method, 0 < θ ≤ 1. Only used if tstepper='Theta'.
        method : str, default='RAS'
            Waveform relaxation method, either 'RAS' (Restricted Additive Schwarz) or 'AS' (Additive Schwarz).
        maxiter : int, default=100
            Maximum number of waveform relaxation iterations.
        tol : float, default=1e-3
            Convergence tolerance for the stopping criterion.
        criterion : str, default='boundary'
            Type of convergence criterion, e.g., 'boundary' for checking subdomain boundary changes.
        """
        
        self.mesh = mesh
        self.n = n
        self.overlap = overlap
        self.subdomains, self.ltog, _ = mesh.decompose(n = n, overlap = overlap)  # list of subdomains of type `Mesh` and dict of local to global mappings
        self.f = func
        self.dt = dt
        self.t0 = t0
        self.T = T
        self.dirichlet = dirichlet_bc # Dirichlet boundary condition for the whole domain, for the main problem
        self.initial = icond  # initial condition for the whole domain, for the main problem, Initial condition vector at t0 for all nodes of whole domain
        self.degree = degree
        self.tstepper = tstepper
        self.theta = theta
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.criterion = criterion # stopping criterion
        self.ntime = int((T - t0)/dt) + 1 # total number of time nodes
        self.nspace = mesh.nvertices() + mesh.nedges()*(degree - 1) + mesh.nelements()*(degree - 1)*(degree - 2)//2 # total number of space nodes (for whole domain)
    
    def construct_dirichlet_bc(self, domainID: int, maps: dict, data: dict) -> dict:

        """
        Construct Dirichlet boundary values for a specific subdomain.

        This function computes the Dirichlet boundary values for the nodes of a given 
        subdomain, taking into account nodes that are shared between multiple subdomains.
        For nodes shared by multiple subdomains, the boundary value is computed as the 
        average of the corresponding values. If a subdomain ID in `maps` is 0, the value 
        is taken from `self.dirichlet`.

        Parameters
        ----------
        domainID : int
            The ID of the subdomain for which Dirichlet data is constructed.
        maps : dict
            A dictionary of subdomain mappings, typically obtained from 
            `self.mesh.subdomain_mapping(n=self.n, overlap=self.overlap)`. 
            For a given subdomain `maps[domainID]`, each key is a local boundary node index, 
            and the corresponding value is a list of tuples `(subdomain_id, node_index)` 
            indicating which subdomains share this node and its index in those subdomains.
        data : dict
            A dictionary where keys are subdomain IDs and values are numpy arrays of shape 
            `(num_nodes, num_time_steps)`. Column `j` of `data[i]` contains the solution 
            `u(x, t_j)` for the nodes of subdomain `i`.

        Returns
        -------
        dirichlet_bc : dict
            A dictionary mapping local boundary node indices of `domainID` to their 
            Dirichlet boundary values. Shared nodes are assigned the average value across 
            the corresponding subdomains. Nodes associated with subdomain 0 use values 
            from `self.dirichlet`.

        Notes
        -----
        - Assumes that `data` contains the solution for all relevant subdomains and time steps.
        - Shared nodes are averaged over all subdomains listed in `maps`.
        - Subdomain 0 is treated as a special case using `self.dirichlet`.
        """
        # Not correct!
        dirichlet_bc = {}
        for dindex, dlist in maps[domainID].items():
            share = len(dlist) # number of subdomains that shares dindex node
            if share > 1:
                # Average over all subdomains sharing this node
                val = 0
                for (i, j) in dlist:
                    if i == 0:
                        val += (1/share)*self.dirichlet[j]
                    else:
                        val += (1/share)*data[i][j, :]
                dirichlet_bc[dindex] = val
            else:
                i, j = dlist[0]
                if i == 0:
                    dirichlet_bc[dindex] = self.dirichlet[j]
                else:
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

        return self.initial[self.ltog[domainID]]
    
    def combine(self, data: dict) -> np.ndarray:

        """
        Assemble a global solution from subdomain solutions.

        Parameters
        ----------
        data : dict
            Dictionary containing subdomain solutions. Keys are subdomain IDs (starting from 1),
            and values are arrays of shape `(local_dofs, ntime)` representing the local solution 
            for each subdomain at all time steps.

        Returns
        -------
        global_solution : ndarray, shape (nspace, ntime)
            Global solution assembled from subdomain solutions. Each column represents the solution
            at a specific time step: `global_solution[:, i] = u(x, t_i)`.
            - For RAS (`self.method = 'RAS'`), only interior DOFs of each subdomain contribute; overlaps
            are ignored during assembly.
            - For AS (`self.method = 'AS'`), contributions from all subdomains are added and overlaps
            are averaged.

        Raises
        ------
        ValueError
            If `self.method` is not 'RAS' or 'AS'.

        Notes
        -----
        - The function uses `self.ltog` for mapping local subdomain DOFs to global DOFs.
        - Overlap/interface DOFs are handled according to the chosen assembly method:
        RAS restricts to interior contributions, AS averages overlaps.
        """

        # Initialize the global solution array of shape (nspace, ntime).
        # Each column corresponds to the solution at a specific time step: global_solution[:, i] = u(x, t_i),
        # with the first column representing the initial condition at t0.
        global_solution = np.zeros((self.nspace, self.ntime))

        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog

        if self.method == 'RAS':
            seen = set()
            for i in range(1, self.n + 1):
                for local_index, global_index in enumerate(local_to_global[i]):
                    if global_index not in seen:
                        global_solution[global_index, :] = data[i][local_index, :]
                        seen.add(global_index)
        elif self.method == 'AS':
            count = np.zeros(self.nspace)  # counts how many subdomains contribute to each global DOF
            for i in range(1, self.n + 1):
                for local_index, global_index in enumerate(local_to_global[i]):
                    global_solution[global_index, :] += data[i][local_index, :]
                    count[global_index] += 1
            # divide by number of contributions to average overlaps
            for idx in range(self.nspace):
                if count[idx] > 0:
                    global_solution[idx, :] /= count[idx]
        else:
            raise ValueError(f"Invalid method '{self.method}'. Must be 'RAS' or 'AS'.")

        return global_solution
    
    def boundary_criterion(self, data_old: dict, data_new: dict) -> float:

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
    
    def initial_data(self) -> dict:
        idata = {}
        for subdomain in self.subdomains:
            nspace = subdomain.nvertices() + subdomain.nedges()*(self.degree - 1) + subdomain.nelements()*(self.degree - 1)*(self.degree - 2)//2
            idata[subdomain.domainID] = np.zeros((nspace, self.ntime))
        return idata

    def solve(self) -> np.ndarray:

        logger.info("="*80)
        logger.info("[Waveform Relaxation] Starting solver")
        logger.info(f"Number of subdomains: {len(self.subdomains)} | max iterations: {self.maxiter} | tolerance: {self.tol:.2e}")
        logger.info("="*80)

        maps = self.mesh.subdomain_mapping(self.subdomains)

        initial_data = self.initial_data()

        for iter in range(self.maxiter):
            new_data = {}
            for subdomain in self.subdomains:
                domainid = subdomain.domainID
                initial_cond = self.construct_initial(domainid)
                dirichlet_bc = self.construct_dirichlet_bc(domainID = domainid, maps = maps, data = initial_data)
                logger.debug(f"The constructed dirichlet boundary conditions for a subdomain {domainid} in iteration {iter + 1}: {dirichlet_bc}")
                subdomain_heat = HeatProblem(
                    mesh = subdomain, 
                    func = self.f, 
                    dt = self.dt, 
                    t0 = self.t0, 
                    T = self.T, 
                    dirichlet_bc = dirichlet_bc, 
                    icond = initial_cond, 
                    tstepper = self.tstepper,
                    theta = self.theta)
                new_data[domainid] = subdomain_heat.solve()

            error = self.boundary_criterion(initial_data, new_data)

            logger.info(f"[Waveform Relaxation] Iteration {iter + 1}: error = {error:.6e}")

            if error < self.tol:
                logger.info(f"[Waveform Relaxation] Converged after {iter + 1} iterations with error = {error:.6e}")
                logger.info("="*80)
                break
            else:
                initial_data = new_data
        else:
            logger.warning(f"[Waveform Relaxation] Reached max iterations ({self.maxiter}) with error = {error:.6e}")
        
        logger.info("[Waveform Relaxation] Solver finished successfully")
        logger.info("="*80)

        return self.combine(initial_data)