import numpy as np
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fom.heat import HeatProblem
from rom.heatrom import HeatROM
from utils.errornorms import ErrorNorms
from utils.logger import setup_logger

logger = setup_logger(__name__, level = 'info')

class ReducedWaveformRelaxation():

    def __init__(self, femspace: FEMSpace, n: int, overlap: int, r: int, nsnap: int, func, dt: float, t0: float, T: float, 
                 dirichlet_bc: dict, icond: np.ndarray, tstepper: str = 'Theta', theta: float = 0.5, method: str = 'RAS', maxiter: int = 100, 
                 tol: float = 1e-3, criterion: str = 'boundary'):
        """
        Initialize a Reduced Order Waveform Relaxation solver for time-dependent PDEs over a decomposed domain.

        Parameters
        ----------
        femspace : FEMSpace
            The finite element space representing the full computational domain.
        n : int
            Number of subdomains to decompose the mesh into.
        overlap : int
            Number of overlapping layers between subdomains (0 for non-overlapping decomposition).
        r : int
            Reduced order
        nsnap : int
            Number of snapshots to use for POD basis construction.
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
        tstepper : str, default='Theta'
            Time-stepping method to use.
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
        self.femspace = femspace
        self.n = n
        self.overlap = overlap
        self.r = r
        self.nsnap = nsnap
        self.subdomains, self.ltog, self.maps, _ = self.femspace.mesh.decompose(n = n, overlap = overlap)  # list of subdomains of type `Mesh` and dict of local to global mappings
        self.f = func
        self.dt = dt
        self.t0 = t0
        self.T = T
        self.dirichlet = dirichlet_bc # Dirichlet boundary condition for the whole domain, for the main problem
        self.initial = icond  # initial condition for the whole domain, for the main problem, Initial condition vector at t0 for all nodes of whole domain
        self.tstepper = tstepper
        self.theta = theta
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.criterion = criterion # stopping criterion
        self.ntime = int((T - t0)/dt) + 1 # total number of time nodes
        self.nspace = self.femspace.mesh.nnodes() # total number of space nodes (for whole domain)
        self.basis = self.setup_bases()  # dictionary: domainID -> basis matrix (Store reduced bases for all subdomains)
        self.error_history = []
        self.time = np.arange(self.t0, self.T + self.dt, self.dt)
    
    def restriction_matrix(self, domainID: int) -> np.ndarray:
        """
        Construct the restriction matrix for a specific subdomain.
        The restriction matrix maps the global degrees of freedom (DOFs) to the local DOFs
        of the specified subdomain. (It includes all DOFs, both interior and boundary.)

        Parameters
        ----------
        domainID : int
            Identifier of the subdomain.
        
        Returns
        -------
        np.ndarray
            Restriction matrix of shape (local_dofs, global_dofs) for the specified subdomain.
        """
        restriction_matrix = np.zeros((self.subdomains[domainID - 1].nnodes(), self.nspace)) 
        local_to_global = self.ltog[domainID]
        for local_node, global_node in enumerate(local_to_global):
            restriction_matrix[local_node, global_node] = 1
        return restriction_matrix
    
    def snapshot_matrix(self) -> np.ndarray:
        Heat_solver = HeatProblem(
                    femspace = self.femspace,
                    func = self.f, 
                    dt = (self.T - self.t0)/(self.nsnap - 1), 
                    t0 = self.t0, 
                    T = self.T, 
                    dirichlet_bc = self.dirichlet, 
                    icond = self.initial, 
                    tstepper = 'Theta',
                    theta = self.theta)
        return Heat_solver.solve()
    
    def pod_basis(self, domainID: int, smatrix: np.ndarray) -> np.ndarray:
        """
        Construct the POD basis for a specific subdomain.

        The POD basis is built by restricting the full-order FEM solution
        to the given subdomain and performing Proper Orthogonal Decomposition (POD)
        on this restricted data to generate a local reduced space associated with the subdomain.

        Parameters
        ----------
        domainID : int
            Identifier of the subdomain.

        Returns
        -------
        np.ndarray
            POD basis matrix for the subdomain, whose columns form
            a basis of the local reduced space.
        """
        # Restriction matrix for the subdomain
        R = self.restriction_matrix(domainID)

        # Collect snapshots restricted to the subdomain
        snapshot_matrix = R @ smatrix  # Shape: (local_dofs, nsnap)

        # Perform SVD on the snapshot matrix
        U, _, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)

        # Select the first r modes as the POD basis
        pod_basis = U[:, :self.r]  # Shape: (local_dofs, r)

        return pod_basis

    def construct_basis(self, domainID: int, smatrix: np.ndarray, method: str = 'POD') -> np.ndarray:
        """
        Construct the reduced basis for a specific subdomain.

        The reduced basis is built by restricting the full-order FEM solution
        to the given subdomain and using this restricted data to generate
        a local reduced space associated with the subdomain.

        Parameters
        ----------
        domainID : int
            Identifier of the subdomain.

        Returns
        -------
        np.ndarray
            Reduced basis matrix for the subdomain, whose columns form
            a basis of the local reduced space.
        """

        if method == 'POD':
            return self.pod_basis(domainID, smatrix)
        else:
            raise ValueError(f"Invalid basis construction method '{method}'. Only 'POD' is supported.")

    def setup_bases(self):
        basis = {}
        for domainID in range(1, self.n + 1):
            subd = self.subdomains[domainID - 1]
            n = subd.nnodes()                  # total nodes
            bnodes = set(self.maps[domainID].keys())  # boundary node indices
            interior_nodes = [i for i in range(n) if i not in bnodes]
            r = len(interior_nodes)            # reduced dim = number of interior nodes

            # Initialize Phi_i with zeros
            mat = np.zeros((n, r))

            # Fill identity on interior DOFs
            for col, i in enumerate(interior_nodes):
                mat[i, col] = 1.0

            basis[domainID] = mat

        return basis
    
    # def setup_bases(self):
    #     basis = {}
    #     # smatrix = self.snapshot_matrix()  # collect snapshots for the whole domain
    #     for domainID in range(1, self.n + 1):
    #         n = self.subdomains[domainID - 1].nnodes()
    #         bnodes = self.maps[domainID].keys()
    #         mat = np.zeros((n, n - len(bnodes)))
    #         for i in range(n):
    #             if i not in bnodes:
    #                 mat[i, i] = 1.0
    #         basis[domainID] = mat #np.eye(self.subdomains[domainID - 1].nnodes()) #self.construct_basis(domainID, smatrix) #np.eye(self.subdomains[domainID - 1].nnodes())
    #     return basis

    def construct_dirichlet_bc(self, domainID: int, maps: dict, data: dict) -> dict:
        """
        Construct Dirichlet boundary values for a specific subdomain, accounting for shared nodes.

        This function constructs Dirichlet boundary data for the boundary nodes of a given
        subdomain `domainID`. For nodes shared by multiple subdomains, the treatment depends
        on the chosen Schwarz method:
        
        - If **any** of the shared entries corresponds to subdomain 0, the Dirichlet value
          prescribed in `self.dirichlet` is used directly (this takes precedence).
        - Otherwise, the values from all sharing subdomains are combined:
            - For the Additive Schwarz method (`method == 'AS'`), the values are summed.
            - For the Restricted Additive Schwarz method (`method == 'RAS'`), the values
              are averaged.

        Parameters
        ----------
        domainID : int
            The ID of the subdomain for which Dirichlet data is constructed.
        maps : dict
            A dictionary of subdomain mappings, typically obtained from `self.maps`. 
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
            Dictionary mapping local boundary node indices of `domainID` to numpy arrays
            containing the Dirichlet boundary values over all time steps.(including initial)

        Notes
        -----
        - Assumes that `data` contains the solution for all relevant subdomains and time steps.
        - Nodes shared by multiple subdomains are combined according to the selected Schwarz method.
        - Subdomain 0 is treated as a special case: its Dirichlet values override any combination with other subdomains.
        """
        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog[domainID]
        dirichlet_bc = {}
        for dindex, dlist in maps[domainID].items():
            share = len(dlist) # number of subdomains that shares dindex node
            # Check if any entry uses Dirichlet value
            dirichlet_entry = next(((i, j) for (i, j) in dlist if i == 0), None)
            if dirichlet_entry is not None:
                # If any i==0, assign Dirichlet value directly
                _, j = dirichlet_entry
                dirichlet_bc[dindex] = self.dirichlet[j]
            elif share > 1:
                # Average over all subdomains sharing this node
                val = sum(data[i][j, :] for (i, j) in dlist) 
                if self.method == 'AS':
                    dirichlet_bc[dindex] = val
                    dirichlet_bc[dindex][0] = self.initial[local_to_global[dindex]]
                elif self.method == 'RAS':
                    dirichlet_bc[dindex] = val/share
                    dirichlet_bc[dindex][0] = self.initial[local_to_global[dindex]]
                else:
                    raise ValueError(f"Invalid method '{self.method}'. Must be 'RAS' or 'AS'.")
            else:
                # Only one subdomain, take its value
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

        # Apply Dirichlet boundary conditions
        for idx, values in self.dirichlet.items():
            global_solution[idx, :] = values

        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog

        if self.method == 'RAS':
            count = np.zeros(self.nspace)  # counts how many subdomains contribute to each global DOF
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_vertices()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index, :] += data[i][local_index, :] # notice that if you remove +, then you will get perfect plot, but for AS, local solutions should be plotted
                        count[global_index] += 1
            for ix in range(self.nspace):
                if count[ix] > 0:
                    global_solution[ix, :] /= count[ix]
        elif self.method == 'AS':
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_vertices()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index, :] += data[i][local_index, :] # notice that if you remove +, then you will get perfect plot, but for AS, local solutions should be plotted
        else:
            raise ValueError(f"Invalid method '{self.method}'. Must be 'RAS' or 'AS'.")
        
        # Apply initial conditions - initial condition is given considering the ordering of global solution
        global_solution[:, 0] = self.initial 

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
    
    def dirichlet_vector(self, domainID: int, data: dict) -> np.ndarray:
        dvector = np.zeros((self.subdomains[domainID - 1].nnodes(), self.ntime))
        for dindex, dvalue in data.items():
            dvector[dindex, :] = dvalue
        return dvector
    
    def initial_data(self) -> dict:
        idata = {}
        for subdomain in self.subdomains:
            idata[subdomain.domainID] = np.zeros((subdomain.nnodes(), self.ntime))
        return idata

    def offline(self):
        
        # Here we will compute all offline computations

        if self.femspace.dim == 2:
            diffusion = lambda x, y: np.eye(2)
            reaction  = lambda x, y: 1
        else:  # 1D
            diffusion = lambda x: 1
            reaction  = lambda x: 1

        m1 = {} # Phi.T M_i
        m2 = {} # Phi.T K_i = Phi.T (M_i + delta t A_i)
        m3 = {} # Phi.T M_i Phi
        m4 = {} # Phi.T K_i Phi = Phi.T (M_i + delta t A_i) Phi

        for subdomain in self.subdomains:
            domainid = subdomain.domainID
            subspace = FEMSpace(mesh = subdomain, domain = self.femspace.domain, space = self.femspace.space, degree = self.femspace.degree)
            assembler = Assembler(subspace)
            mass = assembler.global_mass_matrix(reaction = reaction)
            stiffness = assembler.global_stiffness_matrix(diffusion = diffusion)
            m1[domainid] = self.basis[domainid].T @ mass
            m2[domainid] = self.basis[domainid].T @ (mass + self.dt*stiffness)
            m3[domainid] = m1[domainid] @ self.basis[domainid]
            m4[domainid] = m2[domainid] @ self.basis[domainid]

        return m1, m2, m3, m4
    
    def solve(self, history: bool = False, uh = None, exact = None) -> np.ndarray:

        logger.info("="*80)
        logger.info("[Reduced Order Waveform Relaxation] Starting solver")
        logger.info(
            f"Number of subdomains: {len(self.subdomains)} | "
            f"method: {self.method} | "
            f"max iterations: {self.maxiter} | "
            f"tolerance: {self.tol:.2e}"
        )
        logger.info("="*80)

        error: float = float("inf")

        m1, m2, m3, m4 = self.offline()

        initial_data = self.initial_data()

        for iter in range(self.maxiter):
            new_data = {}
            for subdomain in self.subdomains:
                domainid = subdomain.domainID
                initial_cond = self.construct_initial(domainid)
                dirichlet_data = self.construct_dirichlet_bc(domainID = domainid, maps = self.maps, data = initial_data)
                dirichlet_bc = self.dirichlet_vector(domainID = domainid, data = dirichlet_data)
                logger.debug(f"The constructed dirichlet boundary conditions for a subdomain {domainid} in iteration {iter + 1}: {dirichlet_bc}")
                subdomain_heat = HeatROM(
                    femspace = FEMSpace(mesh = subdomain, domain = self.femspace.domain, space = self.femspace.space, degree = self.femspace.degree),
                    basis = self.basis[domainid],
                    func = self.f, 
                    dt = self.dt, 
                    t0 = self.t0, 
                    T = self.T, 
                    dirichlet_bc = dirichlet_bc, 
                    icond = initial_cond, 
                    tstepper = self.tstepper,
                    theta = self.theta, 
                    offline = [m1[domainid], m2[domainid], m3[domainid], m4[domainid]])
                new_data[domainid] = subdomain_heat.solve()

            error = self.boundary_criterion(initial_data, new_data)

            logger.info(f"[Reduced Order Waveform Relaxation] Iteration {iter + 1}: error = {error:.6e}")

            if error < self.tol:
                logger.info(f"[Reduced Order Waveform Relaxation] Converged after {iter + 1} iterations with error = {error:.6e}")
                logger.info("="*80)
                break
            else:
                initial_data = new_data

            if history: # store error history
                rowr_sol = self.combine(initial_data)
                est = ErrorNorms(femspace = self.femspace, u1 = rowr_sol, u2 = uh, u_exact = exact, time = self.time)
                self.error_history.append(est.compute(norm = 'l2'))
        else:
            logger.warning(f"[Reduced Order Waveform Relaxation] Reached max iterations ({self.maxiter}) with error = {error:.6e}")
        
        logger.info("[Reduced Order Waveform Relaxation] Solver finished successfully")
        logger.info("="*80)

        return self.combine(initial_data)