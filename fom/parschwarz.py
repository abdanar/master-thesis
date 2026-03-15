import numpy as np
from fom.poisson import PoissonProblem
from fem.femspace import FEMSpace
from utils.logger import setup_logger
from utils.errornorms import ErrorNorms
from mpi4py import MPI

logger = setup_logger(__name__, level = 'info')

class Schwarz():

    def __init__(self, femspace: FEMSpace, n: int, overlap: int, func, dirichlet_bc: dict, direction: str = 'vertical', method: str = 'RAS',
                  omega: float = 1.0, maxiter: int = 100, tol: float = 1e-3, criterion: str = 'boundary', mpi_enabled: bool = False):
        """
        Initialize a Schwarz solver for time-dependent PDEs over a decomposed domain.

        Parameters
        ----------
        femspace : FEMSpace
            The finite element space representing the full computational domain.
        n : int
            Number of subdomains to decompose the mesh into.
        direction : str
            Specifies how the structured mesh should be partitioned. 
            This function works **only for structured rectangular meshes**.
            - 'vertical'   : split the mesh along the x-direction (columns).
            - 'horizontal' : split the mesh along the y-direction (rows).
            Default is 'vertical'.
        overlap : int
            Number of overlapping layers between subdomains (0 for non-overlapping decomposition).
        func : callable
            Source function of the PDE, e.g., f(x, t).
        dirichlet_bc : dict
            Dirichlet boundary conditions for the whole domain.
            Keys are global node indices, values are the corresponding Dirichlet values.
        method : str, default='RAS'
            Waveform relaxation method, either 'RAS' (Restricted Additive Schwarz) or 'AS' (Additive Schwarz).
        omega : float, optional
            Relaxation parameter for the Schwarz iteration. The global iterate is updated as
                u^{k+1} = (1 - omega) u^k + omega * u_tilde^{k+1}.
            Values 0 < omega <= 1 are allowed. Using omega < 1 stabilizes the additive Schwarz method for multiple overlapping subdomains.
            Default is 1.0 (no relaxation).
        maxiter : int, default=100
            Maximum number of waveform relaxation iterations.
        tol : float, default=1e-3
            Convergence tolerance for the stopping criterion.
        criterion : str, default='boundary'
            Type of convergence criterion, e.g., 'boundary' for checking subdomain boundary changes.
        mpi_enabled : bool
            If True, uses MPI for parallel Schwarz iterations. Each MPI rank owns one or more subdomains.
        """
        self.femspace = femspace
        self.n = n
        self.overlap = overlap
        self.subdomains, self.ltog, self.maps, _ = self.femspace.mesh.decompose(n = n, overlap = overlap, direction = direction)  # list of subdomains of type `Mesh` and dict of local to global mappings
        self.f = func
        self.dirichlet = dirichlet_bc # Dirichlet boundary condition for the whole domain, for the main problem
        self.method = method
        self.omega = omega
        self.maxiter = maxiter
        self.tol = tol
        self.criterion = criterion # stopping criterion
        self.nspace = self.femspace.mesh.nnodes() # total number of space nodes (for whole domain)
        self.error_history = []
        self.mpi_enabled = mpi_enabled
        if mpi_enabled:
            self.comm = MPI.COMM_WORLD # MPI communicator
            self.rank = self.comm.Get_rank() # process ID, identifies different processes, starts from 0 to size-1
            self.size = self.comm.Get_size() # total number of processes
            self.localIDs = self._assign_subdomains() # list of subdomain IDs assigned to the current MPI rank
            self.local_subdomains = [subdomain for subdomain in self.subdomains if subdomain.domainID in self.localIDs] # keep only local subdomains
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            self.localIDs = list(range(1, n + 1))
            self.local_subdomains = self.subdomains

    def _assign_subdomains(self) -> list[int]:
        """
        Assign local subdomains to the current MPI rank using block (contiguous) decomposition.

        Each MPI process may own one or more subdomains. This function divides `self.n` total subdomains 
        (numbered from 1 to n) among `self.size` MPI ranks as evenly as possible, giving each rank a 
        contiguous block of subdomains. If `n` is not divisible by `size`, the first `r` ranks receive 
        one extra subdomain.

        Returns
        -------
        list[int]
            List of subdomain indices assigned to the current MPI rank (1-based).

        Example
        -------
        n = 10 subdomains, size = 3 ranks
        Rank 0 → [1,2,3,4]
        Rank 1 → [5,6,7]
        Rank 2 → [8,9,10]
        """
        q, r = divmod(self.n, self.size)
        start = self.rank * q + min(self.rank, r) + 1  # shift by +1 for 1-based indexing
        end = start + q + (1 if self.rank < r else 0)
        return list(range(start, end))

    def owner(self, domainID: int) -> int:
        """
        Determine the MPI rank that owns a given subdomain.

        In a block decomposition of `self.n` subdomains (numbered 1..n) across `self.size` MPI ranks, 
        each rank owns a contiguous block of subdomains. This function computes which rank owns 
        a specific subdomain ID `domainID`.

        Parameters
        ----------
        domainID : int
            Global subdomain index (1-based) whose owner rank is to be determined.

        Returns
        -------
        int
            MPI rank that owns the specified subdomain.

        Raises
        ------
        ValueError
            If `domainID` is outside the valid range [1, self.n].

        Example
        -------
        n = 10 subdomains, size = 3 ranks
        Subdomain 5 → owner() returns 1
        Subdomain 10 → owner() returns 2
        """
        if not (1 <= domainID <= self.n):
            raise ValueError("Invalid subdomain index")
        q, r = divmod(self.n, self.size)
        for rank in range(self.size):
            start = rank * q + min(rank, r) + 1
            end = start + q + (1 if rank < r else 0)
            if start <= domainID < end:
                return rank
        raise ValueError(f"Could not determine owner for the subdomain {domainID}")
    
    def construct_dirichlet_bc(self, domainID: int, maps: dict, data: dict) -> dict:
        """
        Construct Dirichlet boundary values for a specific subdomain, accounting for shared nodes.

        This function computes the Dirichlet boundary values for the nodes of a given subdomain.
        For nodes shared by multiple subdomains, the boundary value is computed as the average of 
        the corresponding values from all subdomains sharing that node. However, if **any** of 
        the shared entries corresponds to subdomain 0, the Dirichlet value from `self.dirichlet` 
        is used directly.

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
            `(num_nodes, )`. Column `j` of `data[i]` contains the solution 
            `u(x)` for the nodes of subdomain `i`.

        Returns
        -------
        dirichlet_bc : dict
            A dictionary mapping local boundary node indices of `domainID` to their 
            Dirichlet boundary values. Shared nodes are assigned the average value across 
            the corresponding subdomains unless a Dirichlet entry from subdomain 0 exists, 
            which takes precedence.

        Notes
        -----
        - Assumes that `data` contains the solution for all relevant subdomains and time steps.
        - Nodes shared among multiple subdomains are normally averaged unless subdomain 0 is present.
        - Subdomain 0 is treated as a special case: its Dirichlet values override any averages.
        """    
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
                val = sum(data[i][j] for (i, j) in dlist) 
                if self.method == 'AS':
                    dirichlet_bc[dindex] = val
                elif self.method == 'RAS':
                    dirichlet_bc[dindex] = val/share
                else:
                    raise ValueError(f"Invalid method '{self.method}'. Must be 'RAS' or 'AS'.")
            else:
                # Only one subdomain, take its value
                i, j = dlist[0]
                dirichlet_bc[dindex] = data[i][j]
        return dirichlet_bc
    
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
        global_solution : ndarray, shape (nspace, )
            Global solution assembled from subdomain solutions. Each column represents the solution
            at a specific time step: `global_solution[:i] = u(x, t_i)`.
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
        global_solution = np.zeros(self.nspace)

        # Apply Dirichlet boundary conditions
        for idx, values in self.dirichlet.items():
            global_solution[idx] = values

        # The dictionary that contains arrays for local dof to global dof mappings for each subdomains with keys to be domainID
        local_to_global = self.ltog

        if self.method == 'RAS':
            count = np.zeros(self.nspace)  
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_vertices()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index] += data[i][local_index]
                        count[global_index] += 1
            for ix in range(self.nspace):
                if count[ix] > 0:
                    global_solution[ix] /= count[ix]
        elif self.method == 'AS':
            for subdomain in self.subdomains:
                i = subdomain.domainID
                bdindices = subdomain.boundary_vertices()
                for local_index, global_index in enumerate(local_to_global[i]):
                    if local_index not in bdindices:
                        global_solution[global_index] += data[i][local_index] # sum contributions from all subdomains, to have perfect solution on overlaps remove + sign
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
            idata[subdomain.domainID] = np.zeros((subdomain.nnodes(), 1))
        return idata

    def solve(self, history: bool = False, uh = None, exact = None) -> np.ndarray:

        logger.info("="*80)
        logger.info("[Schwarz] Starting solver")
        logger.info(
            f"Number of subdomains: {len(self.subdomains)} | "
            f"method: {self.method} | "
            f"relaxation (omega): {self.omega:.2f} | "
            f"max iterations: {self.maxiter} | "
            f"tolerance: {self.tol:.2e}"
        )
        logger.info("="*80)

        error: float = float("inf")

        initial_data = self.initial_data()

        for iter in range(self.maxiter):
            new_data = {}
            for subdomain in self.subdomains:
                domainid = subdomain.domainID
                dirichlet_bc = self.construct_dirichlet_bc(domainID = domainid, maps = self.maps, data = initial_data)
                logger.debug(f"The constructed dirichlet boundary conditions for a subdomain {domainid} in iteration {iter + 1}: {dirichlet_bc}")
                subdomain_heat = PoissonProblem(
                    femspace = FEMSpace(mesh = subdomain, domain = self.femspace.domain, space = self.femspace.space, degree = self.femspace.degree),
                    func = self.f,
                    dirichlet_bc = dirichlet_bc)
                new_data[domainid] = subdomain_heat.solve()

            error = self.boundary_criterion(initial_data, new_data)
            logger.info(f"[Schwarz] Iteration {iter + 1}: error = {error:.6e}")

            if error < self.tol:
                logger.info(f"[Schwarz] Converged after {iter + 1} iterations with error = {error:.6e}")
                logger.info("="*80)
                break
            else:
                if self.omega == 1.0:
                    initial_data = new_data
                else:
                    for i in initial_data:
                        initial_data[i] = (1 - self.omega) * initial_data[i] + self.omega * new_data[i]

            if history: # store error history
                schwarz_sol = self.combine(initial_data)
                est = ErrorNorms(femspace = self.femspace, u1 = schwarz_sol, u2 = uh, u_exact = exact)
                self.error_history.append(est.compute(norm = 'l2'))
        else:
            logger.warning(f"[Schwarz] Reached max iterations ({self.maxiter}) with error = {error:.6e}")
        
        logger.info("[Schwarz] Solver finished successfully")
        logger.info("="*80)

        return self.combine(initial_data)