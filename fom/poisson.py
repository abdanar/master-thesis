import numpy as np
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import *
from fem.boundary import DirichletBC
import utils.logger as log 

logger = log.setup_logger(__name__, level = 'info')

class PoissonProblem:
    def __init__(self, femspace: FEMSpace, func):
        """
        FEM solver for the Poisson equation:
            - 1D: - u'' = f(x),   x ∈ Ω
            - 2D: - Δu(x, y) = f(x, y),   (x, y) ∈ Ω
        with Dirichlet boundary conditions specified by `dirichlet_bc`.

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object
        func : callable
            Source term f(x) for 1D or f(x, y) for 2D
        """
        self.femspace = femspace
        self.f = func
        self.dim = femspace.dim

    def solve(self, lift: str = 'nodal', solver: LinearSolver = DirectSolver(), **kwargs):
        """
        Solves the Poisson problem using FEM discretization and specified boundary condition handling.

        Parameters
        ----------
        lift : str
            Method for handling Dirichlet boundary conditions. Options are:
                - 'nodal': Directly modify the system to enforce Dirichlet BCs at the specified nodes.
                - 'harmonic': Compute a harmonic lifting function that satisfies the Dirichlet BCs and 
                   solve for the homogeneous part of the solution.
            Default is 'nodal'.
        solver : LinearSolver
            Linear solver to use for solving the linear system. Must be an instance of a class that 
            inherits from `LinearSolver`. Default is `DirectSolver()`.
        **kwargs
            Additional keyword arguments to pass to the boundary condition application method 
            (e.g., solver parameters for iterative solvers).
        Returns
        -------
        np.ndarray
            The computed solution vector at the FEM nodes.
        """
        if self.dim == 1:
            diffusion = lambda x: 1
        elif self.dim == 2:
            diffusion = lambda x, y: np.eye(2)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
        pde_assembler = Assembler(self.femspace)
        lhs = pde_assembler.global_stiffness_matrix(diffusion = diffusion).tocsc()
        rhs = pde_assembler.global_load_vector(self.f)
        if lift == 'nodal':
            boundary = DirichletBC(femspace = self.femspace, f = self.f, markers = self.femspace.mesh.segment_markers)
            A, b = boundary.apply(lhs, rhs, **kwargs)
            return solver.solve(A, b, **kwargs)
        # elif lift == 'harmonic':
        #     boundary = HarmonicDirichletBC(self.femspace, self.dirichlet)
        #     A, b = boundary.apply(Kstiff = lhs, rhs = rhs, solver = solver, **kwargs)
        #     u_hom = solver.solve(A, b, **kwargs)
        #     return u_hom + boundary.lift
        else:
            raise ValueError(f"Invalid lift type '{lift}'. Supported types are 'nodal' and 'harmonic'.")
        
    
class HarmonicDirichletBC:
    """
    Harmonic lifting implementation for Dirichlet boundary conditions.

    Instead of inserting boundary values directly into the system,
    this approach constructs a lifting function u_D that satisfies
        -Δ u_D = 0  in Ω
           u_D = g  on Γ_D
    where g are the prescribed Dirichlet values.

    The full solution is decomposed as
        u = u_h + u_D
    where
    - u_D : harmonic lifting satisfying Dirichlet BCs
    - u_h : homogeneous solution with zero Dirichlet boundary conditions
    Substituting into the original system
        K u = rhs
    yields the modified homogeneous problem
        K u_h = rhs - K u_D
    which can be solved using standard homogeneous Dirichlet enforcement.

    Advantages
    ----------
    - Produces a smooth extension of boundary values into the domain.
    - Often improves numerical stability for problems with strong boundary layers.

    Notes
    -----
    The stiffness matrix used for computing the lifting must correspond to the
    Laplace operator so that the lifting satisfies the harmonic equation.
    """
    def __init__(self, femspace: FEMSpace, dirichlet_nodes: dict):
        self.dirichlet_nodes = dirichlet_nodes
        self._nodal_bc = NodalDirichletBC(femspace, dirichlet_nodes)
        self._homogeneous_bc = NodalDirichletBC(femspace, {node: 0.0 for node in dirichlet_nodes})
        self.lift = None

    def lifting(self, Kstiff: Matrix, solver: LinearSolver = DirectSolver(), **kwargs) -> Vector:
        """
        This function computes the full harmonic lifting vector u_D by solving the Laplace problem:
            -Δ u_D = 0 in Ω,
               u_D = g on Γ_D,
        where g are the prescribed Dirichlet values. The system matrix Kstiff should be 
        the stiffness matrix corresponding to the Laplacian operator. The resulting 
        u_D will satisfy the Dirichlet BCs and can be used to modify the original 
        problem to solve for the homogeneous part.

        Parameters
        ----------
        Kstiff : Matrix
            The global stiffness matrix corresponding to the Laplacian operator, 
            used for computing the harmonic lifting. This should be the stiffness 
            matrix of the pure Laplace problem (without any modifications for BCs) 
            to ensure that the lifting satisfies -Δ u_D = 0.
        solver : LinearSolver, optional
            The linear solver to use for solving the Laplace problem to compute 
            the harmonic lifting. Default is DirectSolver, but iterative solvers 
            can also be used for larger problems. For available solvers, see 
            fem.linearsolver.LinearSolver and its subclasses.
        **kwargs : Any
            Additional parameters for the linear solver, such as tolerance, max iterations, etc. 
            These will be passed to the solver's solve method.

        Returns
        -------
        u_D : Vector
            The full harmonic lifting vector of shape (n, m).

        Notes
        -----
        - The returned value is the full lifting vector defined 
          on all DOFs, not just the boundary. It has the prescribed 
          values at the boundary nodes and satisfies the Laplace 
          equation in the interior.
        """
        logger.debug("Computing harmonic lifting (solving Laplace problem)")
        if not isinstance(solver, LinearSolver):
            raise TypeError(f"solver must be a LinearSolver instance, got {type(solver).__name__}")
        rhs_lifte = np.zeros((Kstiff.shape[0], self._nodal_bc._bc_values.shape[1]))
        K_lift, rhs_lift = self._nodal_bc.apply(Kstiff, rhs_lifte)
        lift = solver.solve(K_lift, rhs_lift)
        logger.debug("The full harmonic lifting vector computed")
        return lift

    def apply(self, Kstiff: Matrix, rhs: Vector, solver: LinearSolver = DirectSolver(), **kwargs) -> tuple[Matrix, Vector]:
        """
        This function applies the harmonic Dirichlet decomposition to the stiffness 
        matrix Kstiff and load vector(s) rhs. Once you apply this function, the returned 
        K_mod and rhs_mod correspond to the modified system for the homogeneous part 
        of the solution, where the effect of the Dirichlet BCs has been incorporated 
        through the lifting. You can then solve K_mod u_hom = rhs_mod for the homogeneous
        solution u_hom, and recover the full solution as u = u_hom + u_D, where u_D is 
        the harmonic lifting computed by the lifting() method.

        Parameters
        ----------
        Kstiff : Matrix
            The global stiffness matrix corresponding to the Laplacian operator, 
            used for computing the harmonic lifting. This should be the stiffness 
            matrix of the pure Laplace problem (without any modifications for BCs) 
            to ensure that the lifting satisfies -Δ u_D = 0.
        rhs : Vector
            Global right-hand side vector(s).
        solver : LinearSolver, optional
            The linear solver to use for computing the harmonic lifting. Default is 
            DirectSolver, but iterative solvers can also be used for larger problems. 
            For available solvers, see fem.linearsolver.LinearSolver and its subclasses.
        **kwargs : Any
            Additional parameters for the linear solver, such as tolerance, max iterations, etc. 
            These will be passed to the solver's solve method.

        Returns
        -------
        K_mod : Matrix
            Modified system matrix with homogeneous Dirichlet BCs.
        rhs_mod : Vector
            Modified RHS: rhs - K * u_D.

        Notes
        -----
        The lifting is cached after the first computation and reused in
        subsequent calls unless `reset()` is invoked.
        """
        logger.debug("Applying harmonic Dirichlet BC")
        if self.lift is None:
            self.lift = self.lifting(Kstiff, solver, **kwargs)
        rhs_mod = rhs - Kstiff @ self.lift
        K_mod, rhs_mod = self._homogeneous_bc.apply(Kstiff, rhs_mod)
        logger.debug("Harmonic Dirichlet BC applied")
        return K_mod, rhs_mod
    
    def reset(self):
        """
        Invalidate the cached harmonic lifting vector.

        Call this method when the lifting needs to be recomputed, e.g., after
        changing the stiffness matrix or solver. The next call to ``apply()`` 
        or ``lifting()`` will trigger a fresh computation.

        Examples
        --------
        >>> bc = HarmonicDirichletBC(femspace, dirichlet_nodes)
        >>> bc.apply(K1, rhs1, solver=DirectSolver())  # computes and caches lifting
        >>> bc.apply(K1, rhs2)                          # reuses cached lifting
        >>> bc.reset()                                   # clear cache
        >>> bc.apply(K2, rhs3, solver=CGSolver())       # recomputes with new operator
        """
        self.lift = None
