import numpy as np
from typing import Callable
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import LinearSolver, DirectSolver
from fem.boundary import DirichletBC
import utils.logger as log 
from typing import Optional

logger = log.setup_logger(__name__, level = 'info')

class PoissonProblem:
    def __init__(self, femspace: FEMSpace, f: Callable, g: Callable | dict):
        """
        FEM solver for the Poisson equation:
            - 1D: - u'' = f(x),   x ∈ Ω
            - 2D: - Δu(x, y) = f(x, y),   (x, y) ∈ Ω
        with Dirichlet boundary conditions
            - 1D: u(x) = g(x) for x on the boundary of Ω
            - 2D: u(x, y) = g(x, y) for (x, y) on the boundary of Ω

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object
        f : Callable
            Source term f(x) for 1D or f(x, y) for 2D
        g : Callable or dict
            The Dirichlet boundary condition function or dictionary.
            - If it is a function, it should be defined as g(x) for 1D or g(x, y) for 2D problems.
            - If it is a dictionary, the keys should be global node indices corresponding to the boundary nodes, 
              and the values should be the Dirichlet values at those nodes. For example: {0: 0.0, 5: 1.0, ...}

        Warning to users
        ----------------
        If `solve` is called only once, then this class introduce unnecessary overhead from 
        assembling the system matrix, applying boundary conditions and converting to CSR/CSC formats. 
        However, if `solve` is called multiple times with different boundary conditions (by providing `g_new`), 
        then this class is efficient as it allows reusing the assembled system and only updates
        the boundary conditions without needing to reassemble the entire system. 
        This can be particularly beneficial for parametric studies or time-dependent problems 
        where boundary conditions evolve, as it avoids redundant computations and can significantly 
        reduce the overall runtime.
        """
        self.femspace = femspace
        self.f = f
        self.g = g
        self.dim = femspace.dim

        # Define the diffusion coefficient (identity for standard Poisson problem)
        if self.dim == 1:
            diffusion = lambda x: 1
        elif self.dim == 2:
            diffusion = lambda x, y: np.eye(2)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

        # Assemble the system matrix and load vector for the Poisson problem
        assembler = Assembler(self.femspace)
        self.lhs_base = assembler.global_stiffness_matrix(diffusion=diffusion)
        self.rhs_base = assembler.global_load_vector(self.f)

        # Convert LHS to both CSR and CSC formats for efficient use with different solvers
        self.lhs_csr = self.lhs_base.tocsr()  # for iterative solvers (fast row slicing)
        self.lhs_csc = self.lhs_base.tocsc()  # for direct solvers (LU, column operations)

        # Initialize the boundary condition handler (for nodal lifting)
        self.boundary = DirichletBC(femspace=self.femspace, g=self.g)

        # Store modified LHS with Dirichlet nodes applied (for nodal lifting)
        self.lhs_csr_mod, _ = self.boundary.apply(K = self.lhs_csr, modify_K = True)
        self.lhs_csc_mod, _ = self.boundary.apply(K = self.lhs_csc, modify_K = True)
    
    def solve(self, lift: str = 'nodal', solver: LinearSolver = DirectSolver(), g_new: Optional[Callable | dict] = None, **kwargs):
        """
        Solves the Poisson problem using FEM discretization and specified boundary condition handling.

        It supports two methods for enforcing Dirichlet boundary conditions:

        1. 'nodal': Directly modifies the system matrix and RHS to enforce Dirichlet BCs at the specified nodes.
        2. 'harmonic': Computes a harmonic lifting function that satisfies the Dirichlet BCs and solves for the homogeneous part of the solution.

        Most importantly, the method allows updating the Dirichlet boundary values on-the-fly by providing a new function or dictionary `g_new`. 
        If `g_new` is provided, it updates the boundary handler with the new values before solving. This enables dynamic changes to the boundary 
        conditions without needing to reassemble the entire system, which can be efficient for parametric studies or time-dependent problems where 
        boundary conditions evolve.

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
        g_new : Callable or dict, optional
            New Dirichlet boundary condition function or dictionary to update the boundary handler before solving.
            - If it is a function, it should be defined as g_new(x) for 1D or g_new(x, y) for 2D problems.
            - If it is a dictionary, the keys should be global node indices corresponding to the boundary nodes, 
              and the values should be the new Dirichlet values at those nodes. For example: {0: 0.0, 5: 1.0, ...}
            If `g_new` is None, the existing boundary conditions defined by `self.g` will be used without modification.
        **kwargs
            Additional keyword arguments to pass to the boundary condition application method 
            (e.g., solver parameters for iterative solvers).
            
        Returns
        -------
        np.ndarray, shape (n_vertices,)
            The computed solution vector at the FEM nodes.
        """
        # Update Dirichlet boundary values in the boundary handler (if needed)
        if g_new is not None:
            self.boundary.update_dirichlet_values(g_new)

        if lift == 'nodal':
            # Note: Since the mesh and thus the Dirichlet nodes never change, lhs_mod (the LHS with Dirichlet nodes applied) does NOT need to be recomputed.
            _, rhs = self.boundary.apply(K = self.lhs_csc, rhs = self.rhs_base, modify_K = False, **kwargs)
            if isinstance(solver, DirectSolver):
                return solver.solve(self.lhs_csc_mod, rhs, **kwargs) # type: ignore
            else:
                return solver.solve(self.lhs_csr_mod, rhs, **kwargs) # type: ignore
        # elif lift == 'harmonic':
        #     hom_boundary = DirichletBC(self.femspace, g = self.g)
        #     A, b = hom_boundary.apply(Kstiff = lhs, rhs = rhs, solver = solver, **kwargs)
        #     u_hom = solver.solve(A, b, **kwargs)
        #     return u_hom + boundary.lift
        else:
            raise ValueError(f"Invalid lift type '{lift}'. Supported types are 'nodal' and 'harmonic'.")
    
# class HarmonicDirichletBC:
#     """
#     Harmonic lifting implementation for Dirichlet boundary conditions.

#     Instead of inserting boundary values directly into the system,
#     this approach constructs a lifting function u_D that satisfies
#         -Δ u_D = 0  in Ω
#            u_D = g  on Γ_D
#     where g are the prescribed Dirichlet values.

#     The full solution is decomposed as
#         u = u_h + u_D
#     where
#     - u_D : harmonic lifting satisfying Dirichlet BCs
#     - u_h : homogeneous solution with zero Dirichlet boundary conditions
#     Substituting into the original system
#         K u = rhs
#     yields the modified homogeneous problem
#         K u_h = rhs - K u_D
#     which can be solved using standard homogeneous Dirichlet enforcement.

#     Advantages
#     ----------
#     - Produces a smooth extension of boundary values into the domain.
#     - Often improves numerical stability for problems with strong boundary layers.

#     Notes
#     -----
#     The stiffness matrix used for computing the lifting must correspond to the
#     Laplace operator so that the lifting satisfies the harmonic equation.
#     """
#     def __init__(self, femspace: FEMSpace, dirichlet_nodes: dict):
#         self.dirichlet_nodes = dirichlet_nodes
#         self._nodal_bc = NodalDirichletBC(femspace, dirichlet_nodes)
#         self._homogeneous_bc = NodalDirichletBC(femspace, {node: 0.0 for node in dirichlet_nodes})
#         self.lift = None

#     def lifting(self, Kstiff: Matrix, solver: LinearSolver = DirectSolver(), **kwargs) -> Vector:
#         """
#         This function computes the full harmonic lifting vector u_D by solving the Laplace problem:
#             -Δ u_D = 0 in Ω,
#                u_D = g on Γ_D,
#         where g are the prescribed Dirichlet values. The system matrix Kstiff should be 
#         the stiffness matrix corresponding to the Laplacian operator. The resulting 
#         u_D will satisfy the Dirichlet BCs and can be used to modify the original 
#         problem to solve for the homogeneous part.

#         Parameters
#         ----------
#         Kstiff : Matrix
#             The global stiffness matrix corresponding to the Laplacian operator, 
#             used for computing the harmonic lifting. This should be the stiffness 
#             matrix of the pure Laplace problem (without any modifications for BCs) 
#             to ensure that the lifting satisfies -Δ u_D = 0.
#         solver : LinearSolver, optional
#             The linear solver to use for solving the Laplace problem to compute 
#             the harmonic lifting. Default is DirectSolver, but iterative solvers 
#             can also be used for larger problems. For available solvers, see 
#             fem.linearsolver.LinearSolver and its subclasses.
#         **kwargs : Any
#             Additional parameters for the linear solver, such as tolerance, max iterations, etc. 
#             These will be passed to the solver's solve method.

#         Returns
#         -------
#         u_D : Vector
#             The full harmonic lifting vector of shape (n, m).

#         Notes
#         -----
#         - The returned value is the full lifting vector defined 
#           on all DOFs, not just the boundary. It has the prescribed 
#           values at the boundary nodes and satisfies the Laplace 
#           equation in the interior.
#         """
#         logger.debug("Computing harmonic lifting (solving Laplace problem)")
#         if not isinstance(solver, LinearSolver):
#             raise TypeError(f"solver must be a LinearSolver instance, got {type(solver).__name__}")
#         rhs_lifte = np.zeros((Kstiff.shape[0], self._nodal_bc._bc_values.shape[1]))
#         K_lift, rhs_lift = self._nodal_bc.apply(Kstiff, rhs_lifte)
#         lift = solver.solve(K_lift, rhs_lift)
#         logger.debug("The full harmonic lifting vector computed")
#         return lift

#     def apply(self, Kstiff: Matrix, rhs: Vector, solver: LinearSolver = DirectSolver(), **kwargs) -> tuple[Matrix, Vector]:
#         """
#         This function applies the harmonic Dirichlet decomposition to the stiffness 
#         matrix Kstiff and load vector(s) rhs. Once you apply this function, the returned 
#         K_mod and rhs_mod correspond to the modified system for the homogeneous part 
#         of the solution, where the effect of the Dirichlet BCs has been incorporated 
#         through the lifting. You can then solve K_mod u_hom = rhs_mod for the homogeneous
#         solution u_hom, and recover the full solution as u = u_hom + u_D, where u_D is 
#         the harmonic lifting computed by the lifting() method.

#         Parameters
#         ----------
#         Kstiff : Matrix
#             The global stiffness matrix corresponding to the Laplacian operator, 
#             used for computing the harmonic lifting. This should be the stiffness 
#             matrix of the pure Laplace problem (without any modifications for BCs) 
#             to ensure that the lifting satisfies -Δ u_D = 0.
#         rhs : Vector
#             Global right-hand side vector(s).
#         solver : LinearSolver, optional
#             The linear solver to use for computing the harmonic lifting. Default is 
#             DirectSolver, but iterative solvers can also be used for larger problems. 
#             For available solvers, see fem.linearsolver.LinearSolver and its subclasses.
#         **kwargs : Any
#             Additional parameters for the linear solver, such as tolerance, max iterations, etc. 
#             These will be passed to the solver's solve method.

#         Returns
#         -------
#         K_mod : Matrix
#             Modified system matrix with homogeneous Dirichlet BCs.
#         rhs_mod : Vector
#             Modified RHS: rhs - K * u_D.

#         Notes
#         -----
#         The lifting is cached after the first computation and reused in
#         subsequent calls unless `reset()` is invoked.
#         """
#         logger.debug("Applying harmonic Dirichlet BC")
#         if self.lift is None:
#             self.lift = self.lifting(Kstiff, solver, **kwargs)
#         rhs_mod = rhs - Kstiff @ self.lift
#         K_mod, rhs_mod = self._homogeneous_bc.apply(Kstiff, rhs_mod)
#         logger.debug("Harmonic Dirichlet BC applied")
#         return K_mod, rhs_mod
    
#     def reset(self):
#         """
#         Invalidate the cached harmonic lifting vector.

#         Call this method when the lifting needs to be recomputed, e.g., after
#         changing the stiffness matrix or solver. The next call to ``apply()`` 
#         or ``lifting()`` will trigger a fresh computation.

#         Examples
#         --------
#         >>> bc = HarmonicDirichletBC(femspace, dirichlet_nodes)
#         >>> bc.apply(K1, rhs1, solver=DirectSolver())  # computes and caches lifting
#         >>> bc.apply(K1, rhs2)                          # reuses cached lifting
#         >>> bc.reset()                                   # clear cache
#         >>> bc.apply(K2, rhs3, solver=CGSolver())       # recomputes with new operator
#         """
#         self.lift = None
