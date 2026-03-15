import numpy as np
from typing import Callable
import scipy as sc
from scipy.sparse import sparray
from fem.linearsolver import LinearSolver, DirectSolver, JacobiSolver, CGSolver
from fem.assembler import Assembler

# -------------------------------------------------------------------
# IMPORTANT: Constant vs time-dependent mass and stiffness matrices
# -------------------------------------------------------------------
#
# The `Theta` time stepper supports both constant-in-time and
# time-dependent FEM matrices through the unified type:
#
#     MatrixValue    = np.ndarray | sparray
#     MatrixFunction = Callable[[float], np.ndarray | sparray]
#     MatrixType     = MatrixValue | MatrixFunction
#
# You may therefore pass either:
#   - a matrix directly (constant in time), or
#   - a callable M(t), A(t) returning the matrix at time t.
#
# Internally, constant matrices are automatically wrapped as
# time-independent functions:
#
#     M -> (lambda t: M)
#     A -> (lambda t: A)
#
# so the stepper can uniformly call M(t_n), A(t_n) at each time step.
#
# -------------------------------------------------------------------
# 1. Constant-in-time matrices
# -------------------------------------------------------------------
# If the mass or stiffness matrix does not depend on time, simply
# assemble it once and pass it directly:
#
#     M = assembler.global_mass_matrix(reaction=reaction)
#     A = assembler.global_stiffness_matrix(diffusion=diffusion)
#
# The `Theta` solver will:
#   - detect that the matrices are constant,
#   - precompute the left-hand side
#         M + θ Δt A
#   - reuse its LU factorization at every time step for efficiency.
#
# -------------------------------------------------------------------
# 2. Time-dependent matrices
# -------------------------------------------------------------------
# If coefficients depend on time, pass callables returning the matrix
# at the requested time t.
#
# Example: time-dependent reaction term
#
#     def reaction_time(x, y, t):
#         return 1.0 + 0.1 * t
#
#     M = lambda t: assembler.global_mass_matrix(
#             reaction=lambda x, y: reaction_time(x, y, t)
#         )
#
# Example: time-dependent diffusion coefficient
#
#     def diffusion_time(x, y, t):
#         return (1.0 + 0.5 * t) * np.eye(2)
#
#     A = lambda t: assembler.global_stiffness_matrix(
#             diffusion=lambda x, y: diffusion_time(x, y, t)
#         )
#
# In this case:
#   - M(t) and A(t) are evaluated at each time level,
#   - no LU factorization is reused,
#   - the left-hand side is rebuilt at every step.
#
# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------
# - The time stepper itself does not distinguish between constant
#   and time-dependent matrices; it always calls M(t), A(t).
# - Performance optimizations (LU reuse) are enabled automatically
#   when both M and A are constant matrices.
# - This design keeps the interface simple, general, and robust
#   for both stationary and non-stationary PDEs.
# -------------------------------------------------------------------

MatrixValue = np.ndarray | sparray
MatrixFunction = Callable[[float], np.ndarray]
MatrixType = MatrixValue | MatrixFunction

class Theta:

    def __init__(self, M: MatrixType, A: MatrixType, assembler: Assembler, F: MatrixFunction, t0: float, dt: float, dirichlet_bc: dict, theta: float):
                
        """
        Time integration using the θ-method for time-dependent FEM systems.

        This class advances the semi-discrete problem

            M(t) u'(t) + A(t) u(t) = F(t)

        in time using the θ-method. The parameter θ controls the implicitness
        of the scheme.

        The solver supports both **constant** and **time-dependent** mass and stiffness matrices.  
        For constant matrices, LU factorization is reused automatically for efficiency. (not yet)

        Parameters
        ----------
        M : MatrixType
            Mass matrix, either a constant matrix (dense or sparse) or a callable
            function M(t) returning the mass matrix at time t.
        A : MatrixType
            Stiffness/convection/reaction matrix, either constant or callable A(t)
            returning the matrix at time t.
        assembler : Assembler
            FEM assembler used to construct the load vector and apply
            Dirichlet boundary conditions.
        F : callable
            Load vector function F(t) returning the load vector at time t.
        t0 : float
            Initial time of the simulation.
        dt : float
            Time step size.
        dirichlet_bc : dict
            Dictionary of Dirichlet boundary conditions prescribed at all
            time steps. Keys are global node indices, and values are arrays
            of length n_steps containing the boundary values in time.
        theta : float
            Parameter of the θ-method.
            - θ = 1.0   corresponds to the Backward Euler method
            - θ = 0.5   corresponds to the Crank-Nicolson method

        Notes
        -----
        - Optimized for constant mass/stiffness matrices with LU factorization reuse.(not yet)
        - Time-dependent matrices M(t) and A(t) are provided as callable
          functions and are evaluated at the required time levels.
        - The θ-method requires data from the previous time step
          (mass matrix, system matrix, and load vector).
        """
        self.M = M if callable(M) else (lambda t: M)
        self.A = A if callable(A) else (lambda t: A)
        self.assembler = assembler
        self.F = F
        self.t0 = t0
        self.dt = dt
        self.dirichlet_bc = dirichlet_bc
        self.theta = theta
        self.dim = assembler.dim

    def step(self, u_n: np.ndarray, F_n: np.ndarray, t_n: float, A_n: MatrixValue | None = None, M_n: MatrixValue | None = None):

        """
        Advance the solution by one time step using the θ-method.

        Solves the semi-discrete system:

            M(t) u'(t) + A(t) u(t) = F(t),

        using the θ-method time discretization:

            (M_{n+1} + θ Δt A_{n+1}) u_{n+1} = (M_n - (1-θ) Δt A_n) u_n + Δt [ θ F_{n+1} + (1-θ) F_n ].

        Special cases:
            θ = 1   → Backward Euler
            θ = 1/2 → Crank-Nicolson

        Parameters
        ----------
        u_n : np.ndarray
            Solution vector at time t_n.
        F_n : np.ndarray
            Load vector F(t_n).
        t_n : float
            Current time corresponding to u_n.
        A_n : MatrixValue or None, optional
            System matrix A(t_n) from the previous step. Required for θ < 1.
        M_n : MatrixValue or None, optional
            Mass matrix M(t_n) from the previous step. Required for θ < 1.

        Returns
        -------
        u : np.ndarray
            Solution vector at time t_{n+1}.
        A : MatrixValue
            System matrix A(t_{n+1}) (dense or sparse).
        M : MatrixValue
            Mass matrix M(t_{n+1}) (dense or sparse).
        F : np.ndarray
            Load vector F(t_{n+1}).
        """

        theta = self.theta
        dt = self.dt
        t = t_n + dt # t_n+1
        i = int(round((t - self.t0) / self.dt)) # n+1

        M = self.M(t) # M(t_n+1)
        A = self.A(t) # A(t_n+1)
        F = self.F(t) # F(t_n+1)

        lhs = M + theta*dt*A
        rhs = (M_n - (1 - theta)*dt*A_n)@u_n + dt*(theta*F + (1 - theta)*F_n).ravel()
        
        # Boundary condition is computed at t_n+1
        dirichlet = {k: v[i] for k, v in self.dirichlet_bc.items()}

        lhs, rhs = self.assembler.apply_Dirichlet_bc(lhs, rhs, dirichlet)

        u = DirectSolver().solve(lhs, rhs)  # u_n+1

        return u, A, M, F