import numpy as np
from typing import Callable
import scipy as sc
from scipy.sparse import sparray
from fem.linearsolver import LinearSolver, DirectSolver, JacobiSolver, CGSolver
from fem.assembler import Assembler

MatrixValue = np.ndarray | sparray
MatrixFunction = Callable[[float], np.ndarray]
MatrixType = MatrixValue | MatrixFunction

class ModifiedTheta:

    def __init__(self, M: MatrixType, A: MatrixType, assembler: Assembler, F: MatrixFunction, basis: np.ndarray, t0: float, dt: float, dirichlet_bc: dict, theta: float):
                
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
        self.basis = basis
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

        u = DirectSolver().solve(self.basis.T @ lhs @ self.basis, self.basis.T @ rhs)  # u_n+1

        return self.basis @ u, A, M, F