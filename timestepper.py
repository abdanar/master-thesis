from abc import ABC, abstractmethod
from linearsolver import LinearSolver
from assembler import Assembler

# -------------------------------------------------------------------
# IMPORTANT: Time-dependent mass and stiffness matrices
# -------------------------------------------------------------------
# The TimeStepper class expects M(t) and A(t) as callable functions of time.
# Even if your mass or stiffness matrices are constant in time (space-only FEM),
# you need to wrap them as functions so the stepper interface works consistently.
#
# 1. Constant-in-time matrices:
#    Simply wrap the pre-assembled matrix in a lambda that ignores t:
#
#       M_static = assembler.global_mass_matrix(reaction=reaction)
#       A_static = assembler.global_stiffness_matrix(diffusion=diffusion)
#
#       M_time = lambda t: M_static   # returns same mass matrix at any time t
#       A_time = lambda t: A_static  # returns same stiffness matrix at any time t
#
# 2. Time-dependent matrices:
#    If the PDE coefficients depend on time, define a function that constructs
#    the matrix for the given time t. For example, a reaction term increasing
#    linearly in time:
#
#       def reaction_time(x, y, t):
#           return 1 + 0.1*t
#
#       M_time = lambda t: assembler.global_mass_matrix(
#                              reaction=lambda x, y: reaction_time(x, y, t))
#
#    Similarly, for a time-dependent diffusion coefficient:
#
#       def diffusion_time(x, y, t):
#           return (1 + 0.5*t) * np.eye(2)
#
#       A_time = lambda t: assembler.global_stiffness_matrix(
#                              diffusion=lambda x, y: diffusion_time(x, y, t))
#
# Notes:
# - This allows TimeStepper to call M(t_n) and A(t_n) at each time step.
# - The stepper interface does not need to know if the matrix is truly time-dependent;
#   it just calls the function for the current time.
# - This approach keeps your code general and compatible with both constant
#   and time-dependent coefficients.
# -------------------------------------------------------------------


class TimeStepper(ABC):

    def __init__(self, M, A, assembler: Assembler, f, t0: float, dt: float, dirichlet_bc: dict):

        """
        Parameters
        ----------
        M : callable
            Mass matrix as a function of time: M(t)
        A : callable
            Stiffness/convection/reaction matrix as a function of time: A(t)
        assembler : Assembler
            FEM assembler object, used to compute load vector and apply BCs
        f : callable
            Source term f(x, y, t)
        dt : float
            Time step size
        dirichlet_bc : dict
            Dictionary of Dirichlet BCs at all time steps.
            Keys are global node indices; values are np.ndarray of length n_steps:
            {0: [0.0, 0.45, 3.4], 5: [1.0, 3.4, 23.4]}
        t0 : float, optional
            Initial time of simulation (default is 0.0)
        """

        self.M = M
        self.A = A
        self.assembler = assembler
        self.f = f
        self.dt = dt
        self.dirichlet_bc = dirichlet_bc
        self.t0 = t0  # needed to compute step index

    @abstractmethod
    def step(self, u_n, t_n):
        pass

class BackwardEuler(TimeStepper):

    def step(self, u_n, t_n):
        
        """
        Advance the solution by one time step using the Backward Euler method
        for a time-dependent FEM problem.

        Solves the semi-discrete system:

            M(t) u'(t) + A(t) u(t) = F(t),

        using the fully implicit Backward Euler discretization:

            (M_{n+1} + Δt A_{n+1}) u_{n+1} = M_{n+1} u_n + Δt F_{n+1}

        where:
            - M(t) : time-dependent mass matrix
            - A(t) : time-dependent stiffness/convection/reaction matrix
            - F(t) : time-dependent load vector
            - u_n   : solution vector at previous time step t_n
            - Δt    : time step size

        Parameters
        ----------
        u_n : np.ndarray
            Solution vector at the previous time step t_n.
        t_n : float
            Current time corresponding to u_n.

        Returns
        -------
        u_np1 : np.ndarray
            Solution vector at the next time step t_{n+1}.

        Notes
        -----
        - Dirichlet boundary conditions are applied at t_{n+1} using
        `self.dirichlet_bc`, which should store values for all time steps
        as a dictionary mapping node indices to arrays of length n_steps.
        - Time-dependent matrices M(t) and A(t) are obtained by calling
        the provided callable functions `self.M(t)` and `self.A(t)`.
        - Fully implicit: previous step matrices or load vectors are not needed.
        """

        t = t_n + self.dt # t_n+1

        i = int(round((t - self.t0)/self.dt)) # n+1

        M = self.M(t) # M(t_n+1)
        A = self.A(t) # A(t_n+1)

        func = lambda x, y: self.f(x, y, t) # f(x, y, t_n+1)

        F = self.assembler.global_load_vector(func = func) # F_n+1

        lhs = M + self.dt*A
        rhs = M@u_n + self.dt*F.ravel()

        # Boundary condition is computed at t_n+1
        dirichlet = {k: v[i] for k, v in self.dirichlet_bc.items()}

        lhs, rhs = self.assembler.apply_Dirichlet_bc(lhs, rhs, dirichlet)

        return LinearSolver(lhs, rhs).solve()

class CrankNicolson(TimeStepper):

    def step(self, u_n, A_n, M_n, F_n, t_n):
        
        """
        Advance the solution by one time step using the Crank–Nicolson method.

        Solves the semi-discrete system:

            M(t) u'(t) + A(t) u(t) = F(t),

        using the Crank-Nicolson time discretization:

            (M_{n+1} + Δt/2 * A_{n+1}) u_{n+1} =
                (M_n - Δt/2 * A_n) u_n + Δt/2 * (F_n + F_{n+1})

        where:
            - M(t) : time-dependent mass matrix
            - A(t) : time-dependent stiffness/convection/reaction matrix
            - F(t) : time-dependent load vector
            - u_n   : solution at previous time step
            - Δt    : time step size

        Parameters
        ----------
        u_n : np.ndarray
            Solution vector at the previous time step t_n.
        A_n : np.ndarray
            Stiffness/convection/reaction matrix at previous time step t_n.
        M_n : np.ndarray
            Mass matrix at previous time step t_n.
        F_n : np.ndarray
            Load vector at previous time step t_n.
        t_n : float
            Current time corresponding to u_n.

        Returns
        -------
        u_np1 : np.ndarray
            Solution vector at the next time step t_{n+1}.
        A_np1 : np.ndarray
            Stiffness/convection/reaction matrix at t_{n+1}.
        M_np1 : np.ndarray
            Mass matrix at t_{n+1}.
        F_np1 : np.ndarray
            Load vector at t_{n+1}.

        Notes
        -----
        - Dirichlet boundary conditions are applied at t_{n+1} using
        `self.dirichlet_bc`, which should store values for all time steps
        as a dictionary mapping node indices to arrays of length n_steps.
        - Previous step matrices (A_n, M_n) and load vector (F_n) are
        required for Crank–Nicolson, and the method returns the current
        step matrices/load vector for caching in the next iteration.
        - This method is semi-implicit, using information from both the
        previous and current time steps to achieve second-order accuracy
        in time.
        """

        t = t_n + self.dt # t_n+1

        i = int(round((t - self.t0)/self.dt)) # n+1

        M = self.M(t) # M(t_n+1)
        A = self.A(t) # A(t_n+1)

        func = lambda x, y: self.f(x, y, t) # f(x, y, t_n+1)

        F = self.assembler.global_load_vector(func = func)    # F_n+1

        lhs = M + 0.5*self.dt*A
        rhs = (M_n - 0.5*self.dt*A_n)@u_n + 0.5*self.dt*(F_n + F)

        # Boundary condition is computed at t_n+1
        dirichlet = {k: v[i] for k, v in self.dirichlet_bc.items()}

        lhs, rhs = self.assembler.apply_Dirichlet_bc(lhs, rhs, dirichlet)

        u = LinearSolver(lhs, rhs).solve() # u_n+1

        return u, A, M, F