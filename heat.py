import numpy as np
from mesh import Mesh
from assembler import Assembler
from linearsolver import LinearSolver
from timestepper import TimeStepper, BackwardEuler, CrankNicolson

class HeatProblem:
    def __init__(self, mesh: Mesh, func, dt: float, t0: float, T: float, dirichlet_bc: dict, icond: dict, tstepper: str = 'BackwardEuler'):

        """
        FEM solver for the time-dependent heat equation:
            u_t(x, y, t) - ∇·(∇u(x, y, t)) = f(x, y, t),   (x, y) ∈ Ω, t ∈ (t0, T)
        with:
            - Dirichlet boundary conditions specified at all times
            - Initial condition u(x, y, t0) = u0(x, y)
        The semi-discrete system (after spatial FEM discretization) is:
            M u'(t) + A u(t) = F(t)
        where:
            - M : mass matrix
            - A : stiffness matrix
            - F(t) : load vector due to source term f(x, y, t)

        Parameters
        ----------
        mesh : Mesh
            FEM mesh object
        func : callable
            Source term f(x, y, t)
        dt : float
            Time step size
        t0 : float
            Initial time
        T : float
            Final time
        dirichlet_bc : dict
            Dirichlet BCs at all time steps. Keys are global node indices;
            values are np.ndarray of length n_steps: {0: [0.0, 0.45, 3.4], ...}
        icond : np.ndarray
            Initial condition vector at t0
        tstepper : str
            Time integration method ('BackwardEuler' or 'CrankNicolson')
        """

        self.mesh = mesh
        self.f = func
        self.dt = dt
        self.t0 = t0
        self.T = T
        self.dirichlet = dirichlet_bc
        self.initial = icond
        self.tstepper = tstepper

    def assemble_space(self):

        """
        Assemble space-dependent matrices (M, A) using the assembler.
        Returns:
            M : np.ndarray
                Mass matrix
            A : np.ndarray
                Stiffness matrix
            assembler : Assembler
                FEM assembler object
        """

        assembler = Assembler(self.mesh)
        diffusion = lambda x, y: np.eye(2)
        reaction = lambda x, y: 1

        M = assembler.global_mass_matrix(reaction=reaction)
        A = assembler.global_stiffness_matrix(diffusion=diffusion)

        return M, A, assembler

    def solve(self):

        """
        Solve the PDE using the specified time-stepping method.

        Returns
        -------
        u_history : list of np.ndarray
            Solution vectors at all time steps
        """

        M, A, assembler = self.assemble_space()
        n_steps = int(np.ceil((self.T - self.t0) / self.dt))
        u_history = []

        # Initial solution
        u_n = self.initial.copy()
        u_history.append(u_n)

        # Initialize previous step matrices and load vector for Crank-Nicolson
        F_prev = assembler.global_load_vector(lambda x, y: self.f(x, y, self.t0))
        A_prev, M_prev = A, M

        # Select time-stepper
        if self.tstepper == 'BackwardEuler':
            stepper = BackwardEuler(M=lambda t: M, A=lambda t: A,
                                    assembler=assembler, f=self.f,
                                    dt=self.dt, t0=self.t0, dirichlet_bc=self.dirichlet)
        elif self.tstepper == 'CrankNicolson':
            stepper = CrankNicolson(M=lambda t: M, A=lambda t: A,
                                    assembler=assembler, f=self.f,
                                    dt=self.dt, t0=self.t0, dirichlet_bc=self.dirichlet)
        else:
            raise ValueError(f"Unknown time-stepper: {self.tstepper}")

        # Time-stepping loop
        t_n = self.t0
        for step in range(n_steps):
            if self.tstepper == 'BackwardEuler':
                u_n = stepper.step(u_n, t_n)
            elif self.tstepper == 'CrankNicolson':
                u_n, A_prev, M_prev, F_prev = stepper.step(u_n, A_prev, M_prev, F_prev, t_n)
            t_n += self.dt
            u_history.append(u_n.copy())

        return u_history