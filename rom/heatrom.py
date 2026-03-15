import sys
import numpy as np
from tqdm import trange
from fom.heat import HeatProblem
import fem.timestepper as ts
import utils.logger as log 
from fem.femspace import FEMSpace
from fem.assembler import Assembler
from fem.linearsolver import LinearSolver, DirectSolver, JacobiSolver, CGSolver
from rom.reducedstepper import ModifiedTheta

logger = log.setup_logger(__name__, level = 'info')

class HeatROM:
    def __init__(self, femspace: FEMSpace, basis: np.ndarray, func, dt: float, t0: float, T: float, dirichlet_bc: np.ndarray, icond: np.ndarray, 
                 offline, tstepper: str = 'Theta', theta: float = 0.5):
        self.femspace = femspace
        self.m1, self.m2, self.m3, self.m4 = offline
        self.f = func
        self.dt = dt
        self.t0 = t0
        self.T = T
        self.dirichlet = dirichlet_bc
        self.initial = icond
        self.tstepper = tstepper
        self.theta = theta
        self.ntime = int((T - t0)/dt) + 1 # total number of time nodes (including boundaries)
        self.dim = femspace.dim
        self.basis = basis
        self.r = basis.shape[1] # reduced order
        self.assembler = Assembler(self.femspace)
        if self.dim == 2:
            self.Fr = lambda t: basis.T@self.assembler.global_load_vector(lambda x, y: self.f(x, y, t))
        else:
            self.Fr = lambda t: basis.T@self.assembler.global_load_vector(lambda x: self.f(x, t))
    
    def solve(self) -> np.ndarray:
        nsteps = self.ntime - 1
        solution = np.zeros((self.femspace.mesh.nnodes(), self.ntime))
        logger.debug(f"Using {self.tstepper} time-stepping | Starting time integration | nsteps = {nsteps}, dt = {self.dt}")
        u = self.basis.T @ (self.initial - self.dirichlet[:, 0])
        solution[:, 0] = self.initial
        t = self.t0
        with trange(nsteps, desc = "\033[92m Reduced Heat Solver\033[0m", unit="step",ascii = "░▒█", ncols = 100, disable = not sys.stdout.isatty()) as pbar:
            for step in pbar:
                pbar.set_postfix_str(f"\033[93mt={t:.3e}\033[0m")  # yellow t
                rhs = self.m3 @ u + self.dt * self.Fr(t + self.dt).ravel() - self.m2 @ self.dirichlet[:, step + 1] + self.m1 @ self.dirichlet[:, step]
                u = np.linalg.solve(self.m4, rhs)
                t += self.dt
                solution[:, step + 1] = self.basis @ u + self.dirichlet[:, step + 1]
        return solution