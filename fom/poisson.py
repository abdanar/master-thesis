import sys
import numpy as np
from tqdm import trange
from fem.femspace import FEMSpace
from fem.assembler import Assembler
import logger as log 

logger = log.setup_logger(__name__, level = 'info')

class PoissonProblem:
    def __init__(self, femspace: FEMSpace, func, dirichlet_bc: dict):
        """
        FEM solver for the Poisson equation:
            - 1D: - u'' = f(x),   x ∈ Ω
            - 2D: - Δu(x, y) = f(x, y),   (x, y) ∈ Ω

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object
        func : callable
            Source term f(x) for 1D or f(x, y) for 2D
        dirichlet_bc : dict
            Dirichlet boundary conditions. Each key is a global node index, and
            the corresponding value specifies the solution value imposed at that node.
        """
        self.femspace = femspace
        self.f = func
        self.dirichlet = dirichlet_bc
        self.dim = femspace.dim

    def solve(self):

        if self.dim == 2:
            diffusion = lambda x, y: np.eye(2)
        else:  # 1D
            diffusion = lambda x: 1

        pde_assembler = Assembler(self.femspace)

        lhs = pde_assembler.global_stiffness_matrix(diffusion = diffusion)

        rhs = pde_assembler.global_load_vector(self.f)

        A, b = pde_assembler.apply_Dirichlet_bc(K = lhs, rhs = rhs, dirichlet_nodes = self.dirichlet)

        return np.linalg.solve(A, b)

