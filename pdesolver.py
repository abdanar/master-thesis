import numpy as np
from linearsolver import LinearSolver
from mesh import Mesh
from assembler import Assembler

class PoissonProblem:
    def __init__(self, mesh: Mesh, func, dirichlet_bc, neumann_bc, robin_bc):
        self.mesh = mesh
        self.f = func
        self.dirichlet = dirichlet_bc
        self.neumann = neumann_bc
        self.robin = robin_bc

    def solve(self):
        
        # Constant isotropic diffusion using lambda
        diffusion = lambda x, y: np.eye(2)
        
        # Define an assembler
        assembler = Assembler(self.mesh)

        # Compute the stiffness matrix
        stiffness_matrix = assembler.global_stiffness_matrix(diffusion = diffusion)

        # Compute the load vector
        load_vector = assembler.global_load_vector(func = self.f)

        # Apply the Dirichlet boundary condition
        lhsmatrix, rhs = assembler.apply_Dirichlet_bc(stiffness_matrix, load_vector, self.dirichlet)

        # Solve the linear system
        u = LinearSolver(lhsmatrix, rhs).solve()
    
        return u






