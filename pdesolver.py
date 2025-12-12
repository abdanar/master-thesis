from mesh import Mesh
from assembler import Assembler

class PoissonProblem:
    def __init__(self, mesh: Mesh, func, dirichlet_bc = None, neumann_bc = None, robin_bc = None):
        self.mesh = mesh
        self.f = func
        self.dirichlet = dirichlet_bc
        self.neumann = neumann_bc
        self.robin = robin_bc

    def linsystem(self):

        assembler = Assembler(self.mesh)
        stiffness_matrix = assembler.global_stiffness_matrix()
        load_vector = assembler.global_load_vector()
        K, rhs = assembler.apply_Dirichlet_bc(stiffness_matrix, load_vector, dirichlet_bc)


