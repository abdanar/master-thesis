from mesh import Mesh

class PoissonProblem:
    def __init__(self, mesh: Mesh, func, dirichlet_bc = None, neumann_bc = None, robin_bc = None):
        self.f = func
        self.dirichlet = dirichlet_bc
        self.neumann = neumann_bc
        self.robin = robin_bc
    