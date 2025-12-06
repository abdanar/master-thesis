import numpy as np
from mesh import Mesh
from refelement import ReferenceElement
from phyelement import PhysicalElement
from localint import LocalIntegrator

class Assembler:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    # The total number of basis functions, or total number of global degrees of freedom
    def ndof(self, degree: int):
        return self.mesh.nvertices() + self.mesh.nedges()*(degree - 1) + self.mesh.nelements()*(degree - 1)*(degree - 2)//2
    
    # below functions can be gathered under one function and become significantly faster!

    def global_stiffness_matrix(self, diffusion, convection, reaction, func, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global stiffness matrix
        n = self.ndof(degree)
        K_global = np.zeros((n, n))

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lstiffness = LocalIntegrator(phy_element, diffusion, convection, reaction, func, quadrature_order).local_stiffness_matrix()
            for i_local, i_global in enumerate(triangle):
                for j_local, j_global in enumerate(triangle):
                    K_global[i_global, j_global] += lstiffness[i_local, j_local]
        return K_global
    
    def global_convection_matrix(self, diffusion, convection, reaction, func, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global convection matrix
        n = self.ndof(degree)
        C_global = np.zeros((n, n))

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lconvection = LocalIntegrator(phy_element, diffusion, convection, reaction, func, quadrature_order).local_convection_matrix()
            for i_local, i_global in enumerate(triangle):
                for j_local, j_global in enumerate(triangle):
                    C_global[i_global, j_global] += lconvection[i_local, j_local]
        return C_global
    
    def global_mass_matrix(self, diffusion, convection, reaction, func, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global mass matrix
        n = self.ndof(degree)
        M_global = np.zeros((n, n))

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lmass = LocalIntegrator(phy_element, diffusion, convection, reaction, func, quadrature_order).local_mass_matrix()
            for i_local, i_global in enumerate(triangle):
                for j_local, j_global in enumerate(triangle):
                    M_global[i_global, j_global] += lmass[i_local, j_local]
        return M_global
    
    
    def global_load_vector(self, diffusion, convection, reaction, func, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global stiffness matrix
        n = self.ndof(degree)
        F_global = np.zeros((n, 1))

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lload = LocalIntegrator(phy_element, diffusion, convection, reaction, func, quadrature_order).local_load_vector()
            for i_local, i_global in enumerate(triangle):
                F_global[i_global] += lload[i_local]
        return F_global
    

    # Apply boundary conditions
    def apply_Dirichlet_bc(self, K, f, dirichlet_nodes):
        """
        dirichlet_nodes: dict {node_index: value}
        """
        for node, value in dirichlet_nodes.items():
            K[node, :] = 0        # zero out the row
            K[node, node] = 1     # set diagonal to 1
            f[node] = value       # set RHS
        return K, f