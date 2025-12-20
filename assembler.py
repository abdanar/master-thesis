import numpy as np
from mesh import Mesh
from refelement import ReferenceElement
from phyelement import PhysicalElement
from localint import LocalIntegrator
from logger import setup_logger

logger = setup_logger(__name__, level = 'info')

class Assembler:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    # The total number of basis functions, or total number of global degrees of freedom - including boundary 
    def ndof(self, degree: int):
        return self.mesh.nvertices() + self.mesh.nedges()*(degree - 1) + self.mesh.nelements()*(degree - 1)*(degree - 2)//2
    
    # below functions can be gathered under one function and become significantly faster!

    def global_stiffness_matrix(self, diffusion, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global stiffness matrix
        n = self.ndof(degree)
        K_global = np.zeros((n, n))

        logger.debug(f"Assembling global stiffness matrix for degree={degree} with {self.mesh.nelements()} elements")

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lstiffness = LocalIntegrator(phy_element, quadrature_order).local_stiffness_matrix(diffusion)
            for i_local, i_global in enumerate(triangle):
                for j_local, j_global in enumerate(triangle):
                    K_global[i_global, j_global] += lstiffness[i_local, j_local]
        logger.debug("Global stiffness matrix assembly complete")
        return K_global
    
    def global_convection_matrix(self, convection, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global convection matrix
        n = self.ndof(degree)
        C_global = np.zeros((n, n))

        logger.debug(f"Assembling global convection matrix for degree={degree}")

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lconvection = LocalIntegrator(phy_element, quadrature_order).local_convection_matrix(convection)
            for i_local, i_global in enumerate(triangle):
                for j_local, j_global in enumerate(triangle):
                    C_global[i_global, j_global] += lconvection[i_local, j_local]
        logger.debug("Global convection matrix assembly complete")
        return C_global
    
    def global_mass_matrix(self, reaction, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global mass matrix
        n = self.ndof(degree)
        M_global = np.zeros((n, n))

        logger.debug(f"Assembling global mass matrix for degree={degree}")

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lmass = LocalIntegrator(phy_element, quadrature_order).local_mass_matrix(reaction)
            for i_local, i_global in enumerate(triangle):
                for j_local, j_global in enumerate(triangle):
                    M_global[i_global, j_global] += lmass[i_local, j_local] # if there is any problem, switch the the positions of i_global, j_global!
        logger.debug("Global mass matrix assembly complete")
        return M_global
    
    
    def global_load_vector(self, func, quadrature_order = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):

        # Define global stiffness matrix
        n = self.ndof(degree)
        F_global = np.zeros((n, 1))

        logger.debug(f"Assembling global load vector for degree={degree}")

        ref_element = ReferenceElement(domain, space, degree)
        _, triangles = self.mesh.upgrade(domain, space, degree)
        for triangle in triangles:
            phy_element = PhysicalElement(self.mesh.vertices[triangle], ref_element)
            lload = LocalIntegrator(phy_element, quadrature_order).local_load_vector(func)
            for i_local, i_global in enumerate(triangle):
                F_global[i_global] += lload[i_local]
        logger.debug("Global load vector assembly complete")
        return F_global
    

    # Apply Dirichlet boundary conditions
    def apply_Dirichlet_bc(self, K, rhs, dirichlet_nodes: dict) -> tuple[np.ndarray, np.ndarray]:

        """
        Apply Dirichlet (essential) boundary conditions to a finite element system.

        This function enforces prescribed values at specified degrees of freedom
        by modifying the global stiffness matrix and right-hand side vector
        in a "strong" sense. It zeros out the corresponding rows and columns
        of the matrix, sets the diagonal to 1, and sets the right-hand side
        to the prescribed boundary value.

        Parameters
        ----------
        K : numpy.ndarray
            The global stiffness (or system) matrix of shape (n, n), where n is
            the total number of degrees of freedom including boundary nodes.
        rhs : numpy.ndarray
            The global right-hand side vector of shape (n,) or (n, 1) corresponding
            to the system K u = rhs.
        dirichlet_nodes : dict
            A dictionary mapping global node indices to prescribed values.
            Example: {0: 0.0, 5: 1.0} means node 0 is fixed at 0 and node 5 at 1.

        Returns
        -------
        K_mod : numpy.ndarray
            The modified stiffness matrix with Dirichlet conditions applied.
        rhs_mod : numpy.ndarray
            The modified right-hand side vector with Dirichlet conditions applied.

        Notes
        -----
        - This function preserves symmetry of the matrix by zeroing both the row
        and the column corresponding to the Dirichlet node.
        - The original contributions in K and rhs at the Dirichlet nodes are overwritten.
        - This is the standard "strong imposition" method for essential boundary conditions.
        """

        logger.debug(f"Applying Dirichlet boundary conditions to {len(dirichlet_nodes)} nodes")

        # The total number of nodes, including boudnary
        nT = K.shape[0]

        # Change RHS considering the Dirichlet boundary conditions
        for i in range(nT):
            if i in dirichlet_nodes:
                rhs[i] = dirichlet_nodes[i]
            else:
                for node, value in dirichlet_nodes.items():
                    rhs[i] -= K[i, node]*value

        # Change LHS considering the Dirichlet boundary conditions
        for node, value in dirichlet_nodes.items():
            K[node, :] = 0     # zero out the row
            K[:, node] = 0     # zero out the column
            K[node, node] = 1  # set the diagonal to 1

        logger.debug("Dirichlet boundary conditions applied")
        
        return K, rhs

    # Apply Neumann boundary conditions - one needs to define separate function for extra term in load vector, which is a line integral