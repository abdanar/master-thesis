from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import numpy as np
from scipy.sparse import coo_array
from fem.localint import LocalIntegrator
from fem.phyelement import PhysicalElement
from fem.refelement import ReferenceElement
from utils.logger import get_logger
logger = get_logger(__name__)
if TYPE_CHECKING:
    from fem.femspace import FEMSpace

#----------------- Assembler for Finite Element Method (FEM) -------------------------
# The `Assembler` class is responsible for assembling the global stiffness matrix, 
# convection matrix, reaction matrix, and load vector for a given finite element space. 
# It uses the local integrator to compute the contributions from each element and 
# assembles them into global sparse matrices and vectors. 
#
# The assembler handles both 1D and 2D meshes and supports different types of PDEs by 
# allowing users to specify diffusion, convection, and reaction coefficients as functions.
# The assembly process is optimized for efficiency, especially for large meshes, by
# using sparse matrix formats and minimizing redundant computations.
#
# The `coo_array` format is used for efficient assembly of the global matrices, which can 
# then be converted to more efficient formats (e.g., CSR or CSC) for solving the linear systems.
# Depending on the solver type, i.e., direct or iterative, the assembler can be configured to 
# produce matrices in the appropriate format for optimal performance.
# --------------------------------------------------------------------------------------

class Assembler:
    def __init__(self, femspace: "FEMSpace"):
        self.mesh = femspace.mesh
        self.degree = femspace.degree
        self.dim = femspace.dim
        self.domain = femspace.domain
        self.space = femspace.space
        self.nnodes = self.mesh.nnodes()

    def _build_physical_element(self, element: np.ndarray, ref_element: ReferenceElement) -> PhysicalElement:
        if self.dim == 1:
            verts = self.mesh.vertices[[element[0], element[-1]]]
        else:
            verts = self.mesh.vertices[element[:3]]
        return PhysicalElement(verts, ref_element)

    # Define global stiffness matrix
    def global_stiffness_matrix(self, diffusion: Callable, quadrature_order: int = 2) -> coo_array:
        logger.debug(f"Assembling global stiffness matrix for degree={self.degree} with {self.mesh.nelements()} elements")
        elements = self.mesh.elements
        ref_element = ReferenceElement(self.dim, self.domain, self.space, self.degree)
        rows, cols, vals = [], [], []
        for element in elements:
            phy_element = self._build_physical_element(element, ref_element)
            lstiffness = LocalIntegrator(phy_element, quadrature_order).local_stiffness_matrix(diffusion)
            for i_local, i_global in enumerate(element):
                for j_local, j_global in enumerate(element):
                    rows.append(i_global)
                    cols.append(j_global)
                    vals.append(lstiffness[i_local, j_local])
        K_global = coo_array((vals, (rows, cols)), shape=(self.nnodes, self.nnodes))
        logger.debug("Global stiffness matrix assembly complete")
        return K_global
    
    # Define global convection matrix
    def global_convection_matrix(self, convection: Callable, quadrature_order: int = 2) -> coo_array:
        logger.debug(f"Assembling global convection matrix for degree={self.degree}")
        elements = self.mesh.elements
        ref_element = ReferenceElement(self.dim, self.domain, self.space, self.degree)
        rows, cols, vals = [], [], []
        for element in elements:
            phy_element = self._build_physical_element(element, ref_element)
            lconvection = LocalIntegrator(phy_element, quadrature_order).local_convection_matrix(convection)
            for i_local, i_global in enumerate(element):
                for j_local, j_global in enumerate(element):
                    rows.append(i_global)
                    cols.append(j_global)
                    vals.append(lconvection[i_local, j_local])
        C_global = coo_array((vals, (rows, cols)), shape=(self.nnodes, self.nnodes))
        logger.debug("Global convection matrix assembly complete")
        return C_global
    
    # Define global reaction matrix
    def global_reaction_matrix(self, reaction: Callable, quadrature_order: int = 2) -> coo_array:
        logger.debug(f"Assembling global reaction matrix for degree={self.degree}")
        elements = self.mesh.elements
        ref_element = ReferenceElement(self.dim, self.domain, self.space, self.degree)
        rows, cols, vals = [], [], []
        for element in elements:
            phy_element = self._build_physical_element(element, ref_element)
            lreaction = LocalIntegrator(phy_element, quadrature_order).local_reaction_matrix(reaction)
            for i_local, i_global in enumerate(element):
                for j_local, j_global in enumerate(element):
                    rows.append(i_global)
                    cols.append(j_global)
                    vals.append(lreaction[i_local, j_local])
        R_global = coo_array((vals, (rows, cols)), shape=(self.nnodes, self.nnodes))
        logger.debug("Global reaction matrix assembly complete")
        return R_global
    
    # Define global load vector
    def global_load_vector(self, func: Callable, quadrature_order: int = 2) -> np.ndarray:
        logger.debug(f"Assembling global load vector for degree={self.degree}")
        F_global = np.zeros(self.nnodes)
        ref_element = ReferenceElement(self.dim, self.domain, self.space, self.degree)
        elements = self.mesh.elements
        for element in elements:
            phy_element = self._build_physical_element(element, ref_element)
            lload = LocalIntegrator(phy_element, quadrature_order).local_load_vector(func)
            for i_local, i_global in enumerate(element):
                F_global[i_global] += lload[i_local]
        logger.debug("Global load vector assembly complete")
        return F_global