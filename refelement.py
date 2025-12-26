import numpy as np

# --------------------------------------------------------------------------
# Reference Element for Finite Element Method (FEM)
#
# This class defines a single **reference element**. A reference element
# serves as a standard template for all elements of the same type in the mesh.
# Actual physical elements are mapped from this reference element using
# an affine or more general transformation.
#
# The finite element is defined by:
#   K - element domain (e.g., interval, triangle, quadrilateral)
#   S - space of shape functions (e.g., Lagrange, Hermite)
#   N - the set of nodal variables (nodes in the reference element)
#
# Supported dimensions:
#   1D - interval elements
#   2D - triangular elements (counterclockwise node ordering)
#
# The reference element stores:
#   - Nodes in the reference domain (`reference_nodes`)
#   - Shape functions (`phi`) evaluated at any reference point
#   - Gradients of shape functions (`grad_phi`) at any reference point
#
# Notes:
#   - Node ordering: vertex nodes first, then edge nodes, then interior nodes
#   - Degree 1: only vertex nodes
#   - Degree 2: vertex + edge nodes, no interior nodes
#   - Degree >= 3: vertex + edge + interior nodes
# --------------------------------------------------------------------------

class ReferenceElement:
    def __init__(self, dim: int = 1, domain = None, space: str = 'Lagrange', degree: int = 1):
        """
        Reference element for Lagrange FEM.

        Parameters
        ----------
        dim : int
            Dimension of the element (1, 2, 3, ...)
        domain : str
            Type of reference element, e.g., 'interval', 'triangle', 'quadrilateral', 'tetrahedron'
            If None, default to 'interval' for 1D and 'triangle' for 2D
        space : str
            Function space, currently only 'Lagrange'
        degree : int
            Polynomial degree
        """
        if dim not in [1, 2]:
            raise ValueError(f"Unsupported dimension: {dim}. Only 1D and 2D supported.")
        self.dim = dim
        if domain is None:
            self.domain = 'interval' if dim == 1 else 'triangle'
        else:
            self.domain = domain
        self.space = space
        self.degree = degree

        # Reference nodes and number of basis functions
        self.ref_nodes = self.reference_nodes()
        self.nbasis = (degree + 1)*(degree + 2)//2

    def reference_nodes(self):
        """
        Compute the reference nodes of the element for Lagrange shape functions.

        The nodes are returned in a **counterclockwise order** for 2D triangular elements:
            1. Vertex nodes (corner points of the element)
            2. Edge nodes (evenly spaced along edges, following counterclockwise order)
            3. Interior nodes (inside the element, ordered from bottom to top and left to right in each row;
               only present for degree ≥ 3)

        For 1D interval elements, nodes are equally spaced on [0, 1].

        Returns
        -------
        nodes : np.ndarray
            Array of reference nodes. Shape:
                - (degree + 1, 1) for 1D interval
                - ((degree + 1)*(degree + 2)//2, 2) for 2D triangle

            Ordering:
                - 1D: nodes[0] = 0, nodes[-1] = 1, equally spaced in between
                - 2D triangle:
                    - First 3 nodes: vertices [(0,0), (1,0), (0,1)]
                    - Next 3*(degree-1) nodes: edge nodes in counterclockwise order (0->1, 1->2, 2->0)
                    - Remaining nodes: interior nodes arranged row by row (only for degree ≥ 3)

        Notes
        -----
        - Degree 1: only vertex nodes, no edge or interior nodes
        - Degree 2: vertex + edge nodes, no interior nodes
        - Degree ≥ 3: vertex + edge + interior nodes
        - The returned array is always 2D: shape (nbasis, dim), so that indexing is consistent across 1D and 2D.

        Raises
        ------
        NotImplementedError
            If the combination of `self.domain` and `self.dim` is unsupported.

        Examples
        --------
        1D interval, degree 3:
            >>> ref_elem = ReferenceElement(dim=1, degree=3)
            >>> ref_elem.reference_nodes()
            array([[0.0],
                [0.3333],
                [0.6667],
                [1.0]])

        2D triangle, degree 2 (no interior nodes):
            >>> ref_elem = ReferenceElement(dim=2, degree=2)
            >>> ref_elem.reference_nodes()
            array([[0, 0],    # vertices
                [1, 0],
                [0, 1],
                [0.5, 0],  # edge node (0->1)
                [0.5, 0.5],# edge node (1->2)
                [0, 0.5]]) # edge node (2->0)
        """
        if self.dim == 1 or self.domain == 'interval':
            nodes = np.linspace(0, 1, self.degree + 1)[:, None]
            return nodes
        elif self.dim == 2 and self.domain == 'triangle':
            # Note: There are (degree + 1)(degree + 2)/2 nodes for Lagrange elements of given degree on a triangle
            vertex_nodes = np.array([[0, 0], [1, 0], [0, 1]]) # counterclockwise order
            division = np.linspace(0., 1., num = self.degree + 1)[1: -1]
            if self.degree > 1:
                # Define edge nodes for each edge of the triangle -> counterclockwise order
                edge_nodes = np.zeros((3*self.degree - 3, 2))
                # (0,0) -> (1,0) edge nodes
                edge_nodes[:self.degree-1] = np.column_stack([division, np.zeros_like(division)])
                # (1,0) -> (0,1) edge nodes
                edge_nodes[self.degree-1:2*(self.degree-1)] = np.column_stack([1 - division, division])
                # (0,1) -> (0,0) edge nodes
                edge_nodes[2*(self.degree-1):] = np.column_stack([np.zeros_like(division), division])
                # Define interior nodes for the triangle -> from bottom to top rows and left to right in each row
                idx = 0
                interior_nodes = np.zeros((int((self.degree - 1)*(self.degree - 2)/2), 2))
                for row in range(self.degree - 2):
                    coldiv = division[:-1-row]
                    interior_nodes[idx: idx + len(coldiv)] = np.column_stack([division[row]*np.ones_like(coldiv), coldiv])
                    idx += len(coldiv)
            else:
                edge_nodes = np.zeros((0, 2))
                interior_nodes = np.zeros((0, 2))
            return np.vstack([vertex_nodes, edge_nodes, interior_nodes])
        else:
            raise NotImplementedError(f"Reference nodes not implemented for domain '{self.domain}' and dim={self.dim}.")

    # Define a function that computes the shape functions at a given point in the reference element
    def phi(self, ref_point: np.ndarray):
        
        # For reference triangle with Lagrange shape functions
        # Number of nodal basis shape functions nbasis = (degree + 1)(degree + 2)/2
        # The output is an array of shape (nbasis, ) given by [phi_0, phi_1, ..., phi_(nbasis-1)]
        # where phi_i is the value of the i-th shape function at the given reference point
        # so one can access the value of the i-th shape function at the reference point by phi[i]

        nbasis = (self.degree + 1)*(self.degree + 2)//2
        phi = np.zeros(nbasis)
        x_ref, y_ref = ref_point
        if self.degree == 1:
            phi[0] = 1 - x_ref - y_ref # phi_0(x, y) = 1 - x - y
            phi[1] = x_ref             # phi_1(x, y) = x
            phi[2] = y_ref             # phi_2(x, y) = y
        else:
            ref_nodes = self.reference_nodes()
            # construct Vandermonde like matrix for Lagrange basis functions in 2D and solve the system to get coefficients -> will be added later
        return phi
    
    # Define a function that computes the gradients of the shape functions at a given point in the reference element
    def grad_phi(self, ref_point: np.ndarray):

        # Compute the gradients of the shape functions at a given point in the reference element
        # Number of nodal basis shape functions nbasis = (degree + 1)(degree + 2)/2
        # The output is an array of shape (nbasis, 2) given by [gradphi_0, gradphi_1, ..., gradphi_(nbasis-1)],
        # where gradphi_i = [grad_{x}phi_i, grad_{y}phi_i] is the gradient of the i-th shape function at the given reference point
        # so one can access the gradient of the i-th shape function at the reference point by grad_phi[i]

        nbasis = (self.degree + 1)*(self.degree + 2)//2
        grad_phi = np.zeros((nbasis, 2))
        x_ref, y_ref = ref_point
        if self.degree == 1:
            grad_phi = np.array([[-1, -1], [1, 0], [0, 1]]) # gradients of linear shape functions are constant
        else:
            ref_nodes = self.reference_nodes()
            # construct Vandermonde like matrix for Lagrange basis functions in 2D and solve the system to get coefficients -> will be added later
        return grad_phi
    


    
