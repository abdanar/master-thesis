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
    def __init__(self, dim: int = 2, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):
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
        self.ref_nodes = self.reference_nodes()
        if self.dim == 1:
            self.nbasis = degree + 1
        elif self.dim == 2:
            self.nbasis = (degree + 1)*(degree + 2)//2

    def reference_nodes(self) -> np.ndarray:
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
            return np.linspace(0, 1, self.nbasis)[:, None] # the order of the points is exactly same as the order of element nodes, i.e., left vertex, interior nodes (left → right), right vertex
        elif self.dim == 2 and self.domain == 'triangle':
            # Note: There are (degree + 1)(degree + 2)/2 nodes for Lagrange elements of given degree on a triangle
            vertex_nodes = np.array([[0, 0], [1, 0], [0, 1]]) # counterclockwise order
            division = np.linspace(0., 1., num = self.degree + 1)[1: -1]
            if self.degree > 1:
                # Define edge nodes for each edge of the triangle -> counterclockwise order
                edge_nodes = np.zeros((3*self.degree - 3, 2))
                # (0,0) -> (1,0) edge nodes (from left to right)
                edge_nodes[:self.degree-1] = np.column_stack([division, np.zeros_like(division)])
                # (1,0) -> (0,1) edge nodes (from right to left diagonally)
                edge_nodes[self.degree-1:2*(self.degree-1)] = np.column_stack([1 - division, division])
                # (0,1) -> (0,0) edge nodes (from above to below)
                edge_nodes[2*(self.degree-1):] = np.column_stack([np.zeros_like(division), division])[::-1]
                # Define interior nodes for the triangle -> from bottom to top rows and left to right in each row
                idx = 0
                interior_nodes = np.zeros(((self.degree - 1)*(self.degree - 2)//2, 2))
                for row in range(self.degree - 2): # for each row, the order is from left to right, and from bottom row to top row
                    coldiv = division[:-1-row]
                    interior_nodes[idx: idx + len(coldiv)] = np.column_stack([coldiv, division[row]*np.ones_like(coldiv)])
                    idx += len(coldiv)
            else:
                edge_nodes = np.zeros((0, 2))
                interior_nodes = np.zeros((0, 2))
            return np.vstack([vertex_nodes, edge_nodes, interior_nodes])
        else:
            raise NotImplementedError(f"Reference nodes not implemented for domain '{self.domain}' and dim={self.dim}.")

    def phi(self, ref_point) -> np.ndarray:
        """
        Evaluate Lagrange shape functions at a point in the reference element.

        This method returns the values of all nodal Lagrange basis functions
        evaluated at `ref_point`. The ordering of the basis functions is such
        that φ_i corresponds to the i-th reference node returned by
        `self.reference_nodes()`.

        Supported reference elements
        -----------------------------
        - 1D interval [0, 1]
        - 2D reference triangle with vertices (0,0), (1,0), (0,1)

        Parameters
        ----------
        ref_point : scalar or np.ndarray
            Evaluation point in reference coordinates.
            - 1D: scalar
            - 2D: shape (2,)

        Returns
        -------
        phi : np.ndarray
            Array of shape (nbasis,), where phi[i] = φ_i(ref_point).
        """
        if self.dim == 1 or self.domain == 'interval':
            if self.degree == 1:
                return np.array([1 - ref_point, ref_point])
            else:
                nbasis = self.nbasis
                # Solve 1D Vandermonde system for Lagrange basis
                nodes = self.reference_nodes()[:, 0]
                V = np.vander(nodes, increasing = True).T  # shape (nbasis, nbasis)
                phi = np.linalg.solve(V, np.array([ref_point**i for i in range(nbasis)]))
                return phi
        elif self.dim == 2 and self.domain == 'triangle':
            x_ref, y_ref = ref_point
            if self.degree == 1:
                return np.array([1 - x_ref - y_ref, x_ref, y_ref])
            else:
                # Solve 2D Vandermonde system for Lagrange basis
                ref_nodes = self.reference_nodes()
                # Build monomial basis (1, x, y, x^2, xy, y^2, ...)
                monomials = []
                for i in range(self.degree + 1):
                    for j in range(self.degree + 1 - i):
                        monomials.append((i, j))
                V = np.array([[node[0]**p[0]*node[1]**p[1] for p in monomials] for node in ref_nodes])
                rhs = np.array([x_ref**p[0]*y_ref**p[1] for p in monomials])
                phi = np.linalg.solve(V, rhs)
                return phi
        else:
            raise NotImplementedError(f"phi not implemented for domain '{self.domain}' and dim={self.dim}.")
    
    def grad_phi(self, ref_point) -> np.ndarray:
        """
        Evaluate gradients of Lagrange shape functions at a point
        in the reference element.

        This method computes the gradients ∇φ_i of all nodal Lagrange
        basis functions evaluated at the reference point `ref_point`.
        The ordering is consistent with `self.reference_nodes()`, i.e.,
        ∇φ_i corresponds to the i-th reference node.

        Supported reference elements
        -----------------------------
        - 1D reference interval [0, 1]
        - 2D reference triangle with vertices (0,0), (1,0), (0,1)

        In 1D, the gradient is the derivative with respect to the
        reference coordinate ξ.
        In 2D, the gradient is taken with respect to (x, y).

        Parameters
        ----------
        ref_point : scalar or np.ndarray
            Evaluation point in reference coordinates.
            - 1D: scalar ξ
            - 2D: array-like of length 2, representing (x, y)

        Returns
        -------
        grad_phi : np.ndarray
            Array containing the gradients of the shape functions:
            - 1D: shape (nbasis, 1), where grad_phi[i, 0] = dφ_i/dξ
            - 2D: shape (nbasis, 2), where
                grad_phi[i] = [∂φ_i/∂x, ∂φ_i/∂y]

        Notes
        -----
        - For degree 1 elements, gradients are constant and given explicitly.
        - For higher degrees, gradients are computed by differentiating
          the monomial basis and solving the corresponding Vandermonde system.
        - The implementation assumes Lagrange nodal basis functions
          defined on the reference element.
        """
        if self.dim == 1 or self.domain == "interval":
            # Linear case: constant gradients
            if self.degree == 1:
                return np.array([[-1.0], [1.0]])
            else:
                nbasis = self.nbasis
                grad_phi = np.zeros((nbasis, 1))
                xi = float(ref_point)
                # Higher order: differentiate Vandermonde system
                nodes = self.reference_nodes()[:, 0]
                # Vandermonde matrix: V_{ij} = ξ_j^i
                V = np.vander(nodes, N=nbasis, increasing=True).T
                # Derivatives of monomials at xi: d/dx (x^k) = k x^{k-1}
                b = np.zeros(nbasis)
                for k in range(1, nbasis):
                    b[k] = k * xi**(k - 1)
                grad_phi[:, 0] = np.linalg.solve(V, b)
                return grad_phi
        elif self.dim == 2 and self.domain == "triangle":
            if self.degree == 1:
                return np.array([[-1, -1], [1, 0], [0, 1]]) # gradients of linear shape functions are constant
            else:
                x_ref, y_ref = ref_point
                grad_phi = np.zeros((self.nbasis, 2))
                ref_nodes = self.reference_nodes()
                # Monomial basis ordered by total degree
                monomials = [(i, j) for i in range(self.degree + 1) for j in range(self.degree + 1 - i)]
                # Vandermonde matrix
                V = np.array([[node[0]**p * node[1]**q for (p, q) in monomials] for node in ref_nodes])
                # Right-hand sides: gradients of monomials at (x, y)
                bx = np.array([p * x_ref**(p - 1) * y_ref**q if p > 0 else 0.0 for (p, q) in monomials])
                by = np.array([q * x_ref**p * y_ref**(q - 1) if q > 0 else 0.0 for (p, q) in monomials])
                grad_phi[:, 0] = np.linalg.solve(V, bx)
                grad_phi[:, 1] = np.linalg.solve(V, by)
                return grad_phi
        else:
            raise NotImplementedError(f"grad_phi not implemented for domain '{self.domain}' and dim={self.dim}.")

    def shape_functions(self):
        """
        Return a dictionary of Lagrange shape functions for the reference element,
        keyed by the local node index and node coordinates.

        Each shape function corresponds to a reference node and can be evaluated
        at any point in the reference element. The dictionary keys are tuples
        `(i, coords)` where `i` is the local node index and `coords` is either
        a scalar (1D) or a tuple of floats (2D or higher).

        Returns
        -------
        phi_dict : dict[tuple[int, float or tuple[float]], Callable]
            Dictionary mapping `(node_index, node_coords)` to callable shape functions.
            - Key: `(i, coords)`
                - `i` : local node index (0 <= i < `self.nbasis`)
                - `coords` : node coordinates in reference element
                    - 1D interval: scalar
                    - 2D triangle: tuple of floats
            - Value: callable `phi(ref_point)` returning the value of the
            corresponding Lagrange shape function at a given reference point.

        Notes
        -----
        - Node ordering matches `self.reference_nodes()`.
        - Each callable uses a closure (`i=i`) to safely reference its node.
        - The input `ref_point` should have shape:
            - 1D interval: scalar or array-like with 1 element
            - 2D triangle: array-like with shape `(2,)`

        Example
        -------
            >>> ref_elem = ReferenceElement(dim=2, domain='triangle', degree=2)
            >>> phi_dict = ref_elem.shape_functions()
            >>> x_hat = np.array([0.3, 0.2])
            >>> for key, phi_i in phi_dict.items():
            >>>     index, coords = key
            >>>     val = phi_i(x_hat)
            >>>     print(f"Node index: {index}, coords: {coords}, phi({x_hat}) = {val}")
        """
        phi_dict = {}
        nodes = self.reference_nodes()
        for i, node in enumerate(nodes):
            key = (i, tuple(node)) if node.shape[0] > 1 else (i, node[0])
            phi_dict[key] = lambda x, i = i: self.phi(x)[i]
        return phi_dict