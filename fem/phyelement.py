import numpy as np
from fem.refelement import ReferenceElement

# ------------ Physical Element for Finite Element Method (FEM) ----------------
# This class defines a single **physical element** in 1D or 2D.
# A physical element is obtained by mapping a reference element
# from the standard domain to its actual coordinates in the mesh.
#
# The finite element is defined by:
#   K - physical element domain (interval in 1D, triangle in 2D)
#   P - space of shape functions (Lagrange FEM)
#   N - set of nodal variables (nodes in the physical element)
#
# Supported dimensions:
#   1D - interval elements
#   2D - triangular elements (counterclockwise vertex ordering)
#
# The physical element stores:
#   - Coordinates of vertices (`vertices`)
#   - Reference element object (`ref_element`)
#   - Methods for mapping points between reference and physical elements
#   - Jacobian, inverse Jacobian, and determinant of the transformation
#   - Evaluation of shape functions and their gradients at reference points
# --------------------------------------------------------------------------

# -------------------------- Optimization Notes ---------------------------------
# The current implementation is straightforward and prioritizes clarity. 
# Both 1D and 2D cases, as well as single and multiple point mappings, are handled 
# in a unified way. However, this may introduce some overhead due to conditional 
# checks and less efficient handling of multiple points in 2D. Therefore, the following optimizations could be considered:
# 1. Separate methods for 1D and 2D mappings to avoid runtime checks
# 2. Vectorized handling of multiple points in 2D to eliminate Python loops and leverage NumPy's optimized operations
# 3. Caching the Jacobian and its inverse for each element to avoid redundant calculations during shape function evaluations
# 4. Precomputing shape function values and gradients at quadrature points to speed up assembly processes
# 5. Using more efficient data structures or libraries (e.g., Numba, Cython) for critical sections of the code
# ------------------------------------------------------------------------------

class PhysicalElement:
    def __init__(self, vertices: np.ndarray, ref_element: ReferenceElement):
        """
        Physical element for 1D and 2D Lagrange FEM.

        Attributes
        ----------
        vertices : np.ndarray
            Physical coordinates of element vertices, only triangle corners are needed.
            - 1D: shape (2,) for interval.
            - 2D: shape (3, 2) for triangle.
        ref_element : ReferenceElement
            Reference element defining shape functions φ and gradients ∇φ.
        """
        self.vertices = vertices
        self.ref_element = ref_element

    def reference_to_physical(self, ref_point):   
        """
        Map a point or points from the reference element to the physical element.

        The reference element (e.g., unit interval or unit triangle) is mapped to the
        physical element defined by its vertex coordinates via an affine transformation.

        In 1D:
            - Reference element: [0, 1]
            - Physical element: [x0, x1] (vertices of the element)
            - Mapping: x = x0 + (x1 - x0) * ξ
            - Reference vertex 0 maps to physical vertex 0, reference vertex 1 maps to physical vertex 1

        In 2D (triangle):
            - Reference triangle vertices: (0,0), (1,0), (0,1)
            - Physical triangle vertices: vert[0], vert[1], vert[2]
            - Affine mapping: F(ξ, η) = R @ [ξ, η] + b
                - R = [[x1-x0, x2-x0],
                    [y1-y0, y2-y0]]   (edge vectors from vertex 0)
                - b = [x0, y0]           (anchor at vertex 0)
            - Vertex mapping:
                Reference (0,0) → Physical vert[0]
                Reference (1,0) → Physical vert[1]
                Reference (0,1) → Physical vert[2]

        Parameters
        ----------
        ref_point : scalar (1D) or array-like (2D: shape (2,) or (n, 2))
            Coordinates in the reference element.
                - 1D: scalar or array of shape (n,) or (n, 1) for n points
                - 2D: array-like of shape (2,) for a single point, or (n, 2) for n points (each row is a point)

        Returns
        -------
        float or np.ndarray
            Coordinates in the physical element.
            - 1D: scalar if a single point is provided, or array of shape (n,) or (n, 1) for multiple points
            - 2D: array of shape (2,) for a single point, or (n, 2) for multiple points

        Notes
        -----
        - The method handles both single points and multiple points.
        - For multiple points, the affine transformation is applied to each row of `ref_point`.
        - Broadcasting ensures that the translation vector is correctly added in all cases.
        - In 1D, the mapping is x = x0 + (x1 - x0) * ξ.
        - In 2D, the mapping is F(ξ, η) = R @ [ξ, η] + b with R = [[x1-x0, x2-x0], [y1-y0, y2-y0]] and b = [x0, y0].
        """
        if self.ref_element.dim == 1:
            x0, x1 = self.vertices[0], self.vertices[1]
            return x0 + (x1 - x0)*ref_point
        elif self.ref_element.dim == 2: # here note that even with upgraded elements, vertices are still triangle corners
            vert = self.vertices
            R = np.array([[vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
                        [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]]])
            b = vert[0]
            if ref_point.shape[0] == 2:
                return R@ref_point + b
            elif ref_point.shape[1] == 2:
                return ref_point @ R.T + b[:, np.newaxis].T
            else:
                raise ValueError("ref_point has invalid shape for 2D mapping.")
        else:
            raise NotImplementedError(f"reference_to_physical not implemented for dim={self.ref_element.dim}")
        
    def physical_reference_nodes(self):
        """
        Map all reference element nodes to their corresponding coordinates
        in the physical element using the affine transformation.

        Returns
        -------
        np.ndarray, shape (n_nodes, dim)
            Physical coordinates of all Lagrange nodes.
        """
        return self.reference_to_physical(self.ref_element.reference_nodes())

    def physical_to_reference(self, phys_point: float | np.ndarray) -> float | np.ndarray:
        """
        Map a point or points from the physical element back to the reference element.

        Parameters
        ----------
        phys_point : np.ndarray or float
            Coordinates in the physical element.
            - 1D: float or shape (n,) for n points
            - 2D: shape (2,) for a single point or (n, 2) for n points

        Returns
        -------
        np.ndarray or float
            Coordinates of the corresponding point(s) in the reference element.
            - 1D: float or array of shape (n,)
            - 2D: array of shape (2,) for a single point or (n, 2) for multiple points 
              (currently does not support multiple 2D points)

        Raises
        ------
        NotImplementedError
            If the element dimension is not 1 or 2.
        """
        if self.ref_element.dim == 1:
            x0, x1 = self.vertices[0], self.vertices[1]
            return (phys_point - x0)/(x1 - x0)
        elif self.ref_element.dim == 2:
            return np.linalg.inv(self.jacobian())@(phys_point - self.vertices[0])
        else:
            raise NotImplementedError(f"physical_to_reference not implemented for dim={self.ref_element.dim}")

    def jacobian(self) -> float | np.ndarray:
        """
        Compute the Jacobian of the transformation from the reference element to the physical element.

        For 1D interval element with vertices x0, x1:
            F(ξ) = x0 + (x1 - x0) * ξ
            J = x1 - x0  (scalar)

        For 2D triangular element with vertices
            vert[0] = (x0, y0), vert[1] = (x1, y1), vert[2] = (x2, y2):
            F(ξ, η) = vert[0] + (vert[1] - vert[0]) * ξ + (vert[2] - vert[0]) * η
            J = [[dF1/dξ, dF1/dη],
                [dF2/dξ, dF2/dη]]  (2x2 array)

        Returns
        -------
        np.ndarray or float
            - 1D: scalar, the Jacobian of the interval element
            - 2D: shape (2, 2), Jacobian matrix for the triangular element
        """
        if self.ref_element.dim == 1:
            return self.vertices[1] - self.vertices[0]
        elif self.ref_element.dim == 2:
            vert = self.vertices
            J = np.array([[vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
                        [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]]])
            return J
        else:
            raise NotImplementedError(f"Jacobian not implemented for dim={self.ref_element.dim}")
        
    def jacobian_inv(self) -> float | np.ndarray:
        """
        Compute the inverse of the Jacobian of the element transformation.

        For 1D elements, the Jacobian is a scalar dx/dξ, so its inverse
        is also a scalar. For 2D triangular elements, the Jacobian is
        a 2x2 matrix, and the inverse is a 2x2 matrix.

        Returns
        -------
        float or np.ndarray
            - 1D: scalar inverse of dx/dξ
            - 2D: 2x2 inverse matrix J^{-1}
        """
        return 1.0/self.jacobian() if self.ref_element.dim == 1 else np.linalg.inv(self.jacobian())
    
    def det_jacobian(self):
        """
        Compute the determinant of the Jacobian of the transformation from the reference element
        to the physical element.

        For 1D interval elements, the determinant is simply the Jacobian (scalar).
        For 2D triangular elements, the determinant is the usual 2x2 determinant of the Jacobian matrix.

        Returns
        -------
        float
            Determinant of the Jacobian.

        Raises
        ------
        NotImplementedError
            If the element dimension is not 1 or 2.
        """
        return self.jacobian() if self.ref_element.dim == 1 else np.linalg.det(self.jacobian())

    def phi_physical(self, ref_point: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the shape functions of the element at a given reference point.

        The shape functions are defined on the reference element, and this method
        simply calls the reference element's shape functions. The number of basis
        functions depends on the element type and polynomial degree:
            - 1D interval element of degree p: nbasis = p + 1
            - 2D triangular element of degree p: nbasis = (p + 1)(p + 2)/2

        Parameters
        ----------
        ref_point : np.ndarray or float
            Coordinates of the point in the reference element.
            - 1D: scalar
            - 2D: array of shape (2,)

        Returns
        -------
        np.ndarray, shape (nbasis,)
            Values of all shape functions at the given reference point.
            Access the i-th shape function by `phi_physical[i]`.
        """
        return self.ref_element.phi(ref_point)

    def grad_phi_physical(self, ref_point: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the gradients of the shape functions at a given reference point,
        mapped to the physical element.

        The gradients in the physical element are obtained via the transformation
            - 1D: grad_phi_phys = grad_phi_ref/J
            - 2D: grad_phi_phys = J^{-T}*grad_phi_ref (by construction, we need grad_phi_ref@J^{-1} instead)
        where J is the Jacobian of the mapping from the reference element 
        to the physical element.

        Parameters
        ----------
        ref_point : np.ndarray or float
            Coordinates of the point in the reference element.
            - 1D: scalar
            - 2D: array of shape (2,)

        Returns
        -------
        np.ndarray
            Gradients of all basis functions at the given reference point.
            - 1D: shape (nbasis, 1) 
            - 2D: shape (nbasis, 2), each row = [dphi_i/dx, dphi_i/dy]
        """
        if self.ref_element.dim == 1:
            return self.ref_element.grad_phi(ref_point)/self.jacobian()
        elif self.ref_element.dim == 2:
            return self.ref_element.grad_phi(ref_point)@self.jacobian_inv()
        else:
            raise NotImplementedError(f"grad_phi_physical not implemented for dim={self.ref_element.dim}")

    def local_to_global_mapping(self, element: np.ndarray) -> dict:
        """
        Return the mapping from local node indices to global node indices for a mesh element.

        In finite element methods, each element has a set of local nodes
        numbered from 0 to n_nodes-1. This method provides the correspondence
        between these local node indices and the global node indices in the mesh.

        Parameters
        ----------
        element : np.ndarray
            Array of global node indices defining the element.  
            The order of nodes must match the reference element.

        Returns
        -------
        dict[int, int]
            Dictionary mapping:
                - Key: local node index (0 ≤ i < n_nodes)
                - Value: corresponding global node index in the mesh

        Notes
        -----
        - This mapping is used for assembling global matrices and vectors from
          element-local contributions.
        - The local node ordering must be consistent with the reference element.

        Example
        -------
            >>> # 2D triangular mesh with 4 vertices and 2 elements
            >>> mesh.vertices
            array([[0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]])
            >>> mesh.elements
            array([[0, 1, 2],
                [1, 3, 2]])
            >>> phys_elem0 = femspace.get_physical_element(0)  # first triangle
            >>> mapping0 = phys_elem0.local_to_global_mapping(mesh.elements[0])
            >>> print(mapping0)
            {0: 0, 1: 1, 2: 2}
            >>> phys_elem1 = femspace.get_physical_element(1)  # second triangle
            >>> mapping1 = phys_elem1.local_to_global_mapping(mesh.elements[1])
            >>> print(mapping1)
            {0: 1, 1: 3, 2: 2}
            >>> # Local node 0 of the second triangle corresponds to global vertex 1, etc.
        """
        return {i : g for i, g in enumerate(element)}
    
    def shape_functions(self, element: np.ndarray) -> dict:
        """
        Return the Lagrange shape functions for this physical element, keyed by global node indices.

        This method constructs callable Lagrange shape functions associated with the
        nodes of a single physical element. Each function can be evaluated at any
        point in physical coordinates. The dictionary keys are the global node
        indices, making this convenient for assembling global matrices and vectors.

        Parameters
        ----------
        element : np.ndarray
            Array of global vertex indices defining the element.  
            The ordering of indices must match the reference element so that
            local Lagrange basis functions are correctly associated.

            - 1D interval: shape (nbasis,)
            - 2D triangle: shape (nbasis,)

        Returns
        -------
        phi_dict : dict[int, Callable]
            Dictionary mapping:

            - Key: global node index
            - Value: callable `phi(x_phys)` that returns the value of the
            corresponding Lagrange shape function at a given physical point
            `x_phys`.

        Notes
        -----
        - Local-to-global correspondence is determined by the order of `element`.
        - Each callable uses a closure (`i=i`) to correctly bind its local basis function.
        - Physical coordinates are internally mapped to reference coordinates
        via `self.physical_to_reference` before evaluating the reference basis.
        - No reordering or validation of element connectivity is performed.

        Example
        -------
            >>> # 2D triangular mesh with 4 vertices and 2 elements
            >>> mesh.vertices
            array([[0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]])
            >>> mesh.elements
            array([[0, 1, 2],
                [1, 3, 2]])
            >>> phys_elem0 = femspace.get_physical_element(0)
            >>> phi_dict0 = phys_elem0.shape_functions(mesh.elements[0])
            >>> x_phys = np.array([0.25, 0.25])
            >>> for g, phi_g in phi_dict0.items():
            ...     print(f"global {g}, phi(x_phys) = {phi_g(x_phys)}")
            global 0, phi(x_phys) = 0.5625
            global 1, phi(x_phys) = 0.1875
            global 2, phi(x_phys) = 0.25
        """
        phi_dict = {}
        for i, g in enumerate(element): # this works because the order of element indices matches self.ref_element.phi order
            phi_dict[g] = lambda x, i=i: self.ref_element.phi(self.physical_to_reference(x))[i] # must be changed
        return phi_dict

    def shape_function_gradients(self, element: np.ndarray) -> dict:
        """
        Return gradients of Lagrange shape functions in physical coordinates,
        keyed by global node indices.
        """
        phi_grad_dict = {}
        for i, g in enumerate(element):
            if self.ref_element.dim == 1: # must be changed
                phi_grad_dict[g] = lambda x, i=i: (self.ref_element.grad_phi(x)/self.jacobian())[i]
            elif self.ref_element.dim == 2:
                phi_grad_dict[g] = lambda x, i=i: (self.ref_element.grad_phi(x)@self.jacobian_inv())[i]
            else:
                raise NotImplementedError(f"grad_phi_physical not implemented for dim={self.ref_element.dim}")
        return phi_grad_dict