import numpy as np
from refelement import ReferenceElement

# --------------------------------------------------------------------------
# Physical Element for Finite Element Method (FEM)
#
# This class defines a single **physical element** in 1D or 2D.
# A physical element is obtained by mapping a reference element
# from the standard domain to its actual coordinates in the mesh.
#
# The finite element is defined by:
#   K - physical element domain (interval in 1D, triangle in 2D)
#   S - space of shape functions (Lagrange FEM)
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

class PhysicalElement:
    def __init__(self, vertices: np.ndarray, ref_element: ReferenceElement):
        """
        Physical element for 1D and 2D Lagrange FEM.

        Attributes
        ----------
        vertices : np.ndarray
            Physical coordinates of element vertices.
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