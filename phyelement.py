import numpy as np
from refelement import ReferenceElement

# The finite element is defined by
# T - element domain (e.g., triangle, quadrilateral)
# S - space of shape functions (e.g., Lagrange, Hermite)
# N - the set of nodal variables.

class PhysicalElement:
    def __init__(self, vertices: np.ndarray, ref_element: ReferenceElement):
        """
        vertices: physical coordinates of element vertices
        ref_element: reference element containing φ and ∇φ
        """
        self.vertices = vertices
        self.ref_element = ref_element

    # Define a reference map F: T_hat -> T, where T_hat is the reference element and T is the physical element
    def reference_to_physical(self, ref_point: np.ndarray):
         
        """
        Map a point or points from the reference element to the physical element.

        The reference element (e.g., the unit triangle with vertices at (0,0), (1,0), (0,1))
        is mapped to the physical element defined by its vertex coordinates via an affine transformation:

            F(ξ, η) = R @ [ξ; η] + b

        where
            - R = [[x1 - x0, x2 - x0],
                [y1 - y0, y2 - y0]] is the linear transformation matrix,
            - b = [x0, y0] is the translation vector,
            - (ξ, η) are coordinates in the reference element,
            - (x, y) are coordinates in the physical element.

        Parameters
        ----------
        ref_point : np.ndarray
            Coordinates in the reference element. Can be:
            - shape (2,) for a single point,
            - shape (n, 2) for n points (each row is a point).

        Returns
        -------
        np.ndarray
            Coordinates in the physical element. Returns:
            - shape (2,) for a single input point,
            - shape (n, 2) for multiple input points.

        Raises
        ------
        ValueError
            If `ref_point` does not have a valid shape.

        Notes
        -----
        - The method handles both single points and multiple points.
        - For multiple points, the affine transformation is applied to each row of `ref_point`.
        - Broadcasting ensures that the translation vector `b` is correctly added in all cases.
        """

        vert = self.vertices
        R = np.array([[vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
                      [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]]])
        b = vert[0]

        if ref_point.shape[0] == 2:
            return R@ref_point + b
        elif ref_point.shape[1] == 2:
            return ref_point @ R.T + b[:, np.newaxis].T
        else:
            raise ValueError(f"Invalid shape for ref_point: {ref_point.shape}")
        
    # Compute nodes of a physical triangular element with Lagrange shape functions of a given degree
    def physical_reference_nodes(self):
        
        """
        Map all reference element nodes to their corresponding coordinates
        in the physical triangular element using the affine transformation.

        Returns
        -------
        np.ndarray, shape (n_nodes, 2)
            Physical coordinates of all Lagrange nodes of this element.
            - The first 3 rows correspond to the vertices.
            - The remaining rows (if any) correspond to edge and interior nodes
            for higher-degree Lagrange elements.
        """
        return self.reference_to_physical(self.ref_element.reference_nodes())

    # Define the inverse map F^{-1}: K -> K_hat
    def physical_to_reference(self, phys_point: np.ndarray):
        """
        Map a point from the physical element back to the reference element.

        This function computes the inverse of the affine mapping
        defined by `reference_to_physical`. For a linear triangle,
        the mapping from the reference triangle T_hat to the physical
        triangle T is:
            F(ξ, η) = A @ [ξ; η] + b
        where
            A = [[x1 - x0, x2 - x0],
                [y1 - y0, y2 - y0]],
            b = [x0, y0],
        and vert = [[x0, y0], [x1, y1], [x2, y2]] are the physical vertices.

        The inverse mapping F^{-1}: T -> T_hat is then:
            [ξ; η] = A^{-1} @ (phys_point - b)

        Parameters
        ----------
        phys_point : np.ndarray, shape (2,)
            Coordinates of a point in the physical element.

        Returns
        -------
        np.ndarray, shape (2,)
            Coordinates of the corresponding point in the reference element.
        """
        
        return np.linalg.inv(self.jacobian())@(phys_point - self.vertices[0])
    
    # Compute the Jacobian matrix J of the transformation F at a given point in the reference element
    def jacobian(self):
        """
        Compute the Jacobian matrix J of the transformation from the reference
        element K_hat to the physical element K.

        For a triangular element with vertices
            vert[0] = (x0, y0), vert[1] = (x1, y1), vert[2] = (x2, y2),
        the reference map F: K_hat -> K is
            F(xi, eta) = vert[0] + (vert[1] - vert[0]) * xi + (vert[2] - vert[0]) * eta

        The Jacobian matrix J is
            J = [[dF1/dxi, dF1/deta],
                [dF2/dxi, dF2/deta]]

        Returns
        -------
        np.ndarray, shape (2, 2)
            The Jacobian matrix of the mapping from reference to physical element.
        """

        vert = self.vertices
        J = np.array([[vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
                      [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]]])
        return J
    
    def jacobian_inv(self):
        """
        Compute the inverse of the Jacobian matrix of the element transformation.

        Returns
        -------
        np.ndarray, shape (2, 2)
            The inverse of the Jacobian matrix J^{-1}.
        """
        return np.linalg.inv(self.jacobian())
    
    def det_jacobian(self):
        """
        Compute the determinant of the Jacobian matrix of the element transformation.

        The determinant is used for scaling integrals when transforming from the
        reference element to the physical element.

        Returns
        -------
        float
            The determinant det(J) of the Jacobian matrix.
        """
        return np.linalg.det(self.jacobian())
    
    # Define a function that computes the shape functions in the physical element at the given reference point
    def phi_physical(self, ref_point: np.ndarray):
        """
        Evaluate the shape functions of the element at a given reference point.

        The shape functions are defined on the reference element, and this method
        simply calls the reference element's shape functions. For a triangular 
        Lagrange element of degree p, the number of basis functions is nbasis = (p + 1)(p + 2)/2.

        Parameters
        ----------
        ref_point : np.ndarray, shape (2,)
            Coordinates of the point in the reference element K_hat.

        Returns
        -------
        np.ndarray, shape (nbasis,)
            Values of all shape functions at the given reference point.
            Access the i-th shape function by `phi_physical[i]`.
        """
        return self.ref_element.phi(ref_point)
    
    # Define a function that computes the gradients of the shape functions in the physical element at the given reference point
    def grad_phi_physical(self, ref_point: np.ndarray):
        """
        Evaluate the gradients of the shape functions at a given reference point,
        mapped to the physical element.

        The gradients in the physical element are obtained via the
        transformation
            grad_phi_phys = J^{-T} * grad_phi_ref
        where J is the Jacobian of the mapping from the reference element 
        to the physical element.

        Parameters
        ----------
        ref_point : np.ndarray, shape (2,)
            Coordinates of the point in the reference element K_hat.

        Returns
        -------
        np.ndarray, shape (nbasis, 2)
            Gradients of all basis functions at the given reference point.
            Each row grad_phi_phys[i] = [dphi_i/dx, dphi_i/dy].
        """
        grad_phi_ref = self.ref_element.grad_phi(ref_point)
        return grad_phi_ref @ self.jacobian_inv()