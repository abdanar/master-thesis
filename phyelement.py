import numpy as np
from refelement import ReferenceElement

# The finite element is defined by
# K - element domain (e.g., triangle, quadrilateral)
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

    # Define a reference map F: K_hat -> K, where K_hat is the reference element and K is the physical element
    def reference_to_physical(self, ref_point: np.ndarray): 
        # We have a unique mapping from reference to physical element, defined by the physical element's nodes
        # Explanation: (0, 0) -> vert[0] = [x0, y0], (1, 0) -> vert[1] = [x1, y1], (0, 1) -> vert[2] = [x2, y2],
        # where vert is given as vert = [[x0, y0], [x1, y1], [x2, y2]].
        # The mapping can be expressed as:
        # F(x_ref, y_ref) = A*[x_ref; y_ref] + b, where A is a 2x2 matrix and b is a 2x1 vector defined as:
        # A = [[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]]
        # b = [x0; y0]

        vert = self.vertices
        R = np.array([[vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
                      [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]]])
        b = vert[0]
        return R@ref_point + b

    # Define the inverse map F^{-1}: K -> K_hat
    def physical_to_reference(self, phys_point: np.ndarray):

        # vert = self.vertices
        # P = np.array([[vert[2, 1] - vert[0, 1], vert[0, 0] - vert[2, 0]],
        #               [vert[0, 1] - vert[1, 1], vert[1, 0] - vert[0, 0]]])
        # b = vert[0]
        # output -> (1/np.linalg.det(P))*P@(phys_point - b)

        return np.linalg.inv(self.jacobian())@(phys_point - self.vertices[0])
    
    # Compute the Jacobian matrix J of the transformation F at a given point in the reference element
    def jacobian(self):
        # The reference map F can be expressed as:
        # F(x, y) = (x_0 + (x_1 - x_0)*x + (x_2 - x_0)*y, y_0 + (y_1 - y_0)*x + (y_2 - y_0)*y) = (F1(x, y), F2(x, y))
        # where (x_0, y_0), (x_1, y_1), (x_2, y_2) are the coordinates of the vertices of the physical element.
        # The Jacobian matrix J is given by the partial derivatives of F with respect to x and y:
        # J = [[dF1/dx, dF1/dy], [dF2/dx, dF2/dy]] = A defined in reference_to_physical method

        vert = self.vertices
        J = np.array([[vert[1, 0] - vert[0, 0], vert[2, 0] - vert[0, 0]],
                      [vert[1, 1] - vert[0, 1], vert[2, 1] - vert[0, 1]]])
        return J
    
    def jacobian_inv(self):
        return np.linalg.inv(self.jacobian())
    
    def det_jacobian(self):
        return np.linalg.det(self.jacobian())
    
    # Define a function that computes the shape functions at a given point in the physical element 
    def phi_physical(self, phys_point: np.ndarray):
        return self.ref_element.phi(self.physical_to_reference(phys_point))
    
    # Define a function that computes the gradients of the shape functions at a given point in the physical element
    def grad_phi_physical(self, phys_point: np.ndarray):
        # Transform gradient from reference element to physical element using the Jacobian
        # grad_phi_phys(phys_point) = J^{-T}*grad_phi_ref(physical_to_reference(phys_point))
        # where J is the Jacobian matrix of the transformation from reference to physical element (`jacobian` function above)
        # The output is an array of shape (nbasis, 2) given by [gradphi_physical_0, gradphi_physical_1, ..., gradphi_physical_(nbasis-1)],
        # where gradphi_physical_i = [grad_{x}phi_physical_i, grad_{y}phi_physical_i] is the gradient of the i-th shape function at the given physical point

        grad_phi_ref = self.ref_element.grad_phi(self.physical_to_reference(phys_point))
        return self.jacobian_inv().T@grad_phi_ref.T