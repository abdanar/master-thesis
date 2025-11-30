import numpy as np

# The finite element is defined by
# K - element domain (e.g., triangle, quadrilateral)
# S - space of shape functions (e.g., Lagrange, Hermite)
# N - the set of nodal variables.

class FiniteElement:
    def __init__(self, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):
        self.domain = domain
        self.degree = degree
        self.space = space

    # Compute nodes of a triangular element with Lagrange shape functions of a given degree
    def reference_nodes(self): # counterclockwise order
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

    # Define a reference map F: K_hat -> K, where K_hat is the reference element and K is the physical element
    def reference_to_physical(self, vert: np.ndarray, ref_point: np.ndarray): 
        # We have a unique mapping from reference to physical element, defined by the physical element's nodes
        # Explanation: (0, 0) -> vert[0] = [x0, y0], (1, 0) -> vert[1] = [x1, y1], (0, 1) -> vert[2] = [x2, y2],
        # where vert is given as vert = [[x0, y0], [x1, y1], [x2, y2]].
        # The mapping can be expressed as:
        # F(x_ref, y_ref) = A*[x_ref; y_ref] + b, where A is a 2x2 matrix and b is a 2x1 vector defined as:
        # A = [[x1 - x0, x2 - x0], [y1 - y0, y2 - y0]]
        # b = [x0; y0]

        A = np.zeros((2, 2))
        A[0, 0] = vert[1, 0] - vert[0, 0] # x1 - x0
        A[1, 0] = vert[1, 1] - vert[0, 1] # y1 - y0
        A[0, 1] = vert[2, 0] - vert[0, 0] # x2 - x0
        A[1, 1] = vert[2, 1] - vert[0, 1] # y2 - y0

        b = np.zeros(2)
        b[0] = vert[0, 0] # x0
        b[1] = vert[0, 1] # y0

        return A@ref_point + b

    # Define the inverse map F^{-1}: K -> K_hat
    def physical_to_reference(self, vert: np.ndarray, phys_point: np.ndarray):

        A_inv = np.zeros((2, 2))
        A_inv[0, 0] = vert[2, 1] - vert[0, 1] # y2 - y0
        A_inv[1, 0] = vert[0, 1] - vert[1, 1] # y0 - y1
        A_inv[0, 1] = vert[0, 0] - vert[2, 0] # x0 - x2
        A_inv[1, 1] = vert[1, 0] - vert[0, 0] # x1 - x0

        b = np.zeros(2)
        b[0] = vert[0, 0] # x0
        b[1] = vert[0, 1] # y0

        return (1/np.linalg.det(A_inv))*A_inv@(phys_point - b)

    # Compute the Jacobian matrix J of the transformation F at a given point in the reference element
    def jacobian(self, vert: np.ndarray):

        # The reference map F can be expressed as:
        # F(x, y) = (x_0 + (x_1 - x_0)*x + (x_2 - x_0)*y, y_0 + (y_1 - y_0)*x + (y_2 - y_0)*y) = (F1(x, y), F2(x, y))
        # where (x_0, y_0), (x_1, y_1), (x_2, y_2) are the coordinates of the vertices of the physical element.
        # The Jacobian matrix J is given by the partial derivatives of F with respect to x and y:
        # J = [[dF1/dx, dF1/dy], [dF2/dx, dF2/dy]] = A defined in reference_to_physical method

        Jacobian = np.zeros((2, 2))
        Jacobian[0, 0] = vert[1, 0] - vert[0, 0] # x1 - x0
        Jacobian[1, 0] = vert[1, 1] - vert[0, 1] # y1 - y0
        Jacobian[0, 1] = vert[2, 0] - vert[0, 0] # x2 - x0
        Jacobian[1, 1] = vert[2, 1] - vert[0, 1] # y2 - y0

        return Jacobian

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
    
    # Define a function that computes the shape functions at a given point in the physical element 
    def phi_physical(self, vert: np.ndarray, phys_point: np.ndarray):
        ref_point = self.physical_to_reference(vert, phys_point)
        return self.phi(ref_point)
    
    # Define a function that computes the gradients of the shape functions at a given point in the physical element
    def grad_phi_physical(self, vert: np.ndarray, phys_point: np.ndarray):

        # Transform gradient from reference element to physical element using the Jacobian
        # grad_phi_phys(phys_point) = J^{-T}*grad_phi_ref(physical_to_reference(phys_point))
        # where J is the Jacobian matrix of the transformation from reference to physical element (`jacobian` function above)
        # The output is an array of shape (nbasis, 2) given by [gradphi_physical_0, gradphi_physical_1, ..., gradphi_physical_(nbasis-1)],
        # where gradphi_physical_i = [grad_{x}phi_physical_i, grad_{y}phi_physical_i] is the gradient of the i-th shape function at the given physical point

        ref_point = self.physical_to_reference(vert, phys_point)
        Jacobian = self.jacobian(vert)
        grad_phi_ref = self.grad_phi(ref_point)
        grad_phi_phys = np.linalg.inv(Jacobian).T @ grad_phi_ref.T
        return grad_phi_phys.T

    # Define a function to compute the integral on a reference triangle required for entries of a matrix, i.e., integrals appearing on a discrete weak formulation
    def integrate(self):
        return None

    # Construct local mass matrix for a given triangle.
    def local_mass(self, vert: np.ndarray):
        # Here we construct the local mass matrix A, i.e., 
        # A[i, j] = |det(J)|*int_{ref_triangle} phi_physical[i]*phi_physical[j] d(x_ref, y_ref)
        #         = 2|T_physical|*int_{ref_triangle} phi_physical[i]*phi_physical[j] d(x_ref, y_ref)

        nbasis = (self.degree + 1)*(self.degree + 2)//2
        mass_matrix = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            for j in range(nbasis):
                mass_matrix[i, j] = self.integrate()
        return mass_matrix
    
    # Construct local stiffness matrix for a given triangle.
    def local_stifness(self, ):
        # Here we construct the local stiffness matrix A, i.e., 
        # A[i, j] = |det(J)|*int_{ref_triangle} grad_phi_physical[i]*grad_phi_physical[j] d(x_ref, y_ref)
        #         = 2|T_physical|*int_{ref_triangle} grad_phi_physical[i]*grad_phi_physical[j] d(x_ref, y_ref)

        nbasis = (self.degree + 1)*(self.degree + 2)//2
        stiffness_matrix = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            for j in range(nbasis):
                stiffness_matrix[i, j] = self.integrate()
        return stiffness_matrix

    
