import numpy as np

# The finite element is defined by
# K - element domain (e.g., triangle, quadrilateral)
# S - space of shape functions (e.g., Lagrange, Hermite)
# N - the set of nodal variables.

class ReferenceElement:
    def __init__(self, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):
        self.domain = domain
        self.space = space
        self.degree = degree
        self.ref_nodes = self.reference_nodes()

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
    


    
