from phyelement import PhysicalElement

class LocalIntegrator():
    def __init__(self, element: PhysicalElement, diffusion = None, convection = None, reaction = None, func = None, quadrature_order = None):
        self.element = element
        self.f = func
        self.diff = diffusion
        self.conv = convection
        self.react = reaction
   
    
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