import numpy as np
from phyelement import PhysicalElement
from quadrature import triangle_quadrature

class LocalIntegrator():
    def __init__(self, element: PhysicalElement, diffusion, convection, reaction, func, quadrature_order = 2):
        self.element = element
        self.f = func
        self.diff = diffusion
        self.conv = convection
        self.react = reaction
        self.qorder = quadrature_order

    
    def local_stiffness_matrix(self):

        """
        Compute the local stiffness matrix for the finite element.

        The local stiffness matrix K is defined as:
            K[i, j] = ∫_T (∇φ_i(x))^T A(x) ∇φ_j(x) dx
        where:
            - φ_i, φ_j are the shape functions of the element,
            - ∇φ_i, ∇φ_j are their gradients in the physical element,
            - A(x) is the diffusion (or conductivity) coefficient evaluated at x,
            - T is the physical element domain.

        Numerical integration is performed using a quadrature rule on the reference element,
        and the gradients are mapped to the physical element using the Jacobian.

        Returns
        -------
        K : np.ndarray
            Local stiffness matrix of shape (nbasis, nbasis), where nbasis is the number of basis functions.
        """

        # get quadrature nodes and weights on the reference triangle
        ref_pts, weights = triangle_quadrature(self.qorder)

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # define local stiffness matrix
        nbasis = self.element.ref_element.nbasis
        K = np.zeros((nbasis, nbasis))
    
        # loop over all quadrature points
        for node, weight in zip(ref_pts, weights):

            # Evaluate gradient of shape functions at this quadrature point, shape (nbasis, 2)
            grad_phi_vals = self.element.grad_phi_physical(node)

            # Evaluate diffusion coefficient A(x) at this physical point, shape (2, 2)
            diff_vals = self.diff(self.element.reference_to_physical(node))

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                for j in range(nbasis):
                    K[i, j] += weight * detJ * grad_phi_vals[i] @ diff_vals @ grad_phi_vals[j]
        return K
    
    def local_convection_matrix(self):
        return None


    def local_mass_matrix(self):

        """
        Compute the local mass matrix for the finite element.

        The local mass matrix M is defined as:
            M[i, j] = ∫_T c(x) * φ_i(x) * φ_j(x) dx
        where:
            - φ_i, φ_j are the shape functions of the element,
            - c(x) is a scalar reaction or mass coefficient evaluated at x,
            - T is the physical element domain.

        Numerical integration is performed using a quadrature rule on the reference element,
        and the shape functions are mapped to the physical element.

        Returns
        -------
        M : np.ndarray
            Local mass matrix of shape (nbasis, nbasis), where nbasis is the number of basis functions.
        """

        # get quadrature nodes and weights on the reference triangle
        ref_pts, weights = triangle_quadrature(self.qorder)

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # initialize local stiffness matrix
        nbasis = self.element.ref_element.nbasis
        M = np.zeros((nbasis, nbasis))

        # loop over all quadrature points
        for node, weight in zip(ref_pts, weights):
            
            # Evaluate shape functions at this quadrature point, shape (nbasis,)
            phi_vals = self.element.phi_physical(node)

            # Evaluate reaction coefficient c(x) at this physical point
            react_val = self.react(self.element.reference_to_physical(node))

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                for j in range(nbasis):
                    M[i, j] += weight * detJ * react_val * phi_vals[i] * phi_vals[j]
        return M
         
        
    def load_vector(self):
        """
        Compute the local load vector F for the element.

        The local load vector is defined as:
            F[i] = ∫_T f(x) * phi_i(x) dx

        where:
            - T is the physical element,
            - phi_i(x) is the i-th shape function,
            - f(x) is the source function evaluated at the physical point x.

        Numerical integration is performed using quadrature points and weights
        defined on the reference element, and the result is scaled by the determinant
        of the Jacobian.

        Returns
        -------
        np.ndarray, shape (nbasis, 1)
            The local load vector for the element.
        """
        
        # get quadrature nodes and weights on the reference triangle
        ref_pts, weights = triangle_quadrature(self.qorder)

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # initialize local stiffness matrix
        nbasis = self.element.ref_element.nbasis
        F = np.zeros((nbasis, 1))

        # loop over all quadrature points
        for node, weight in zip(ref_pts, weights):
            
            # Evaluate shape functions at this quadrature point, shape (nbasis,)
            phi_vals = self.element.phi_physical(node)

            # Evaluate f(x) at this physical point
            func_val = self.f(self.element.reference_to_physical(node))

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                F[i] += weight * detJ * func_val * phi_vals[i]
        return F



   
    
