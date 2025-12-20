import numpy as np
from phyelement import PhysicalElement
from quadrature import triangle_quadrature
from logger import setup_logger

logger = setup_logger(__name__, level = 'info')

class LocalIntegrator:
    def __init__(self, element: PhysicalElement, quadrature_order = 2):

        """
        Local finite element integrator for convection-diffusion-reaction problems in 2D.

        This class computes element-local matrices and load vectors for the PDE:

            -∇·(A(x) ∇u(x)) + b(x)·∇u(x) + c(x) u(x) = f(x),   x ∈ T

        where:
            - A(x) : diffusion (conductivity) coefficient, 2x2 matrix,
            - b(x) : convection/advection coefficient, length-2 vector,
            - c(x) : reaction or mass coefficient, scalar,
            - f(x) : source term, scalar.

        The integrals are evaluated over a physical triangular element using
        quadrature rules on the reference element and mapping via the element's Jacobian.

        Methods
        -------
        local_stiffness_matrix(diffusion)
            Compute the element stiffness matrix K[i,j] = ∫_T (∇φ_i)^T A ∇φ_j dx.
        local_convection_matrix(convection)
            Compute the element convection matrix C[i,j] = ∫_T (b · ∇φ_j) φ_i dx.
        local_mass_matrix(reaction)
            Compute the element mass/reaction matrix M[i,j] = ∫_T c φ_i φ_j dx.
        local_load_vector(func)
            Compute the element load vector F[i] = ∫_T f φ_i dx.

        Parameters
        ----------
        element : PhysicalElement
            The physical element object providing geometry and basis function evaluations.
        quadrature_order : int, optional
            The order of quadrature used for numerical integration (default is 2).

        Notes
        -----
        - Each method only requires the coefficient(s) it needs (diffusion, convection,
        reaction, or source term) and returns a NumPy array of shape (nbasis, nbasis)
        for matrices or (nbasis,) for the load vector.
        - Gradients and shape functions are mapped to the physical element using the
        Jacobian determinant and transformation from the reference element.
        - Quadrature points and weights are precomputed in the constructor for efficiency.
        - Suitable for linear or higher-order Lagrange triangular elements.
        """

        self.element = element
        self.qorder = quadrature_order
        self.ref_pts, self.weights = triangle_quadrature(self.qorder) # get quadrature nodes and weights on the reference triangle

    
    def local_stiffness_matrix(self, diffusion) -> np.ndarray:

        """
        Compute the element-local stiffness matrix for a 2D triangular finite element.

        The local stiffness matrix K is defined as:

            K[i, j] = ∫_T (∇φ_i(x))^T A(x) ∇φ_j(x) dx

        where:
            - φ_i, φ_j are the element's shape functions,
            - ∇φ_i, ∇φ_j are the gradients of the shape functions mapped
            to the physical element,
            - A(x) is the 2x2 diffusion (or conductivity) coefficient evaluated at x,
            - T is the physical element domain.

        Numerical integration is performed using a quadrature rule on the reference
        triangle and scaled by the determinant of the Jacobian for mapping to
        the physical element.

        Parameters
        ----------
        diffusion : callable
            A function A(x) that takes a 2D point x (numpy array of shape (2,))
            and returns a 2x2 NumPy array representing the diffusion coefficient.

        Returns
        -------
        K : np.ndarray, shape (nbasis, nbasis)
            The local stiffness matrix for the element, where nbasis is the
            number of basis functions associated with the element.

        Notes
        -----
        - This method assumes the element provides gradient evaluations at
        quadrature points via `grad_phi_physical`.
        - Quadrature points and weights are precomputed in the constructor.
        - The resulting matrix is symmetric if A(x) is symmetric.
        """

        logger.debug(f"Computing local stiffness matrix")

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # define local stiffness matrix
        nbasis = self.element.ref_element.nbasis
        K = np.zeros((nbasis, nbasis))
    
        # loop over all quadrature points
        for node, weight in zip(self.ref_pts, self.weights):

            # Evaluate gradient of shape functions at this quadrature point, shape (nbasis, 2)
            grad_phi_vals = self.element.grad_phi_physical(node)

            # Map the reference quadrature point `node` to its physical coordinates `(x, y)` in the physical element.
            x, y = self.element.reference_to_physical(node)

            # Evaluate diffusion coefficient A(x, y) at this physical point, shape (2, 2)
            diff_vals = diffusion(x, y)

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                for j in range(nbasis):
                    K[i, j] += weight * detJ * grad_phi_vals[i] @ diff_vals @ grad_phi_vals[j]
        return K
    

    def local_convection_matrix(self, convection) -> np.ndarray:

        """
        Compute the element-local convection (advection) matrix for a 2D triangular element.

        The local convection matrix C is defined as:

            C[i, j] = ∫_T (b(x) · ∇φ_j(x)) * φ_i(x) dx

        where:
            - φ_i, φ_j are the element's shape functions,
            - ∇φ_j is the gradient of φ_j mapped to the physical element,
            - b(x) is the 2D convection/advection vector evaluated at x,
            - T is the physical element domain.

        This corresponds to the weak form of the PDE term b(x)·∇u(x) after
        multiplication by the test function φ_i and integration over the element.

        Parameters
        ----------
        convection : callable
            A function b(x) that takes a 2D point x (numpy array of shape (2,))
            and returns a length-2 NumPy array representing the convection vector.

        Returns
        -------
        C : np.ndarray, shape (nbasis, nbasis)
            The element-local convection matrix, where nbasis is the number of
            basis functions associated with the element.

        Notes
        -----
        - Quadrature points and weights are precomputed in the constructor.
        - The resulting matrix is generally **not symmetric**, even for constant b(x).
        - conv_vals @ grad_phi_vals[j] computes the dot product between the convection
        vector and the gradient of the j-th basis function.
        """

        logger.debug(f"Computing local convection matrix")

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # define local convection matrix
        nbasis = self.element.ref_element.nbasis
        C = np.zeros((nbasis, nbasis))
    
        # loop over all quadrature points
        for node, weight in zip(self.ref_pts, self.weights):

            # Evaluate shape functions at this quadrature point, shape (nbasis,)
            phi_vals = self.element.phi_physical(node)

            # Evaluate gradient of shape functions at this quadrature point, shape (nbasis, 2)
            grad_phi_vals = self.element.grad_phi_physical(node)

            # Map the reference quadrature point `node` to its physical coordinates `(x, y)` in the physical element.
            x, y = self.element.reference_to_physical(node)

            # Evaluate convection coefficient b(x, y) at this physical point, shape (2, )
            conv_vals = convection(x, y)

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                for j in range(nbasis):
                    C[i, j] += weight * detJ * conv_vals @ grad_phi_vals[j] * phi_vals[i]
        return C


    def local_mass_matrix(self, reaction) -> np.ndarray:

        """
        Compute the element-local mass (reaction) matrix for a 2D triangular element.

        The local mass matrix M is defined as:

            M[i, j] = ∫_T c(x) * φ_i(x) * φ_j(x) dx

        where:
            - φ_i, φ_j are the element's shape functions,
            - c(x) is the scalar reaction or mass coefficient evaluated at x,
            - T is the physical element domain.

        Numerical integration is performed using a quadrature rule on the reference
        triangle and scaled by the determinant of the Jacobian for mapping to the
        physical element.

        Parameters
        ----------
        reaction : callable
            A function c(x) that takes a 2D point x (numpy array of shape (2,))
            and returns a scalar representing the reaction/mass coefficient.

        Returns
        -------
        M : np.ndarray, shape (nbasis, nbasis)
            The element-local mass matrix, where nbasis is the number of basis
            functions associated with the element.

        Notes
        -----
        - Quadrature points and weights are precomputed in the constructor.
        - The resulting matrix is symmetric if c(x) is scalar-valued and positive.
        """

        logger.debug(f"Computing local mass matrix")

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # initialize local mass matrix
        nbasis = self.element.ref_element.nbasis
        M = np.zeros((nbasis, nbasis))

        # loop over all quadrature points
        for node, weight in zip(self.ref_pts, self.weights):
            
            # Evaluate shape functions at this quadrature point, shape (nbasis,)
            phi_vals = self.element.phi_physical(node)

            # Map the reference quadrature point `node` to its physical coordinates `(x, y)` in the physical element.
            x, y = self.element.reference_to_physical(node)

            # Evaluate reaction coefficient c(x, y) at this physical point
            react_val = reaction(x, y)

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                for j in range(nbasis):
                    M[i, j] += weight * detJ * react_val * phi_vals[i] * phi_vals[j]
        return M
         
        
    def local_load_vector(self, func) -> np.ndarray:

        """
        Compute the element-local load (source) vector for a 2D triangular element.

        The local load vector F is defined as:

            F[i] = ∫_T f(x) * φ_i(x) dx

        where:
            - φ_i is the i-th shape function of the element,
            - f(x) is the source term evaluated at the physical point x,
            - T is the physical element domain.

        Numerical integration is performed using quadrature points and weights on
        the reference triangle, scaled by the determinant of the Jacobian for mapping
        to the physical element.

        Parameters
        ----------
        func : callable
            A function f(x) that takes a 2D point x (numpy array of shape (2,))
            and returns a scalar representing the source term.

        Returns
        -------
        F : np.ndarray, shape (nbasis, 1)
            The element-local load vector, where nbasis is the number of basis
            functions associated with the element.

        Notes
        -----
        - Quadrature points and weights are precomputed in the constructor.
        - The resulting vector can be added to the global load vector in assembly.
        """

        logger.debug(f"Computing local load vector")

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # initialize local load vector
        nbasis = self.element.ref_element.nbasis
        F = np.zeros((nbasis, 1))

        # loop over all quadrature points
        for node, weight in zip(self.ref_pts, self.weights):
            
            # Evaluate shape functions at this quadrature point, shape (nbasis,)
            phi_vals = self.element.phi_physical(node)

            # Map the reference quadrature point `node` to its physical coordinates `(x, y)` in the physical element.
            x, y = self.element.reference_to_physical(node)

            # Evaluate f(x, y) at this physical point
            func_val = func(x, y)

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                F[i] += weight * detJ * func_val * phi_vals[i]
        return F
    