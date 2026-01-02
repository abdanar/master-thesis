import numpy as np
from phyelement import PhysicalElement
from quadrature import triangle_quadrature, interval_quadrature
from logger import setup_logger

logger = setup_logger(__name__, level = 'info')

class LocalIntegrator:
    def __init__(self, element: PhysicalElement, quadrature_order: int = 2):
        """
        Local finite element integrator for convection-diffusion-reaction problems in 1D or 2D.

        This class computes element-local matrices and load vectors for the PDE:

            - 1D: -d/dx(a(x) du/dx) + b(x) du/dx + c(x) u(x) = f(x),   x ∈ I
            - 2D: ∇·(A(x) ∇u(x)) + b(x)·∇u(x) + c(x) u(x) = f(x),   x ∈ T

        where:
            - a(x) or A(x) : diffusion (conductivity) coefficient, scalar in 1D, 2x2 matrix in 2D,
            - b(x) : convection/advection coefficient, scalar in 1D, length-2 vector in 2D,
            - c(x) : reaction or mass coefficient, scalar,
            - f(x) : source term, scalar,
            - I : physical interval element in 1D,
            - T : physical triangular element in 2D.

        The integrals are evaluated over a physical element using quadrature rules on the reference 
        element and mapping via the element's Jacobian.

        Methods
        -------
        local_stiffness_matrix(diffusion)
            Compute the element stiffness matrix:
                - 1D: K[i,j] = ∫_I D(x) dφ_i/dx dφ_j/dx dx
                - 2D: K[i,j] = ∫_T (∇φ_i)^T A ∇φ_j dx
        local_convection_matrix(convection)
            Compute the element convection matrix:
                - 1D: C[i,j] = ∫_I b(x) dφ_j/dx φ_i dx
                - 2D: C[i,j] = ∫_T (b · ∇φ_j) φ_i dx
        local_mass_matrix(reaction)
            Compute the element mass/reaction matrix:
                - 1D/2D: M[i,j] = ∫_I/∫_T c(x) φ_i φ_j dx
        local_load_vector(func)
            Compute the element load vector:
                - 1D/2D: F[i] = ∫_I/∫_T f(x) φ_i dx

        Parameters
        ----------
        element : PhysicalElement
            The physical element object providing geometry and basis function evaluations.
        quadrature_order : int, optional
            Requested quadrature order. If the provided order is too low for the polynomial
            degree of the finite element space, it is automatically increased to ensure
            sufficient integration accuracy.

        Notes
        -----
        - The effective quadrature order is chosen as max(quadrature_order, 2 * p),
          where p is the polynomial degree of the reference element.
        - This guarantees accurate integration of stiffness, convection, and mass terms
          for higher-order Lagrange elements with constant coefficients.
        - Each method only requires the coefficient(s) it needs (diffusion, convection,
          reaction, or source term) and returns a NumPy array of shape (nbasis, nbasis)
          for matrices or (nbasis,) for the load vector.
        - Gradients and shape functions are mapped to the physical element using the
          Jacobian determinant and transformation from the reference element.
        - Quadrature points and weights are precomputed in the constructor for efficiency.
        - Suitable for linear or higher-order Lagrange elements in 1D or triangular elements in 2D.
        """
        self.element = element
        self.qorder = quadrature_order if quadrature_order >= 2*element.ref_element.degree else 2*element.ref_element.degree
        if self.element.ref_element.dim == 1 and self.element.ref_element.domain == 'interval':
            self.ref_pts, self.weights = interval_quadrature(self.qorder) 
        elif self.element.ref_element.dim == 2 and self.element.ref_element.domain == 'triangle':
            self.ref_pts, self.weights = triangle_quadrature(self.qorder) # get quadrature nodes and weights on the reference triangle
        else:
            raise NotImplementedError("Quadrature for the given element domain and dimension is not implemented.")

    def local_stiffness_matrix(self, diffusion) -> np.ndarray:
        """
        Compute the element-local stiffness matrix for 1D or 2D Lagrange elements.

        The local stiffness matrix K is defined as:

            - 1D: K[i, j] = ∫_I a(x) * dφ_i/dx * dφ_j/dx dx
            - 2D: K[i, j] = ∫_T (∇φ_i(x, y))^T A(x, y) ∇φ_j(x, y) dx dy

        where:
            - φ_i, φ_j are the element's shape functions,
            - ∇φ_i, ∇φ_j are the gradients of the shape functions mapped
              to the physical element,
            - a(x) : scalar diffusion coefficient in 1D,
            - A(x, y) is the 2x2 diffusion (or conductivity) coefficient evaluated at (x, y),
            - T is the physical triangular element domain in 2D,
            - I is the physical interval element domain in 1D.

        Numerical integration is performed using precomputed quadrature points
        and weights on the reference element, scaled by the determinant of
        the Jacobian of the transformation to the physical element.

        Parameters
        ----------
        diffusion : callable
            - 1D: a(x) -> scalar diffusion coefficient at point x
            - 2D: A(x, y) -> 2x2 diffusion matrix at point (x, y)

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
        - The resulting matrix is symmetric if the diffusion coefficient is symmetric.
        - Supports 1D interval elements and 2D triangular elements.
        """
        logger.debug(f"Computing local stiffness matrix")

        # determinant of the Jacobian
        detJ = self.element.det_jacobian()

        # define local stiffness matrix
        nbasis = self.element.ref_element.nbasis
        K = np.zeros((nbasis, nbasis))
    
        # loop over all quadrature points
        for node, weight in zip(self.ref_pts, self.weights):

            # Evaluate gradient of shape functions at this quadrature point, shape (nbasis, self.element.ref_element.dim)
            grad_phi_vals = self.element.grad_phi_physical(node)

            if self.element.ref_element.dim == 1:

                # Map the reference quadrature point `node` to its physical coordinate `x` in the physical element.
                x = self.element.reference_to_physical(node)

                # Evaluate diffusion coefficient a(x) at this physical point, shape scalar
                diff_vals = diffusion(x)

                # Loop over all pairs of basis functions
                for i in range(nbasis):
                    for j in range(nbasis):
                        K[i, j] += weight * detJ * grad_phi_vals[i] * diff_vals * grad_phi_vals[j]
            
            elif self.element.ref_element.dim == 2:

                # Map the reference quadrature point `node` to its physical coordinates `(x, y)` in the physical element.
                x, y = self.element.reference_to_physical(node)

                # Evaluate diffusion coefficient A(x, y) at this physical point, shape scalar (2, 2)
                diff_vals = diffusion(x, y)

                # Loop over all pairs of basis functions
                for i in range(nbasis):
                    for j in range(nbasis):
                        K[i, j] += weight * detJ * grad_phi_vals[i] @ diff_vals @ grad_phi_vals[j]
            else:
                raise NotImplementedError(f"local_stiffness_matrix not implemented for dim={self.element.ref_element.dim}")
        return K 

    def local_convection_matrix(self, convection) -> np.ndarray:
        """
        Compute the element-local convection (advection) matrix for a 2D triangular element.

        The local convection matrix C is defined as:

            - 1D: C[i, j] = ∫_I b(x) * dφ_j/dx * φ_i(x) dx
            - 2D: C[i, j] = ∫_T (b(x, y) · ∇φ_j(x, y)) * φ_i(x, y) dx dy

        where:
            - φ_i, φ_j are the element's shape functions,
            - ∇φ_j is the gradient of φ_j mapped to the physical element,
            - b(x) or b(x, y) is the convection/advection evaluated at the physical point,
            - I is the 1D interval element, T is the 2D triangular element.

        This corresponds to the weak form of the PDE term b(x, y)·∇u(x, y) after
        multiplication by the test function φ_i and integration over the element.

        Parameters
        ----------
        convection : callable
            Convection/advection function:
            - 1D: convection(x) -> scalar
            - 2D: convection(x, y) -> length-2 NumPy array

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

            if self.element.ref_element.dim == 1:
                
                # Map the reference quadrature point `node` to its physical coordinate `x` in the physical element.
                x = self.element.reference_to_physical(node)

                # Evaluate convection coefficient b(x) at this physical point, scalar
                conv_vals = convection(x)

                # Loop over all pairs of basis functions
                for i in range(nbasis):
                    for j in range(nbasis):
                        C[i, j] += weight * detJ * conv_vals * grad_phi_vals[j] * phi_vals[i]
            elif self.element.ref_element.dim == 2:

                # Map the reference quadrature point `node` to its physical coordinates `(x, y)` in the physical element.
                x, y = self.element.reference_to_physical(node)

                # Evaluate convection coefficient b(x, y) at this physical point, shape (2,)
                conv_vals = convection(x, y)

                # Loop over all pairs of basis functions
                for i in range(nbasis):
                    for j in range(nbasis):
                        C[i, j] += weight * detJ * conv_vals @ grad_phi_vals[j] * phi_vals[i]
            else:
                raise NotImplementedError(f"local_convection_matrix not implemented for dim={self.element.ref_element.dim}")
        return C

    def local_mass_matrix(self, reaction) -> np.ndarray:
        """
        Compute the element-local mass (reaction) matrix for 1D or 2D Lagrange elements.

        The local mass matrix M is defined as:

            - 1D: M[i, j] = ∫_I c(x) * φ_i(x) * φ_j(x) dx
            - 2D: M[i, j] = ∫_T c(x, y) * φ_i(x, y) * φ_j(x, y) dx dy

        where:
            - φ_i, φ_j are the element's shape functions,
            - c is the scalar reaction or mass coefficient evaluated at x,
            - T is the physical triangular element domain in 2D,
            - I is the physical interval element domain in 1D.

        Numerical integration is performed using precomputed quadrature points
        and weights on the reference element, scaled by the determinant of
        the Jacobian of the transformation to the physical element.

        Parameters
        ----------
        reaction : callable
            - 1D: c(x) -> scalar reaction/mass coefficient at point x
            - 2D: c(x, y) -> scalar reaction/mass coefficient at point (x, y) (numpy array of shape (2,))

        Returns
        -------
        M : np.ndarray, shape (nbasis, nbasis)
            The element-local mass matrix, where nbasis is the number of basis
            functions associated with the element.

        Notes
        -----
        - Assumes the element provides shape function evaluations at quadrature points via
          `phi_physical`.
        - Quadrature points and weights are precomputed in the constructor.
        - The resulting matrix is symmetric if c is scalar-valued and positive.
        - Supports 1D interval elements and 2D triangular elements.
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

            # Map the reference quadrature point to its physical coordinates
            phys_point = self.element.reference_to_physical(node)

            # Evaluate the reaction coefficient at this physical point
            if self.element.ref_element.dim == 1:
                react_val = reaction(phys_point)           # 1D: phys_point is scalar
            elif self.element.ref_element.dim == 2:
                react_val = reaction(*phys_point)         # 2D: phys_point is array-like (x, y)
            else:
                raise NotImplementedError(f"local_mass_matrix not implemented for dim={self.element.ref_element.dim}")

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                for j in range(nbasis):
                    M[i, j] += weight * detJ * react_val * phi_vals[i] * phi_vals[j]
        return M
          
    def local_load_vector(self, func) -> np.ndarray:
        """
        Compute the element-local load (source) vector for 1D or 2D Lagrange elements.

        The local load vector F is defined as:

            - 1D: F[i] = ∫_I f(x) * φ_i(x) dx
            - 2D: F[i] = ∫_T f(x, y) * φ_i(x, y) dx dy

        where:
            - φ_i is the i-th shape function of the element,
            - f is the source term evaluated at the physical point,
            - I is the 1D interval element, T is the 2D triangular element.

        Numerical integration is performed using quadrature points and weights on
        the reference element, scaled by the determinant of the Jacobian for mapping
        to the physical element.

        Parameters
        ----------
        func : callable
            Source term function:
            - 1D: func(x) -> scalar
            - 2D: func(x, y) -> scalar

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
            
            # Map reference quadrature point to physical coordinates
            phys_point = self.element.reference_to_physical(node)

            # Evaluate the source term at this physical point
            if self.element.ref_element.dim == 1:
                func_val = func(phys_point)
            elif self.element.ref_element.dim == 2:
                func_val = func(*phys_point)
            else:
                raise NotImplementedError(f"local_load_vector not implemented for dim={self.element.ref_element.dim}")

            # Loop over all pairs of basis functions
            for i in range(nbasis):
                F[i] += weight * detJ * func_val * phi_vals[i]
        return F
    