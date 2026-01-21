import numpy as np
from fem.femspace import FEMSpace
from fem.quadrature import interval_quadrature, triangle_quadrature

class ErrorNorms:
    def __init__(self, femspace: FEMSpace, u1: np.ndarray, u2 = None, u_exact = None, grad_u_exact = None, mode: str = 'auto'):
        """
        Compute various error norms for FEM solutions.

        Supports L², H¹-seminorm, H¹, and L∞ norms. Comparison can be made either
        with an exact solution or another FEM solution, making it suitable for
        Schwarz iterations or reduced-order methods.

        Parameters
        ----------
        femspace : FEMSpace
            FEM space object defining mesh, basis, and element mappings.
        u1 : np.ndarray
            Primary FEM solution to compare.
        u2 : np.ndarray, optional
            Secondary FEM solution for FEM-to-FEM error computation.
        u_exact : callable, optional
            Exact solution function `u_exact(x)` for FEM-to-exact comparison.
            Should accept either scalar (1D) or array-like (2D) coordinates.
        grad_u_exact : callable, optional
            Gradient of the exact solution, required for H¹-based norms.
            Should return a vector at given coordinates.
        mode : {'auto', 'exact', 'fem'}, default='auto'
            Comparison mode:
                - 'auto': uses u2 if provided, else u_exact
                - 'exact': compare with exact solution
                - 'fem': compare with another FEM solution (u2)
        """
        self.femspace = femspace
        self.mesh = femspace.mesh
        self.u1 = u1
        self.u2 = u2
        self.u_exact = u_exact
        self.grad_u_exact = grad_u_exact

        # Determine comparison mode
        if mode == 'auto':
            if u2 is not None:
                self.mode = 'fem'
            elif u_exact is not None:
                self.mode = 'exact'
            else:
                self.mode = 'self'
        else:
            self.mode = mode

        # Quadrature points and weights
        self.qorder = 2*self.femspace.degree
        if self.femspace.dim == 1 and self.femspace.domain == 'interval':
            self.ref_pts, self.weights = interval_quadrature(self.qorder) 
        elif self.femspace.dim == 2 and self.femspace.domain == 'triangle':
            self.ref_pts, self.weights = triangle_quadrature(self.qorder)
        else:
            raise NotImplementedError("Quadrature for the given element domain and dimension is not implemented.")
        
    def _diff(self, elem_index, x_phys):
        """Universal difference function u1 - u2 or u1 - u_exact, or u1 alone."""
        u1_x = self.femspace.evaluate_solution_on_element(self.u1, elem_index, x_phys)
        if self.mode == 'fem':
            u2_x = self.femspace.evaluate_solution_on_element(self.u2, elem_index, x_phys)
            return u1_x - u2_x
        elif self.mode == 'exact':
            if self.u_exact is None:
                raise ValueError("Exact solution must be provided for 'exact' mode.")
            return u1_x - self.u_exact(*x_phys) if self.femspace.dim == 2 else u1_x - self.u_exact(x_phys)
        else:
            return u1_x
        
    def _grad_diff(self, elem_index, x_phys):
        """Universal gradient difference: grad(u1) - grad(u2), grad(u1) - grad(u_exact), or grad(u1) alone."""
        grad_uh = self.femspace.evaluate_grad_solution_on_element(self.u1, elem_index, x_phys)
        if self.mode == 'fem':
            grad_u2 = self.femspace.evaluate_grad_solution_on_element(self.u2, elem_index, x_phys)
            return grad_uh - grad_u2
        elif self.mode == 'exact':
            if self.grad_u_exact is None:
                raise ValueError("Gradient of exact solution required for H1 norms.")
            grad_u = self.grad_u_exact(x_phys)
            return grad_uh - grad_u
        else:
            return grad_uh
  
    def l2_error(self):
        """
        Compute the L² norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        The behavior depends on which comparator is provided:

            - If `u2` is given: ||u1 - u2||_{L2} = sqrt(∫_Ω (u1(x) - u2(x))² dx)
            - If `u_exact` is given: ||u1 - u_exact||_{L2} = sqrt(∫_Ω (u1(x) - u_exact(x))² dx)
            - If neither is given: ||u1||_{L2} = sqrt(∫_Ω (u1(x))² dx)

        This method uses element-wise quadrature over the mesh, mapping reference 
        quadrature points to physical coordinates, and supports 1D intervals and 2D triangles.
        """
        err = 0.0

        # Loop over elements
        for elem_index in range(self.mesh.nelements()):
            
            # Get the physical element corresponding to this mesh element
            phys_elem = self.femspace.get_physical_element(elem_index)

            # Determinant of the Jacobian for mapping reference -> physical element
            detJ = phys_elem.det_jacobian()

            # Loop over quadrature points and weights
            for node, weight in zip(self.ref_pts, self.weights):

                # Map quadrature point from reference element to physical coordinates
                phys_point = phys_elem.reference_to_physical(node)

                # Accumulate element contribution to the squared L2 error
                err += weight*detJ*self._diff(elem_index, phys_point)**2

        # Return the L2 norm (square root of accumulated squared error)
        return float(np.sqrt(err))  

    def h1_semi_error(self):
        """
        Compute the H¹-seminorm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        The behavior depends on which comparator is provided:

            - If `u2` is given: |u1 - u2|_{H1} = sqrt(∫_Ω ||∇(u1(x) - u2(x))||² dx)
            - If `grad_u_exact` is given: |u1 - u_exact|_{H1} = sqrt(∫_Ω ||∇(u1(x) - u_exact(x))||² dx)
            - If neither is given: |u1|_{H1} = sqrt(∫_Ω ||∇u1(x)||² dx)

        This method uses element-wise quadrature over the mesh, mapping reference 
        quadrature points to physical coordinates, and supports 1D intervals and 2D triangles.

        Notes
        -----
        - For H¹-based norms, the gradient of the exact solution (`grad_u_exact`) is required 
        if `u_exact` is used as the comparator.
        """
        err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            detJ = phys_elem.det_jacobian()
            for node, weight in zip(self.ref_pts, self.weights):
                phys_point = phys_elem.reference_to_physical(node)
                diff = self._grad_diff(elem_index, phys_point)
                err += weight*detJ*np.dot(diff, diff)
        return float(np.sqrt(err))
    
    def h1_error(self):
        """
        Compute the H¹-norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        The behavior depends on which comparator is provided:

            - If `u2` is given: ||u1 - u2||_{H1} = sqrt(∫_Ω (u1 - u2)² dx + ∫_Ω ||∇(u1 - u2)||² dx)
            - If `u_exact` and `grad_u_exact` are given: ||u1 - u_exact||_{H1} = sqrt(∫_Ω (u1 - u_exact)² dx + ∫_Ω ||∇(u1 - u_exact)||² dx)
            - If neither is given: ||u1||_{H1} = sqrt(∫_Ω (u1)² dx + ∫_Ω ||∇u1||² dx)

        This method uses element-wise quadrature over the mesh, mapping reference 
        quadrature points to physical coordinates, and supports 1D intervals and 2D triangles.

        Notes
        -----
        - Both the exact solution (`u_exact`) and its gradient (`grad_u_exact`) are required
        if H¹-norm comparison with the exact solution is desired.
        """
        return float(np.sqrt(self.l2_error()**2 + self.h1_semi_error()**2))
    
    def linf_error(self):
        """
        Compute the L∞ (maximum) norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        The behavior depends on which comparator is provided:

            - If `u2` is given: ||u1 - u2||_{L∞} ≈ max_{x ∈ Ω} |u1(x) - u2(x)|
            - If `u_exact` is given: ||u1 - u_exact||_{L∞} ≈ max_{x ∈ Ω} |u1(x) - u_exact(x)|
            - If neither is given: ||u1||_{L∞} ≈ max_{x ∈ Ω} |u1(x)|

        The maximum is evaluated at quadrature points of each element. For a more accurate 
        estimate, one may evaluate on a finer set of points per element.

        Notes
        -----
        - Supports 1D intervals and 2D triangles.
        - Element-wise evaluation avoids global searches and improves efficiency.
        """
        max_err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            for node in self.ref_pts:
                phys_point = phys_elem.reference_to_physical(node)
                err = abs(self._diff(elem_index, phys_point))
                if err > max_err:
                    max_err = err
        return max_err

    def compute(self, norm: str = "l2"):
        """
        Compute a selected norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        This method provides a general interface to any supported norm.

        Parameters
        ----------
        norm : {'l2', 'h1', 'h1_semi', 'linf'}, default='l2'
            The norm type to compute:
                - 'l2'      : L² norm
                - 'h1_semi' : H¹-seminorm
                - 'h1'      : full H¹ norm
                - 'linf'    : L∞ (maximum) norm

        Returns
        -------
        float
            The computed norm value according to the selected type.

        Notes
        -----
        - The behavior depends on which comparator is provided (u2, u_exact, or neither).
        - Element-wise evaluation using quadrature is used for efficiency.
        - Supports 1D intervals and 2D triangles.
        """
        norm = norm.lower()
        if norm == "l2":
            return self.l2_error()
        elif norm in "linf":
            return self.linf_error()
        elif norm == "h1_semi":
            return self.h1_semi_error()
        elif norm == "h1":
            return self.h1_error()
        else:
            raise ValueError(f"Unknown norm '{norm}'. Choose from 'l2', 'linf', 'h1_semi', 'h1'.")