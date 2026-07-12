import numpy as np
from numba import njit
from fem.femspace import FEMSpace
from fem.quadrature import interval_quadrature, triangle_quadrature
from typing import Callable, Optional, Literal
from enum import Enum
import inspect
from utils.logger import get_logger
logger = get_logger(__name__)

# Define norm types for error computation
class NormType(str, Enum):
    L2 = "l2"
    H1 = "h1"
    H1_SEMI = "h1_semi"
    LINF = "linf"

class ErrorNorms:
    def __init__(self, femspace: FEMSpace, u1: np.ndarray, u2: Optional[np.ndarray] = None, u_exact: Optional[Callable] = None, 
                 grad_u_exact: Optional[Callable] = None, time: Optional[np.ndarray] = None, mode: Literal['auto', 'exact', 'fem', 'self'] = 'auto'):
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
        u_exact : Callable, optional
            Exact solution function `u_exact(x)` for FEM-to-exact comparison.
            Should accept either scalar (1D) or array-like (2D) coordinates.
        grad_u_exact : Callable, optional
            Gradient of the exact solution, required for H¹-based norms.
            Should return a vector at given coordinates.
        time : np.ndarray, optional
            Time points corresponding to the solution snapshots, required for time-dependent norms, i.e., t_0, t_1, ..., t_n = T.
        mode : {'auto', 'exact', 'fem', 'self'}, default='auto'
            Comparison mode:
                - 'auto': uses u2 if provided, else u_exact
                - 'exact': compare with exact solution
                - 'fem': compare with another FEM solution (u2)
                - 'self': compute norm of u1 alone (no comparison)

        Function Definition Guidelines
        ---------------------------
        - Exact solution (or gradient) functions should be defined as either:
            1. u_exact(x): for stationary problems, where x is the spatial coordinate(s).
            2. u_exact(t, x): for time-dependent problems, where t is the time variable and x is the spatial coordinate(s).
        - The wrapper will automatically detect the function signature and adapt the interface accordingly, 
        allowing for seamless integration with both stationary and time-dependent problems. 
        """
        assert mode in ['auto', 'exact', 'fem', 'self'], "Mode must be one of 'auto', 'exact', 'fem', 'self'."
        self.femspace = femspace
        self.mesh = femspace.mesh
        self.u1 = self._ensure_2d(u1)
        self.u2 = self._ensure_2d(u2) if u2 is not None else None
        self.u_exact = self._as_time_function(self.adapt_function(u_exact)) if u_exact is not None else None
        self.grad_u_exact = self._as_time_function(self.adapt_function(grad_u_exact)) if grad_u_exact is not None else None
        self.time = time
        self.nt = 1 if time is None else len(time)

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

    # later `adapt_function` will be removed, and whole codebase will be refactored to use the unified time-dependent
    # function interface, but for now this allows us to support both the old style and new style function definitions 
    # while we transition the codebase 
    @ staticmethod
    def adapt_function(f):
        """
        Adapt function signatures to a unified form:
        - f(x, y)     ->   g(x_vec) where x_vec = [x, y]
        - f(x, y, t)  ->   g(t, x_vec) where x_vec = [x, y]
        - f(x, t)     ->   g(t, x)
        """
        sig = inspect.signature(f)
        n_params = len(sig.parameters)
        if n_params == 2:  # could be (x, y) OR (x, t)
            param_names = list(sig.parameters.keys())
            if param_names[1] == 't':  # f(x, t) -> g(t, x)
                return lambda t, x: f(x, t)
            else:  # f(x, y) -> g(x_vec)
                return lambda x_vec: f(x_vec[0], x_vec[1])
        elif n_params == 3: # f(x, y, t) -> g(t, x_vec)
            return lambda t, x_vec: f(x_vec[0], x_vec[1], t)
        else:
            return f  # assume it's already in the correct form
        
    def _ensure_2d(self, u):
        """
        Ensure that the solution array is 2D (ndof, nt). If it's 1D, reshape it to (ndof, 1).
        This allows for consistent handling of time-dependent solutions, even if only one 
        time step is provided.
        """
        u = np.asarray(u)
        if u.ndim == 1:
            return u[:, None]  # (ndof,) → (ndof, 1)
        if u.ndim == 2:
            return u
        raise ValueError("Solution must be 1D or 2D.")

    def _as_time_function(self, func: Callable) -> Callable:
        """
        Wrap a user-defined function into a unified time-dependent interface u(t, x).

        This utility allows the solver to accept both stationary and time-dependent
        functions while enforcing a consistent internal format.

        Supported function signatures
        ----------------------------
        1. Stationary case: u(x)
        2. Time-dependent case: u(t, x)

        Behavior
        --------
        - If the function has one argument (x), it is treated as time-independent
        and converted to:
                u(t, x) = u(x)
        - If the function has two arguments (t, x), it is used directly.
        - Any other signature raises a ValueError.

        Parameters
        ----------
        func : Callable
            User-defined function representing either u(x) or u(t, x).

        Returns
        -------
        Callable
            A unified function u(t, x) compatible with FEM time-dependent evaluation.

        Notes
        -----
        - The spatial variable x is always a single object:
            1D: scalar or array-like
            2D/3D: vector-like array (x[0], x[1], ...)
        - Time variable t is provided for interface consistency even in stationary cases.
        """
        sig = inspect.signature(func)
        n_params = len(sig.parameters)
        if n_params == 1:
            return lambda t, x: func(x)
        elif n_params == 2:
            return func
        else:
            raise ValueError("Function must be u(x) or u(t, x)")
        
    def _diff(self, elem_index: int, x_phys, t_index: int = 0):
        """
        Universal difference function: 

        - If mode is 'fem': returns u1(`x_phys`) - u2(`x_phys`) at time index `t_index`
        - If mode is 'exact': returns u1(`x_phys`) - u_exact(`x_phys`, `time[t_index]`)
        - If mode is 'self': returns u1(`x_phys`) at time index `t_index` (for norms of a single solution)

        Parameters
        ----------
        elem_index : int
            Index of the element for evaluating the solution.
        x_phys : array-like
            Physical coordinates where the solution is evaluated.
        t_index : int, optional
            Time index for time-dependent solutions, default is 0 (first time step).
        """
        u1_x = self.femspace.evaluate_solution_on_element(self.u1[:, t_index], elem_index, x_phys)
        if self.mode == 'fem':
            u2_x = self.femspace.evaluate_solution_on_element(self.u2[:, t_index], elem_index, x_phys) # type: ignore
            return u1_x - u2_x
        elif self.mode == 'exact':
            t = self.time[t_index] if self.time is not None else 0.0
            if self.u_exact is None:
                raise ValueError("Exact solution must be provided for 'exact' mode.")
            return u1_x - self.u_exact(t, x_phys)
        else:  # mode == 'self'
            return u1_x
        
    def _grad_diff(self, elem_index: int, x_phys, t_index: int = 0):
        """
        Universal gradient difference function for H¹-based norms:
        
        - If mode is 'fem': returns ∇u1(`x_phys`) - ∇u2(`x_phys`) at time index `t_index`
        - If mode is 'exact': returns ∇u1(`x_phys`) - ∇u_exact(`x_phys`, `time[t_index]`)
        - If mode is 'self': returns ∇u1(`x_phys`) at time index `t_index` (for H¹ norms of a single solution)

        Parameters
        ----------
        elem_index : int
            Index of the element for evaluating the solution gradient.
        x_phys : array-like
            Physical coordinates where the solution gradient is evaluated.
        t_index : int, optional
            Time index for time-dependent solutions, default is 0 (first time step).
        """
        grad_uh = self.femspace.evaluate_grad_solution_on_element(self.u1[:, t_index], elem_index, x_phys)
        if self.mode == 'fem':
            grad_u2 = self.femspace.evaluate_grad_solution_on_element(self.u2[:, t_index], elem_index, x_phys) # type: ignore
            return grad_uh - grad_u2
        elif self.mode == 'exact':
            if self.grad_u_exact is None:
                raise ValueError("Gradient of exact solution required for H1 norms.")
            t = self.time[t_index] if self.time is not None else 0.0
            grad_u = self.grad_u_exact(t, x_phys)
            return grad_uh - grad_u
        else:  # mode == 'self'
            return grad_uh
    @njit
    def l2_error(self) -> float:
        """
        Compute the L² norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        The behavior depends on which comparator is provided:

            - If `u2` is given: ||u1 - u2||_{L2} = sqrt(∫_Ω (u1(x) - u2(x))² dx)
            - If `u_exact` is given: ||u1 - u_exact||_{L2} = sqrt(∫_Ω (u1(x) - u_exact(x))² dx)
            - If mode is 'self': ||u1||_{L2} = sqrt(∫_Ω (u1(x))² dx)

        This method uses element-wise quadrature over the mesh, mapping reference 
        quadrature points to physical coordinates, and supports 1D intervals and 2D triangles.
        """
        logger.debug("Computing L2 error using element-wise quadrature...")
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
    
    def l2_error_time(self):
        """
        Compute the L² error integrated over time for time-dependent solutions. The L² error is computed 
        at each time step and then integrated in time using the trapezoidal rule. The formula is:

            sqrt(∫_0^T ||u1(t) - u2(t)||_{L2}² dt)
        or
            sqrt(∫_0^T ||u1(t) - u_exact(t)||_{L2}² dt)

        depending on the mode. If `self.time` is not provided or has only one time point, it falls back 
        to the static L² error computation.

        Notes
        -----
        - This method assumes that the time snapshots in `u1`, `u2`, and the exact solution (if used) are ordered 
        according to the time points in `time`. 
        - The time integration is performed using the trapezoidal rule for better accuracy.
        """
        if self.time is None or self.nt == 1:
            logger.warning("Time array is not provided or has only one time point. Falling back to static L2 error computation.")
            return self.l2_error()  # fallback to static norm 
        logger.debug("Computing L2 error integrated over time using trapezoidal rule...")       
        # Vectorized time integration using trapezoidal rule
        dt = np.diff(self.time)
        l2_error_sq = np.empty(self.nt)
        for i in range(self.nt):
            l2_error_sq[i] = self.l2_error_at_time(i) ** 2
        err_time = 0.5 * np.sum(dt*(l2_error_sq[:-1] + l2_error_sq[1:]))
        return float(np.sqrt(err_time))

    def l2_error_at_time(self, t_index: int):
        """
        This method computes the L2 error at a specific time index, which is used in the time integration 
        of the L2 error. It loops over all elements and quadrature points to compute the spatial L2 error 
        at that time step. The formula is: 

            sqrt(∫_Ω (u1(x, time[t_index]) - u2(x, time[t_index]))² dx) 
        or 
            sqrt(∫_Ω (u1(x, time[t_index]) - u_exact(x, time[t_index]))² dx) 

        depending on the mode.
        """
        logger.debug(f"Computing L2 error at time index {t_index} (time = {self.time[t_index] if self.time is not None else 'N/A'})...")
        err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            detJ = phys_elem.det_jacobian()
            for node, weight in zip(self.ref_pts, self.weights):
                phys_point = phys_elem.reference_to_physical(node)
                err += weight*detJ*self._diff(elem_index, phys_point, t_index)**2
        return float(np.sqrt(err))

    def linf_l2_error(self) -> float:
        """
        Compute the L∞ error of the L² norm across time for time-dependent solutions. 
        The L² error is computed at each time step, and the maximum value across all 
        time steps is returned. The formula is:

            max_{t ∈ (t_0, ..., t_{n})} ||u1(t) - u2(t)||_{L2}
        or
            max_{t ∈ (t_0, ..., t_{n})} ||u1(t) - u_exact(t)||_{L2}
        depending on the mode. If `self.time` is not provided or has only one time point, it falls back
        to the static L² error computation.
        """
        if self.time is None or self.nt == 1:
            logger.warning("Time array is not provided or has only one time point. Falling back to static L2 error computation.")
            return self.l2_error()  # fallback to static norm
        logger.debug("Computing L∞ error of the L2 norm across time by taking maximum across time steps...")
        max_err = 0.0
        for t_index in range(self.nt):
            max_err = max(max_err, self.l2_error_at_time(t_index))
        return max_err
    
    def linf_error(self) -> float:
        """
        Compute the L∞ (maximum) error between two FEM solutions or between a FEM solution and 
        the exact solution.

        The behavior depends on which comparator is provided:

            - If `u2` is given: max_{x ∈ Ω} |u1(x) - u2(x)|
            - If `u_exact` is given: max_{x ∈ Ω} |u1(x) - u_exact(x)|
            - If mode is 'self': max_{x ∈ Ω} |u1(x)|
        
        This method loops over all elements and quadrature points to compute the maximum 
        pointwise error. 
        """
        logger.debug("Computing L∞ error by looping over elements and quadrature points...")
        max_err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            for node in self.ref_pts:
                phys_point = phys_elem.reference_to_physical(node)
                err = abs(self._diff(elem_index, phys_point))
                if err > max_err:
                    max_err = err
        return max_err

    def linf_error_time(self) -> float:
        """
        Compute the L∞ error integrated over time for time-dependent solutions. The L∞ error is computed 
        at each time step and the maximum error across all time steps is returned. The formula is:

            max_{t ∈ (t_0, ..., t_{n})} ||u1(t) - u2(t)||_{L∞}
        or
            max_{t ∈ (t_0, ..., t_{n})} ||u1(t) - u_exact(t)||_{L∞}

        depending on the mode. If `time` is not provided or has only one time point, 
        it falls back to the static L∞ error computation.
        """
        if self.time is None or self.nt == 1:
            logger.warning("Time array is not provided or has only one time point. Falling back to static L∞ error computation.")
            return self.linf_error()  # fallback to static norm
        logger.debug("Computing L∞ error integrated over time by taking maximum across time steps...")
        max_err = 0.0
        for t_index in range(self.nt):
            max_err = max(max_err, self.linf_error_at_time(t_index))
        return max_err

    def linf_error_at_time(self, t_index: int) -> float:
        """
        Compute the L∞ error at a specific time index. This method loops over all elements and 
        quadrature points to compute the maximum pointwise error at that time step. The formula is:

            max_{x ∈ Ω} |u1(x, time[t_index]) - u2(x, time[t_index])|
        or
            max_{x ∈ Ω} |u1(x, time[t_index]) - u_exact(x, time[t_index])|

        depending on the mode.
        """
        logger.debug(f"Computing L∞ error at time index {t_index} (time = {self.time[t_index] if self.time is not None else 'N/A'})...")
        max_err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            for node in self.ref_pts:
                phys_point = phys_elem.reference_to_physical(node)
                err = abs(self._diff(elem_index, phys_point, t_index))
                if err > max_err:
                    max_err = err
        return max_err

    def h1_semi_error(self) -> float:
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
        logger.debug("Computing H¹-seminorm error using element-wise quadrature...")
        err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            detJ = phys_elem.det_jacobian()
            for node, weight in zip(self.ref_pts, self.weights):
                phys_point = phys_elem.reference_to_physical(node)
                diff = self._grad_diff(elem_index, phys_point)
                err += weight*detJ*np.dot(diff, diff)
        return float(np.sqrt(err))

    def h1_semi_error_time(self) -> float:
        """
        Compute the H¹-seminorm error integrated over time for time-dependent solutions. The H¹-seminorm error is computed 
        at each time step and then integrated in time using the trapezoidal rule. The formula is:

            sqrt(∫_0^T |u1(t) - u2(t)|_{H1}² dt)
        or
            sqrt(∫_0^T |u1(t) - u_exact(t)|_{H1}² dt)
        depending on the mode. If `self.time` is not provided or has only one time point, it falls back
        to the static H¹-seminorm error computation.

        Notes
        -----
        - This method assumes that the time snapshots in `u1`, `u2`, and the exact solution (if used) are ordered according to the time points in `time`.
        - The time integration is performed using the trapezoidal rule for better accuracy.
        """
        if self.time is None or self.nt == 1:
            logger.warning("Time array is not provided or has only one time point. Falling back to static H¹-seminorm error computation.")
            return self.h1_semi_error()  # fallback to static norm       
        logger.debug("Computing H¹-seminorm error integrated over time using trapezoidal rule...") 
        dt = np.diff(self.time)
        h1s_error_sq = np.empty(self.nt)
        for i in range(self.nt):
            h1s_error_sq[i] = self.h1_semi_error_at_time(i) ** 2
        err_time = 0.5 * np.sum(dt*(h1s_error_sq[:-1] + h1s_error_sq[1:]))
        return float(np.sqrt(err_time))
    
    def h1_semi_error_at_time(self, t_index: int) -> float:
        """
        Compute the H¹-seminorm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution at a specific time index.

        The behavior depends on which comparator is provided:

            - If `u2` is given: |u1 - u2|_{H1} = sqrt(∫_Ω ||∇(u1(x, time[t_index]) - u2(x, time[t_index]))||² dx)
            - If `grad_u_exact` is given: |u1 - u_exact|_{H1} = sqrt(∫_Ω ||∇(u1(x, time[t_index]) - u_exact(x, time[t_index]))||² dx)
            - If neither is given: |u1|_{H1} = sqrt(∫_Ω ||∇u1(x, time[t_index])||² dx)

        This method uses element-wise quadrature over the mesh, mapping reference 
        quadrature points to physical coordinates, and supports 1D intervals and 2D triangles.

        Notes
        -----
        - For H¹-based norms, the gradient of the exact solution (`grad_u_exact`) is required 
        if `u_exact` is used as the comparator.
        """
        logger.debug(f"Computing H¹-seminorm error at time index {t_index} (time = {self.time[t_index] if self.time is not None else 'N/A'})...")
        err = 0.0
        for elem_index in range(self.mesh.nelements()):
            phys_elem = self.femspace.get_physical_element(elem_index)
            detJ = phys_elem.det_jacobian()
            for node, weight in zip(self.ref_pts, self.weights):
                phys_point = phys_elem.reference_to_physical(node)
                diff = self._grad_diff(elem_index, phys_point, t_index)
                err += weight*detJ*np.dot(diff, diff)
        return float(np.sqrt(err))
    
    def h1_error(self) -> float:
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

    def h1_error_time(self) -> float:
        """
        Compute the H¹-norm error integrated over time for time-dependent solutions. The H¹-norm error is computed 
        at each time step and then integrated in time using the trapezoidal rule. The formula is:

            sqrt(∫_0^T ||u1(t) - u2(t)||_{H1}² dt)
        or
            sqrt(∫_0^T ||u1(t) - u_exact(t)||_{H1}² dt)

        depending on the mode. If `self.time` is not provided or has only one time point, it falls back to the static H¹-norm 
        error computation using the `h1_error()` method.

        Notes
        -----
        - This method assumes that the time snapshots in `u1`, `u2`, and the exact solution (if used) are ordered according to the time points in `time`.
        - The time integration is performed using the trapezoidal rule for better accuracy.
        """
        if self.time is None or self.nt == 1:
            logger.warning("Time array is not provided or has only one time point. Falling back to static H¹-norm error computation.")
            return self.h1_error()  # fallback to static norm
        logger.debug("Computing H¹-norm error integrated over time using trapezoidal rule...")        
        dt = np.diff(self.time)
        h1_error_sq = np.empty(self.nt)
        for i in range(self.nt):
            h1_error_sq[i] = self.h1_error_at_time(i) ** 2
        err_time = 0.5 * np.sum(dt*(h1_error_sq[:-1] + h1_error_sq[1:]))
        return float(np.sqrt(err_time))

    def h1_error_at_time(self, t_index: int) -> float:
        """
        Compute the H¹-norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution at a specific time index.

        The behavior depends on which comparator is provided:

            - If `u2` is given: ||u1 - u2||_{H1} = sqrt(∫_Ω (u1(x, time[t_index]) - u2(x, time[t_index]))² dx + ∫_Ω ||∇(u1(x, time[t_index]) - u2(x, time[t_index]))||² dx)
            - If `u_exact` and `grad_u_exact` are given: ||u1 - u_exact||_{H1} = sqrt(∫_Ω (u1(x, time[t_index]) - u_exact(x, time[t_index]))² dx + ∫_Ω ||∇(u1(x, time[t_index]) - u_exact(x, time[t_index]))||² dx)
            - If neither is given: ||u1||_{H1} = sqrt(∫_Ω (u1(x, time[t_index]))² dx + ∫_Ω ||∇u1(x, time[t_index])||² dx)
        
        This method uses element-wise quadrature over the mesh, mapping reference
        quadrature points to physical coordinates, and supports 1D intervals and 2D triangles.

        Notes
        -----
        - Both the exact solution (`u_exact`) and its gradient (`grad_u_exact`) are required
        if H¹-norm comparison with the exact solution is desired.
        """
        logger.debug(f"Computing H¹-norm error at time index {t_index} (time = {self.time[t_index] if self.time is not None else 'N/A'})...")
        l2 = self.l2_error_at_time(t_index)
        h1s = self.h1_semi_error_at_time(t_index)
        return float(np.sqrt(l2**2 + h1s**2))

    def compute(self, norm: NormType = NormType.L2, t_index: Optional[int] = None) -> float:
        """
        Compute a selected norm of a FEM solution or the error between two FEM solutions 
        or between a FEM solution and the exact solution.

        This method provides a general interface to any supported norm.

        Parameters
        ----------
        norm : NormType, default=NormType.L2
            The norm type to compute:
                - NormType.L2      : L² norm
                - NormType.H1_SEMI : H¹-seminorm
                - NormType.H1      : full H¹ norm
                - NormType.LINF    : L∞ (maximum) norm
        
        Returns
        -------
        float
            The computed norm value.
        """
        assert t_index is None or (self.time is not None and 0 <= t_index < self.nt), "Invalid time index."
        time_map = {NormType.L2: self.l2_error_at_time,
                    NormType.LINF: self.linf_error_at_time,
                    NormType.H1_SEMI: self.h1_semi_error_at_time,
                    NormType.H1: self.h1_error_at_time}
        time_integrated_map = {NormType.L2: self.l2_error_time,
                               NormType.LINF: self.linf_error_time,
                               NormType.H1_SEMI: self.h1_semi_error_time,
                               NormType.H1: self.h1_error_time}
        if t_index is not None:
            return time_map[norm](t_index)
        return time_integrated_map[norm]()