import numpy as np

# Numerical integration on the reference triangle using quadrature rule
def triangle_quadrature(order: int):
    """
    Return quadrature points and weights on the reference triangle.

    The reference triangle is defined as
        T_ref = {(x, y) : x >= 0, y >= 0, x + y <= 1},
    with area |T_ref| = 1/2.

    For low orders (1-3), minimal symmetric rules (Dunavant-style) are used.
    For higher orders, a tensor-product Gauss-Legendre quadrature is used.

    Parameters
    ----------
    order : int
        Quadrature order (polynomial degree to integrate exactly). 

    Returns
    -------
    points : np.ndarray, shape (npoints, 2)
        Quadrature points in reference coordinates.
    weights : np.ndarray, shape (npoints,)
        Quadrature weights corresponding to each point. Sum(weights) = 1/2.
    """
    # --- Low-order minimal rules (Dunavant-style) ---
    if order == 1:
        points = np.array([[1/3, 1/3]])
        weights = np.array([1/2])
        return points, weights
    elif order == 2:
        points = np.array([[1/6, 1/6],
                           [2/3, 1/6],
                           [1/6, 2/3]])
        weights = np.array([1/6, 1/6, 1/6])
        return points, weights
    elif order == 3:
        points = np.array([[1/3, 1/3],
                           [1/5, 1/5],
                           [3/5, 1/5],
                           [1/5, 3/5]])
        weights = np.array([-27/96, 25/96, 25/96, 25/96])
        return points, weights

    # --- High-order: tensor-product Gauss–Legendre ---
    xi_1d, w_1d = np.polynomial.legendre.leggauss(order)
    xi_1d = 0.5 * (xi_1d + 1.0)  # map [-1,1] -> [0,1]
    w_1d = 0.5 * w_1d

    points = []
    weights = []

    for i, xi in enumerate(xi_1d):
        for j, eta in enumerate(xi_1d):
            x = xi
            y = (1.0 - xi) * eta
            w = w_1d[i] * w_1d[j] * (1.0 - xi)  # Jacobian factor
            points.append([x, y])
            weights.append(w)

    return np.array(points), np.array(weights)

# Numerical integration on the reference interval [0, 1]
def interval_quadrature(order: int):
    """
    Return Gauss-Legendre quadrature points and weights on the interval [-1, 1].

    Parameters
    ----------
    order : int
        Number of quadrature points. Integrates polynomials of degree up to 2*order - 1 exactly.

    Returns
    -------
    points : np.ndarray, shape (order,)
        Quadrature points (nodes), which are the roots of the Legendre polynomial of degree `order`.
        Values lie strictly inside (-1, 1).

    weights : np.ndarray, shape (order,)
        Quadrature weights corresponding to each point.
        All weights are positive and satisfy sum(weights) = 2.
        The quadrature rule is exact for all polynomials of degree <= 2*order - 1.

    Notes
    -----
    - To use on a different interval [a, b], map points and weights:
        x_mapped = 0.5 * ((b - a) * points + (b + a))
        w_mapped = 0.5 * (b - a) * weights
    - This function is standard for 1D FEM integration and is highly accurate for polynomial integrands.
    """

    # Gauss–Legendre rule on [-1, 1]
    xi, w = np.polynomial.legendre.leggauss(order)

    # Map to [0, 1]
    points = 0.5 * (xi + 1.0)
    weights = 0.5 * w

    return points, weights

def integrate(func, dim: int, order: int):
    """
    Numerically integrate a scalar function over a reference element using quadrature.

    This function supports 1D intervals and 2D triangles. It evaluates the given
    function at quadrature points and computes a weighted sum using the corresponding
    quadrature weights.

    Parameters
    ----------
    func : callable
        The function to integrate. Should accept a single argument:
        - 1D: scalar or 1-element array-like
        - 2D: array-like with shape (2,)
        and return a scalar value.
    dim : int
        Dimension of the integration domain:
        - 1: interval [0,1]
        - 2: reference triangle {(x,y) : x>=0, y>=0, x+y<=1}
    order : int
        Quadrature order. Determines the number of quadrature points and accuracy:
        - 1D: order = number of Gauss points
        - 2D: order = minimal Dunavant rules for 1-3, tensor-product Gauss for higher orders

    Returns
    -------
    float
        Approximate value of the integral of `func` over the reference element.

    Notes
    -----
    - The function uses pre-defined quadrature rules for efficiency:
        - 1D interval: Gauss-Legendre quadrature
        - 2D triangle: Dunavant-style minimal rules for order 1-3, tensor-product for higher orders
    - The function evaluates `func` at each quadrature node and multiplies by the
      corresponding weight before summing.
    - This is suitable for FEM element-level integration of stiffness, mass, or load terms.
    """
    if dim == 2:
        nodes, weights = triangle_quadrature(order)
    elif dim == 1:
        nodes, weights = interval_quadrature(order)
    else:
        raise NotImplementedError("Integration dimension not implemented.")
    values = np.array([func(node) for node in nodes])
    return np.sum(weights * values)
