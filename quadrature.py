import numpy as np


# Numerical integration on the reference triangle using quadrature rule

def triangle_quadrature(order: int):
    if order == 1:
        # Q(f) = |T_reference| x f(x_barycenter) = w1 x f(x1)
        return (np.array([[1/3, 1/3]]), np.array([1/2]))
    elif order == 2:
        # Q(f) = |T_reference|/2 x (f(1/6, 1/6) + f(2/3, 1/6) + f(1/6, 2/3)) = w1 x f(x1) + w2 x f(x2) + w3 x f(x3)
        return (np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]), np.array([1/6, 1/6, 1/6]))
    elif order == 3:
        return (np.array([[1/3, 1/3], [1/5, 1/5], [3/5, 1/5], [1/5, 3/5]]), np.array([-27/96, 25/96, 25/96, 25/96]))
    else:
        raise NotImplementedError("Quadrature order not implemented.")

# Numerical integration on the reference interval [0, 1]

def interval_quadrature(order: int):
    """
    Quadrature rules on the reference interval [0, 1].

    Parameters
    ----------
    order : int
        Quadrature order.

    Returns
    -------
    points : np.ndarray
        Quadrature points in the reference interval.
    weights : np.ndarray
        Corresponding quadrature weights.
    """
    if order == 1:
        # Q(f) = |K_ref| * f(1/2)
        return np.array([0.5]), np.array([1.0])

    elif order == 2:
        # Exact for polynomials up to degree 3
        a = 1/(2*np.sqrt(3))
        return (
            np.array([0.5 - a, 0.5 + a]),
            np.array([0.5, 0.5])
        )

    elif order == 3:
        # Exact for polynomials up to degree 5
        a = np.sqrt(3/5)/2
        return (
            np.array([0.5 - a, 0.5, 0.5 + a]),
            np.array([5/18, 8/18, 5/18])
        )

    else:
        raise NotImplementedError("Quadrature order not implemented.")

def integrate(func, order):
    nodes, weights = triangle_quadrature(order)
    values = np.array([func(node) for node in nodes])
    return np.sum(weights * values)
