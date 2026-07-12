import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
from utils.history import History, MetricType, SpatialMode, TemporalMode
from utils.errornorms import NormType
from visualization.visualize import plot_wireframe, plot_tri, plot_contour

def plot_contour_and_error_2D(x, y, heat_solution: np.ndarray, oswr, iterations: list, suptitle: str = "", 
                              t: int = -1, fig_kw: dict = {"figsize": (7, 14)}, plt_kw: dict = {}):
    # Create a figure with subplots for each iteration, with 2 columns for solutions and errors
    fig, axes = plt.subplots(len(iterations), 2, **(fig_kw if fig_kw is not None else {}))
    # Set the super title for the figure
    if suptitle:
        fig.suptitle(suptitle)

    for i, iter in enumerate(iterations):
        # Plot the combined solution on the same figure on the first column
        plot_tri(x, y, oswr.combine(oswr.iterates[iter])[:, t], ax = axes[i, 0], xlabel = r'$x$', ylabel = r'$y$', contour = True, plot_kwargs = (plt_kw if plt_kw is not None else {}))
        # Plot the  errors on the same figure on the second column
        err = heat_solution - oswr.combine(oswr.iterates[iter])
        # Plot the subdomain errors for the current iteration
        plot_tri(x, y, err[:, t], ax = axes[i, 1], xlabel = r'$x$', ylabel = r'$y$', contour = True, plot_kwargs = (plt_kw if plt_kw is not None else {}))
    return fig, axes

def plot_contour_and_error(vertices, time_grid, heat_solution: np.ndarray, oswr, iterations: list, 
                            suptitle: str = "", fig_kw: dict = {"figsize": (7, 14)}, plt_kw: dict = {}, cont_kw: dict = {}):
    """
    Plot the contour plot of the 1D combined subdomain solutions and errors for each 
    iteration in `iterations` in two columns, with the solutions on the left and the 
    errors on the right.

    Parameters:
    -----------
    vertices: np.ndarray
        The spatial vertices of the domain.
    time_grid: np.ndarray
        The time grid for the solution.
    heat_solution: np.ndarray
        The reference heat solution.
    oswr: object
        The OSWR solver object.
    iterations: list
        A list of iteration numbers to plot.
    zlim: tuple, optional
        The z-axis limits for the plots (default is (-1, 1)).
    suptitle: str, optional
        The super title for the figure (default is an empty string).
    cont_kw: dict, optional
        Additional keyword arguments for the contour plots (default is an empty dictionary).
    fig_kw: dict, optional
        Additional keyword arguments for the figure (default is {"figsize": (7, 14)}).
    plt_kw: dict, optional
    
    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure object containing the subplots.
    axes: np.ndarray
        The array of axes objects for the subplots.
    """
    # Create a figure with subplots for each iteration, with 2 columns for solutions and errors
    fig, axes = plt.subplots(len(iterations), 2, **(fig_kw if fig_kw is not None else {}))

    # Set the super title for the figure
    if suptitle:
        fig.suptitle(suptitle)

    for i, iter in enumerate(iterations):
        # Plot the combined solution on the same figure on the first column
        plot_contour(vertices, time_grid, oswr.combine(oswr.iterates[iter]).T, ax = axes[i, 0], xlabel = r'$x$', ylabel = r'$t$', plot_kwargs = (plt_kw if plt_kw is not None else {}))
        # Plot the  errors on the same figure on the second column
        err = heat_solution - oswr.combine(oswr.iterates[iter])
        # Plot the subdomain errors for the current iteration
        plot_contour(vertices, time_grid, err.T, ax = axes[i, 1], xlabel = r'$x$', ylabel = r'$t$', **(cont_kw if cont_kw is not None else {}), plot_kwargs = (plt_kw if plt_kw is not None else {}))
    return fig, axes

def plot_solution_and_error(vertices, time_grid, heat_solution: np.ndarray, iterates: dict, ltog: dict, iterations: list, 
                            suptitle: str = "", fig_kw: dict = {"figsize": (7, 14)}, plt_kw: dict = {}):
    """
    Plot the wireframe of the 1D subdomain solutions and errors for each iteration in `iterations`
    in two columns, with the solutions on the left and the errors on the right.

    Parameters:
    -----------
    vertices: np.ndarray
        The spatial vertices of the domain.
    time_grid: np.ndarray
        The time grid for the solution.
    heat_solution: np.ndarray
        The reference heat solution.
    iterates: dict
        A dictionary containing the subdomain solutions for each iteration.
    ltog: dict
        A dictionary mapping local subdomain indices to global indices.
    iterations: list
        A list of iteration numbers to plot.
    zlim: tuple, optional
        The z-axis limits for the plots (default is (-1, 1)).
    suptitle: str, optional
        The super title for the figure (default is an empty string).
    fig_kw: dict, optional
        Additional keyword arguments for the figure (default is {"figsize": (7, 14)}).
    plt_kw: dict, optional
        Additional keyword arguments for the wireframe plots (default is an empty dictionary).
    
    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure object containing the subplots.
    axes: np.ndarray
        The array of axes objects for the subplots.
    """
    # Create a figure with subplots for each iteration, with 2 columns for solutions and errors
    fig, axes = plt.subplots(len(iterations), 2, subplot_kw = {"projection": "3d"}, **(fig_kw if fig_kw is not None else {}))

    # Set the super title for the figure
    if suptitle:
        fig.suptitle(suptitle)

    # Get the number of subdomains from the length of the ltog dictionary
    nsub = len(ltog)

    # Create empty arrays to store the subdomain solutions for the current iteration, initialized with NaN values
    Z = np.full((nsub, *heat_solution.shape), np.nan, dtype = float)

    # Create empty arrays to store the subdomain errors for the current iteration, initialized with NaN values
    Z_err = np.full((nsub, *heat_solution.shape), np.nan, dtype = float)

    for i, iter in enumerate(iterations):
        # Plot the subdomain solutions on the same figure on the first column
        for j in range(1, nsub + 1):
            # Extend the subdomain solutions to the global domain
            Z[j-1, ltog[j], :] = iterates[iter][j]
            # Plot the subdomain solutions for the current iteration
            plot_wireframe(vertices, time_grid, Z[j-1].T, ax = axes[i, 0], xlabel = r'$x$', ylabel = r'$t$', xlim = (min(vertices), max(vertices)), 
                           ylim = (time_grid[0], time_grid[-1]), **(plt_kw if plt_kw is not None else {}))
        # Plot the subdomain errors on the same figure on the second column
        for k in range(1, nsub + 1):
            # Compute the error between the iterate and the heat solution for the current iteration
            err = heat_solution[ltog[k], :] - iterates[iter][k]
            # Extend the subdomain errors to the global domain
            Z_err[k-1, ltog[k], :] = err
            # Plot the subdomain errors for the current iteration
            plot_wireframe(vertices, time_grid, Z_err[k-1].T, ax = axes[i, 1], xlabel = r'$x$', ylabel = r'$t$', xlim = (min(vertices), max(vertices)), 
                           ylim = (time_grid[0], time_grid[-1]), **(plt_kw if plt_kw is not None else {}))
    return fig, axes

def max_error(history: History) -> np.ndarray:
    # Extract the maximum absolute error for each iteration from the history object
    for mspec in history.values.keys():
        if mspec.name == MetricType.ABSOLUTE_ERROR and mspec.norm == NormType.LINF and mspec.spatial == SpatialMode.SUBDOMAINS and mspec.temporal == TemporalMode.STATIC:
            subvalues = history.values[mspec]["subdomains"] # dictionary of shape {domainID: shape (niter,)}
    assert isinstance(subvalues, dict), "Expected subvalues to be a dictionary of shape {domainID: shape (niter,)}" # type: ignore
    niter = next(iter(subvalues.values())).shape[0]
    error = np.zeros(niter)
    for j in range(niter):
        for k in subvalues.keys():
            error[j] = max(error[j], subvalues[k][j])
    return error

def theoretical_bound(delta: float, T: float, dim: int, iterations: np.ndarray, heat_solution: np.ndarray, ltog: dict, initial_data: dict) -> np.ndarray:
    """
    This function computes the theoretical bound on the error of the Schwarz Waveform 
    Relaxation (SWR) method as a function of the iteration number. 

    Parameters:
    -----------
    delta: float
        The overlap size between subdomains.
    T: float
        The final time of the simulation.
    dim: int
        The spatial dimension of the problem.
    iterations: np.ndarray
        An array of iteration numbers.
    heat_solution: np.ndarray
        The reference heat solution for comparison.
    ltog: dict
        A dictionary mapping local subdomain indices to global indices.
    initial_data: dict
        A dictionary containing the initial data for each subdomain, where the 
        keys are the subdomain IDs and the values are the initial data for 
        each subdomain.

    Returns:
    --------
    bound: np.ndarray
        An array containing the theoretical bound on the error for each iteration.
    """
    error = 0
    nsub = len(ltog)
    rate = erfc((delta*iterations)/(2*np.sqrt(dim*T)))
    for k in range(1, nsub + 1):
        suberr = np.abs(heat_solution[ltog[k], :] - initial_data[k]).max()
        error = max(error, suberr)
    return rate*error