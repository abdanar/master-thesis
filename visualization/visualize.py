import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from fem.mesh import Mesh
from typing import Callable, Optional

class MeshVisualizer:
    def __init__(self, meshobj: Mesh):
        if not isinstance(meshobj, Mesh):
            raise TypeError("MeshVisualizer expects a Mesh object")
        self.mesh = meshobj

    # Build a color array representing the areas of the triangles in the mesh.
    def carray_areas(self) -> np.ndarray:
        return self.mesh.measures()

    # Build a color array where boundary triangles get color `1` and interior triangles get color `0`.
    def carray_boundary(self) -> np.ndarray:
        colors = np.zeros(self.mesh.elements.shape[0], dtype=int)
        bdtriangles = self.mesh.boundary_elements()
        for i, triangle in enumerate(self.mesh.elements):
            if tuple(triangle) in bdtriangles:
                colors[i] = 1
        return colors
    
    # Build a color array representing the non-overlapping subdomain decomposition of the mesh.
    def carray_decomposition(self, n: int) -> np.ndarray:
        _, _, _, _, membership = self.mesh.decompose(n)
        return np.array(membership)

    def visualize(self, carray: np.ndarray):
        plt.figure(figsize = (6,6), dpi = 150)
        plt.tripcolor(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], triangles = self.mesh.elements, facecolors = carray, edgecolors = "k")
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.show()

class SolutionVisualizer:
    def __init__(self, meshobj: Mesh, u: np.ndarray, dt: Optional[float] = None):
        """
        Parameters
        ----------
        meshobj : Mesh
            The mesh object the solution is defined on.
        u : np.ndarray
            FEM solution array. For time-dependent problems, shape should be (nspace, ntime). 
            For steady-state problems, shape can be (nspace,) or (nspace, 1).
        dt : float, optional
            Time step size
        """
        self.mesh = meshobj
        self.u = u
        self.dt = dt
        self.ntime = u.shape[1] if dt is not None and u.ndim == 2 else 1

    def visualize(self, cmap = 'viridis', levels = 50, figsize = (6, 5), dpi = 150):

        """
        Plot FEM solution for 2D triangular mesh with linear Lagrange elements (degree 1).

        Parameters
        ----------
        cmap : str, optional
            Colormap for contour plot (default 'viridis').
        levels : int, optional
            Number of contour levels (default 50).
        """

        # Use mesh vertices and elements
        vertices = self.mesh.vertices
        elements = self.mesh.elements

        x = vertices[:, 0]
        y = vertices[:, 1]

        # Create a triangulation
        triang = mtri.Triangulation(x, y, elements)

        # Plot filled contour
        plt.figure(figsize=(6,5))
        plt.tricontourf(triang, self.u.ravel(), levels = levels, cmap = cmap)
        plt.triplot(triang, color = 'k', linewidth = 0.5, alpha = 0.3)
        plt.colorbar(label = 'Solution u')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('FEM Solution (Linear Lagrange)')
        plt.show()

    def visualize_3d(self, cmap = 'viridis'):

        """
        Plot FEM solution for 2D triangular mesh as a 3D surface.

        Parameters
        ----------
        cmap : str, optional
            Colormap for surface plot (default 'viridis').
        """

        # Use mesh vertices and elements
        vertices = self.mesh.vertices
        elements = self.mesh.elements

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = self.u.ravel()  # Ensure z is 1D

        # Create a triangulation
        triang = mtri.Triangulation(x, y, elements)

        # Create 3D figure
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_trisurf(triang, z, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Solution u')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title('FEM Solution 3D Surface')

        plt.show()


    def _plot_error_1D_static(self, exact, figsize: tuple = (7, 4), dpi: int = 100, 
                              xlabel: str = 'x', ylabel: str = 'error', title: str = '1D Steady-State Error', **kwargs):
        x = self.mesh.vertices
        err = self.u[:, 0] - exact(x)
        plt.figure(figsize = figsize, dpi = dpi)
        plt.plot(x, err, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def _plot_error_1D_time(self, exact, figsize: tuple = (7, 4), dpi: int = 100, 
                            xlabel: str = 'x', ylabel: str = 'error', title: str = '1D Time-Dependent Error', **kwargs):
        x = self.mesh.vertices
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        line, = ax.plot(x, self.u[:, 0] - exact(x, 0), **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Slider
        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime - 1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            err = self.u[:, step] - exact(x, step*self.dt) # type: ignore
            line.set_ydata(err)
            ax.set_title(f'{title} | t={step*self.dt:.3f}') # type: ignore
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()

    def _plot_error_2D_static(self, exact, figsize: tuple = (7, 6), dpi: int = 100, 
                              xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'error', title: str = '2D Steady-State Error', **kwargs):
        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]
        triang = mtri.Triangulation(x, y, self.mesh.elements)
        err = self.u[:, 0] - exact(x, y)
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(triang, err, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u_h - u_exact')
        plt.show()

    def _plot_error_2D_time(self, exact, figsize: tuple = (7, 6), dpi: int = 100, 
                            xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'error', title: str = '2D Time-Dependent Error', **kwargs):
        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]
        triang = mtri.Triangulation(x, y, self.mesh.elements)
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(triang, self.u[:, 0] - exact(x, y, 0), **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u_h - u_exact')

        # Slider
        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime - 1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            err = self.u[:, step] - exact(x, y, step * self.dt) # type: ignore
            ax.cla()
            ax.plot_trisurf(triang, err, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            ax.set_title(f'{title} | t={step * self.dt:.3f}') # type: ignore
            fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 10, label = zlabel)
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()
    
    def plot_error(self, exact: Callable, figsize: tuple = (7, 6), dpi: int = 100, 
                   xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'error', title: str = 'Error', **kwargs):
        """
        Plot the FEM solution error |u_h - u_exact| for both steady-state and time-dependent problems.

        The error is computed at mesh vertices only, which is well-defined for Schwarz methods 
        and higher-order FEM. Works for both 1D and 2D problems. For time-dependent problems, 
        an interactive time slider is provided to visualize the error at each time step.

        Parameters
        ----------
        exact : Callable
                For time-dependent problems, should be exact(x, t) for 1D or exact(x, y, t) for 2D. 
                For steady-state problems, should be exact(x) for 1D or exact(x, y) for 2D.
        figsize : tuple, optional
                Figure size for the plot (default (7, 6)).
        dpi : int, optional
                Dots per inch for the plot (default 100).
        xlabel : str, optional
                Label for the x-axis (default 'x').
        ylabel : str, optional
                Label for the y-axis (default 'y').
        zlabel : str, optional
                Label for the z-axis (default 'error').
        title : str, optional
                Title of the plot (default 'Error').
        **kwargs : dict
                Additional keyword arguments to pass to the specific plotting functions, 
                e.g., cmap for colormap, levels for contour levels, etc.
        """
        if self.mesh.dim == 1 and self.ntime == 1:
            self._plot_error_1D_static(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, **kwargs)
        elif self.mesh.dim == 1 and self.ntime > 1:
            self._plot_error_1D_time(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, **kwargs)
        elif self.mesh.dim == 2 and self.ntime == 1:
            self._plot_error_2D_static(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, **kwargs)
        elif self.mesh.dim == 2 and self.ntime > 1:
            self._plot_error_2D_time(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, **kwargs)
        else:
            raise ValueError("Unsupported mesh dimension or time steps")
    
    def plot_convergence(self, error_history, logscale: bool = True, figsize: tuple = (6, 4), dpi: int = 150,
                         xlabel: str = "Iteration number", ylabel: str = r"$\| u_h - u \|_{L^2}$", 
                         title: str = r"Iteration vs $L^{2}$ Error", grid: bool = True, styles = None, **kwargs):
        """
        Plot convergence history of Schwarz iterations. This function is intended to be used with the error history tracked during Schwarz iterations, 
        which can be stored in `self.error_history` and `self.error_subdomains`. It supports plotting the global error history as well as the error 
        history for each subdomain in one plot if available.

        Parameters
        ----------
        error_history : list or array or dict
            List of error values per iteration, or a dictionary with subdomain IDs as keys and lists of error values as values.
        logscale : bool, default True
            If True, use semilog-y scale (recommended for convergence plots).
        figsize : tuple, default (6, 4)
            Figure size in inches.
        dpi : int, default 150
            Dots per inch for the figure.
        xlabel : str, default "Iteration number"
            Label for the x-axis.
        ylabel : str
            Label for the y-axis.
        title : str, default "Iteration vs L2 Error"
            Plot title.
        styles : dict, optional
            A dictionary mapping subdomain IDs to style dictionaries for plotting 
            (e.g., {'subdomain1': {'color': 'r', 'linestyle': '--'}, 'subdomain2': {'color': 'b', 'linestyle': '-'}}).
        **kwargs : dict
            Additional keyword arguments to pass to plt.plot / plt.semilogy
            e.g., marker='o', color='r', linewidth=2, linestyle='--', alpha=0.7
        """
        plt.figure(figsize=figsize, dpi=dpi)
        if isinstance(error_history, dict):
            for subdomain, errors in error_history.items():
                iterations = range(1, len(errors) + 1)
                style = dict(kwargs)
                if styles and subdomain in styles:
                    style.update(styles[subdomain])
                if logscale:
                    plt.semilogy(iterations, errors, label=f"Subdomain {subdomain}", **style)
                else:
                    plt.plot(iterations, errors, label=f"Subdomain {subdomain}", **style)
            plt.xlim(1, len(error_history[1]))
            plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        else:
            iterations = range(1, len(error_history) + 1)
            if logscale:
                plt.semilogy(iterations, error_history, **kwargs)
            else:
                plt.plot(iterations, error_history, **kwargs)
            plt.xlim(1, len(error_history))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if grid:
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(direction='in', which='both', top=True, right=True)
        plt.tight_layout()
        plt.show()

    # add vtk versions as well