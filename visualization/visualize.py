import numpy as np
import meshio
import os
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize  # <- correct import
import matplotlib.cm as cm
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

    def plot_comparison(self, u_exact_func = None, cmap = 'viridis'):
        """
        Plot FEM solution for 2D triangular mesh as a 3D surface,
        optionally comparing with the exact solution side by side.

        Parameters
        ----------
        u_exact_func : callable, optional
            Function u_exact(x, y) returning the exact solution.
            If provided, a side-by-side comparison will be plotted.
        cmap : str, optional
            Colormap for surface plot (default 'viridis').
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.tri as mtri

        vertices = self.mesh.vertices
        elements = self.mesh.elements

        x = vertices[:, 0]
        y = vertices[:, 1]
        z_fem = self.u.ravel()  # FEM solution

        triang = mtri.Triangulation(x, y, elements)

        if u_exact_func is None:
            # Single plot
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_trisurf(triang, z_fem, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Solution u')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
            ax.set_title('FEM Solution 3D Surface')
            plt.show()
        else:
            # Side-by-side plots
            fig = plt.figure(figsize=(14,6))

            # FEM solution
            ax1 = fig.add_subplot(121, projection='3d')
            surf1 = ax1.plot_trisurf(triang, z_fem, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='u_h')
            ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
            ax1.set_title('FEM Solution')

            # Exact solution
            z_exact = u_exact_func(x, y)
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = ax2.plot_trisurf(triang, z_exact, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='u_exact')
            ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
            ax2.set_title('Exact Solution')

            plt.show()

    def visualize_1d_time_compare(self, exact_func, nx=200):
        """
        Compare 1D FEM solution with exact solution over time.
        
        exact_func(x, t)
        """

        idx = np.argsort(self.mesh.vertices)
        x_mesh = self.mesh.vertices[idx]
        x_grid = np.linspace(x_mesh.min(), x_mesh.max(), nx)

        fig, ax = plt.subplots(figsize=(8,5))
        plt.subplots_adjust(bottom=0.25)

        # Initial plot
        z_exact = exact_func(x_grid, 0.0)
        z_num = self.u[:, 0][idx]

        line_exact, = ax.plot(x_grid, z_exact, 'r-', label='Exact')
        line_num, = ax.plot(x_mesh, z_num, 'bo-', label='Numerical')
        ax.set_xlabel('x'); ax.set_ylabel('u')
        ax.set_title('Time = 0.0')
        ax.legend()
        
        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime-1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            t = step*self.dt
            line_exact.set_ydata(exact_func(x_grid, t))
            line_num.set_ydata(self.u[:, step][idx])
            ax.set_title(f'Time = {t:.3f}')
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()

    def visualize_3d_time(self, cmap = 'viridis'):
        """
        Interactive 3D visualization with time slider.
        """
        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]
        elements = self.mesh.elements

        triang = mtri.Triangulation(x, y, elements)

        fig = plt.figure(figsize = (8,6), dpi = 100)
        ax = fig.add_subplot(111, projection = '3d')
        plt.subplots_adjust(bottom = 0.25)

        # Initial surface
        z = self.u[:, 0]
        surf = ax.plot_trisurf(triang, z, cmap = cmap, edgecolor = 'k', linewidth = 0.2, antialiased = True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f'FEM Solution 3D Surface | t = 0.0')

        fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 10, label = 'Solution u')

        # Slider axes
        ax_slider = plt.axes((0.25, 0.1, 0.5, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime - 1, valinit = 0, valstep = 1)

        def update(val):
            step = int(time_slider.val)
            ax.clear()
            z = self.u[:, step]
            surf = ax.plot_trisurf(triang, z, cmap = cmap, edgecolor = 'k', linewidth = 0.2, antialiased = True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f'FEM Solution 3D Surface | t = {step*self.dt:.3f}')
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()

    def visualize_3d_time_compare(self, exact_func, cmap='viridis', nx=100, ny=100):
        """
        Compare numeric FEM solution (triangular mesh) with smooth exact solution
        on a fine grid for better visualization.
        
        Parameters:
        -----------
        exact_func : callable
            exact_func(x, y, t)
        cmap : str
            colormap
        nx, ny : int
            resolution of the grid for smooth exact solution
        """
        x_mesh = self.mesh.vertices[:, 0]
        y_mesh = self.mesh.vertices[:, 1]
        elements = self.mesh.elements
        triang = mtri.Triangulation(x_mesh, y_mesh, elements)

        # Create fine grid for smooth exact solution
        x_grid = np.linspace(0, 1, nx)
        y_grid = np.linspace(0, 1, ny)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        # Initial surfaces
        z_exact = exact_func(X_grid, Y_grid, 0.0)
        z_num = self.u[:, 0]

        surf1 = ax1.plot_surface(X_grid, Y_grid, z_exact, cmap=cmap, edgecolor='none')
        ax1.set_title('Exact Solution (Smooth)')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='u')

        surf2 = ax2.plot_trisurf(triang, z_num, cmap=cmap, edgecolor='k', linewidth=0.2)
        ax2.set_title('Numerical Solution')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='u')

        # Slider
        ax_slider = plt.axes((0.25, 0.1, 0.5, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime-1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            t = step*self.dt

            ax1.cla()
            ax2.cla()

            # Smooth exact
            z_exact = exact_func(X_grid, Y_grid, t)
            ax1.plot_surface(X_grid, Y_grid, z_exact, cmap=cmap, edgecolor='none')
            ax1.set_title(f'Exact Solution | t={t:.3f}')
            ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')

            # Numeric
            z_num = self.u[:, step]
            ax2.plot_trisurf(triang, z_num, cmap=cmap, edgecolor='k', linewidth=0.2)
            ax2.set_title(f'Numerical Solution | t={t:.3f}')
            ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')

            fig.canvas.draw_idle()

        time_slider.on_changed(update)
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
            err = self.u[:, step] - exact(x, step*self.dt)
            line.set_ydata(err)
            ax.set_title(f'{title} | t={step*self.dt:.3f}')
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
            err = self.u[:, step] - exact(x, y, step * self.dt)
            ax.cla()
            ax.plot_trisurf(triang, err, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            ax.set_title(f'{title} | t={step * self.dt:.3f}')
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
                         xlabel: str = "Iteration", ylabel: str = r"$\| u_h - u \|_{L^2}$", 
                         title: str = r"Iteration vs $L^{2}$ Error", **kwargs):
        """
        Plot convergence history of Schwarz iterations.

        Parameters
        ----------
        error_history : list or array
            List of error values per iteration.
        logscale : bool, default True
            If True, use semilog-y scale (recommended for convergence plots).
        figsize : tuple, default (6, 4)
            Figure size in inches.
        dpi : int, default 150
            Dots per inch for the figure.
        title : str, default "Iteration vs L2 Error"
            Plot title.
        **kwargs : dict
            Additional keyword arguments to pass to plt.plot / plt.semilogy
            e.g., marker='o', color='r', linewidth=2, linestyle='--', alpha=0.7
        """
        iterations = range(1, len(error_history) + 1)
        plt.figure(figsize=figsize, dpi=dpi)
        if logscale:
            plt.semilogy(iterations, error_history, **kwargs)
        else:
            plt.plot(iterations, error_history, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # add vtk versions as well