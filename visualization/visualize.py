from typing import Callable, Optional
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.widgets import Slider
from fem.mesh import Mesh

class MeshVisualizer:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

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
    def carray_decomposition(self, n: int, edge_weights = None) -> np.ndarray:
        _, _, _, _, membership = self.mesh.decompose(n = n, edge_weights = edge_weights)
        return np.array(membership)
    
    # Annotate the triangles with their numbers.
    def _triangle_numbers(self, mesh: Mesh, ax, color = 'black', fontsize = 6):        
        for i, element in enumerate(mesh.elements):
            centroid = mesh.vertices[element].mean(axis = 0)
            ax.text(centroid[0], centroid[1], str(i + 1), color = color, fontsize = fontsize, ha = 'center', va = 'center', clip_on = True)
    
    # Annotate the vertices with their numbers.
    def _vertex_numbers(self, mesh: Mesh, ax, color = 'black', fontsize = 7):
        for i, (x, y) in enumerate(mesh.vertices):
            ax.text(x, y, str(i + 1), color = color, fontsize = fontsize, ha = 'right', va = 'bottom', clip_on = True)

    # Plot the mesh vertices as dots.
    def _vertex_markers(self, mesh: Mesh, ax, color = 'black', size = 15):
        ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], s = size, c = color)

    def plot_mesh(self, carray: Optional[np.ndarray] = None, figsize: tuple = (6, 6), dpi: int = 150, 
                show_vertex_markers: bool = True, show_node_numbers: bool = True, show_element_numbers: bool = True,
                axis_off: bool = True, save_path: Optional[str] = None):
        """
        Visualize the mesh with optional coloring and annotations.

        Parameters
        ----------
        carray : np.ndarray, optional
            An array of shape (ntriangles,) to color the triangles. If None, no coloring is applied (default: None).
        figsize : tuple, optional
            Figure size in inches (width, height) (default: (6, 6)).
        dpi : int, optional
            Resolution of the figure in dots per inch (default: 150).
        show_vertex_markers : bool, optional
            If True, plot the vertices as markers (default: True).
        show_node_numbers : bool, optional
            If True, annotate the vertices with their numbers (default: True).
        show_element_numbers : bool, optional
            If True, annotate the elements with their numbers (default: True).
        axis_off : bool, optional
            If True, turn off the axis (default: True).
        save_path : str, optional
            If provided, save the figure to the specified path (default: None).
        """
        plt.figure(figsize = figsize, dpi = dpi)
        if carray is None:
            plt.triplot(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements, color = 'k', linewidth = 0.5)
        else:
            plt.tripcolor(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements, carray, edgecolors = 'k')
        self._vertex_numbers(self.mesh, plt.gca()) if show_node_numbers else None
        self._triangle_numbers(self.mesh, plt.gca()) if show_element_numbers else None
        self._vertex_markers(self.mesh, plt.gca()) if show_vertex_markers else None
        plt.axis('off') if axis_off else None
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.show()

    def plot_subdomains(self, subdomains: dict[int, Mesh], membership: Optional[np.ndarray] = None, 
                        figsize: tuple = (6, 6), dpi: int = 150, cmap: str = "Set3", 
                        ncols: int = 3, include_global_mesh: bool = True, show_vertex_markers: bool = True, 
                        show_node_numbers: bool = True, show_element_numbers: bool = True,
                        axis_off = True, save_path: Optional[str] = None):
        """
        Visualize the subdomain decomposition of the mesh supporting both non-overlapping and 
        overlapping subdomains. Optionally include the global mesh with subdomains colored by membership.

        This function creates a subplot for each subdomain and optionally the global mesh. The global mesh is colored 
        according to the membership array of non-overlapping subdomains, while the subdomains are outlined with their 
        own triangles. Node and element numbers can be annotated for clarity. The figure can be saved to a specified path.

        Parameters
        ----------
        subdomains : dict[int, Mesh]
            A dictionary mapping subdomain IDs to their corresponding Mesh objects.
        membership : np.ndarray, optional
            An array of shape (ntriangles,) where each entry is the subdomain ID that the triangle belongs to. 
            Required if include_global_mesh is True to color the global mesh by subdomain membership.
        figsize : tuple, optional
            Figure size in inches (width, height) (default: (6, 6)).
        dpi : int, optional
            Resolution of the figure in dots per inch (default: 150).
        cmap : str, optional
            Colormap used for coloring the global mesh by subdomain membership (default: "Set3").
        ncols : int, optional
            Number of columns in the subplot grid (default: 3).
        include_global_mesh : bool, optional
            If True, include a subplot of the global mesh colored by subdomain membership (default: True).
        show_vertex_markers : bool, optional
            If True, plot the vertices as markers on all subplots (default: True).
        show_node_numbers : bool, optional
            If True, annotate the vertices with their numbers (default: True).
        show_element_numbers : bool, optional
            If True, annotate the triangles with their numbers (default: True).
        axis_off : bool, optional
            If True, turn off the axis for all subplots (default: True).
        save_path : str, optional
            If provided, saves the figure to the given path (e.g., "subdomains.png" or "subdomains.pdf"). 
            If None, the figure is not saved (default: None).
        """
        nsub = len(subdomains)
        nplots = nsub + 1 if include_global_mesh else nsub
        nrows = (nplots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
        axes = axes.flatten()
        if include_global_mesh:
            ax = axes[0]
            assert membership is not None, "membership array is required to plot global mesh with subdomain coloring"
            triang = mtri.Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)
            ax.tripcolor(triang, membership, edgecolors = 'k', cmap = cmap)
            self._vertex_numbers(self.mesh, ax) if show_node_numbers else None
            self._triangle_numbers(self.mesh, ax) if show_element_numbers else None
            self._vertex_markers(self.mesh, ax) if show_vertex_markers else None
            ax.axis('off') if axis_off else None
            ax.set_title("Global Mesh")
            ax.set_aspect('equal')
        for i, (domainid, submesh) in enumerate(subdomains.items()):
            ax = axes[i + 1] if include_global_mesh else axes[i]
            ax.triplot(submesh.vertices[:, 0], submesh.vertices[:, 1], triangles = submesh.elements, 
                       color = 'k', linewidth = 0.5, alpha = 0.7)
            self._vertex_numbers(submesh, ax) if show_node_numbers else None
            self._triangle_numbers(submesh, ax) if show_element_numbers else None
            self._vertex_markers(submesh, ax) if show_vertex_markers else None
            ax.axis('off') if axis_off else None
            ax.set_title(f"Subdomain {domainid}")
            ax.set_aspect('equal')
        plt.tight_layout()
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
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

    def visualize(self,
                xlabel: str = "Iteration number",
                ylabel: str = r"$\| u_h - u \|_{L^2}$",
                title: str = r"Iteration vs $L^{2}$ Error",
                grid: bool = True,
                logscale: bool = False,
                cmap: str = 'viridis',
                levels: int = 50,
                figsize=(6, 5),
                dpi: int = 150,
                save_path: Optional[str] = None,
                **kwargs):
        """
        Visualize the FEM solution for 1D or 2D meshes and optionally save the figure.

        For 1D meshes, a line plot of the solution is produced.
        For 2D meshes, a filled contour plot over a triangulation is generated.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis (default: "Iteration number").
        ylabel : str, optional
            Label for the y-axis (default: r"$\\| u_h - u \\|_{L^2}$").
        title : str, optional
            Title of the plot (default: r"Iteration vs $L^{2}$ Error").
        grid : bool, optional
            If True, display a background grid (default: True).
        logscale : bool, optional
            If True, apply logarithmic scaling:
            - 1D: logarithmic y-axis
            - 2D: logarithmic color scaling (uses |u| + ε to avoid log(0)) (default: False).
        cmap : str, optional
            Colormap used for 2D contour plots (default: 'viridis').
        levels : int, optional
            Number of contour levels for 2D plots (default: 50).
        figsize : tuple, optional
            Figure size in inches (width, height) (default: (6, 5)).
        dpi : int, optional
            Resolution of the figure in dots per inch (default: 150).
        save_path : str, optional
            If provided, saves the figure to the given path (e.g., "plot.png" or "plot.pdf").
            If None, the figure is not saved (default: None).
        **kwargs : dict
            Additional keyword arguments passed to the plotting functions:
            - 1D: matplotlib.pyplot.plot (e.g., color='b', linestyle='-')
            - 2D: matplotlib.pyplot.tricontourf (e.g., edgecolors='k')

        Raises
        ------
        ValueError
            If the mesh dimension is not supported (only 1D and 2D are supported).

        Notes
        -----
        - For FEM consistency, solution values are flattened using `.ravel()`.
        - Log scaling in 2D uses `|u| + 1e-14` to avoid numerical issues.
        - Saving with `.pdf` is recommended for publication-quality (vector graphics).
        """
        plt.figure(figsize=figsize, dpi=dpi)
        if self.mesh.dim == 1:
            x = self.mesh.vertices
            u = self.u.ravel()
            plt.plot(x, u, **kwargs)
            if logscale:
                plt.yscale('log')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
        elif self.mesh.dim == 2:
            from matplotlib.colors import LogNorm
            x = self.mesh.vertices[:, 0]
            y = self.mesh.vertices[:, 1]
            u = self.u.ravel()
            triang = mtri.Triangulation(x, y, self.mesh.elements)
            if logscale:
                u_plot = np.abs(u) + 1e-14
                contour = plt.tricontourf(triang, u_plot, levels=levels, cmap=cmap, norm=LogNorm(vmin=u_plot.min(), vmax=u_plot.max()))
            else:
                contour = plt.tricontourf(triang, u, levels=levels, cmap=cmap)
            plt.triplot(triang, color='k', linewidth=0.3, alpha=0.5)
            plt.colorbar(contour, label='Solution $u$')
            plt.gca().set_aspect('equal')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
        else:
            raise ValueError("Unsupported mesh dimension for visualization")

        # Common settings
        if grid:
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(direction='in', which='both', top=True, right=True)
        plt.tight_layout()

        # Save figure
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

        plt.show()

    def visualize_3d(self, cmap = 'viridis', xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'u', title: str = 'FEM Solution 3D Surface'):
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
        fig = plt.figure(figsize=(8,6), dpi = 150)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_trisurf(triang, z, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Solution u')
    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

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
            self._plot_error_1D_static(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
        elif self.mesh.dim == 1 and self.ntime > 1:
            self._plot_error_1D_time(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)
        elif self.mesh.dim == 2 and self.ntime == 1:
            self._plot_error_2D_static(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, **kwargs)
        elif self.mesh.dim == 2 and self.ntime > 1:
            self._plot_error_2D_time(exact, figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, **kwargs)
        else:
            raise ValueError("Unsupported mesh dimension or time steps")
    
    @staticmethod
    def plot_convergence(error_history: np.ndarray | dict[int, np.ndarray], logscale: bool = True, figsize: tuple = (6, 4), dpi: int = 150,
                         xlabel: str = "Iteration number", ylabel: str = r"$\| u_h - u \|_{L^2}$", 
                         title: str = r"Iteration vs $L^{2}$ Error", grid: bool = True, save_path: Optional[str] = None,
                         styles: Optional[dict] = None, **kwargs):
        """
        Plot convergence history of Schwarz iterations. This function is intended to be used with the error history tracked during Schwarz iterations, 
        which can be stored in `self.error_history` and `self.error_subdomains`. It supports plotting the global error history as well as the error 
        history for each subdomain in one plot if available.

        Parameters
        ----------
        error_history : np.ndarray or dict[int, np.ndarray]
            If a numpy array of shape (niter,), it is treated as the global error history.
            If a dictionary of shape {domainID: np.ndarray}, it is treated as the subdomain error history, where each value is a numpy array of shape (niter,).
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
        grid : bool, default True
            If True, display a grid on the plot.
        save_path : str, optional
            If provided, saves the figure to the given path (e.g., "convergence_plot.png" or "convergence_plot.pdf"). If None, the figure is not saved.
        **kwargs : dict
            Additional keyword arguments to pass to plt.plot / plt.semilogy
            e.g., marker='o', color='r', linewidth=2, linestyle='--', alpha=0.7
        """
        assert isinstance(error_history, (np.ndarray, dict)), "error_history must be a numpy array or a dictionary"
        plt.figure(figsize=figsize, dpi=dpi)
        if isinstance(error_history, np.ndarray):
            iterations = range(1, len(error_history) + 1)
            if logscale:
                plt.semilogy(iterations, error_history, **kwargs)
            else:
                plt.plot(iterations, error_history, **kwargs)
            plt.xlim(1, len(error_history))
        else:
            for domainid, value in error_history.items():
                style = dict(kwargs)
                if styles and domainid in styles:
                    style.update(styles[domainid])
                if logscale:
                    plt.semilogy(range(1, len(value) + 1), value, label=f"Subdomain {domainid}", **style)
                else:
                    plt.plot(range(1, len(value) + 1), value, label=f"Subdomain {domainid}", **style)
            plt.xlim(1, len(next(iter(error_history.values()))))
            plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if grid:
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(direction='in', which='both', top=True, right=True)
        plt.tight_layout()
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.show()

    # add vtk versions as well