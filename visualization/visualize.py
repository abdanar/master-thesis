from typing import Optional, Literal
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.widgets import Slider
from fem.mesh import Mesh
from fem.femspace import FEMSpace
from matplotlib.colors import Normalize

from utils.logger import get_logger
logger = get_logger(__name__)

# Configure matplotlib to use LaTeX for rendering text with support for mathematical symbols and equations, 
# and set the font to a serif style for a more traditional look. This enhances the visual quality of the plots, 
# especially when displaying mathematical expressions in axis labels, titles, and annotations.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
    """})

class MeshVisualizer:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    # Build a color array representing the areas of the triangles in the mesh.
    def carray_areas(self) -> np.ndarray:
        return self.mesh.measures()

    # Build a color array where boundary triangles get color `1` and interior triangles get color `0`.
    def carray_boundary(self) -> np.ndarray:
        colors = np.zeros(self.mesh.elements.shape[0], dtype = int)
        elements = self.mesh.elements
        bdelements = self.mesh.boundary_elements()
        element_view = elements.view([('', elements.dtype)] * elements.shape[1]).ravel()
        bdelement_view = bdelements.view([('', bdelements.dtype)] * bdelements.shape[1]).ravel()
        indices = np.nonzero(np.isin(element_view, bdelement_view))[0]
        colors[indices] = 1
        return colors
        
    # Annotate the triangles with their numbers.
    def _triangle_numbers(self, mesh: Mesh, ax, color = 'black', fontsize = 6):        
        for i, element in enumerate(mesh.elements):
            centroid = mesh.vertices[element].mean(axis = 0)
            ax.text(centroid[0], centroid[1], str(i), color = color, fontsize = fontsize, ha = 'center', va = 'center', clip_on = True)
    
    # Annotate the vertices with their numbers.
    def _vertex_numbers(self, mesh: Mesh, ax, color = 'black', fontsize = 7):
        for i, (x, y) in enumerate(mesh.vertices):
            ax.text(x, y, str(i), color = color, fontsize = fontsize, ha = 'right', va = 'bottom', clip_on = True)

    # Plot the mesh vertices as dots.
    def _vertex_markers(self, mesh: Mesh, ax, color = 'black', size = 15):
        ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], s = size, c = color)

    def plot_mesh(self, carray: Optional[np.ndarray] = None, figsize: tuple = (6, 6), dpi: int = 150, 
                show_vertex_markers: bool = True, show_node_numbers: bool = True, show_element_numbers: bool = True,
                show_color_bar: bool = False, axis_off: bool = True, cbar_fraction: float = 0.05, cbar_pad: float = 0.05,
                save_path: Optional[str] = None, **kwargs):
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
        show_color_bar : bool, optional
            If True, display a color bar when coloring is applied (default: False).
        axis_off : bool, optional
            If True, turn off the axis (default: True).
        cbar_fraction : float, optional
            Fraction of the original axes to use for the colorbar (default: 0.05).
        cbar_pad : float, optional
            Padding between the axes and colorbar (default: 0.05).
        save_path : str, optional
            If provided, save the figure to the specified path (default: None).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting functions.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if carray is None:
            ax.triplot(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements, color='k', linewidth=0.5, **kwargs)
            tpc = None
        else:
            tpc = ax.tripcolor(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements, facecolors = carray, edgecolors = 'k', **kwargs)
        if show_node_numbers:
            self._vertex_numbers(self.mesh, ax)
        if show_element_numbers:
            self._triangle_numbers(self.mesh, ax)
        if show_vertex_markers:
            self._vertex_markers(self.mesh, ax)
        if tpc is not None and show_color_bar:
            cbar = fig.colorbar(tpc, ax=ax, fraction = cbar_fraction, pad = cbar_pad)
        if axis_off:
            ax.axis('off')
        ax.set_aspect('equal')
        fig.tight_layout()
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
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
    def __init__(self, mesh: Mesh, u: np.ndarray, dt: Optional[float] = None, femspace: Optional[FEMSpace] = None):
        """
        Parameters
        ----------
        mesh : Mesh
            The mesh object the solution is defined on.
        u : np.ndarray
            FEM solution array.
            - For steady-state problems, should be of shape (nspace,) where nspace is the number of mesh vertices.
            - For time-dependent problems, should be of shape (nspace, ntime) where ntime is the number of time steps.
        dt : float, optional
            Time step size
        femspace : FEMSpace, optional
            The finite element space associated with the solution (default: None).
        """
        self.mesh = mesh
        self.u = u
        self.dt = dt
        self.femspace = femspace
        self.ntime = u.shape[1] if dt is not None and u.ndim == 2 else 1
        assert u.shape[0] == mesh.vertices.shape[0], "The number of rows in u must match the number of mesh vertices"
        assert (dt is not None and u.ndim == 2) or (dt is None and u.ndim == 1), "For time-dependent problems, u must be 2D with shape (nspace, ntime). For steady-state problems, u must be 1D with shape (nspace,)"

    def _eval_at_points(self, eval_points: np.ndarray, t_index: Optional[int] = None, coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        This function evaluates the FEM solution `self.u` at given points `eval_points` using the provided `self.femspace`.
        It uses the shape functions of the finite element space to compute the solution values at the specified points.

        Parameters
        ----------
        eval_points : np.ndarray
            Points at which to evaluate the solution.
            - If the mesh is 1D, should be of shape (n_eval,).
            - If the mesh is 2D, should be of shape (n_eval, 2) where each row is (x, y).
        t_index : int, optional
            Time step index for time-dependent problems (default: None). If None, it evaluates the steady-state solution.
        coeffs : np.ndarray, optional
            Coefficients to use for evaluation instead of `self.u`. Should have the same shape as `self.u` (default: None).
        """
        assert self.femspace is not None, "FEM space is required to evaluate at given eval_points"
        values = np.zeros(eval_points.shape[0])
        if coeffs is not None:
            assert coeffs.shape[0] == self.u.shape[0], "Coefficient array must have the same number of rows as u"
            weights = coeffs if t_index is None else coeffs[:, t_index]
        else:
            weights = self.u if t_index is None else self.u[:, t_index]
        for i, point in enumerate(eval_points):
            values[i] = self.femspace.evaluate_solution(U = weights, x = point)
        return values

    def _plot_1D_static(self, use_femspace: bool = False, eval_points: Optional[np.ndarray] = None, 
                        data: Optional[dict] = None, figsize: tuple = (7, 4), dpi: int = 100, 
                        slabel: Optional[str] = None , xlabel: str = r'$x$', ylabel: str = r'$u(x)$', title: str = 'FEM Solution',
                        xmin: Optional[float] = None, xmax: Optional[float] = None, ymin: Optional[float] = None, ymax: Optional[float] = None, 
                        styles: Optional[dict[str, dict]] = None, logscale: bool = False, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        if use_femspace:
            assert self.femspace is not None, "FEM space is required to evaluate at given eval_points"
            assert eval_points is not None, "eval_points must be provided when use_femspace is True"
            x_values = eval_points
            y_values = self._eval_at_points(eval_points)
        else:
            x_values = self.mesh.vertices
            y_values = self.u
        fig = plt.figure(figsize = figsize, dpi = dpi, constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(x_values, y_values, label=slabel, **kwargs)
        if data is not None:
            for key, values in data.items():
                if use_femspace:
                    values = self._eval_at_points(eval_points, coeffs = values) # type: ignore
                assert len(values) == len(x_values), f"{key} has wrong size"
                style = styles.get(key, {}) if styles else {}
                ax.plot(x_values, values, **style)
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logscale:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if data is not None:
            ax.legend()
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        ax.tick_params(direction='in', which='both', top=True, right=True)
        plt.show()

    def _plot_1D_time(self, use_femspace: bool = False, eval_points: Optional[np.ndarray] = None, 
                    data: Optional[dict] = None, figsize: tuple = (7, 4), dpi: int = 100, 
                    slabel: Optional[str] = None, xlabel: str = r'$x$', ylabel: str = r'$u$', title: str = 'Evolution of Solution', 
                    xmin: Optional[float] = None, xmax: Optional[float] = None, ymin: Optional[float] = None, ymax: Optional[float] = None, 
                    styles: Optional[dict] = None, logscale: bool = False, grid: bool = True, **kwargs):
        if use_femspace:
            assert self.femspace is not None, "FEM space is required to evaluate at given eval_points"
            assert eval_points is not None, "eval_points must be provided when use_femspace is True"
            values = np.zeros((eval_points.shape[0], self.ntime))
            for t in range(self.ntime): # precompute values at all time steps for efficiency
                values[:, t] = self._eval_at_points(eval_points, t_index = t)
            if data is not None:
                data_values = {}
                for label, arr in data.items():
                    assert arr.shape[1] == self.ntime, f"{label} wrong time dimension"
                    assert arr.shape[0] == self.u.shape[0], f"{label} wrong spatial dimension"
                    data_values[label] = np.zeros((eval_points.shape[0], self.ntime))
                    for t in range(self.ntime):
                        data_values[label][:, t] = self._eval_at_points(eval_points, t_index = t, coeffs = arr) # type: ignore
            x_values = eval_points
            y_values = values[:, 0]
        else:
            x_values = self.mesh.vertices
            y_values = self.u[:, 0]
            
        # Initial plot
        fig = plt.figure(figsize = figsize, dpi = dpi, constrained_layout=True)
        gs = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [20, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        line_main, = ax.plot(x_values, y_values, label=slabel, **kwargs)
        extra_lines = {}
        if data is not None:
            for key, arr in data.items():
                arr = arr if not use_femspace else data_values[key] # type: ignore use precomputed values for efficiency
                style = styles.get(key, {}) if styles else {}
                line, = ax.plot(x_values, arr[:, 0], **style)
                extra_lines[key] = line
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(direction = "in", which = "both", top = True, right = True)
        if data is not None:
            ax.legend()
        if logscale:
            ax.set_yscale("log")
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        
        # Slider
        slider_ax = fig.add_subplot(gs[1])
        slider_ax.set_facecolor("none")
        for spine in slider_ax.spines.values():
            spine.set_visible(False)
        slider_ax.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)
        slider = Slider(ax=slider_ax, label="Time step", valmin=0, valmax=self.ntime - 1, valinit=0, valstep=1, facecolor="#0c8b21")
        
        # Update function for slider
        def update(val):
            t = int(slider.val)
            if use_femspace:
                assert eval_points is not None, "eval_points must be provided when use_femspace is True"
                y = values[:, t]  # type:ignore use precomputed values for efficiency
            else:
                y = self.u[:, t]
            line_main.set_ydata(y)
            if data is not None:
                for key, line in extra_lines.items():
                    line.set_ydata(data[key][:, t] if not use_femspace else data_values[key][:, t]) # type: ignore use precomputed values for efficiency
            ax.set_ylabel(f"${ylabel}(x, {t * self.dt:.3f})$") # type: ignore
            # keep limits fixed (important!)
            if ymin is not None or ymax is not None:
                ax.set_ylim(bottom=ymin, top=ymax)
            if xmin is not None or xmax is not None:
                ax.set_xlim(left=xmin, right=xmax)
            fig.canvas.draw_idle()
        slider.on_changed(update)
        plt.show()

    def _plot_2D_static(self, use_femspace: bool = False, eval_points: Optional[np.ndarray] = None, 
                        plot_type: Literal["3d", "contour"] = "3d", contour_levels: int = 20, figsize: tuple = (7, 6), dpi: int = 100, 
                        xlabel: str = r'$x$', ylabel: str = r'$y$', zlabel: str = r'$u(x, y)$', title: str = '2D Steady-State Solution', 
                        xmin: Optional[float] = None, xmax: Optional[float] = None, ymin: Optional[float] = None, ymax: Optional[float] = None,
                        zmin: Optional[float] = None, zmax: Optional[float] = None, logscale: bool = False, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        
        assert plot_type in ["3d", "contour"], "plot_type must be either '3d' or 'contour'"
        
        # Evaluate solution at either mesh vertices or specified evaluation points using the FEM space for smoother visualization.
        if use_femspace:
            assert self.femspace is not None, "FEM space is required to evaluate at given eval_points"
            assert eval_points is not None, "eval_points must be provided when use_femspace is True"
            x_values = eval_points[:, 0]
            y_values = eval_points[:, 1]
            z_values = self._eval_at_points(eval_points)
        else:
            x_values = self.mesh.vertices[:, 0]
            y_values = self.mesh.vertices[:, 1]
            z_values = self.u

        # Color mapping normalization based on the range of z_values for consistent coloring across different plots.
        cmap = kwargs.pop("cmap", None) or "viridis"
        if logscale:
            z_values = np.log10(np.maximum(z_values, 1e-12))
        norm = Normalize(vmin=np.min(z_values), vmax=np.max(z_values))

        # Create the figure
        fig = plt.figure(figsize = figsize, dpi = dpi, constrained_layout=True)

        # Plot either a filled contour plot or a 3D surface plot based on the specified plot_type.
        if plot_type == "contour":
            ax = fig.add_subplot(111)
            surf = ax.tricontourf(x_values, y_values, z_values, levels = contour_levels, norm = norm, cmap = cmap, **kwargs)
        else:
            ax = fig.add_subplot(111, projection = '3d')
            surf = ax.plot_trisurf(x_values, y_values, z_values, norm = norm, cmap = cmap, **kwargs)

        # Set axis limits if provided, otherwise they will be determined automatically by matplotlib.
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        if plot_type == "3d" and (zmin is not None or zmax is not None):
            ax.set_zlim(bottom = zmin, top = zmax)

        # Set axis labels and title for the plot.
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plot_type == "3d":
            ax.set_zlabel(zlabel)
        ax.set_title(title)

        # Configure tick parameters to improve readability, especially when the plot is dense or has a lot of data points.
        ax.tick_params(direction='in', which='both', top=True, right=True)

        # Add a color bar to the plot to indicate the mapping of colors to solution values.
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
        
        # Apply logarithmic scaling to the z-axis for 3D plots if logscale is True.
        if logscale and plot_type == "3d":
            ax.set_zscale("log") # type: ignore log scale for z-axis in 3D plot
        
        # Add a grid to the plot for better visibility of the data points
        if grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Save the figure to the specified path if save_path is provided, ensuring that the directory exists.
        if save_path is not None:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

        plt.show()

    def _plot_2D_time(self, use_femspace: bool = False, eval_points: Optional[np.ndarray] = None, 
                            plot_type: Literal["3d", "contour"] = "3d", contour_levels: int = 20, figsize: tuple = (7, 6), dpi: int = 100, 
                            xlabel: str = r'$x$', ylabel: str = r'$y$', zlabel: str = r'$u(x, y)$', title: str = 'Evolution of Solution', 
                            xmin: Optional[float] = None, xmax: Optional[float] = None, ymin: Optional[float] = None, ymax: Optional[float] = None, 
                            zmin: Optional[float] = None, zmax: Optional[float] = None, logscale: bool = False, grid: bool = True, **kwargs):
        
        assert plot_type in ["3d", "contour"], "plot_type must be either '3d' or 'contour'"
        
        # Evaluate solution at either mesh vertices or specified evaluation points using the FEM space for smoother visualization.
        if use_femspace:
            assert self.femspace is not None, "FEM space is required to evaluate at given eval_points"
            assert eval_points is not None, "eval_points must be provided when use_femspace is True"
            values = np.zeros((eval_points.shape[0], self.ntime))
            for t in range(self.ntime):
                values[:, t] = self._eval_at_points(eval_points, t_index = t)
            x_values = eval_points[:, 0]
            y_values = eval_points[:, 1]
            z_values = values[:, 0]
        else:
            x_values = self.mesh.vertices[:, 0]
            y_values = self.mesh.vertices[:, 1]
            z_values = self.u[:, 0]

        # Color mapping normalization based on the range of z_values for consistent coloring across different plots.
        cmap = kwargs.pop("cmap", None) or "viridis"
        if logscale:
            z_values = np.log10(np.maximum(z_values, 1e-12))
        norm = Normalize(vmin=np.min(z_values), vmax=np.max(z_values))
        
        # Create the figure and plot either a filled contour plot or a 3D surface plot based on the specified plot_type.
        t0, z0 = 0, all_values[:, 0]
        fig = plt.figure(figsize = figsize, dpi = dpi, constrained_layout=True)
        gs = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [16, 1], hspace=0.25)

        # For the initial plot, we create either a contour or surface plot based on the specified plot_type.
        if plot_type == "contour":
            ax = fig.add_subplot(gs[0])
            surf = ax.tricontourf(x_values, y_values, z_values, levels = contour_levels, norm = norm, cmap = cmap, **kwargs)
        else:
            ax = fig.add_subplot(gs[0], projection='3d')
            surf = ax.plot_trisurf(x_values, y_values, z_values, cmap = cmap, norm = norm, **kwargs)
        
        # Set axis limits if provided, otherwise they will be determined automatically by matplotlib.
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        if plot_type == "3d" and (zmin is not None or zmax is not None):
            ax.set_zlim(bottom = zmin, top = zmax)

        # Set axis labels and title for the plot.    
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plot_type == "3d":
            ax.set_zlabel(zlabel)
        ax.set_title(title)

        # Add a color bar to the plot to indicate the mapping of colors to solution values.
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)

        # Configure tick parameters to improve readability, especially when the plot is dense or has a lot of data points.
        ax.tick_params(direction='in', which='both', top=True, right=True)
        
        # Apply logarithmic scaling to the z-axis for 3D plots if logscale is True.
        if logscale:
            ax.set_zscale("log") # type: ignore log scale for z-axis in 3D plot
        
        # Add a grid to the plot for better visibility of the data points
        if grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Slider
        slider_ax = fig.add_subplot(gs[1])
        slider_ax.set_facecolor("none")
        for spine in slider_ax.spines.values():
            spine.set_visible(False)
        slider_ax.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)
        slider = Slider(ax=slider_ax, label="Time step", valmin=0, valmax=self.ntime - 1, valinit=0, valstep=1, facecolor="#0c8b21")
        
        # keep reference to surface
        surf_container = {"surf": surf}

        # update function for slider to update the surface plot based on the selected time step. 
        def update(val):
            step = int(slider.val)
            # remove old surface only
            surf_container["surf"].remove()
            if use_femspace:
                assert eval_points is not None, "eval_points must be provided when use_femspace is True"
                z_values = values[:, step] # type: ignore use precomputed values for efficiency
            else:
                z_values = self.u[:, step]
            if plot_type == "contour":
                surf_container["surf"] = ax.tricontourf(x_values, y_values, z_values, levels = contour_levels, norm = norm, cmap = cmap, **kwargs)
            else:
                surf_container["surf"] = ax.plot_trisurf(x_values, y_values, z_values, norm = norm, cmap = cmap, **kwargs)
            if plot_type == "3d":
                ax.set_zlabel(f"$u(x, y, {step * self.dt:.3f})$") # type: ignore
            if zmin is not None or zmax is not None:
                ax.set_zlim(bottom=zmin, top=zmax)
            fig.canvas.draw_idle()
        slider.on_changed(update)

        plt.show()
    
    def plot(self, use_femspace: bool = False, eval_points: Optional[np.ndarray] = None, 
                plot_type: Literal["3d", "contour"] = "3d", contour_levels: int = 20, 
                data: Optional[dict] = None, styles: Optional[dict] = None, figsize: tuple = (7, 6), dpi: int = 100, 
                slabel: Optional[str] = None, xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'z', title: str = 'Solution', 
                xmin: Optional[float] = None, xmax: Optional[float] = None, ymin: Optional[float] = None, ymax: Optional[float] = None, 
                zmin: Optional[float] = None, zmax: Optional[float] = None, logscale: bool = False, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        """
        Plot the `self.u` values for 1D or 2D meshes, optionally plotting the error between `self.u` and 
        a reference solution `data`.
        
        The error is computed as `self.u - data` at mesh vertices and visualized either as 
        a line plot (1D) or a 3D surface plot (2D). For time-dependent problems, 
        an interactive time slider is provided to visualize at each time step.

        Parameters
        ----------
        use_femspace : bool, optional
            If True, evaluate the solution at the quadrature points of the FEM space for smoother visualization. 
            If False, visualize the solution at the mesh vertices. (default: False)
        eval_points : np.ndarray, optional
            Points at which to evaluate the solution when use_femspace is True.
            - For 1D meshes, should be of shape (n_eval,).
            - For 2D meshes, should be of shape (n_eval, 2) where each row is (x, y).
            Required if use_femspace is True.
        plot_type : Literal["3d", "contour"], optional
            For 2D meshes, specify whether to plot a 3D surface or a filled contour plot (default: "3d"). Ignored for 1D meshes.
        contour_levels : int, optional
            Number of contour levels to use when plot_type is "contour" (default: 20). Ignored for 1D meshes and 3D plots.
        data : dict, optional
            A dictionary of reference solutions to compare against, where keys are labels and values are numpy arrays of the same shape as `self.u`. 
        styles : dict, optional
            A dictionary mapping labels in `data` to style dictionaries for plotting (e.g., {'reference': {'label': 'Reference Solution', 'color': 'r', 'linestyle': '--'}}). Ignored if `data` is None.
        figsize : tuple, optional
                Figure size for the plot (default (7, 6)).
        dpi : int, optional
                Dots per inch for the plot (default 100).
        slabel : Optional[str], optional
                Label for the solution plot (default 'u').
        xlabel : str, optional
                Label for the x-axis (default 'x').
        ylabel : str, optional
                Label for the y-axis (default 'y').
        zlabel : str, optional
                Label for the z-axis (default 'u(x, y)').
        title : str, optional
                Title of the plot (default 'Solution').
        xmin, xmax, ymin, ymax, zmin, zmax : float, optional
                Axis limits for the plot. If None, limits are determined automatically (default None).
        logscale : bool, optional
                If True, use a logarithmic scale for the y-axis (default False).
        grid : bool, optional
                If True, display a grid on the plot (default True).
        save_path : str, optional
                If provided, saves the figure to the given path (e.g., "error_plot.png" or "error_plot.pdf"). 
                If None, the figure is not saved (default None).
        **kwargs : dict
                Additional keyword arguments to pass to the specific plotting functions, 
                e.g., cmap for colormap, levels for contour levels, etc.
        """
        if self.mesh.dim == 1 and self.ntime == 1:
            self._plot_1D_static(use_femspace, eval_points, data, figsize, dpi, slabel, xlabel, ylabel, title, 
                                xmin, xmax, ymin, ymax, styles, logscale, grid, save_path, **kwargs)
        elif self.mesh.dim == 1 and self.ntime > 1:
            self._plot_1D_time(use_femspace, eval_points, data, figsize, dpi, slabel, xlabel, ylabel, title, 
                               xmin, xmax, ymin, ymax, styles, logscale, grid, **kwargs)
        elif self.mesh.dim == 2 and self.ntime == 1:
            self._plot_2D_static(use_femspace=use_femspace, eval_points=eval_points, plot_type=plot_type, contour_levels=contour_levels, 
                                figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title,
                                logscale=logscale, grid=grid, save_path=save_path, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, **kwargs)
        elif self.mesh.dim == 2 and self.ntime > 1:
            self._plot_2D_time(use_femspace=use_femspace, eval_points=eval_points, plot_type=plot_type, contour_levels=contour_levels, 
                               figsize=figsize, dpi=dpi, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, 
                               logscale=logscale, grid=grid, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, **kwargs)
        else:
            raise ValueError("Unsupported mesh dimension or time steps")
    
    @staticmethod
    def plot_iteration(data: np.ndarray | dict, logscale: bool = True, figsize: tuple = (6, 4), dpi: int = 150,
                        xlabel: str = "Iteration number", ylabel: str = r"$\| u_h - u \|_{L^2}$", 
                        title: str = r"Iteration vs $L^{2}$ Error", grid: bool = True, save_path: Optional[str] = None,
                        styles: Optional[dict] = None, **kwargs):
        """
        Plot the iteration vs data values. 

        If data is a numpy array, it is plotted as a single line. If data is a dictionary, each entry is plotted as a separate 
        line with optional styles. This is useful when you want to visualize multiple obtained data values 

        Parameters
        ----------
        data : np.ndarray or dict
            If a numpy array of shape (niter,)
            If a dictionary of shape {figID: np.ndarray} where each value is a numpy array of shape (niter,).
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
            A dictionary mapping figure IDs to style dictionaries for plotting 
            (e.g., {'fig1': {'label': 'Figure 1', 'color': 'r', 'linestyle': '--'}, 'fig2': {'label': 'Figure 2', 'color': 'b', 'linestyle': '-'}}).
        grid : bool, default True
            If True, display a grid on the plot.
        save_path : str, optional
            If provided, saves the figure to the given path (e.g., "convergence_plot.png" or "convergence_plot.pdf"). If None, the figure is not saved.
        **kwargs : dict
            Additional keyword arguments to pass to plt.plot / plt.semilogy
            e.g., marker='o', color='r', linewidth=2, linestyle='--', alpha=0.7
        """
        assert isinstance(data, (np.ndarray, dict)), "data must be a numpy array or a dictionary"
        plt.figure(figsize=figsize, dpi=dpi)
        if isinstance(data, np.ndarray):
            iterations = range(1, len(data) + 1)
            if logscale:
                plt.semilogy(iterations, data, **kwargs)
            else:
                plt.plot(iterations, data, **kwargs)
            #plt.xlim(1, len(data))
        else:
            for domainid, value in data.items():
                style = dict(kwargs)
                if styles and domainid in styles:
                    style.update(styles[domainid])
                if logscale:
                    plt.semilogy(range(1, len(value) + 1), value, **style)
                else:
                    plt.plot(range(1, len(value) + 1), value, **style)
            #plt.xlim(1, len(next(iter(data.values()))))
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