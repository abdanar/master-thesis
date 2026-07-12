from typing import Literal, Optional
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from fem.femspace import FEMSpace
from fem.mesh import Mesh
from utils.logger import get_logger
from visualization.parula import parula

logger = get_logger(__name__)

# Configure matplotlib to use LaTeX for rendering text with support for mathematical symbols and equations, 
# and set the font to a serif style for a more traditional look. This enhances the visual quality of the plots, 
# especially when displaying mathematical expressions in axis labels, titles, and annotations.
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage{amssymb}
        """,

        # thesis figure sizing
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })        
except:
    logger.debug("LaTeX not available, falling back to default matplotlib rendering.")

def plot_tri(x, y, z, ax = None, 
            wireframe: bool = False, wirewidth = 0.3, 
            contour: bool = False, levels = None, cline: bool = False, cbar: bool = True,
            xlabel: str = "", ylabel: str = "", zlabel: str = "", title: str = "",
            xlim: Optional[tuple] = None, ylim: Optional[tuple] = None, zlim: Optional[tuple] = None,
            logscale: bool = False, cmap = 'parula', elev: float = 20, azim: float = 225, 
            fig_kwargs = None, plot_kwargs = None):
    """
    Plot a 3D triangular surface, wireframe plot or 2D filled contour plot.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates of the vertices of the triangular mesh.
    y : np.ndarray
        1D array of y-coordinates of the vertices of the triangular mesh.
    z : np.ndarray
        1D array of z-coordinates (or values) corresponding to each vertex defined by x and y.
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        An existing 3D axis to plot on. If None, a new figure and axis will be created (default: None).
    wireframe : bool, default False
        Whether to plot the wireframe edges of the surface. If wireframe is True, only the wireframe will be plotted.
    wirewidth : float, default 0.3
        Line width for the wireframe edges.
    contour : bool, default False
        Whether to plot filled contours on the surface. If contour is True, only filled contours will be plotted.
    levels : int or array-like
        The number of contour levels or the specific levels to plot for the filled contour plot.
    cline : bool, default False
        Whether to draw contour lines on top of the filled contours when contour is True.
    cbar : bool, default True
        Whether to display a colorbar for the plot.
    xlabel : str, default ""
        Label for the x-axis.
    ylabel : str, default ""
        Label for the y-axis.
    zlabel : str, default ""
        Label for the z-axis.
    title : str, default ""
        Title of the plot.
    xlim : tuple, optional
        Limits for the x-axis as (xmin, xmax). If None, limits are determined automatically.
    ylim : tuple, optional
        Limits for the y-axis as (ymin, ymax). If None, limits are determined automatically.
    zlim : tuple, optional
        Limits for the z-axis as (zmin, zmax). If None, limits are determined automatically.
    logscale : bool, default False
        Whether to use a logarithmic scale for the color mapping of the wireframe.
    cmap : colormap, default 'parula'
        Colormap to use for coloring the surface or wireframe.
        Note that parula is not a built-in colormap in matplotlib, so it is defined in the
        visualization.parula module. You can replace it with any other colormap available 
        in matplotlib (e.g., 'viridis', 'plasma', 'inferno', etc.) or a custom colormap.
    elev : float, default 20
        Elevation angle in the z plane for the 3D plot.
    azim : float, default 225
        Azimuth angle in the x,y plane for the 3D plot.
    fig_kwargs : dict, optional
        Additional keyword arguments for the figure.
    plot_kwargs : dict, optional
        Additional keyword arguments for the surface, wireframe, or filled contour plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis object containing the plot.
    """
    assert not (wireframe and contour), "Cannot plot both wireframe and filled contour at the same time. Please choose either wireframe or contour."
    # Use existing axis or create new one
    if ax is None:
        fig = plt.figure(**(fig_kwargs if fig_kwargs is not None else {}))
        ax = fig.add_subplot(projection='3d' if not contour else None)
    else:
        fig = ax.figure  # reuse existing figure
    
    # Plot the triangular surface, wireframe, or filled contour based on the provided parameters.
    cmap = parula() if cmap == 'parula' else cmap
    z = np.abs(z) + 1e-12 if logscale else z
    norm = colors.LogNorm() if logscale else None
    if wireframe:
        surf = ax.plot_trisurf(x, y, z, cmap = colors.ListedColormap(['white']), norm=norm, linewidth = wirewidth, shade = False, **(plot_kwargs if plot_kwargs else {}))
        m = cm.ScalarMappable(norm = surf.norm, cmap = cmap) 
        surf.set_edgecolors(m.to_rgba(surf.get_array()))
    elif contour:
        surf = ax.tricontourf(x, y, z, levels = levels, cmap = cmap, norm = norm, **(plot_kwargs if plot_kwargs else {}))
        ax.tricontour(x, y, z, levels = levels, colors = 'k', linewidths = 0.5) if cline else None
    else:
        surf = ax.plot_trisurf(x, y, z, cmap = cmap, norm = norm, **(plot_kwargs if plot_kwargs else {}))
        
    # Set axis labels and title for the plot.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set axis limits if provided, otherwise they will be determined automatically by matplotlib
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None

    if not contour:
        # Set the background pane colors to transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Set the background grid line widths
        ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.3

        ax.set_zlabel(zlabel, rotation = 90)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlim(zlim) if zlim is not None else None    
        ax.view_init(elev=elev, azim=azim)
    else:
        fig.colorbar(surf, ax = ax) if cbar else None
    
    return fig, ax

def plot_contour(X, Y, Z, ax = None, levels = None, cline: bool = False, cbar: bool = True,
                xlabel: str = "", ylabel: str = "", title: str = "",
                xlim: Optional[tuple] = None, ylim: Optional[tuple] = None,
                logscale: bool = False, fig_kwargs = None, plot_kwargs = None):
    """
    Plot 2D filled contour plot. 

    Parameters
    ----------
    X : np.ndarray
        The x-coordinates of the data points.
    Y : np.ndarray
        The y-coordinates of the data points.
    Z : np.ndarray
        The height values corresponding to each (x, y) point, which will be represented as contour levels in the plot.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes are created.
    levels : int or array-like
        The number of contour levels or the specific levels to plot.
    cline : bool, optional
        Whether to draw contour lines on top of the filled contours.
    cbar : bool, optional
        Whether to display a colorbar for the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    title : str, optional
        The title of the plot.
    xlim : tuple, optional
        The limits for the x-axis.
    ylim : tuple, optional
        The limits for the y-axis.
    logscale : bool, optional
        Whether to use a logarithmic scale for the color mapping of the contour levels. 
        If True, a logarithmic normalization is applied to the Z values for coloring the contours.
    fig_kwargs : dict, optional
        Additional keyword arguments for the figure.
    plot_kwargs : dict, optional
        Additional keyword arguments for the filled contour plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    # Use existing axis or create new one
    if ax is None:
        fig, ax = plt.subplots(**(fig_kwargs if fig_kwargs else {}))
    else:
        fig = ax.figure  # reuse existing figure

    # Add a small constant to Z to avoid issues with logarithmic scaling when Z contains zero or negative values
    Z = np.abs(Z) + 1e-12 if logscale else Z

    # Plot the filled contour, and if logscale is True, use a logarithmic normalization for the color mapping
    contour = ax.contourf(X, Y, Z, levels = levels, norm = colors.LogNorm() if logscale else None, **(plot_kwargs if plot_kwargs else {}))
    ax.contour(X, Y, Z, levels = levels, colors = 'k', linewidths = 0.5) if cline else None

    # Set axis labels and title for the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set axis limits if provided, otherwise they will be determined automatically by matplotlib
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None

    # Add a color bar to the plot to indicate the mapping of contour colors to Z values
    fig.colorbar(contour, ax = ax) if cbar else None

    return fig, ax

def plot_wireframe(X, Y, Z, ax = None,
                   elev: float = 20, azim: float = 225,
                   xlabel: str = "", ylabel: str = "", zlabel: str = "", title: str = "",
                   xlim: Optional[tuple] = None, ylim: Optional[tuple] = None, zlim: Optional[tuple] = None,
                   logscale: bool = False, cmap = 'parula', wirewidth = 0.3, **kwargs):
    """
    Plot a 3D wireframe.

    Parameters
    ----------
    X : np.ndarray
        1D array of x-coordinates.
    Y : np.ndarray
        1D array of y-coordinates.
    Z : np.ndarray
        2D array of z-coordinates corresponding to X and Y.
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        An existing 3D axis to plot on. If None, a new figure and axis will be created (default: None).
    elev : float, default 20
        Elevation angle in the z plane for the 3D plot.
    azim : float, default 225
        Azimuth angle in the x,y plane for the 3D plot.
    xlabel : str, default ""
        Label for the x-axis.
    ylabel : str, default ""
        Label for the y-axis.
    zlabel : str, default ""
        Label for the z-axis.
    title : str, default ""
        Title of the plot.
    xlim : tuple, optional
        Limits for the x-axis as (xmin, xmax). If None, limits are determined automatically.
    ylim : tuple, optional
        Limits for the y-axis as (ymin, ymax). If None, limits are determined automatically.
    zlim : tuple, optional
        Limits for the z-axis as (zmin, zmax). If None, limits are determined automatically.
    logscale : bool, default False
        Whether to use a logarithmic scale for the color mapping of the wireframe.
    cmap : colormap, default 'parula'
        Colormap to use for coloring the wireframe.
        Note that parula is not a built-in colormap in matplotlib, so it is defined in the
        visualization.parula module. You can replace it with any other colormap available 
        in matplotlib (e.g., 'viridis', 'plasma', 'inferno', etc.) or a custom colormap.
    wirewidth : float, default 0.3
        Line width for the wireframe edges.
    **kwargs : dict
        Additional keyword arguments are passed to the .Figure constructor.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis object containing the plot.
    """
    # Convert X and Y to 2D arrays if they are 1D, which is required for the 3D plotting functions in matplotlib
    X, Y = np.meshgrid(X, Y)

    # Use existing axis or create new one
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.figure  # reuse existing figure
    
    # Create a 3D wireframe plot using the provided X, Y, Z data
    cmap = parula() if cmap == 'parula' else cmap
    Z = np.abs(Z) + 1e-12 if logscale else Z
    norm = colors.LogNorm() if logscale else None
    surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, shade = False, cmap = colors.ListedColormap(['white']), norm=norm, linewidth = wirewidth)
    m = cm.ScalarMappable(norm = surf.norm, cmap = cmap) 
    surf.set_edgecolors(m.to_rgba(surf.get_array()))
    
    # Set the background pane colors to transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Set the background grid line widths
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.3
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.3
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.3
    
    # Set axis labels and title for the plot.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel, rotation = 90)
    ax.zaxis.set_rotate_label(False)
    ax.set_title(title)

    # Set axis limits if provided, otherwise they will be determined automatically by matplotlib
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    ax.set_zlim(zlim) if zlim is not None else None

    # Set the viewing angle for the 3D plot using elevation and azimuth parameters to provide a better perspective of the data
    ax.view_init(elev=elev, azim=azim)

    # Adjust the layout of the figure
    fig.tight_layout()

    return fig, ax

def plot(x, y, ax = None,
        xlabel: str = "", ylabel: str = "", title: str = "",
        xlim: Optional[tuple] = None, ylim: Optional[tuple] = None, xticks = None,
        logscale: bool = False, grid: bool = True, fig_kwargs = None, plot_kwargs = None):
    """
    Plot a 2D line plot.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the data points.
    y : np.ndarray
        The y-coordinates of the data points.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot. If None, a new figure and axes are created.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    title : str, optional
        The title of the plot.
    xlim : tuple, optional
        The limits for the x-axis.
    ylim : tuple, optional
        The limits for the y-axis.
    xticks : array-like, optional
        The ticks for the x-axis.
    logscale : bool, optional
        Whether to use a logarithmic scale for the y-axis.
    grid : bool, optional
        Whether to display grid lines.
    fig_kwargs : dict, optional
        Additional keyword arguments for the figure.
    plot_kwargs : dict, optional
        Additional keyword arguments for the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    # Use existing axis or create new one
    if ax is None:
        fig, ax = plt.subplots(**(fig_kwargs if fig_kwargs else {}))
    else:
        fig = ax.figure  # reuse existing figure

    # Plot the data using a logarithmic scale for the y-axis if logscale is True, otherwise use a linear scale
    ax.plot(x, y, **(plot_kwargs if plot_kwargs else {}))

    # Set 2D grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7) if grid else None

    # Set axis labels and title for the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Set axis limits if provided, otherwise they will be determined automatically by matplotlib
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None

    # Set y-axis to logarithmic scale if logscale is True, which is useful for visualizing data that spans several orders of magnitude
    ax.set_yscale('log') if logscale else None

    # Set x-axis ticks if provided, otherwise they will be determined automatically by matplotlib
    ax.set_xticks(xticks) if xticks is not None else None

    # Set tick parameters to have ticks pointing inward and enable ticks on the top and right sides of the plot
    ax.tick_params(direction='in', which='both', top=True, right=True)

    # Adjust the layout of the figure to prevent overlap of elements and ensure a clean presentation
    fig.tight_layout()

    return fig, ax

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
                title: str = "", xlabel: str = "", ylabel: str = "", facecolors: Optional[np.ndarray] = None,
                edgecolor = 'k', linewidth = 0.5, show_vertex_markers: bool = False, show_node_numbers: bool = False, 
                show_element_numbers: bool = False, show_color_bar: bool = False, axis_off: bool = True, 
                cbar_fraction: float = 0.05, cbar_pad: float = 0.05, **kwargs):
        """
        Plot the triangular mesh with optional coloring and annotations for vertices and elements.

        Parameters
        ----------
        carray : np.ndarray, optional
            An array of shape (ntriangles,) to color the triangles.
            If None, no coloring is applied (default: None).
        figsize : tuple, optional
            Figure size in inches (width, height) (default: (6, 6)).
        dpi : int, optional
            Resolution of the figure in dots per inch (default: 150).
        title : str, optional
            Title of the plot (default: "").
        xlabel : str, optional
            Label for the x-axis (default: "").
        ylabel : str, optional
            Label for the y-axis (default: "").
        facecolors : np.ndarray, optional
            An array of shape (ntriangles, 4) representing RGBA colors for 
            the triangles. If provided, it overrides carray for coloring (default: None).
        edgecolor : str, optional
            Color for the edges of the triangles (default: 'k' for black).
        linewidth : float, optional
            Line width for the edges of the triangles (default: 0.5).
        show_vertex_markers : bool, optional
            If True, plot the vertices as markers (default: False).
        show_node_numbers : bool, optional
            If True, annotate the vertices with their numbers (default: False).
        show_element_numbers : bool, optional
            If True, annotate the elements with their numbers (default: False).
        show_color_bar : bool, optional
            If True, display a color bar when coloring is applied (default: False).
        axis_off : bool, optional
            If True, turn off the axis (default: True).
        cbar_fraction : float, optional
            Fraction of the original axes to use for the colorbar (default: 0.05).
        cbar_pad : float, optional
            Padding between the axes and colorbar (default: 0.05).
        **kwargs : dict, optional
            Additional keyword arguments to pass to the plotting functions.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.
        """
        import matplotlib.colors as mcolors
        import colorcet as cc

        # Validate input shapes and types
        ntri = self.mesh.elements.shape[0]
        assert carray is None or carray.shape[0] == ntri, f"carray must have shape ({ntri},) matching the number of triangles in the mesh"
        if facecolors is not None:
            facecolors = np.asarray(facecolors)
            assert facecolors.shape == (ntri, 4), f"facecolors must have shape ({ntri}, 4) representing RGBA colors for each triangle"
        
        # Determine face colors for the triangles based on carray or facecolors input
        fc = None
        if carray is not None:
            if facecolors is not None:
                # user already provided RGBA → ignore labels
                fc = facecolors
            else:
                # convert labels → RGBA using glasbey
                palette = np.array([mcolors.to_rgba(c) for c in cc.glasbey])
                fc = palette[carray]

        # Create a figure and axis for plotting the mesh. If carray is provided, use it to color the triangles; otherwise, plot the mesh without coloring.
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        triangulation = mtri.Triangulation(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.elements)
        if carray is None:
            ax.triplot(triangulation, color = edgecolor, linewidth = linewidth, **kwargs)
            tpc = None
        else:        
            import matplotlib.collections as mcoll
            pc = mcoll.PolyCollection(self.mesh.vertices[self.mesh.elements], facecolors=fc, edgecolors=edgecolor, linewidths=linewidth, **kwargs)
            ax.add_collection(pc)
            tpc = pc
        
        # Annotattions
        if show_node_numbers:
            self._vertex_numbers(self.mesh, ax)
        if show_element_numbers:
            self._triangle_numbers(self.mesh, ax)
        if show_vertex_markers:
            self._vertex_markers(self.mesh, ax)

        # Add a color bar to the plot if coloring is applied (change this to a discrete color bar if carray contains categorical labels)
        if tpc is not None and show_color_bar:
            norm = mcolors.Normalize(vmin=np.min(carray), vmax=np.max(carray))
            sm = cm.ScalarMappable(norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
        
        # Add style elements to the plot
        if axis_off:
            ax.axis('off')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')

        return fig, ax

    def plot_subdomains(self, subdomains: dict[int, Mesh], membership: Optional[np.ndarray] = None, 
                        figsize: tuple = (6, 6), dpi: int = 150, cmap: str = "Set3", 
                        ncols: int = 3, nrows: int = 1, include_global_mesh: bool = True, show_vertex_markers: bool = False, 
                        show_node_numbers: bool = False, show_element_numbers: bool = False,
                        axis_off = True):
        """
        Visualize the subdomain decomposition of the mesh supporting both non-overlapping and 
        overlapping subdomains. Optionally include the global mesh with subdomains colored by membership.

        This function creates a subplot for each subdomain and optionally the global mesh. The global mesh is colored 
        according to the membership array of non-overlapping subdomains, while the subdomains are outlined with their 
        own triangles. Node and element numbers can be annotated for clarity.

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
        nrows : int, optional
            Number of rows in the subplot grid (default: 1).
        include_global_mesh : bool, optional
            If True, include a subplot of the global mesh colored by subdomain membership (default: True).
        show_vertex_markers : bool, optional
            If True, plot the vertices as markers on all subplots (default: False).
        show_node_numbers : bool, optional
            If True, annotate the vertices with their numbers (default: False).
        show_element_numbers : bool, optional
            If True, annotate the triangles with their numbers (default: False).
        axis_off : bool, optional
            If True, turn off the axis for all subplots (default: True).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the subplots.
        axes : np.ndarray
            An array of axes objects for each subplot.
        """
        nsub = len(subdomains)
        nplots = nsub + 1 if include_global_mesh else nsub
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
        return fig, axes

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
                slabel: Optional[str] = None, xlabel: str = r'$x$', ylabel: str = 'y', zlabel: str = 'z', title: str = 'Solution', 
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