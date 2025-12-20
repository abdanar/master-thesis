import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mesh

class MeshVisualizer:
    def __init__(self, meshobj: mesh.Mesh):
        if not isinstance(meshobj, mesh.Mesh):
            raise TypeError("MeshVisualizer expects a Mesh object")
        self.mesh = meshobj

    # Build a color array representing the areas of the triangles in the mesh.
    def carray_areas(self) -> np.ndarray:
        return self.mesh.areas()

    # Build a color array where boundary triangles get color `1` and interior triangles get color `0`.
    def carray_boundary(self) -> np.ndarray:
        colors = np.zeros(self.mesh.elements.shape[0], dtype=int)
        bdtriangles = self.mesh.boundary_triangles()
        for i, triangle in enumerate(self.mesh.elements):
            if tuple(triangle) in bdtriangles:
                colors[i] = 1
        return colors
    
    # Build a color array representing the subdomain decomposition of the mesh.
    def carray_decomposition(self, n: int) -> np.ndarray:
        _, _, membership = self.mesh.decompose(n)  # Example with n subdomains
        return np.array(membership)
    
    def visualize(self, carray: np.ndarray):
        plt.figure(figsize = (6,6), dpi = 150)
        plt.tripcolor(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], triangles = self.mesh.elements, facecolors = carray, edgecolors = "k")
        plt.colorbar()
        plt.show()

class SolutionVisualizer:
    def __init__(self, meshobj: mesh.Mesh, u: np.ndarray, dt = None):

        """
        Parameters
        ----------
        meshobj : Mesh
            Your FEM mesh
        u : np.ndarray
            Solution array, shape (ndofs, ntime)
        dt : float, optional
            Time step size, default 1.0
        """

        self.mesh = meshobj
        self.u = u
        if u.ndim == 1:
            u = u[:, np.newaxis]
        self.u = u
        self.dt = dt
        self.ntime = u.shape[1]

    def visualize(self, cmap = 'viridis', levels = 50):

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

    def visualize_3d_time(self, cmap='viridis'):
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
            ax.set_title(f'FEM Solution 3D Surface | t = {step * self.dt:.3f}')
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()
    