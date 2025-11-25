import numpy as np
import matplotlib.pyplot as plt
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
        _, membership = self.mesh.decompose(n)  # Example with n subdomains
        return np.array(membership)
    
    def visualize(self, carray: np.ndarray):
        plt.figure(figsize = (6,6), dpi = 150)
        plt.tripcolor(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], triangles = self.mesh.elements, facecolors = carray, edgecolors = "k")
        plt.colorbar()
        plt.show()