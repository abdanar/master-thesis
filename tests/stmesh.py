import triangle as tr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from fem.mesh import Mesh
from visualization.visualize import MeshVisualizer

vert = np.array([[0,0],[4,0],[4,1],[0,1]])
mesh_square = Mesh(vert, options = 'st, dx=0.1, dy=0.1')

def plot(vertices, elements):
    tri = Triangulation(vertices[:,0], vertices[:,1], elements)
    plt.figure(figsize=(6,5))
    plt.triplot(tri, lw=1)
    plt.gca().set_aspect('equal')
    plt.title("Mesh")
    plt.show()

plot(mesh_square.vertices, mesh_square.elements)

mesh_vis = MeshVisualizer(meshobj = mesh_square)

mesh_vis.visualize(carray = mesh_vis.carray_decomposition(2))
