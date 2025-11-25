# main.py
import numpy as np
import mesh
import visualize 
from triangle import triangulate

# Create a simple square mesh
vert = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh_square = mesh.Mesh(vert, options = 'qa0.001')

# Create a donut mesh with a hole in the center
def donut_mesh():
    # computes vertices and triangles for a donut mesh by determining maximum area of triangles
    def circle(N, R):
        i = np.arange(N)
        theta = i * 2 * np.pi / N
        pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
        seg = np.stack([i, i + 1], axis=1) % N
        return pts, seg

    pts0, seg0 = circle(30, 1.4)
    pts1, seg1 = circle(16, 0.6)
    pts = np.vstack([pts0, pts1])
    seg = np.vstack([seg0, seg1 + seg0.shape[0]])
    return pts, seg, [[0, 0]]

vertices, segments, holes = donut_mesh()
mesh_donut = mesh.Mesh(vertices = vertices, segments = segments, holes = holes, options = 'qpa0.01')
print(mesh_donut.holes)

# Get information about the mesh
print(mesh_donut.info()) 

# Get information about subdomains
submeshes, membership = mesh_donut.decompose(10)
for submesh in submeshes:
    print(submesh.info())

# Create visualizer and show decomposition into 10 subdomains 
visualizer_donut = visualize.MeshVisualizer(mesh_donut)
visualizer_donut.visualize(visualizer_donut.carray_decomposition(10))

# Create visualizer and show decomposition into 10 subdomains 
submeshes, membership = mesh_square.decompose(20)
for submesh in submeshes:
    print(submesh.info())

visualizer_sq = visualize.MeshVisualizer(mesh_square)
visualizer_sq.visualize(visualizer_sq.carray_decomposition(20))


