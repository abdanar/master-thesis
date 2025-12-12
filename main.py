# main.py
import numpy as np
import mesh
import visualize 
from triangle import triangulate
from pdesolver import PoissonProblem

# Create a simple square mesh
vert = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh_square = mesh.Mesh(vert, options = 'qa0.001')

#visualizer_sq = visualize.MeshVisualizer(mesh_square)
#visualizer_sq.visualize(visualizer_sq.carray_boundary())

# Define a source function
def func(x, y):
    return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

# Define a Dirichlet boundary condition function
def g(x, y):
    return np.sin(np.pi*x) * np.sin(np.pi*y)

# Define a Dirichlet boundary condition dictionary
dirichlet_bc = dict()
for i, nodes in enumerate(mesh_square.vertices):
    dirichlet_bc[i] = g(nodes[0], nodes[1])

pde = PoissonProblem(mesh_square, func, dirichlet_bc, neumann_bc = None, robin_bc = None)
pde_solution = pde.solve()
print(pde_solution)

visualizer_pde = visualize.SolutionVisualizer(mesh_square, pde_solution)
visualizer_pde.visualize()



# # Create a donut mesh with a hole in the center
# def donut_mesh():
#     # computes vertices and triangles for a donut mesh by determining maximum area of triangles
#     def circle(N, R):
#         i = np.arange(N)
#         theta = i * 2 * np.pi / N
#         pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
#         seg = np.stack([i, i + 1], axis=1) % N
#         return pts, seg

#     pts0, seg0 = circle(30, 1.4)
#     pts1, seg1 = circle(16, 0.6)
#     pts = np.vstack([pts0, pts1])
#     seg = np.vstack([seg0, seg1 + seg0.shape[0]])
#     return pts, seg, [[0, 0]]

# vertices, segments, holes = donut_mesh()
# mesh_donut = mesh.Mesh(vertices = vertices, segments = segments, holes = holes, options = 'qpa0.01')
# print(mesh_donut.holes)

# # Get information about the mesh
# print(mesh_donut.info()) 

# # Get information about subdomains
# submeshes, membership = mesh_donut.decompose(10)
# for submesh in submeshes:
#     print(submesh.info())

# # Create visualizer and show decomposition into 10 subdomains 
# visualizer_donut = visualize.MeshVisualizer(mesh_donut)
# visualizer_donut.visualize(visualizer_donut.carray_decomposition(10))

# # Create visualizer and show decomposition into 10 subdomains 
# submeshes, membership = mesh_square.decompose(20)
# for submesh in submeshes:
#     print(submesh.info())

# visualizer_sq = visualize.MeshVisualizer(mesh_square)
# visualizer_sq.visualize(visualizer_sq.carray_decomposition(20))


