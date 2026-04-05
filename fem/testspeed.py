import numpy as np
import time
from fem.mesh import Mesh
from fem.femspace import FEMSpace

# vertices = np.array([[0,0],[1,0],[1,1],[0,1]])
# segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
# segment_markers = np.array([1, 2, 3, 4])
# mesh2D = Mesh(vertices = vertices, segments = segments, segment_markers = segment_markers, options = 'pqa0.01')

# # Finite element space of degree 1
# femspace2D = FEMSpace(mesh2D, degree = 1)

# start = time.time()
# submeshes, ltog, gtol, maps, _ = femspace2D.mesh.decompose(n = 2, overlap = 1, version = 1)
# print(maps[1])
# print(maps[2])
# print("Time taken for version 1 (in-place modification):", time.time() - start)


# Create a mesh - space mesh generation
vert1D = np.linspace(0, 1, 100)
mesh1D = Mesh(vertices = vert1D, dim = 1)

# Finite element space of degree 1
femspace1D = FEMSpace(mesh1D, domain = 'interval', degree = 1)

start = time.time()
submeshes, ltog, gtol, maps, _ = femspace1D.mesh.decompose(n = 2, overlap = 1, version = 1)
print(maps[1])
print(maps[2])
print("Time taken for version 1 (in-place modification):", time.time() - start)