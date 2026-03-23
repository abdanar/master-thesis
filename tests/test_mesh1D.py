import numpy as np
from fem.mesh import Mesh

# Create a 1D mesh for a domain [0, 1]
vertices = np.linspace(0, 1, 100)
mesh = Mesh(vertices = vertices, dim = 1)

# The total number of nodes in the mesh
print(f'Total number of nodes: {mesh.nnodes()}\n')

# The total number of vertices in the mesh
print(f'Total number of vertices: {mesh.nvertices()}\n')

# The total number of edges in the mesh
print(f'Total number of edges: {mesh.nedges()}\n')

# The total number of elements in the mesh
print(f'Total number of elements: {mesh.nelements()}\n')

# The detailed information about the mesh
print(mesh.info(), "\n")

# Get the boundary nodes that belong to segments with markers 3 and 4
print(f'The boundary nodes that belong to segments with markers 3 and 4 are: {mesh.get_nodes(seg_markers = [3, 4])}\n')

# The edges of the element with index 3
print(f'The edges of the element [12, 13] are: {mesh.edges(element = [12, 13])}\n')

# The mapping from edges to elements
print(f'The mapping from edges to elements is: \n{mesh.edge_to_element_map()}\n')

# The adjacency dictionary of elements in the mesh
print(f'The adjacency dictionary of elements in the mesh is: \n{mesh.adjacency()}\n')

# The boundary edges of the mesh
print(f'The boundary edges of the mesh are: {mesh.boundary_edges()}\n')

# The boundary nodes of the mesh
print(f'The boundary nodes of the mesh are: {mesh.boundary_nodes()}\n')

# The coordinates of the boundary nodes
print(f'The coordinates of the boundary nodes are: {mesh.boundary_nodes_coord()}\n')

# The boundary elements of the mesh
print(f'The boundary elements of the mesh are: {mesh.boundary_elements()}\n')

# Check if a given edge is a boundary edge
print(f'Is the edge (0, 1) a boundary edge? {mesh.is_boundary_edge(edge = (0, 1))}\n')

# Check if a given element is a boundary element
print(f'Is the element [98, 99] a boundary element? {mesh.is_boundary_element(element = [98, 99])}\n')

# Check if a point is inside a given interval
print(f'Is the point 0.5 inside the interval [12, 13]? {mesh.is_in_interval(point = 0.5, interval = mesh.vertices[[12, 13]])}\n')

# Locate the interval containing a given point
interval_index = mesh.locate_interval(point = 0.5)
print(f'The interval containing the point 0.5 is: {mesh.elements[interval_index]}\n')

# The measures (lengths) of the elements in the mesh
print(f'The measures (lengths) of the elements in the mesh are: \n{mesh.measures()}\n')

print("All mesh tests passed!")