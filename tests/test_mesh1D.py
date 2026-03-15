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
print(f'The edges of the element [24, 21, 7] are: {mesh.edges(element = [24, 21, 7])}\n')

# The mapping from edges to elements
print(f'The mapping from edges to elements is: \n{mesh.edge_to_element_map()}\n')

# The adjacency dictionary of elements in the mesh
print(f'The adjacency dictionary of elements in the mesh is: \n{mesh.adjacency()}\n')

# The boundary edges of the mesh
print(f'The boundary edges of the mesh are: {mesh.boundary_edges()}\n')

# The boundary nodes of the mesh
print(f'The boundary nodes of the mesh are: {mesh.boundary_nodes()}\n')

# The coordinates of the boundary nodes
print(f'The coordinates of the boundary nodes are: \n{mesh.boundary_nodes_coord()}\n')

# The boundary elements of the mesh
print(f'The boundary elements of the mesh are: {mesh.boundary_elements()}\n')

# Check if a given edge is a boundary edge
print(f'Is the edge (3, 4) a boundary edge? {mesh.is_boundary_edge(edge = (3, 4))}\n')

# Check if a given element is a boundary element
print(f'Is the element [14, 15, 22] a boundary element? {mesh.is_boundary_element(element = [14, 15, 22])}\n')

# Check if a point is inside a given triangle
print(f'Is the point [0.5, 0.5] inside the triangle [14, 15, 22]? {mesh.is_in_triangle(point = [0.5, 0.5], triangle = mesh.vertices[[14, 15, 22]])}\n')

# Locate the triangle containing a given point
print(f'The triangle containing the point [0.5, 0.5] is: {mesh.locate_triangle(point = [0.5, 0.5])}\n')

# The measures (areas) of the elements in the mesh
print(f'The measures (areas) of the elements in the mesh are: \n{mesh.measures()}\n')

print("All mesh tests passed!")