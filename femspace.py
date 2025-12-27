from mesh import Mesh
from refelement import ReferenceElement
from phyelement import PhysicalElement
import numpy as np
from collections import defaultdict

# --------------------------------------------------------------------------
# FEMSpace class
#
# Defines the finite element space over a given mesh, including support for 
# higher-degree Lagrange elements. Recall the Ciarlet definition of a finite element:
# A finite element is a triplet (K, P, N) where:
# - K : element domain (e.g., interval, triangle)
# - P : space of shape functions (e.g., Lagrange polynomials of degree p)
# - N : set of nodal variables (e.g., point evaluations at nodes)
# This class encapsulates these concepts and provides functionality to upgrade 
# meshes for higher-degree elements. It supports both 1D (interval) and 2D (triangular) meshes.
# --------------------------------------------------------------------------

class FEMSpace:
    def __init__(self, mesh: Mesh, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1):
        self.mesh = mesh.upgrade() if mesh.domainID == 0 else mesh # upgrade only if domainID == 0 (whole domain)
        self.degree = degree
        self.domain = domain
        self.space = space
        self.dim = mesh.dim
        
    def upgrade(self) -> Mesh:
        """
        Upgrade the mesh for higher-degree Lagrange finite elements and return a new Mesh object.
        
        The method supports both 1D (interval) and 2D (triangular) meshes.

        Returns
        -------
        Mesh
            A new Mesh instance representing the upgraded higher-order mesh.
            The returned mesh has updated vertices and element connectivity,
            and the attribute `mesh.degree` is set accordingly.

        Notes
        -----
        - Node ordering is fixed and consistent:
            **1D (interval):**
                [left vertex, interior nodes (left → right), right vertex]
            **2D (triangle):**
                [v0, v1, v2,
                edge nodes on (0,1),
                edge nodes on (1,2),
                edge nodes on (2,0),
                interior nodes]
        - Edge node indices start after the original vertices, i.e., at `self.nvertices()`.
        - Interior node indices start after all original vertices and edge nodes, 
            i.e., at `self.nvertices() + self.nedges() * (degree - 1)`.
        - This function uses dictionaries to track unique edge and interior nodes 
            to avoid duplicate nodes when multiple elements share edges.
        
        Examples
        --------
        **1D Example:**

        Original 1D mesh:
            vertices = [0.0, 1.0, 2.0]
            elements = [[0, 1], [1, 2]]

        Using degree=3 Lagrange elements adds 2 interior nodes per segment:

            Segment [0,1] → interior nodes at 0.333, 0.667
            Segment [1,2] → interior nodes at 1.333, 1.667

        Updated vertices:
            [0.0, 1.0, 2.0, 0.333, 0.667, 1.333, 1.667]

        Updated elements:
            [[0, 3, 4, 1],
            [1, 5, 6, 2]]

        **2D Example (triangle):**

        Suppose we have a 2D mesh with two triangles sharing an edge:

            self.vertices = np.array([
                [0.0, 0.0],  # vertex 0
                [1.0, 0.0],  # vertex 1
                [0.0, 1.0],  # vertex 2
                [1.0, 1.0]   # vertex 3
            ])

            self.elements = np.array([
                [0, 1, 2],  # triangle 0
                [1, 3, 2]   # triangle 1
            ])

        For `degree = 2` (quadratic Lagrange elements):

        - Each triangle gets **3 edge nodes** (midpoints of edges)  
        - Triangle 0 edges: (0,1), (1,2), (2,0) → 3 edge nodes  
        - Triangle 1 edges: (1,3), (3,2), (2,1) → note edge (1,2) already has a node  
        - No interior nodes for degree 2  

        Node ordering for each element:

            updated_elements[0] = [0, 1, 2, 4, 5, 6]  # original vertices + edge nodes
            updated_elements[1] = [1, 3, 2, 7, 8, 4]  # edge node 4 reused from triangle 0

        - `updated_vertices` array contains original 4 vertices plus new edge nodes.  
        - Interior nodes (if degree > 2) would be appended after all edge nodes.
        """
        if self.dim == 1 and self.domain == 'interval':
            updated_vertices = list(self.mesh.vertices)
            updated_elements = np.zeros((self.mesh.nelements(), self.degree + 1), dtype = int)
            next_index = self.mesh.nvertices()
            for i, edge in enumerate(self.mesh.elements):
                a, b = self.mesh.vertices[edge[0]], self.mesh.vertices[edge[1]]
                interior_nodes = np.linspace(a, b, self.degree + 1)[1:-1]
                interior_indices = []
                for node in interior_nodes:
                    updated_vertices.append(node)
                    interior_indices.append(next_index)
                    next_index += 1
                updated_elements[i] = [edge[0]] + interior_indices + [edge[1]]
            updated_vertices = np.array(updated_vertices).reshape(-1)
        elif self.dim == 2 and self.domain == 'triangle':
            nel = self.mesh.nelements()
            nvert = self.mesh.nvertices()
            nedg = self.mesh.nedges()//self.degree # <- number of geometric edges
            pedge = 3*(self.degree - 1) # number of edge nodes per triangle

            updated_elements = np.zeros((nel, 3 + 3*(self.degree - 1) + (self.degree - 1)*(self.degree - 2)//2), dtype = int)
            updated_elements[:, :3] = self.mesh.elements
            updated_vertices = np.zeros((nvert + nedg*(self.degree - 1) + nel*(self.degree - 1)*(self.degree - 2)//2, 2))
            updated_vertices[:nvert, :] = self.mesh.vertices 

            edge_nodes_dict = defaultdict(int)
            interior_nodes_dict = defaultdict(int)

            edge_count = nvert
            interior_count = nvert + nedg*(self.degree - 1)
            for i, element in enumerate(self.mesh.elements):
                nodes = PhysicalElement(vertices = self.mesh.vertices[element], ref_element = ReferenceElement(self.dim, self.domain, self.space, self.degree)).physical_reference_nodes()
                edge_nodes = nodes[3: 3 + pedge, :]
                interior_nodes = nodes[3 + pedge:, :]
                for j, enode in enumerate(edge_nodes):
                    key_node = tuple(enode)
                    if key_node not in edge_nodes_dict:
                        edge_nodes_dict[key_node] = edge_count
                        updated_vertices[edge_count] = enode
                        edge_count += 1
                    updated_elements[i, j + 3] = edge_nodes_dict[key_node]
                for k, inode in enumerate(interior_nodes):
                    key_node = tuple(inode)
                    if key_node not in interior_nodes_dict:
                        interior_nodes_dict[key_node] = interior_count
                        updated_vertices[interior_count] = inode
                        interior_count += 1
                    updated_elements[i, k + pedge + 3] = interior_nodes_dict[key_node]
        else:
            if self.dim not in [1, 2]:
                raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
            else:
                raise ValueError(f"Unsupported domain type: {self.domain}. Use 'interval' for 1D and 'triangle' for 2D meshes.")
        upgraded_mesh = Mesh(vertices = updated_vertices, elements = updated_elements, dim = self.dim, domainID = self.mesh.domainID, options = self.mesh.options)
        upgraded_mesh.degree = self.degree
        return upgraded_mesh