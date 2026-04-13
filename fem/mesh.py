from collections import defaultdict
import itertools as it
import numpy as np
import pymetis
import triangle as tr
from utils.logger import setup_logger

# ---------------- Mesh Class for Finite Element Method (FEM) -------------------
# This class defines a mesh for 1D or 2D domains, including vertices, elements, 
# edges, and boundary segments. The mesh can be initialized from user-provided vertices 
# and optional connectivity, or generated using the `Triangle` library for 2D domains.
# The mesh supports both linear (degree 1) and higher-order elements through an upgrade mechanism.
# The mesh can be upgraded to include additional nodes along edges and within elements 
# for higher-degree Lagrange shape functions. 
#
# For 1D meshes, elements are line segments between consecutive nodes. For 2D meshes, triangulation
# is performed using the `Triangle` library if element connectivity is not provided.
#
# -------------- Naming conventions for mesh attributes and methods --------------
# `_node` -> means index
# `_coord` -> means coordinate
# `vertices` -> geometric vertices (nodes) of the mesh (original input vertices, excluding edge/interior nodes added by upgrade)
# `elements` -> connectivity array defining elements (triangles in 2D, line segments in 1D)
# `edges` -> connectivity array defining edges (line segments in 2D, points in 1D)
# ---------------------------------------------------------------------------------

logger = setup_logger(__name__, level = 'info')

class Mesh:
    def __init__(self, vertices, elements = None, segments = None, segment_markers = None, holes = None, domainID: int = 0, dim: int = 2, options: str = 'qa0.1'):
        """
        Initialize a Mesh object for 1D or 2D domains.

        Parameters
        ----------
        vertices : array-like
            Coordinates of the vertices. In 1D, shape (n,), in 2D, shape (n, 2).
        elements : array-like, optional
            Element connectivity. If provided, triangulation is skipped.
        segments : array-like, optional
            Boundary segments for 2D triangulation.
        segment_markers : array-like, optional
            Markers for the boundary segments.
        holes : array-like, optional
            Holes in the 2D domain.
        domainID : int, default=0
            Identifier for the domain.
        dim : int, default=2
            Dimension of the mesh (1 or 2).
        options : str, default='qa0.1'
            Options to control 2D triangulation (see Triangle API)

        Attributes
        ----------
        degree : int
            Polynomial degree of the finite element space (default is 1, can be upgraded).
        dim : int
            Dimension of the mesh (1 or 2).
        domainID : int
            Identifier for the domain (0 for whole domain, positive integers for subdomains).
        options : str
            Options for 2D triangulation (passed to `Triangle` library).
        holes : array-like
            Holes in the 2D domain (passed to `Triangle` library).
        vertices : np.ndarray
            Coordinates of the mesh vertices. Shape (n_vertices,) for 1D, shape (n_vertices, 2) for 2D.
        elements : np.ndarray
            Element connectivity array. Shape (n_elements, nodes_per_element).
        segments : np.ndarray
            Boundary segments for 2D triangulation. Shape (n_segments, 2). For 1D, this is just the boundary nodes.
        segment_markers : np.ndarray
            Markers for the boundary segments. Shape (n_segments,). For 1D, default markers are 1 for left boundary and 2 for right boundary.

        Note to users
        -------------
        - You are expected to use sequential global node indices for the vertices, starting from 0. The `elements` and `segments` arrays should reference these vertex indices.

        Notes
        -----
        - By default, the mesh represents linear (degree 1) elements. If the mesh is upgraded
            for higher-degree Lagrange shape functions using `upgrade()`, new nodes (edge 
            and interior nodes) will be added and element connectivity will be updated accordingly.
        - In 1D, elements are line segments between consecutive nodes.
        - In 2D, triangulation is performed using the `Triangle` library if `elements` is None. 
            If `elements` is provided, triangulation is skipped.
        - In 2D, the final set of vertices may differ from the user-provided 
            vertices depending on the `options` passed to the Triangle library.
            See the Triangle API for available options: https://rufat.be/triangle/API.html
        - `domainID=0` is reserved for the whole domain; subdomains should use positive integers.
        """
        self.degree = 1
        self.dim = dim
        self.domainID = domainID
        self.options = options
        self.holes = holes
        if dim == 1:
            self.vertices = np.asarray(vertices).squeeze()
            if elements is None:
                n = len(self.vertices)
                self.elements = np.column_stack((np.arange(n-1), np.arange(1, n)))
            else:
                self.elements = np.asarray(elements, dtype = int)
            self.segments = self.boundary_nodes() # in 1D, segments are just the boundary nodes
            self.segment_markers = np.array([1, 2], dtype = int) # default segment markers for 1D: 1 for left boundary, 2 for right boundary
        elif dim == 2:
            self.vertices = np.asarray(vertices)
            if elements is None:
                data = {"vertices": self.vertices}
                if segments is not None:
                    data["segments"] = segments
                if segment_markers is not None:
                    data["segment_markers"] = segment_markers
                if holes is not None:
                    data["holes"] = holes
                triangulation = tr.triangulate(tri = data, opts = options)
                self.vertices = triangulation['vertices']
                self.elements = triangulation['triangles']
                self.segments = triangulation.get("segments", None)
                markers = triangulation.get("segment_markers", None)
                self.segment_markers = None if markers is None else markers.flatten()
            else:
                self.elements = np.asarray(elements, dtype=int)
                if segments is not None:
                    self.segments = np.asarray(segments, dtype=int)
                if segment_markers is not None:
                    self.segment_markers = np.asarray(segment_markers, dtype=int).flatten()
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def nnodes(self) -> int:
        """
        Return the total number of mesh nodes.

        If the mesh has been upgraded, this also includes geometric vertices as well as
        edge and interior (element) nodes.

        Returns
        -------
        int
            Total number of mesh nodes.
        """
        return self.vertices.shape[0]

    def nintnodes(self) -> int:
        """
        Return the total number of interior nodes in the mesh.

        Interior nodes are those that are not on the boundary. 

        Returns
        -------
        int
            Number of interior nodes in the mesh.
        """
        return self.nnodes() - self.nbdnodes()

    def nbdnodes(self) -> int:
        """
        Return the total number of boundary nodes in the mesh.

        Boundary nodes are those that lie on the boundary of the domain. 

        Returns
        -------
        int
            Number of boundary nodes in the mesh.
        """
        return len(self.boundary_nodes())

    def nvertices(self) -> int:
        """
        Return the number of geometric vertices of the mesh.

        Geometric vertices are the original mesh vertices, excluding any
        additional edge or interior nodes introduced by higher-order
        discretizations or mesh upgrades.

        Returns
        -------
        int
            Number of geometric vertices.

        Raises
        ------
        ValueError
            If the mesh dimension is not 1 or 2.
        """
        if self.dim == 1:
            return self.nelements() + 1
        elif self.dim == 2:
            return self.vertices.shape[0] - (self.nedges()//self.degree)*(self.degree - 1) - self.nelements()*(self.degree - 1)*(self.degree - 2)//2
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def nelements(self) -> int:
        """
        Return the total number of elements in the mesh.

        Returns
        -------
        int
            Number of elements in the mesh.
        """
        return self.elements.shape[0]

    def nedges(self) -> int:
        """
        Compute the total number of edges in the mesh.

        This function counts edges consistently for both linear and higher-order (upgraded) meshes.

        For 1D meshes:
            - Each element (line segment) is considered an edge.
            - The total number of edges is `nelements() * degree` for upgraded meshes.

        For 2D meshes:
            - Each triangle has three geometric edges defined by its corner vertices.
            - For higher-order elements (degree > 1), each geometric edge is subdivided 
              into multiple segments corresponding to additional nodes, so the returned 
              count includes all such edge segments.

        Returns
        -------
        int
            Total number of edges or edge segments in the mesh, including subdivisions
            for higher-degree elements.

        Raises
        ------
        ValueError
            If the mesh dimension (`self.dim`) is not 1 or 2.

        Notes
        -----
        - Geometric edges are the edges connecting corner vertices of elements.
        - Upgrading the mesh (degree > 1) adds nodes along edges, effectively increasing
          the number of edge segments.
        - The returned count is consistent with FEM assembly where edge DOFs are relevant.
        """
        if self.dim == 1:
            return self.nelements()*self.degree
        elif self.dim == 2:
            edges = set()
            for triangle in self.elements:
                a, b, c = triangle[:3]
                triedges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
                for edge in triedges:
                    edges.add(edge)
            return len(edges)*self.degree
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def get_nodes(self, seg_markers: list | np.ndarray) -> np.ndarray:
        """
        Return the array of node indices that belong to segments with specified markers.

        Parameters
        ----------
        seg_markers : list | np.ndarray
            List or array of segment markers to filter by. Nodes belonging to segments with these markers will be returned.
        
        Returns
        -------
        nodes : np.ndarray
            Array of node indices that belong to segments with the specified markers.

        Raises
        ------
        ValueError
            If `self.segments` or `self.segment_markers` is not defined, which are required for this method to work.
        """
        if self.segments is None or self.segment_markers is None:
            raise ValueError("Segments and segment_markers must be defined for this method to work.")
        mask = np.isin(self.segment_markers, seg_markers)
        nodes = np.unique(self.segments[mask])
        return nodes

    def edges(self, element : list[int] | np.ndarray) -> set[tuple[int, int]]:
        """
        Return the set of edges for a given element in the mesh.

        Parameters
        ----------
        element : list[int] | np.ndarray
            Global node indices defining the element.
            - For 1D: sequence of vertex indices along the line segment (including interior nodes if upgraded).
            - For 2D linear triangles (degree=1): three corner vertices.
            - For 2D higher-order triangles (degree > 1): [corner vertices, edge nodes, interior nodes]
              stored in order: first the corner vertices a, b, c; then edge nodes along a-b, b-c, c-a;
              then interior nodes (if any).

        Returns
        -------
        edges_set : set of tuples
            Each tuple contains two vertex indices representing an edge.
            - In 1D: all consecutive pairs along the line segment (not sorted).
            - In 2D linear elements: all three edges of the triangle.
            - In 2D higher-order elements: all edges including subdivided segments along edge nodes.
              Edges are stored as sorted tuples to treat (i, j) and (j, i) as the same edge.

        Notes
        -----
        - For 1D, the set contains edges between consecutive vertices (including interior nodes if upgraded).
        - For 2D linear triangles, it contains edges between the three corner vertices.
        - For 2D higher-order triangles:
            - Each geometric edge (a-b, b-c, c-a) is subdivided by the intermediate edge nodes.
            - Consecutive nodes along the edge form edge segments that are included in the set.
            - Interior nodes are ignored for edge construction.
        - The returned set is suitable for FEM assembly, mesh connectivity, and boundary extraction.

        Raises
        ------
        ValueError
            If the mesh dimension (`self.dim`) is not 1 or 2.

        Example
        -------
        ### Linear 1D element
        element = [0, 1]
        edges(element) -> {(0, 1)}
           
        ### Higher-order 1D element (degree=3)
        element = [0, 8, 9, 1]  # 0 and 1 are endpoints, 8, 9 are interior nodes
        edges(element) -> {(0, 8), (8, 9), (9, 1)}

        ### Linear 2D triangle
        element = [0, 1, 2]
        edges(element) -> {(0, 1), (0, 2), (1, 2)}

        ### Higher-order 2D triangle (degree=3)
        element = [ 0, 1, 2,        # corner vertices: a, b, c
                    12, 13,           # edge nodes along a-b
                    14, 15,           # edge nodes along b-c
                    16, 17,           # edge nodes along c-a
                    45               # interior node
                ]
        edges(element) -> {(0, 1), (0, 12), (0, 17), (1, 12), (1, 14), (2, 15), (2, 16), (2, 17), (12, 13), (14, 15), (16, 17)}
        """
        if self.dim == 1:
            edges = set()
            for edge in zip(element[:-1], element[1:]):
                edges.add(tuple(edge))
            return edges
        elif self.dim == 2:
            if self.degree == 1:
                return set(it.combinations(np.sort(element), 2))
            else:
                edges = set()
                a, b, c = element[:3]
                # extract edge nodes
                edge_ab = element[3 : 3 + self.degree - 1]
                edge_bc = element[3 + self.degree - 1 : 3 + 2*(self.degree - 1)]
                edge_ca = element[3 + 2*(self.degree - 1) : 3 + 3*(self.degree - 1)]
                # sequences along edges
                seq_ab = [a] + list(edge_ab) + [b]
                seq_bc = [b] + list(edge_bc) + [c]
                seq_ca = [c] + list(edge_ca) + [a]
                # helper to add consecutive pairs
                def add_edge_pairs(seq):
                    for i in range(len(seq)-1):
                        edges.add(tuple(sorted((seq[i], seq[i+1]))))
                for seq in [seq_ab, seq_bc, seq_ca]:
                    add_edge_pairs(seq)
                return edges
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def vertex_to_element_map(self): # f(vertex) = [elements sharing vertex]
        """
        Build a mapping from vertices to the list of elements that share that vertex.

        This method returns a dictionary mapping each vertex index to a list of element 
        indices that contain that vertex. Elements are indexed by their position in `self.elements`, 
        e.g., the first element is index 0.

        Returns
        -------
        vertex_to_elements : dict
            Dictionary mapping each vertex index to a list of element indices that share that vertex.

        Example
        -------
        Suppose the mesh elements are:
            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]
        Then the vertex-to-element mapping returned will be:
            {
                0: [0],
                1: [0, 1],
                2: [0, 1, 2],
                3: [1, 2],
                4: [2]
            }
        
        Notes
        -----
        - The returned mapping is useful for building adjacency lists, identifying boundary vertices, and mesh partitioning.
        - This method is implemented using a vectorized approach for efficiency, especially for large meshes. 
        - The method works for both 1D and 2D meshes, and it considers only the geometric vertices 
          (not edge or interior nodes added by upgrades) for the mapping.
        """
        elements = self.elements[:, :3] if self.dim == 2 else self.elements[:, [0, -1]]
        all_vertices = elements.ravel()
        element_ids = np.repeat(np.arange(elements.shape[0]), elements.shape[1])
        order = np.argsort(all_vertices)
        all_vertices_sorted = all_vertices[order]
        element_ids_sorted = element_ids[order]
        unique_vertices, counts = np.unique(all_vertices_sorted, return_counts=True)
        splits = np.cumsum(counts)[:-1]
        element_lists = np.split(element_ids_sorted, splits)
        vertex_to_elements = {v: el.astype(int) for v, el in zip(unique_vertices, element_lists)}
        return vertex_to_elements

    def edge_to_element_map(self): # f(edge) = {elements sharing edge}
        """
        Build a mapping from edges to elements sharing that edge, including higher-order edges.

        Each edge in the mesh is represented as a tuple of vertex indices (i, j),
        where i < j. For higher-order elements (degree > 1), this includes all 
        subdivided edge segments created by intermediate edge nodes along each 
        geometric edge.
        
        This method returns a dictionary mapping each edge to the set
        of element indices that contain this edge. Elements are indexed by their
        position in `self.elements`, e.g., the first element is index 0.

        Returns
        -------
        edge_to_element : defaultdict(set)
            Dictionary mapping each edge (tuple of vertex indices) to a set of element
            indices that share the edge.

        Example
        -------
        Linear 1D mesh:
        Suppose the mesh elements are:
            self.elements = [
                [0, 1],  # interval 0
                [1, 2],  # interval 1
                [2, 3]   # interval 2
            ]
        Then the edge-to-element mapping returned will be:
            {
                (0,1): {0},
                (1,2): {1},
                (2,3): {2}
            }

        Linear 2D mesh:
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]

        Then the edge-to-element mapping returned will be:

            {
                (0,1): {0},
                (0,2): {0},
                (1,2): {0,1},
                (1,3): {1},
                (2,3): {1,2},
                (2,4): {2},
                (3,4): {2}
            }

        Higher-order 2D mesh (degree = 2):
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2, 5, 6, 7],   # triangle 0
                [1, 2, 3, 6, 8, 9],   # triangle 1
                [2, 3, 4, 8, 10, 11]  # triangle 2
            ]

        Then the edge-to-element mapping returned will be:

            {
                (0,5): {0}, 
                (1,5): {0},
                (1,6): {0,1},  # shared edge
                (2,6): {0,1},  # shared edge
                (2,7): {0}, 
                (0,7): {0},
                (2,8): {1,2},  # shared edge
                (3,8): {1,2},  # shared edge
                (3,9): {1}, 
                (1,9): {1},
                (3,10): {2}, 
                (4,10): {2},
                (2,11): {2}, 
                (4,11): {2}
            }

        Notes
        -----
        - The order of vertices in each edge tuple is always sorted (min, max) to ensure consistency.
        - Values are sets, so the order of triangle indices is not guaranteed.
        - This mapping is useful for building adjacency lists, identifying interior/boundary
          edges, and mesh partitioning.
        """
        edge_to_element = defaultdict(set)
        for elidx, element in enumerate(self.elements):
            edges = self.edges(element)
            for edge in edges:
                edge_to_element[edge].add(elidx)
        return edge_to_element

    # For 1D it is not correct
    def adjacency(self): # f(element) = {neighboring elements}
        """
        Compute the adjacency dictionary of elements in the mesh.

        Two elements (triangles in 2D or line segments in 1D) are considered neighbors
        if they share an interior edge. Only edges shared by exactly two elements 
        are considered; edges shared by a single element are ignored. This method
        computes the adjacency of elements based on their shared edges.

        Returns
        -------
        adjacency : defaultdict(set)
            Dictionary mapping each element index to a set of neighboring element indices.
                - In 2D, element indices correspond to triangles.
                - In 1D, element indices correspond to line segments.

        Example
        -------
        Linear 1D mesh:
        Suppose the mesh elements are:
            self.elements = [
                [0, 1],  # interval 0
                [1, 2],  # interval 1
                [2, 3]   # interval 2
            ]
        Then the adjacency dictionary returned will be:
            {
                0: {1},   # interval 0 neighbors: interval 1
                1: {0,2}, # interval 1 neighbors: intervals 0 and 2
                2: {1}    # interval 2 neighbors: interval 1
            }
        
        Linear 2D mesh:
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]

        And the edge-to-triangle mapping is:

            {
                (0,1): {0},
                (0,2): {0},
                (1,2): {0,1},
                (1,3): {1},
                (2,3): {1,2},
                (2,4): {2},
                (3,4): {2}
            }

        Then the adjacency dictionary returned by this method will be:

            {
                0: {1},      # triangle 0 neighbors: triangle 1
                1: {0, 2},   # triangle 1 neighbors: triangles 0 and 2
                2: {1}       # triangle 2 neighbors: triangle 1
            }

        Notes
        -----
        - Uses `self.edge_to_element_map()` to get the mapping from edges to elements.
        - Adjacency is symmetric: if `j` is a neighbor of `i`, then `i` is a neighbor of `j`.
        - Values are sets, so the order of neighbors is not guaranteed.
        """
        adjacency = defaultdict(set)
        if self.dim == 1:
            geo_elements = self.elements[:, [0, -1]]
            # map node → set of interval indices (this must be inside edge to element map for 1D since edges are just points)
            edge_to_element = defaultdict(set) # node_to_intervals
            for i, (u, v) in enumerate(geo_elements):
                edge_to_element[u].add(i)
                edge_to_element[v].add(i)
        elif self.dim == 2:
            edge_to_element = self.edge_to_element_map()
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
        for elements in edge_to_element.values():
            if len(elements) == 2:  # interior edge (shared by 2 elements)
                elem1, elem2 = elements
                adjacency[elem1].add(elem2)
                adjacency[elem2].add(elem1)
        return adjacency
    
    def boundary_edges(self) -> np.ndarray:
        """
        Returns the boundary edges of the geometric mesh as an array.

        A boundary edge is an edge that belongs to **only one element** of the original
        geometric mesh. Edges shared by multiple elements are interior and excluded.
        For higher-order elements, only the original geometric nodes are considered:
        - 1D: first and last node of each interval
        - 2D: first three nodes of each triangle

        Returns
        -------
        np.ndarray, shape (nbdedges, 2)
            Array of boundary edges. Each row represents an edge with its two node indices (i, j),
            sorted such that i < j.

        Notes
        -----
        - **1D meshes:** Boundary edges are those containing the endpoints of the mesh,
        i.e., nodes that appear only once among the geometric intervals.
        - **2D meshes:** Only the edges formed by the first three nodes of each triangle
        are considered. Boundary edges are those that appear in exactly one triangle.
        - This method ignores any higher-order nodes that may exist in the elements.

        Examples
        --------
        **1D mesh:**
            elements = np.array([[0, 1],
                                [1, 2],
                                [2, 3]])
            boundary_edges() -> array([[0, 1],
                                    [2, 3]])

        **2D mesh:**
            elements = np.array([[0, 1, 2],
                                [2, 1, 3],
                                [2, 3, 4]])
            boundary_edges() -> array([[0, 1],
                                    [0, 2],
                                    [1, 3],
                                    [2, 4],
                                    [3, 4]])
        """
        if self.dim == 1:
            edges = self.elements[:, [0, -1]]
            edges.sort(axis=1)  # ensure (min, max)
            nodes, counts = np.unique(edges.ravel(), return_counts=True)
            endpoints = nodes[counts == 1]
            mask = np.isin(edges, endpoints).any(axis=1)
            return edges[mask]
        elif self.dim == 2:
            tri_vertices = self.elements[:, :3]
            edges = np.vstack([np.sort(tri_vertices[:, [0, 1]], axis=1),
                               np.sort(tri_vertices[:, [1, 2]], axis=1),
                               np.sort(tri_vertices[:, [2, 0]], axis=1)])
            unique_edges, counts = np.unique(edges, axis = 0, return_counts = True)
            return unique_edges[counts == 1]
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def interior_nodes(self) -> np.ndarray:
        """
        Returns the indices of all interior nodes in the mesh.

        Interior nodes are those that do not lie on the boundary of the mesh.

        Notes
        -----
        - For 1D meshes, interior nodes are those that appear in more than one interval (i.e., not endpoints).
        - For 2D meshes, interior nodes are those that do not belong to any boundary edge. 
          This method relies on `self.boundary_nodes()` to identify boundary nodes and 
          then returns the complement set of nodes that are not on the boundary.
        - The output is a **NumPy array** for direct use in FEM assembly, interior node indexing, or visualization.
        - Always returns nodes in ascending order due to `np.flatnonzero` and the way the mask is constructed.
        - The method used here assumes that nodes are indexed from 0 to `nnodes() - 1` and that `boundary_nodes()` returns a subset of these indices.

        Returns
        -------
        intnodes : np.ndarray, shape (nintnodes,)
            Array of interior node indices, sorted in ascending order.
        """
        mask = np.ones(self.nnodes(), dtype = bool)
        mask[self.boundary_nodes()] = False
        return np.flatnonzero(mask)

    def boundary_nodes(self) -> np.ndarray:
        """
        Returns all the nodes (indices) lying on the boundary of the mesh.

        In 1D, the boundary consists of the **endpoints** of the mesh.  
        In 2D, the boundary nodes are all nodes that belong to **at least one boundary edge**.

        Returns
        -------
        bdnodes : np.ndarray, shape (nbdnodes,)
            Array of boundary node indices, sorted in ascending order.

        Notes
        -----
        - For 1D meshes, the endpoints are identified as the nodes that appear in 
        exactly one interval.
        - For 2D meshes, boundary nodes are identified by first finding the boundary 
        edges (edges that belong to only one triangle) and then collecting all unique 
        nodes that are part of those boundary edges. It supports both linear and 
        higher-order meshes.
        - Always returns nodes in ascending order due to `np.unique`.

        Examples
        --------
        1D mesh:
            self.elements = [
                [0, 1],
                [1, 2],
                [2, 3]
            ]
            boundary_nodes() -> array([0, 3])

        2D mesh:
            self.elements = [
                [0, 1, 2],
                [2, 1, 3],
                [2, 3, 4]
            ]
            boundary_nodes() -> array([0, 1, 2, 3, 4])
        """
        if self.dim == 1:
            arr = self.elements[:, [0, -1]].ravel() # flatten all nodes from intervals
            vals, counts = np.unique(arr, return_counts = True)  # find unique nodes and how often they appear
            return vals[counts == 1] # nodes that appear only once → endpoints
        elif self.dim == 2: # it does not support higher-order meshes since boundary edges are only defined by the first three nodes of each triangle
            if self.degree == 1:
                bdedges = self.boundary_edges()
                return np.unique(bdedges.ravel())
            else:
                k = self.degree - 1 # number of edge nodes per edge
                vertices = self.elements[:, :3]
                # Extract geometric edges defined by the first three nodes of each triangle
                e0 = np.sort(vertices[:, [0, 1]], axis=1)
                e1 = np.sort(vertices[:, [1, 2]], axis=1)
                e2 = np.sort(vertices[:, [2, 0]], axis=1)
                # Stack all geometric edges into a single array for processing. Each edge is represented as a pair of vertex indices (i, j) with i < j.
                edges = np.stack([e0, e1, e2], axis=1)  # shape (n_elem,3,2)
                # Reshape the edges array to have shape (n_elem*3, 2) so that each row corresponds to a single edge. This makes it easier to identify unique edges and count their occurrences.
                edges_2d = edges.reshape(-1, 2)
                # Identify boundary edges by finding unique edges and counting how many times each edge appears.
                edges_2d = np.ascontiguousarray(edges_2d)
                uniq, counts = np.unique(edges_2d, axis=0, return_counts=True)
                bdedges = np.ascontiguousarray(uniq[counts==1])
                # To efficiently check which edges in `edges` are boundary edges, we can use a structured array view to treat each edge as a single entity. 
                dtype = np.dtype([('a', edges_2d.dtype), ('b', edges_2d.dtype)])
                edges_view = edges_2d.view(dtype).ravel()
                bdedges_view = bdedges.view(dtype).ravel()
                # Create a boolean mask indicating which edges in `edges` are boundary edges by checking membership in `bdedges`.
                mask = np.isin(edges_view, bdedges_view)
                # Reshape the mask back to the original edge structure (3 edges per triangle) to identify which triangles contain boundary edges.
                mask_tri = mask.reshape(-1, 3)
                # Extract edge nodes for each geometric edge of the triangle. The edge nodes are stored in the `elements` array immediately following the first three nodes (corner vertices).
                e0_nodes = self.elements[:, 3 : 3 + k]
                e1_nodes = self.elements[:, 3 + k : 3 + 2*k]
                e2_nodes = self.elements[:, 3 + 2*k : 3 + 3*k]
                # Collect all boundary nodes
                bd_nodes = np.concatenate([vertices[mask_tri[:, 0]][:, [0, 1]].ravel(),
                                           vertices[mask_tri[:, 1]][:, [1, 2]].ravel(),
                                           vertices[mask_tri[:, 2]][:, [2, 0]].ravel(),
                                           e0_nodes[mask_tri[:, 0]].ravel(),
                                           e1_nodes[mask_tri[:, 1]].ravel(),
                                           e2_nodes[mask_tri[:, 2]].ravel()])
                return np.unique(bd_nodes)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
    
    def boundary_nodes_coord(self) -> np.ndarray:
        """
        Returns the coordinates of all boundary nodes.

        Returns
        -------
        bdnodes : ndarray, shape (num_boundary_nodes, dim)
            Array of coordinates of boundary nodes.
        """
        return np.array(self.vertices[self.boundary_nodes()])  # shape (num_boundary_nodes, dim)

    def boundary_elements(self) -> np.ndarray:
        """
        Returns the elements that touch the boundary of the mesh.

        An element is considered a boundary element if at least one of its
        nodes lies on the boundary. Uses the geometric boundary nodes
        from `self.boundary_nodes()`.

        Returns
        -------
        np.ndarray, shape (n_boundary_elements, n_nodes_per_element)
            Array of elements (rows) that touch the boundary.

        Notes
        -----
        - Works for 1D and 2D meshes.
        - Only considers geometric nodes; higher-order nodes are ignored.
        - Each row corresponds to an element, preserving original vertex ordering.
        """
        bdnodes = self.boundary_nodes()  # np.ndarray of boundary nodes
        mask = np.isin(self.elements, bdnodes).any(axis=1)
        return self.elements[mask]
    
    def is_boundary_edge(self, edge: tuple) -> bool:
        """
        Check whether a given edge is a boundary edge of the mesh.

        A boundary edge is defined as an edge that belongs to exactly one triangle.

        Parameters
        ----------
        edge : tuple
            A tuple of two node indices (i, j) representing the edge. 
            The order of nodes does not matter.

        Returns
        -------
        bool
            True if the edge is a boundary edge, False otherwise.
        """
        return edge in self.boundary_edges()

    def is_boundary_element(self, element: list | np.ndarray) -> bool:
        """
        Check whether a given element is a boundary element.

        An element is considered a boundary element if at least one of its nodes
        lies on a boundary edge.

        Parameters
        ----------
        element : list or np.ndarray
            A list or array of node indices representing the element.

        Returns
        -------
        bool
            True if the element is a boundary element, False otherwise.
        """
        element = np.asarray(element)
        return bool(np.isin(element, self.boundary_nodes()).any())

    def diam(self) -> float:
        """
        Compute the diameter of the mesh, defined as the maximum distance between any two vertices.

        Returns
        -------
        float
            The diameter of the mesh.
        """
        if self.dim == 1:
            a, b = self.boundary_nodes_coord()
            return np.abs(b-a)
        elif self.dim == 2:
            from scipy.spatial import distance_matrix
            dist_matrix = distance_matrix(self.vertices, self.vertices)
            return np.max(dist_matrix)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def _local_boundary_vertices(self, elements: np.ndarray) -> np.ndarray:
        if self.dim == 1:
            arr = elements.ravel()
            vals, counts = np.unique(arr, return_counts = True)
            return vals[counts == 1]
        elif self.dim == 2:
            tri_nodes = elements[:, :3]
            edges = np.vstack([np.sort(tri_nodes[:, [0, 1]], axis=1),
                                np.sort(tri_nodes[:, [1, 2]], axis=1),
                                np.sort(tri_nodes[:, [2, 0]], axis=1)])
            unique_edges, counts = np.unique(edges, axis = 0, return_counts = True)
            return np.unique(unique_edges[counts == 1].ravel())
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def _extend_subdomains(self, subdomain_elements: list, vertex_to_elements: dict, elements: np.ndarray, overlap: int, copy: bool = True) -> list:
        n = len(subdomain_elements)
        nelements = elements.shape[0]
        new_subdomains = [sd.copy() for sd in subdomain_elements] if copy else subdomain_elements
        # Precompute boolean masks for each subdomain, where True indicates elements that belong to the subdomain
        masks = []
        for sd in new_subdomains:
            mask = np.zeros(nelements, dtype = bool)
            idxs = np.array([np.where((elements == e).all(axis=1))[0][0] for e in sd])
            mask[idxs] = True
            masks.append(mask)
        # Iteratively extend each subdomain by one layer
        for layer in range(overlap):
            for j in range(n):
                # Find boundary vertices of current subdomain
                bvertices = self._local_boundary_vertices(new_subdomains[j])
                # Collect candidate element indices that share boundary vertices
                candidate_indices = np.unique(np.concatenate([vertex_to_elements[v] for v in bvertices]))
                # Filter only new elements not already in subdomain
                new_indices = candidate_indices[~masks[j][candidate_indices]]
                # Add new elements to subdomain and update mask
                if new_indices.size > 0:
                    masks[j][new_indices] = True
                    # Update subdomain elements
                    new_subdomains[j] = elements[masks[j]]
        return new_subdomains

    def decompose(self, n: int, overlap: int = 1, version: int = 1, edge_weights = None): # Element-based partitioning
        """
        Decompose the mesh into `n` subdomains with overlapping layers.

        This method performs an **element-based domain decomposition** for 1D or 2D meshes. 
        Each subdomain is returned as a separate `Mesh` object with its own local node numbering,
        elements. Partitioning is performed using PyMetis's graph partitioning algorithm based 
        on element adjacency. Obtained non-overlapping subdomains via PyMetis can be further extended 
        with overlapping layers by adding neighboring elements that share boundary vertices. This is
        done iteratively, where each layer adds elements that share at least one boundary vertex 
        with the current subdomain. 

        Parameters
        ----------
        n : int
            Number of desired subdomains.
        overlap : int, optional
            Number of layers to extend each subdomain with neighboring elements. Default is 1.
        version : int, optional
            Specifies how the interface boundary is defined. Available options:
            - 1 (default):
                The interface boundary is defined by
                    Γ_{jl} = Γ_j ∩ Ω_l,
                where Ω_l denotes the extended subdomain.
            - 2:
                The interface boundary is defined by
                    Γ_{jl} = Γ_j ∩ Ω_l,
                where Ω_l denotes the original (non-overlapping) subdomain
                before extension.
        edge_weights : list of int or None, optional
            Weights for PyMetis graph partitioning. Default is None.

        Returns
        -------
        submeshes : list of Mesh
            List of `Mesh` objects corresponding to each subdomain.
        ltog : dict[int, np.ndarray]
            Dictionary mapping each subdomain ID to the corresponding global nodes in that subdomain.
            To access the global index of a local node `i` in subdomain `s`, use `ltog[s][i]`.
        gtol : dict[int, np.ndarray]
            Dictionary mapping each subdomain ID to the corresponding local nodes in that subdomain.
            To access the local index of a global node `g` in subdomain `s`, use `gtol[s][g]`. 
            If a global node `g` does not belong to subdomain `s`, then `gtol[s][g]` will be -1.
        subdomain_maps : dict
            Dictionary mapping each subdomain ID to a dictionary of local boundary node mappings.
            The structure for subdomain_maps[s] is as follows:
            subdomain_maps[s] = {
                local_boundary_index_in_s: [
                    (0, global_index),         # if it belongs to the whole domain boundary
                    (t, local_index_in_t),     # if it is shared with another subdomain t
                    ...
                ],
                ...
            }
        membership : np.ndarray of int, shape (n_elements,)
            Array mapping each triangle in the original mesh to its subdomain index.  

        Examples
        --------
        The following examples illustrate the structure of the `subdomain_maps` output for both 1D and 2D meshes for version 1.

        1D example
        ~~~~~~~~~~
        Suppose the interval [0, 1], where the global nodes are [0, 0.25, 0.5, 0.75, 1.0], is divided into 3 subdomains:
            Subdomain 1 (ID=1):
                nodes [0, 0.25, 0.5]        (local indices: 0, 1, 2)
            Subdomain 2 (ID=2):
                nodes [0.25, 0.5, 0.75]     (local indices: 0, 1, 2)
            Subdomain 3 (ID=3):
                nodes [0.5, 0.75, 1.0]      (local indices: 0, 1, 2)
        Boundary nodes (local indices):
            Subdomain 1:
                [0, 0.5]  → [0, 2]
            Subdomain 2:
                [0.25, 0.75] → [0, 2]
            Subdomain 3:
                [0.5, 1.0] → [0, 2]
        Whole domain boundary nodes:
            [0, 1.0] → global indices [0, 4]
        Resulting mapping ``allmaps`` (local indices):
            allmaps = {
                1: {
                    0: [(0, 0)],
                    2: [(2, 0), (3, 0)]
                },
                2: {
                    0: [(1, 1)],
                    2: [(3, 1)]
                },
                3: {
                    0: [(1, 2), (2, 1)],
                    2: [(0, 4)]
                }
            }

        2D example
        ~~~~~~~~~~
        Consider subdomains with IDs 1, 2, and 3.

        - A boundary node of subdomain 1 at local index 5 is shared with:
        subdomain 2 (local index 3), subdomain 3 (local index 7),
        and also lies on the global boundary (index 10):

            allmaps[1][5] = [
                (0, 10),
                (2, 3),
                (3, 7)
            ]

        - A boundary node of subdomain 1 at local index 8 is shared with
        subdomain 3 (local index 4) and lies on the global boundary (index 12):

            allmaps[1][8] = [
                (0, 12),
                (3, 4)
            ]

        - A boundary node of subdomain i at local index k that belongs only to 
        the global boundary:

            allmaps[i][k] = [
                (0, g_index)
            ]
        
        Notes
        -----
        - Whole domain ID is 0, subdomains are numbered 1 to n.
        - Local indexing for each subdomain starts from 0, and follows the order of ordered unique 
          nodes in global indexing within that subdomain.
        - Partitioning is based on element adjacency (element-based partitioning).
        - Overlapping is done by adding neighboring elements sharing boundary vertices (vertex-based overlapping).   
        """
        assert n > 0, "number of subdomains must be positive"
        assert overlap >= 0, "overlap must be non-negative"
        assert version in [1, 2], "version must be either 1 or 2"

        # Create adjacency list and partition using PyMetis
        adjdict = self.adjacency()
        adjlist = [sorted(list(adjdict[i])) for i in range(len(self.elements))]
        _, membership = pymetis.part_graph(nparts = n, adjacency = adjlist, eweights = edge_weights) # cuts can also be retrieved

        # Extract elements for each subdomain based on the partitioning, note that membership is for non-overlapping partitioning, we will add overlap later
        subdomain_elements = [self.elements[np.asarray(membership) == j] for j in range(n)]

        # Precompute vertex -> elements map
        vertex_to_elements = self.vertex_to_element_map()

        # Extend each subdomain by one overlap layer by adding elements that share at least one boundary vertex with the current subdomain (vertex-based overlap).
        subdomain_elements_extended = self._extend_subdomains(subdomain_elements, vertex_to_elements, self.elements, overlap, copy = True if version == 2 else False)

        # Construct submeshes and mapping between local and global node indices for each subdomain
        submeshes = []
        ltog, gtol = {}, {}
        for j, elements in enumerate(subdomain_elements_extended, start = 1):
            global_indices = np.unique(elements) # subdomain nodes in global indexing, sorted in ascending order
            ltog[j] = global_indices # ltog[j][i] gives the global index of the i-th local node in subdomain j 
            local_indices = np.full(self.vertices.shape[0], -1, dtype = int)
            local_indices[global_indices] = np.arange(global_indices.size)
            gtol[j] = local_indices # gtol[j][g] gives the local index of global node g in subdomain j, or -1 if g is not in subdomain j
            vertices = self.vertices[global_indices] # extract the coordinates of the nodes that belong to the subdomain
            elements = local_indices[elements] # relabel elements to local numbering
            submesh = Mesh(vertices = vertices, elements = elements, dim = self.dim, domainID = j, options = self.options)
            submesh.degree = self.degree
            submesh.segments = None
            submesh.segment_markers = None
            submeshes.append(submesh)

        # Compute boundary nodes for all subdomains and the global mesh (`nodes` used here means all nodes including edge nodes for higher degree elements, not just corner vertices)
        boundary_nodes = dict()
        boundary_nodes[0] = self.boundary_nodes() # whole boundary nodes in global indexing
        for k, submesh in enumerate(submeshes, start = 1):
            boundary_nodes[k] = ltog[k][submesh.boundary_nodes()] # boundary nodes of subdomain k in global indexing

        # Construct the mapping for each subdomain, which maps each local node index i in subdomain s to a list of tuples (t, lt) where t is either 0 or another subdomain index, and lt is the corresponding local node index in subdomain t. 
        subdomain_maps = {}
        subdomain_nodes = [np.unique(sdomain) for sdomain in subdomain_elements] # list of arrays of global node indices for each subdomain (with overlap if version 1, without overlap if version 2)
        for s in range(1, n + 1):
            maps = defaultdict(list)
            # Find the exterior boundary nodes of the subdomain in global indexing (see the paper for the definition of exterior boundary)
            exterior_boundary_g = np.intersect1d(boundary_nodes[0], boundary_nodes[s], assume_unique=True)
            # Find the exterior boundary nodes of the subdomain in local indexing
            exterior_boundary = gtol[s][exterior_boundary_g]
            # Define the map: i: [(0, g)] for each exterior boundary node i in local indexing, where g is the corresponding global node index, and 0 indicates that this node is on the global boundary (not an interface node) 
            for loc, g in zip(exterior_boundary, exterior_boundary_g):
                maps[loc].append((0, g))
            # Note: `interface_boundary` gives different results for version 1 and version 2, because in version 1 we modify 
            # the original `subdomain_elements` list in-place when we add new elements to the subdomain, while in version 2 
            # we create a new list of extended subdomains, so the original `subdomain_elements` list remains unchanged and 
            # does not contain the new elements added for overlap. 
            for t in range(1, n + 1):
                if t == s: # skip the same subdomain, we only want to find interface nodes between different subdomains
                    continue
                # Find the interface boundary nodes between subdomain s and t in global indexing
                interface_boundary = np.intersect1d(boundary_nodes[s], subdomain_nodes[t-1], assume_unique=True)
                # Get corresponding local indices in subdomain s and t
                local_s = gtol[s][interface_boundary]   # local indices in subdomain s
                local_t = gtol[t][interface_boundary]   # local indices in subdomain t
                # Define the map: i: [(t, lt)] for each interface boundary node i in local indexing of subdomain s, where lt is the corresponding local node index in subdomain t, and t indicates that this node is on the interface with subdomain t
                for ls, lt, g in zip(local_s, local_t, interface_boundary):
                    maps[ls].append((t, lt))
            subdomain_maps[s] = maps

        # Note that returned `membership` array is for non-overlapping domain decomposition!
        return submeshes, ltog, gtol, subdomain_maps, np.array(membership, dtype = int)

    def is_in_interval(self, point: float, interval: list | np.ndarray, tol: float = 1e-12) -> bool:
        """
        Check if a point lies inside a 1D interval (including endpoints).

        Parameters
        ----------
        point : float
            The coordinate of the point.
        interval : list or np.ndarray, shape (2,)
            Coordinates of the interval endpoints [x0, x1].
        tol : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if point is inside the interval or on its boundary.
        """
        x0, x1 = interval
        return x0 - tol <= point <= x1 + tol

    def is_in_triangle(self, point: list | np.ndarray, triangle: np.ndarray, tol: float = 1e-12) -> bool:
        """
        Check if a point lies inside a triangle (including edges and vertices)
        using barycentric coordinates.

        Parameters
        ----------
        point : list | np.ndarray, shape (2,)
            The (x, y) coordinates of the point.
        triangle : array-like, shape (3, 2)
            Triangle vertices.
        tol : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if point is inside the triangle or on its boundary.
        """
        point = np.asarray(point)
        a, b, c = triangle

        v0 = b - a
        v1 = c - a
        v2 = point - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < tol:
            return False

        invDenom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= -tol and v >= -tol and u + v <= 1.0 + tol)
    
    def locate_triangle(self, point: list | np.ndarray) -> int | None:
        """
        Return the index of the triangle that contains the given point.

        Parameters
        ----------
        point : list | np.ndarray, shape (2,)
            The (x, y) coordinates of the point.

        Returns
        -------
        int or None
            The index of a triangle that contains the point.
            If the point is outside the mesh, return None.

        Notes
        -----
        - Triangle indices refer to the ordering of elements in ``self.elements``.
          In particular, if the function returns index ``i``, the triangle containing
          the point is ``self.elements[i]``.
        - If the point lies strictly inside an element, that element is returned.
        - If the point lies on a shared edge or at a vertex, it belongs to multiple
          elements; in this case, one of the adjacent elements is returned
          (the first encountered in the element loop).
        """
        for i, triangle in enumerate(self.elements):
            vertices = self.vertices[triangle[:3]]
            if self.is_in_triangle(point, vertices):
                return i
        return None
    
    def locate_interval(self, point: float) -> int | None:
        """
        Return the index of the 1D element (interval) that contains the point.

        Parameters
        ----------
        point : float
            The coordinate of the point.

        Returns
        -------
        int or None
            Index of the interval containing the point.
            Returns None if the point is outside the domain.
        """
        for i, edge in enumerate(self.elements):
            interval = np.array([self.vertices[edge[0]], self.vertices[edge[-1]]])  # left and right vertices
            if self.is_in_interval(point, interval):
                return i
        return None
    
    def measures(self) -> np.ndarray:
        """
        Compute the "measure" of each element in the mesh.

        - For 1D meshes, returns the lengths of each edge.
        - For 2D meshes, returns the areas of each triangle.

        Returns
        -------
        np.ndarray
            Array of measures: lengths for 1D, areas for 2D.

        Raises
        ------
        ValueError
            If mesh dimension is not 1 or 2.
        """
        if self.dim == 1:
            lengths = np.zeros(self.nelements()) 
            for i, edge in enumerate(self.elements):
                lengths[i] = abs(self.vertices[edge[-1]] - self.vertices[edge[0]])
            return lengths
        elif self.dim == 2:
            areas = np.zeros(self.nelements()) 
            for i, triangle in enumerate(self.elements):
                a, b, c = self.vertices[triangle[:3]]
                areas[i] = 0.5*abs((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))
            return areas
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
        
    # Human-readable summary
    def __str__(self):
        element_type = "lines" if self.dim == 1 else "triangles"
        num_segments = len(self.segments) if self.segments is not None else 0
        return (f"Mesh: {self.vertices.shape[0]} vertices, "
                f"{self.elements.shape[0]} {element_type}, "
                f"{num_segments} segments")

    # Detailed debugging info
    def __repr__(self):
        return (f"Mesh(nodes={self.vertices}, "
                f"elements={self.elements}, "
                f"segments={self.segments}, holes={self.holes}, "
                f"dim={self.dim}, options='{self.options}')")

    # Information about the mesh (more detailed)
    def info(self):
        """
        Return a human-readable summary of the mesh.
        Works for both 1D (lines) and 2D (triangles) meshes.
        """
        nnodes = self.nnodes()
        nelements = self.nelements()
        num_boundary_nodes = len(self.boundary_nodes())
        total_measure  = np.sum(self.measures()) # Total measure: length for 1D, area for 2D
        element_type = "lines" if self.dim == 1 else "triangles"
        num_boundary_elements = len(self.boundary_elements()) if self.dim == 2 else 2  # In 1D, there are always 2 boundary elements (the two endpoints)
        information = (
        f"Mesh Information:\n"
        f"  Dimension: {self.dim}D\n"
        f"  Domain ID: {self.domainID}\n"
        f"  Number of nodes: {nnodes}\n"
        f"  Number of elements ({element_type}): {nelements}\n"
        f"  Number of boundary nodes: {num_boundary_nodes}\n"
        f"  Number of boundary elements: {num_boundary_elements}\n"
        f"  Total measure ({'length' if self.dim == 1 else 'area'}): {total_measure}")
        return information