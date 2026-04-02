from collections import defaultdict
import itertools as it
import numpy as np
import pymetis
import triangle as tr
import time
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
# `vertices` -> geometric vertices of the mesh (original input vertices, excluding edge/interior nodes added by upgrade)
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
        dofs : np.ndarray
            Global node indices used by the FEM discretization. Initially corresponds to vertex indices, but can include edge/interior nodes after upgrade.

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
        self.dofs = np.unique(self.elements) # global DoF indices used by the FEM discretization (initially just vertex indices, but can include edge/interior nodes after upgrade)

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
    
    # you can add parameter like `original` that returns only for 1degree mesh maps
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
            tri_nodes = self.elements[:, :3]
            edges = np.vstack([np.sort(tri_nodes[:, [0, 1]], axis=1),
                               np.sort(tri_nodes[:, [1, 2]], axis=1),
                               np.sort(tri_nodes[:, [2, 0]], axis=1)])
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
        Returns the indices of all nodes lying on the boundary of the mesh.

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
        - For 2D meshes, the method relies on `self.boundary_edges()` to find all 
        edges on the boundary, then collects all unique nodes from these edges.
        - The output is a **NumPy array** for direct use in FEM assembly, Dirichlet
        boundary conditions, or visualization.
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
            boundary_edges() -> array([[0,1],[0,2],[1,3],[2,4],[3,4]])
            boundary_nodes() -> array([0, 1, 2, 3, 4])
        """
        if self.dim == 1:
            arr = self.elements.ravel() # flatten all nodes from intervals
            vals, counts = np.unique(arr, return_counts = True)  # find unique nodes and how often they appear
            return vals[counts == 1] # nodes that appear only once → endpoints
        elif self.dim == 2:
            bdedges = self.boundary_edges()
            return np.unique(bdedges.ravel())
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
    
    def interval_decompose(self, n: int, overlap: int = 0):
        """
        Split a 1D grid into overlapping subdomains.

        Each subdomain contains its local chunk plus `overlap` points
        from each neighboring subdomain. Consecutive subdomains share
        exactly `overlap + 1` grid points.

        Parameters
        ----------
        n : int
            Number of subdomains to split the 1D grid into.
        overlap : int
            Number of additional points shared with each neighboring subdomain.

        Returns
        -------
        dict of int -> numpy.ndarray
            Dictionary mapping subdomain index (0-based) to its array of
            grid points, including the overlap regions.
        """
        
        x = np.sort(self.vertices)
        nx = x.shape[0]
        
        # Base size of each non-overlapping part
        base = nx//n  # number of intervals
        remainder = nx% n
        
        # Determine number of local points for non-overlapping part
        sizes = []
        for i in range(n):
            sz = base + (1 if i < remainder else 0)
            sizes.append(sz)
    
        subdomains = {}
        start = 0
        for j, subsize in enumerate(sizes):
            if j == 0:
                subdomains[j + 1] = x[0: subsize + overlap]
            elif j == n - 1:
                subdomains[j + 1] = x[-(subsize + overlap):]
            else:
                subdomains[j + 1] = x[start - overlap: start + subsize + overlap]
            start += subsize
        
        return subdomains

    # For optimization, split this function to several parts!
    def decompose(self, n: int, overlap: int = 0, edge_weights = None): # Element-based partitioning
        """
        Decompose the mesh into `n` subdomains using PyMetis, with optional overlapping layers.

        This method performs an **element-based domain decomposition** for 1D or 2D meshes. 
        Each subdomain is returned as a separate `Mesh` object with its own local vertex numbering,
        elements, and a mapping to global mesh vertices. Partitioning is performed using PyMetis's 
        graph partitioning algorithm based on triangle adjacency. Each subdomain has:

        Partitioning steps:
        1. Construct the element adjacency graph (edges for 2D triangles, intervals for 1D elements).
        2. Use PyMetis to partition elements into `n` subdomains.
        3. Optionally add `overlap` layers: neighboring elements sharing at least one boundary vertex
        are added to expand each subdomain.
        4. Construct local meshes with:
            - Local vertex arrays (subset of global vertices used in the subdomain)
            - Triangles/intervals relabeled to local vertex indices
            - `domainID` set to the subdomain index (1 to n)

        Overlapping layers (controlled by `overlap`) add neighboring elements around
        the subdomain boundary. Local-to-global mapping is provided to map local
        subdomain vertices back to the original global mesh vertices.

        Important notes:
        -----------------
        - Whole domain ID is 0, subdomains are numbered 1 to n.
        - Each subdomain Mesh object has:
            - `vertices`: local vertex coordinates used in that subdomain.
            - `elements`: local elements with vertex indices relabeled to local numbering.
            - `domainID`: set to the subdomain index (1 to n).
        - The `local_to_global_mappings` dictionary maps each subdomain ID to the
          corresponding global vertex indices used in that subdomain.
        - Partitioning is based on element adjacency (element-based partitioning).
        - Overlapping is done by adding neighboring elements sharing boundary vertices (vertex-based overlapping).

        Parameters
        ----------
        n : int
            Number of desired subdomains.
        overlap : int, optional
            Number of overlapping layers to add around subdomain boundaries. Default is 0.
            Each layer adds neighboring elements sharing at least one boundary vertex.
        edge_weights : list of int or None, optional
            Weights for PyMetis graph partitioning. Default is None.

        Returns
        -------
        subdomains : list of Mesh
            List of `Mesh` objects corresponding to each subdomain.
        local_to_global_mappings : dict
            Mapping from subdomain ID -> global vertex indices used in that subdomain.
            {subdomain ID: np.array([sorted global indices for whole domain correspoding to the subdomain ID])}
        subdomain_maps : dict
            Mapping of boundary vertices across subdomains and the whole domain.

            For each subdomain `s` (with ID `s = 1, ..., n`), this dictionary stores
            a mapping from *local boundary vertex indices* in subdomain `s`
            to the corresponding location of the same vertex in another domain
            (either another subdomain or the whole domain).

            Structure
            ---------
            subdomain_maps[s] : dict

                Keys
                ----
                local_index_s : int
                    Local index of a boundary vertex in subdomain `s`.

                Values
                ------
                (t, index_t) : tuple of int
                    - `t = 0` indicates that the vertex lies on the boundary of
                      the whole (non-decomposed) domain.
                    - `t > 0` indicates another subdomain ID sharing the same vertex.
                    - `index_t` is the local index of the vertex in domain `t`
                      (or the global vertex index if `t = 0`).

            Interpretation
            --------------
            - Each entry represents a **single correspondence** for a boundary
              vertex of subdomain `s`.
            - If a boundary vertex belongs to the whole domain boundary,
              it is mapped to `(0, global_index)`.
            - If it is shared with another subdomain `t`, it is mapped to
              `(t, local_index_in_t)`.
            - If a vertex is shared with multiple subdomains, the *last processed*
              subdomain overwrites earlier entries.

            Notes
            -----
            - Only boundary vertices of subdomains are included.
            - Boundary detection for subdomains uses `restricted=False`, so
              higher-order edge nodes may appear.
            - The mapping is **not symmetric**: information is stored independently
              for each subdomain.
            - This structure is suitable for:
                - subdomain coupling,
                - interface communication,
                - boundary-condition transfer.

            1D Example
            ----------
            Suppose the domain [0, 1] is divided into 3 subdomains:

            Whole domain nodes: [0, 0.25, 0.5, 0.75, 1.0]

            Subdomain 1 (ID=1): nodes [0, 0.25, 0.5]       # local indices: 0,1,2
            Subdomain 2 (ID=2): nodes [0.25, 0.5, 0.75]    # local indices: 0,1,2
            Subdomain 3 (ID=3): nodes [0.5, 0.75, 1.0]     # local indices: 0,1,2

            Boundary nodes of subdomains (local indices):

            - Subdomain 1: [0, 0.5] → local indices [0,2]
            - Subdomain 2: [0.25, 0.75] → local indices [0,2]
            - Subdomain 3: [0.5, 1.0] → local indices [0,2]

            Whole domain boundary nodes: [0, 1] → global indices [0, 4]

            Resulting mapping allmaps (local indices):

            allmaps = {
                1: {
                    0: [(0, 0)],         # node 0 (local 0) is boundary of whole domain (global 0)
                    2: [(2, 0), (3, 0)]  # node 0.5 (local 2) shared with subdomain 2 (local 0) and 3 (local 0)
                },
                2: {
                    0: [(1, 1)],         # node 0.25 (local 0) shared with subdomain 1 (local 1)
                    2: [(3, 1)]          # node 0.75 (local 2) shared with subdomain 3 (local 1)
                },
                3: {
                    0: [(1, 2), (2, 1)], # node 0.5 (local 0) shared with subdomain 1 (local 2) and 2 (local 1)
                    2: [(0, 4)]          # node 1.0 (local 2) is boundary of whole domain (global 4)
                }
            }
            
            2D Example
            ----------
            Suppose the domain is decomposed into subdomains with IDs 1, 2, and 3:

            - A boundary vertex of subdomain 1 at local index 5 is shared with subdomain 2
            at index 3 and subdomain 3 at index 7, and is also a boundary node of the whole domain
            at index 10:

                allmaps[1][5] = [
                    (0, 10),  # belongs to the whole domain
                    (2, 3),
                    (3, 7)
                ]

            - Another boundary vertex of subdomain 1 at local index 8 is shared only with
            subdomain 3 at index 4 and also belongs to the whole domain at index 12:

                allmaps[1][8] = [
                    (0, 12),
                    (3, 4)
                ]

            - A boundary vertex that appears **only on the whole domain** and not in any subdomain:

                allmaps[subdomainID_i][boundary_index_in_i] = [
                    (0, g_index)
                ]
        membership : np.ndarray of int
            Array mapping each triangle in the original mesh to its subdomain index.
            (length equals number of elements in the original mesh).

        Example
        -------
        >>> mesh = Mesh(vertices=vertices, elements=elements)
        >>> subdomains, _, _, membership = mesh.decompose(n=4, edge_weights=None)
        >>> len(subdomains)  # 4 subdomain Mesh objects
        4
        >>> subdomains[0].domainID  # domain ID of first subdomain
        0
        >>> len(subdomains[0].elements)  # number of triangles in subdomain 0
        15
        >>> membership[:5]  # first 5 triangles and their subdomain assignment
        [0, 0, 1, 2, 1]
        >>> # Example of using membership for plotting colors
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib.tri as mtri
        >>> tri = mtri.Triangulation(vertices[:,0], vertices[:,1], elements)
        >>> plt.tripcolor(tri, facecolors=membership, cmap='tab10')
        >>> plt.show()

        Notes
        -----
        - Triangles in each subdomain use **local vertex numbering**:
        For example, if a subdomain uses global vertex indices [2, 3, 6, 8], then
        local indices 0,1,2,3 correspond to global vertices 2,3,6,8.
        - Overlapping layers expand the subdomain by including neighboring triangles
        sharing boundary vertices.
        - Subdomain numbering **starts from 1**, i.e., subdomain IDs are 1,2,...,n.
        - ID 0 is reserved to indicate the **whole domain** (useful for visualization
            and global assembly).  
        - The `local_to_global_mappings` dict allows reconstruction of a global solution
        from subdomain solutions.
        - The `membership` list is important for plotting and visualization: it maps
        every triangle in the original mesh to its subdomain, so you can use it
        as a color array to show different subdomains in a plot. (non-overlapping version only)
        """
        def local_boundary_nodes(elements, restricted: bool = True) -> set:
            """
            Extract boundary vertices from a collection of mesh elements.

            A vertex is considered a boundary vertex if it belongs to an edge
            (in 1D or 2D) that appears exactly once within the given collection
            of elements.

            The behavior depends on the spatial dimension and on the flag
            ``restricted``.

            Dimension-dependent definition
            -------------------------------
            - 1D (interval mesh):
            Elements are intervals.
            Boundary vertices are vertices that appear exactly once as endpoints
            of all considered intervals.

            - 2D (triangular mesh):
            Elements are triangles.
            Boundary vertices are vertices belonging to edges that occur exactly
            once among all triangles.

            Effect of ``restricted``
            ------------------------
            - ``restricted=True`` (default):
            Only *topological vertices* are used.
            - In 1D: only the first and last vertex of each element are considered.
            - In 2D: only the first three entries of each element (triangle vertices)
                are considered.
            Higher-order nodes (edge or interior nodes) are ignored.

            - ``restricted=False``:
            Higher-order elements are taken into account.
            - In 1D: all consecutive vertex pairs along each element are treated
                as edges.
            - In 2D: edges are reconstructed using both vertices and edge nodes,
                and boundary detection is performed on these refined edges.
            The returned set may therefore include higher-order edge nodes.

            Parameters
            ----------
            elements : iterable of array-like
                Collection of mesh elements.
                - In 1D, each element must contain at least two vertex indices.
                - In 2D, each element must contain at least three vertex indices.
                For higher-order elements, additional entries are interpreted
                according to the polynomial degree when ``restricted=False``.

            restricted : bool, optional
                If True, boundary detection is performed using only topological
                vertices. If False, higher-order edge nodes are included.
                Default is True.

            Returns
            -------
            set
                Set of vertex indices that lie on the boundary of the given
                element collection.

            Raises
            ------
            ValueError
                If the mesh dimension is not supported (only 1D and 2D are allowed).
            """
            if restricted:
                if self.dim == 1:
                    counts = {}
                    for interval in elements:
                        a, b = interval[0], interval[-1]
                        counts[a] = counts.get(a, 0) + 1
                        counts[b] = counts.get(b, 0) + 1
                    return set(v for v, c in counts.items() if c == 1)
                elif self.dim == 2:
                    bdedges = set()
                    for triangle in elements:
                        a, b, c = triangle[:3]
                        triedges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
                        for edge in triedges:
                            if edge in bdedges:
                                bdedges.discard(edge)
                            else:
                                bdedges.add(edge)
                    return set(it.chain.from_iterable(bdedges))
                else:
                    raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
            else:
                bdedges = set()
                if self.dim == 1:
                    for element in elements:
                        for edge in zip(element[:-1], element[1:]):
                            if edge in bdedges:
                                bdedges.discard(tuple(edge))
                            else:
                                bdedges.add(tuple(edge))
                    return set(it.chain.from_iterable(bdedges))
                elif self.dim == 2:
                        for element in elements:
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
                            edges = set()
                            def edge_pairs(seq):
                                for i in range(len(seq)-1):
                                    edges.add(tuple(sorted((seq[i], seq[i+1]))))
                            for seq in [seq_ab, seq_bc, seq_ca]:
                                edge_pairs(seq)
                            
                            for edge in edges:
                                if edge in bdedges:
                                    bdedges.discard(edge)
                                else:
                                    bdedges.add(edge)
                        return set(it.chain.from_iterable(bdedges))
                else:
                    raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")
                
        def nodes(elements):
            """
            Return the unique node indices appearing in a collection of elements.

            This function concatenates all element connectivity arrays and extracts
            the set of unique node indices used by the given elements. The result
            is returned as a sorted NumPy array.

            Parameters
            ----------
            elements : iterable of array-like
                Collection of mesh elements, where each element is an array-like
                object containing node indices.

            Returns
            -------
            numpy.ndarray
                One-dimensional array of unique node indices appearing in
                ``elements``, sorted in ascending order.

            Notes
            -----
            This function does not distinguish between vertex nodes and
            higher-order nodes; all node indices present in the element
            connectivity are included.
            """
            return np.unique(np.concatenate(elements))

        # Create adjacency list and partition using PyMetis
        adjdict = self.adjacency()
        adjlist = [sorted(list(adjdict[i])) for i in range(len(self.elements))]
        _, membership = pymetis.part_graph(nparts = n, adjacency = adjlist, eweights = edge_weights) # cuts can also be retrieved

        # Assign elements to subdomains
        subdomains_elements = {j: [] for j in range(1, n + 1)}
        for tridx, part in enumerate(membership):
            subdomains_elements[part + 1].append(self.elements[tridx])

        # The below code can be optimized, but for clarity I keep it simple, see later if need optimization!
        # Extend each subdomain by one overlap layer by adding elements that share at least one boundary vertex with the current subdomain (vertex-based overlap).
        for layer in range(overlap):
            logger.debug(f"Adding overlapping layer {layer + 1}")
            elements = self.elements
            new_elements = {i: set(tuple(e) for e in subdomains_elements[i]) for i in range(1, n + 1)}
            for sindex, selements in subdomains_elements.items():
                bvertices = local_boundary_nodes(selements)
                for bvertex in bvertices:
                    for element in elements:
                        if bvertex in element:
                            element_tuple = tuple(element)
                            if element_tuple not in new_elements[sindex]:
                                selements.append(element)
                                new_elements[sindex].add(element_tuple)

        boundary_nodes = dict() # domainID: {boundary vertices of the domain with domainID in domainID = 0 (whole mesh) global indexing}
        boundary_nodes[0] = self.boundary_nodes() # whole boundary vertices
        for k in range(1, n + 1):
            boundary_nodes[k] = local_boundary_nodes(subdomains_elements[k], restricted = False)

        # Build local meshes: extract subdomain DOFs, define global <--> local index mapping, and relabel elements to local numbering.
        subdomains = []
        global_to_local_mappings = {}
        local_to_global_mappings = {}
        for j in range(1, n + 1):
            elements = subdomains_elements[j]
            global_indices = nodes(elements) # This is exactly map from localdof to global dof s.t. local_indices = np.arange(0, len(global_indices), dtype=int)
            mapping = {g: l for l, g in enumerate(global_indices)} # This is exactly map from global dof to localdof 
            local_to_global_mappings[j] = global_indices 
            global_to_local_mappings[j] = mapping
            vertices = self.vertices[global_indices]
            elements = np.array([[mapping[v] for v in element] for element in elements], dtype = int)
            local_mesh = Mesh(vertices = vertices, elements = elements, dim = self.dim, domainID = j, options = self.options)
            local_mesh.degree = self.degree
            subdomains.append(local_mesh)

        subdomain_maps = dict()
        for s in range(1, n + 1):
            maps = defaultdict(list)
            wintersection = boundary_nodes[0] & boundary_nodes[s]
            for w in wintersection: # This is for the boundary nodes of the subdomain that shares with whole boundary nodes
                maps[global_to_local_mappings[s][w]].append((0, w))
            for t in range(1, n + 1):
                if t == s:
                    continue
                aintersection = boundary_nodes[s] & set(nodes(subdomains_elements[t]))
                for a in aintersection:
                    maps[global_to_local_mappings[s][a]].append((t, global_to_local_mappings[t][a]))
            subdomain_maps[s] = maps

        # Note that returned `membership` array is for non-overlapping domain decomposition!
        return subdomains, local_to_global_mappings, global_to_local_mappings, subdomain_maps, np.array(membership, dtype = int)

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
        num_nodes = self.nnodes()
        num_elements = self.elements.shape[0]
        num_boundary_nodes = len(self.boundary_nodes())
        total_measure  = np.sum(self.measures()) # Total measure: length for 1D, area for 2D
        element_type = "lines" if self.dim == 1 else "triangles"
        num_boundary_elements = len(self.boundary_elements()) if self.dim == 2 else 2  # In 1D, there are always 2 boundary elements (the two endpoints)
        information = (
        f"Mesh Information:\n"
        f"  Dimension: {self.dim}D\n"
        f"  Domain ID: {self.domainID}\n"
        f"  Number of nodes: {num_nodes}\n"
        f"  Number of elements ({element_type}): {num_elements}\n"
        f"  Number of boundary nodes: {num_boundary_nodes}\n"
        f"  Number of boundary elements: {num_boundary_elements}\n"
        f"  Total measure ({'length' if self.dim == 1 else 'area'}): {total_measure}")
        return information