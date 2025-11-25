import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from collections import defaultdict
import triangle as tr
import pymetis

class Mesh:
    def __init__(self, vertices, segments = None, holes = None, domainID = 0, options: str = 'qa0.1'):

        data = {"vertices": vertices}
        if segments is not None:
            data["segments"] = segments
        if holes is not None:
            data["holes"] = holes

        triangulation = tr.triangulate(tri = data, opts = options)
        self.vertices = triangulation['vertices']
        self.segments = segments
        self.holes = holes
        self.options = options
        self.domainID = domainID
        self.elements = triangulation['triangles']

    # Implement a function that returns the `set` of edges for a given triangle element.
    def edges(self, element) -> set:
        return set(it.combinations(np.sort(element), 2))
    
    # Implement a function that returns the mapping from edges to the triangles that share them.
    def edge_to_tri_map(self): # f(edge) = {tringles sharing edge}
        """
        Build a mapping from edges to triangles sharing that edge.

        Each edge in the mesh is represented as a tuple of vertex indices (i, j),
        where i < j. This method returns a dictionary mapping each edge to the set
        of triangle indices that contain this edge. Triangles are indexed by their
        position in `self.elements`, e.g., the first triangle is index 0.

        Returns
        -------
        edge_to_tri : defaultdict(set)
            Dictionary mapping each edge (tuple of vertex indices) to a set of triangle
            indices that share the edge.

        Example
        -------
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]

        Then the edge-to-triangle mapping returned will be:

            {
                (0,1): {0},
                (0,2): {0},
                (1,2): {0,1},
                (1,3): {1},
                (2,3): {1,2},
                (2,4): {2},
                (3,4): {2}
            }

        Notes
        -----
        - The order of vertices in each edge tuple is always sorted (min, max) to ensure
        consistency.
        - Values are sets, so the order of triangle indices is not guaranteed.
        - This mapping is useful for building adjacency lists, identifying interior/boundary
        edges, and mesh partitioning.
        """
        edge_to_tri = defaultdict(set)
        for tridx, triangle in enumerate(self.elements):
            a, b, c = triangle
            edges = [(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))]
            for edge in edges:
                edge_to_tri[edge].add(tridx)
        return edge_to_tri
        
    # Implement a function that returns the adjacency dictionary mapping each triangle to the set of triangles adjacent to it.
    def adjacency(self): # f(triangle) = {neighboring triangles}
        """
        Compute the adjacency dictionary of triangles in the mesh.

        Each triangle is a neighbor of another triangle if they share an interior edge.
        Only interior edges (shared by exactly two triangles) are considered; boundary
        edges (edges shared by a single triangle) are ignored.

        Returns
        -------
        adjacency : defaultdict(set)
            Dictionary mapping each triangle index to a set of neighboring triangle indices.

        Example
        -------
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
        - Uses `self.edge_to_tri_map()` to get the mapping from edges to triangles.
        - Adjacency is symmetric: if `j` is a neighbor of `i`, then `i` is a neighbor of `j`.
        - Values are sets, so the order of neighbors is not guaranteed.
        """
        edge_to_tri = self.edge_to_tri_map()
        adjacency = defaultdict(set)
        for triangles in edge_to_tri.values():
            if len(triangles) == 2:  # interior edge (shared by 2 triangles)
                tri1, tri2 = triangles
                adjacency[tri1].add(tri2)
                adjacency[tri2].add(tri1)
        return adjacency
    
    # Implement a function that returns the areas of all triangles in the triangulation as a numpy array.
    def areas(self) -> np.ndarray:
        ntri = self.elements.shape[0] # number of triangles in the triangulation
        areas = np.zeros(ntri) 
        for i, triangle in enumerate(self.elements):
            a, b, c = self.vertices[triangle]
            areas[i] = 0.5*abs((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))
        return areas
    
    # Implement a function that returns the set of boundary edges represented as tuples (i, j), where i and j are the indices of the vertices that span the respective edge.
    def boundary_edges(self) -> set:
        """
        Compute the set of boundary edges of the mesh.

        A boundary edge is defined as an edge that belongs to exactly one triangle.
        Interior edges, which are shared by two triangles, are ignored. This method
        iterates over all triangles and toggles each edge in a set: edges that appear
        twice (interior edges) are removed, leaving only boundary edges.

        Returns
        -------
        bdedges : set
            A set of edges represented as tuples of vertex indices (i, j) where i < j.
            Each edge in the set belongs to exactly one triangle (boundary edge).

        Example
        -------
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]

        Then the boundary edges returned by this method will be:

            {(0,1), (0,2), (1,3), (2,4), (3,4)}

        Notes
        -----
        - Each edge is stored as a sorted tuple `(min_vertex, max_vertex)` for consistency.
        - This method is efficient: O(number of triangles) time and O(number of boundary edges) memory.
        - Useful for identifying mesh boundaries, applying boundary conditions, or visualization.
        """
        bdedges = set()
        for triangle in self.elements:
            a, b, c = triangle
            triedges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
            for edge in triedges:
                if edge in bdedges:
                    bdedges.discard(edge)
                else:
                    bdedges.add(edge)
        return bdedges

    # Implement a function that returns the set of (indices of) boundary vertices.
    def boundary_vertices(self) -> set:
        """
        Return the set of vertices that lie on the boundary of the mesh.

        The boundary vertices are defined as all vertices that belong to at least
        one boundary edge. This method first computes the boundary edges using
        `self.boundary_edges()` and then collects all unique vertex indices.

        Returns
        -------
        bdy_vertices : set
            A set of vertex indices that appear on the boundary of the mesh.

        Example
        -------
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]

        If the boundary edges are:

            {(0,1), (0,2), (1,3), (2,4), (3,4)}

        Then the boundary vertices returned by this method will be:

            {0, 1, 2, 3, 4}

        Notes
        -----
        - Uses `itertools.chain.from_iterable` to flatten all boundary edges into a
        single sequence of vertex indices.
        - The result is a set, so each vertex appears only once.
        - Useful for applying boundary conditions, visualization, or identifying
        mesh boundary vertices.
        """
        bdedges = self.boundary_edges()
        return set(it.chain.from_iterable(bdedges))

    # Implement a function that returns the set of boundary triangles.
    def boundary_triangles(self) -> set:
        """
        Return the set of triangles that touch the boundary of the mesh.

        A triangle is considered a boundary triangle if at least one of its
        vertices lies on the boundary. This method first computes the boundary
        vertices using `self.boundary_vertices()` and then collects all triangles
        that share at least one of these vertices.

        Returns
        -------
        bdtri : set
            A set of triangles (each represented as a tuple of vertex indices) that
            touch the boundary of the mesh.

        Example
        -------
        Suppose the mesh elements are:

            self.elements = [
                [0, 1, 2],  # triangle 0
                [2, 1, 3],  # triangle 1
                [2, 3, 4]   # triangle 2
            ]

        If the boundary vertices are:

            {0, 1, 2, 3, 4}

        Then the boundary triangles returned by this method will be:

            {(0, 1, 2), (2, 1, 3), (2, 3, 4)}

        Notes
        -----
        - Uses set intersection (`&`) to check if a triangle shares any boundary vertices.
        - Returns triangles as tuples, preserving vertex indices.
        - Useful for applying boundary conditions, visualization, or identifying
        boundary regions in a mesh.
        """
        bdvertices = self.boundary_vertices()
        bdtri = set()
        for triangle in self.elements:
            if set(triangle) & bdvertices:
                bdtri.add(tuple(triangle))
        return bdtri
    
    # Implement a function that checks whether a given edge is a boundary edge.
    def is_boundary_edge(self, edge: tuple) -> bool:
        """
        Check whether a given edge is a boundary edge of the mesh.

        A boundary edge is defined as an edge that belongs to exactly one triangle.

        Parameters
        ----------
        edge : tuple
            A tuple of two vertex indices (i, j) representing the edge. 
            The order of vertices does not matter.

        Returns
        -------
        bool
            True if the edge is a boundary edge, False otherwise.

        Example
        -------
        If the mesh boundary edges are {(0,1), (0,2), (1,3)}, then:

            self.is_boundary_edge((0,1))  # True
            self.is_boundary_edge((1,2))  # False
        """
        bdedges = self.boundary_edges()
        return edge in bdedges
    
    # Implement a function that checks whether a given triangle element is a boundary triangle.
    def is_boundary_triangle(self, element) -> bool:
        """
        Check whether a given triangle touches the boundary of the mesh.

        A triangle is considered a boundary triangle if at least one of its vertices
        lies on a boundary edge.

        Parameters
        ----------
        element : list or tuple
            A triangle represented as a list or tuple of vertex indices, e.g., [0, 1, 2].

        Returns
        -------
        bool
            True if the triangle is a boundary triangle, False otherwise.

        Example
        -------
        If the boundary vertices are {0, 1, 2, 3}, then:

            self.is_boundary_triangle([0, 1, 2])  # True
            self.is_boundary_triangle([4, 5, 6])  # False
        """
        bdvertices = self.boundary_vertices()
        return bool(set(element) & bdvertices)

    # Implement a function that computes the Jacobian matrices for all triangles in the triangulation as a numpy array.
    def Jacobians(self) -> np.ndarray:
        ntri = self.elements.shape[0] # number of triangles in the triangulation
        Js = np.zeros((ntri, 2, 2)) 
        for i, triangle in enumerate(self.elements):
            a, b, c = self.vertices[triangle]
            Js[i] = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        return Js


    # Implement a function that returns the of `non-overlapping subdomains` using METIS.
    def decompose(self, n: int, edge_weights = None):
        """
        Decompose the mesh into `n` non-overlapping subdomains using PyMetis.

        This method partitions the triangles of the mesh into `n` subdomains,
        creating new Mesh objects for each subdomain. Each subdomain shares the
        same global vertex coordinates but contains only the triangles assigned
        to it. Partitioning is performed using PyMetis's graph partitioning
        algorithm based on triangle adjacency.

        Parameters
        ----------
        n : int
            The number of desired subdomains.
        edge_weights : list of int or None, optional
            Weights of the edges in the mesh for PyMetis partitioning. The length
            must match the number of adjacency connections. Can be None for unweighted partitioning.

        Returns
        -------
        subdomains : list of Mesh
            List of Mesh objects, each representing a subdomain with its triangles
            assigned. The `domainID` attribute of each subdomain is set to the partition index.
        membership : np.ndarray of int
            Array mapping each triangle in the original mesh to its subdomain index.
            `membership[i]` is the subdomain ID of triangle `i`. This is very useful
            for visualization and coloring plots according to subdomain assignment. e.g.,
            membership[i] = subdomain ID of triangle i.

        Example
        -------
        >>> mesh = Mesh(vertices=vertices, elements=elements)
        >>> subdomains, membership = mesh.decompose(n=4, edge_weights=None)
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
        - Each subdomain Mesh object has its `domainID` set to the partition index (0 to n-1).
        - The `membership` list is important for plotting and visualization: it maps
        every triangle in the original mesh to its subdomain, so you can use it
        as a color array to show different subdomains in a plot.
        - Keeping the original global vertex array allows easy mapping between
        subdomain triangles and the original mesh for FEM assembly or interface conditions.
        """
        adjdict = self.adjacency()
        adjlist = [list(adjdict[i]) for i in range(len(self.elements))]

        _, membership = pymetis.part_graph(nparts = n, adjacency = adjlist, eweights = edge_weights) # cuts can also be retrieved

        subdomains = []
        for i in range(n):
            sd = Mesh.__new__(Mesh)  # create an uninitialized Mesh
            sd.vertices = self.vertices           # share the vertices
            sd.elements = []                      # start empty, will append local triangles
            sd.domainID = i
            sd.segments = None
            sd.holes = None
            sd.options = self.options
            subdomains.append(sd)

        for tridx, part in enumerate(membership):
            subdomains[part].elements.append(self.elements[tridx])
        
        for sd in subdomains:
            sd.elements = np.array(sd.elements, dtype=int)

        return subdomains, np.array(membership, dtype=int)
    
    # Human-readable summary
    def __str__(self):
        return (f'''Mesh: {self.vertices.shape[0]} vertices, "
                {self.elements.shape[0]} triangles, "
                {len(self.segments) if self.segments else 0} segments''')
    
    # Detailed debugging info
    def __repr__(self):
        return (f"Mesh(vertices={self.vertices}, "
                f"segments={self.segments}, holes={self.holes}, options='{self.options}')")       

    # information about the mesh (more detailed)
    def info(self):
        num_vertices = self.vertices.shape[0]
        num_elements = self.elements.shape[0]
        total_area = np.sum(self.areas())
        information = f'''Mesh Information:
        Domain ID: {self.domainID}
        Number of vertices: {num_vertices}
        Number of inner vertices: {num_vertices - len(self.boundary_vertices())}
        Number of elements (triangles): {num_elements}
        Number of boundary elements: {len(self.boundary_triangles())}
        Total area: {total_area}'''
        return information