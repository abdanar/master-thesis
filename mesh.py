import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from collections import defaultdict
from phyelement import PhysicalElement
from refelement import ReferenceElement
import triangle as tr
import pymetis
from logger import setup_logger

logger = setup_logger(__name__, level = 'info')

class Mesh:
    def __init__(self, vertices, segments = None, holes = None, domainID = 0, options: str = 'qa0.1'):

        data = {"vertices": vertices}
        if segments is not None:
            data["segments"] = segments
        if holes is not None:
            data["holes"] = holes

        triangulation = tr.triangulate(tri = data, opts = options)
        self.vertices = triangulation['vertices']
        self.elements = triangulation['triangles']
        self.segments = segments
        self.holes = holes
        self.options = options
        self.domainID = domainID

    # Total number of vertices in the triangulation - including boundary nodes
    def nvertices(self):
        return self.vertices.shape[0]
    
    # Total number of (unique) edges in the triangulation
    def nedges(self):
        edges = set()
        for triangle in self.elements:
            a, b, c = triangle
            triedges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
            for edge in triedges:
                edges.add(edge)
        return len(edges)
 
    # Total number triangles in the triangulation
    def nelements(self):
        return self.elements.shape[0]
    
    # Implement a function that returns an array of same size with self.elements that i^th row represents the barycenter of the self.vertices[self.elements[i]]
    def barycenters(self):
        """
        Compute the barycenters (centroids) of triangular elements in the mesh.

        Returns
        -------
        numpy.ndarray
            An array of shape (n_elements, 2), where each row contains the
            [x, y] coordinates of the barycenter of a triangle.
        
        Notes
        -----
        The barycenter of a triangle with vertices v1, v2, v3 is computed as:
        
            barycenter = (v1 + v2 + v3)/3.

        This method assumes that `self.vertices` is a NumPy array of shape (n_vertices, 2)
        and `self.elements` is a NumPy array of shape (n_elements, 3), containing
        the indices of vertices for each triangle.
        """
        barycenters = np.zeros((self.elements.shape[0], 2))
        for i, triangle in enumerate(self.elements):
            vertices = self.vertices[triangle]
            barycenters[i] = np.mean(vertices, axis = 0)  
        return barycenters

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

    # Implement a function that returns the set of (indices of) boundary vertices. - global indices of boundary nodes
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
    
    # Implement a function that returns a list of coordinates of all boundary nodes in the domain
    def boundary_nodes(self) -> list:

        """
        Return the coordinates of all boundary vertices of the domain.

        This method retrieves the vertices corresponding to the boundary indices
        of the domain. Boundary vertices are obtained from the `boundary_vertices()` method.

        Returns
        -------
        bdnodes : list
            A list of arrays, where each array represents the coordinates of a boundary vertex.
        """
        
        bdnodes = []
        for i in self.boundary_vertices():
            bdnodes.append(self.vertices[i])

        return bdnodes

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

    # Implement a function that upgrade vertices and triangles array considering the higher degree Lagrange shape functions
    def upgrade(self, domain: str = 'triangle', space: str = 'Lagrange', degree: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Upgrade the vertices and elements arrays for higher-degree Lagrange finite elements.

        This function adds additional nodes to the mesh (edge and interior nodes) 
        according to the specified polynomial degree for Lagrange shape functions.
        It returns the updated vertex coordinates and element connectivity 
        including the new higher-order nodes.

        Parameters
        ----------
        domain : str, optional
            The type of reference element domain. Default is 'triangle'.
        space : str, optional
            The finite element space type. Default is 'Lagrange'.
        degree : int, optional
            The polynomial degree of the Lagrange shape functions. 
            Must be >= 1. Default is 1.

        Returns
        -------
        updated_vertices : np.ndarray, shape (n_vertices_new, 2)
            The coordinates of all vertices including new edge and interior nodes.
        updated_elements : np.ndarray, shape (n_elements, n_nodes_per_element)
            The updated element connectivity array including indices of new nodes.
            Node ordering: first 3 vertices, then edge nodes, then interior nodes.

        Notes
        -----
        - Edge node indices start after the original vertices, i.e., at `self.nvertices()`.
        - Interior node indices start after all original vertices and edge nodes, 
        i.e., at `self.nvertices() + self.nedges() * (degree - 1)`.
        - This function uses dictionaries to track unique edge and interior nodes 
        to avoid duplicate nodes when multiple elements share edges.
        """

        nel = self.nelements()
        nvert  = self.nvertices()
        nedg = self.nedges()
        pedge = 3*(degree - 1) # number of edge nodes per triangle

        updated_elements = np.zeros((nel, 3 + 3*(degree - 1) + (degree - 1)*(degree - 2)//2), dtype = int)
        updated_elements[:, :3] = self.elements
        updated_vertices = np.zeros((nvert + nedg*(degree - 1) + nel*(degree - 1)*(degree - 2)//2, 2))
        updated_vertices[:nvert, :] = self.vertices 

        edge_nodes_dict = defaultdict(int)
        interior_nodes_dict = defaultdict(int)

        edge_count = nvert
        interior_count = nvert + nedg*(degree - 1)
        for i, element in enumerate(self.elements):
            nodes = PhysicalElement(vertices = self.vertices[element], ref_element = ReferenceElement(domain, space, degree)).physical_reference_nodes()
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
        return updated_vertices, updated_elements

    # Implement a function that computes the Jacobian matrices for all triangles in the triangulation as a numpy array.
    def Jacobians(self) -> np.ndarray:
        ntri = self.elements.shape[0] # number of triangles in the triangulation
        Js = np.zeros((ntri, 2, 2)) 
        for i, triangle in enumerate(self.elements):
            a, b, c = self.vertices[triangle]
            Js[i] = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        return Js

    # Partition the mesh into n subdomains using METIS, optionally adding overlapping layers.
    # Each subdomain has its own local vertex indexing, elements, and mapping to global vertices.
    def decompose(self, n: int, overlap: int = 0, edge_weights = None):

        """
        Decompose the mesh into `n` subdomains using PyMetis, with optional overlapping layers.

        This method partitions the triangles of the mesh into `n` subdomains, creating
        new Mesh objects for each subdomain. Partitioning is performed using PyMetis's 
        graph partitioning algorithm based on triangle adjacency. Each subdomain has:

        - A local vertex array containing all vertices used in its triangles.
        - Triangles renumbered to local vertex indices.
        - A `domainID` set to the subdomain index (1 to n).

        Overlapping layers (controlled by `overlap`) add neighboring triangles around
        the subdomain boundary. Local-to-global mapping is provided to map local
        subdomain vertices back to the original global mesh vertices.

        Parameters
        ----------
        n : int
            Number of desired subdomains.
        overlap : int, optional
            Number of overlapping layers to add around subdomain boundaries. Default is 0.
        edge_weights : list of int or None, optional
            Weights for PyMetis graph partitioning. Default is None.

        Returns
        -------
        subdomains : list of Mesh
            List of Mesh objects for each subdomain.
        local_to_global_mappings : dict
            Mapping from subdomain ID -> global vertex indices used in that subdomain.
            {subdomain ID: np.array([sorted global indices for whole domain correspoding to the subdomain ID])}
        membership : np.ndarray of int
            Array mapping each triangle in the original mesh to its subdomain index.

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

        def local_boundary_vertices(triangles):
            """Return set of boundary vertices from a list of triangles."""
            bdedges = set()
            for triangle in triangles:
                a, b, c = triangle
                triedges = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]
                for edge in triedges:
                    if edge in bdedges:
                        bdedges.discard(edge)
                    else:
                        bdedges.add(edge)
            return set(it.chain.from_iterable(bdedges))

        # Create adjacency list and partition using PyMetis
        adjdict = self.adjacency()
        adjlist = [list(adjdict[i]) for i in range(len(self.elements))]
        _, membership = pymetis.part_graph(nparts = n, adjacency = adjlist, eweights = edge_weights) # cuts can also be retrieved

        # Initialize subdomain Mesh objects
        subdomains = []
        for i in range(1, n + 1):
            sd = Mesh.__new__(Mesh)  # create an uninitialized Mesh
            sd.vertices = []          
            sd.elements = []              
            sd.domainID = i
            sd.segments = None
            sd.holes = None
            sd.options = self.options
            subdomains.append(sd)

        # Assign triangles to subdomains
        subdomains_elements = {j: [] for j in range(1, n + 1)}
        for tridx, part in enumerate(membership):
            subdomains_elements[part + 1].append(self.elements[tridx])

        for layer in range(overlap):
            logger.debug(f"Adding overlapping layer {layer + 1}")
            elements = self.elements
            new_elements = {i: set(tuple(e) for e in subdomains_elements[i]) for i in range(1, n + 1)}
            for sindex, selements in subdomains_elements.items():
                bvertices = local_boundary_vertices(selements)
                for bvertex in bvertices:
                    for element in elements:
                        if bvertex in element:
                            element_tuple = tuple(element)
                            if element_tuple not in new_elements[sindex]:
                                selements.append(element)
                                new_elements[sindex].add(element_tuple)

        local_to_global_mappings = {}
        for subdomain in subdomains:
            elements = subdomains_elements[subdomain.domainID]
            global_indices = np.unique(np.concatenate(elements)) # This is exactly map from localdof to global dof st local_indices = np.arange(0, len(global_indices), dtype=int)
            mapping = {g: l for l, g in enumerate(global_indices)} # This is exactly map from global dof to localdof 
            local_to_global_mappings[subdomain.domainID] = global_indices 
            subdomain.vertices = self.vertices[global_indices]
            subdomain.elements = np.array([[mapping[v] for v in element] for element in elements], dtype=int)

        return subdomains, local_to_global_mappings, np.array(membership, dtype=int)

    def subdomain_mapping(self, n: int, overlap: int = 0) -> dict:

        """
        Construct a mapping of boundary vertices across all subdomains in a decomposed domain.

        For each subdomain, this function creates a dictionary mapping its boundary vertices
        to all subdomains (and optionally the whole domain) that contain the same vertex.

        - A boundary vertex may be shared by multiple subdomains.
        - If a vertex is also on the boundary of the whole (non-decomposed) domain, a tuple
        with first entry 0 is included to indicate this.

        Parameters
        ----------
        n : int
            Number of subdomains along each dimension (or total number, depending on the
            implementation of `decompose`).
        overlap : int, default=0
            Number of overlapping layers of vertices between subdomains.

        Returns
        -------
        allmaps : dict
            Dictionary for **all subdomains**. For each subdomain `i`, the mapping is:

            - Keys: global boundary indices in subdomain `i`.
            - Values: lists of tuples `(j, g_index)` where:
                - `j` is 0 if the vertex belongs to the whole domain, or a subdomain ID for decomposed subdomains.
                - `g_index` is the local index of this boundary vertex in subdomain `j` (or in the whole domain if `j = 0`).

            Format example:

                allmaps[subdomainID_i][boundary_index_in_i] = [
                    (j1, local_index_in_j1),
                    (j2, local_index_in_j2),
                    ...,
                    (0, index_in_whole_domain)  # included if the vertex also belongs to the whole domain
                ]

        Example
        -------
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

        Each subdomain stores, for each boundary vertex, a list of all other subdomains
        (and possibly the whole domain) containing the same vertex, along with their
        corresponding local indices.
        """

        allmaps = {}
        # It is overuse here!, one can accept only subdomains instead of decomposing again!
        subdomains, _, _ = self.decompose(n = n, overlap = overlap)

        # Precompute boundary nodes of the whole domain
        whole_boundary_nodes = self.boundary_nodes()
        whole_vertices_view = self.vertices.view([('', self.vertices.dtype)]*self.vertices.shape[1])

        for subdomain in subdomains:
            sdomainID = subdomain.domainID
            subdomain_maps = {}
            boundary_indices = subdomain.boundary_vertices() # global boundary indices for boundary vertices
            for bindex in boundary_indices:
                boundary_node = subdomain.vertices[bindex]
                subdomain_maps[bindex] = [] # initialize list for this boundary index
                # check if node is on whole domain boundary
                if any(np.all(boundary_node == v) for v in whole_boundary_nodes):
                    position = np.where(whole_vertices_view == boundary_node.view([('', boundary_node.dtype)]*boundary_node.shape[0]))[0][0]
                    subdomain_maps[bindex].append((0, position))   
                for sdomain in subdomains:
                    mask = np.all(sdomain.vertices == boundary_node, axis=1)
                    if np.any(mask):
                        position = np.where(mask)[0][0]
                        subdomain_maps[bindex].append((sdomain.domainID, position))         
            allmaps[sdomainID] = subdomain_maps

        return allmaps
            
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