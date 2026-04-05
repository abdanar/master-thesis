import numpy as np
from fem.mesh import Mesh
import pymetis
from collections import defaultdict
from typing import Optional
import itertools as it
from utils.logger import setup_logger
logger = setup_logger(__name__, level = 'info')

def _local_boundary_vertices(dim: int, elements: np.ndarray) -> np.ndarray:
    if dim == 1:
        arr = elements.ravel()
        vals, counts = np.unique(arr, return_counts = True)
        return vals[counts == 1]
    elif dim == 2:
        tri_nodes = elements[:, :3]
        edges = np.vstack([np.sort(tri_nodes[:, [0, 1]], axis=1),
                            np.sort(tri_nodes[:, [1, 2]], axis=1),
                            np.sort(tri_nodes[:, [2, 0]], axis=1)])
        unique_edges, counts = np.unique(edges, axis = 0, return_counts = True)
        return np.unique(unique_edges[counts == 1].ravel())
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 1D and 2D meshes are supported.")

def _extend_subdomains(subdomain_elements: list, vertex_to_elements: dict, elements: np.ndarray, dim: int, overlap: int, copy: bool = True) -> list:
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
            bvertices = _local_boundary_vertices(dim, new_subdomains[j])
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

def decompose(self, n: int, overlap: int = 0, version: int = 1, edge_weights = None): # Element-based partitioning

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
    subdomain_elements_extended = _extend_subdomains(subdomain_elements, vertex_to_elements, self.elements, self.dim, overlap, copy = True if version == 2 else False)

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
        submeshes.append(submesh)

    # Compute boundary nodes for all subdomains and the global mesh (`nodes` used here means all nodes including edge nodes for higher degree elements, not just corner vertices)
    boundary_nodes = dict()
    boundary_nodes[0] = self.boundary_nodes() # whole boundary nodes in global indexing
    for k, submesh in enumerate(submeshes, start = 1):
        boundary_nodes[k] = submesh.boundary_nodes() # boundary nodes of subdomain k in local indexing

    # Construct the mapping for each subdomain, which maps each local node index i in subdomain s to a list of tuples (t, lt) where t is either 0 or another subdomain index, and lt is the corresponding local node index in subdomain t. 
    subdomain_maps = {}
    subdomain_nodes = [np.unique(sdomain) for sdomain in subdomain_elements] # list of arrays of global node indices for each subdomain (with overlap if version 1, without overlap if version 2)
    for s in range(1, n + 1):
        maps = defaultdict(list)
        # Find the exterior boundary nodes of the subdomain in global indexing (see the paper for the definition of exterior boundary)
        exterior_boundary_g = np.intersect1d(boundary_nodes[0], ltog[s][boundary_nodes[s]], assume_unique=True)
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
            local_s = gtol[s][interface_boundary]       # local indices in subdomain s
            local_t = gtol[t][interface_boundary]       # local indices in subdomain t
            # Define the map: i: [(t, lt)] for each interface boundary node i in local indexing of subdomain s, where lt is the corresponding local node index in subdomain t, and t indicates that this node is on the interface with subdomain t
            for ls, lt, g in zip(local_s, local_t, interface_boundary):
                maps[ls].append((t, lt))
        subdomain_maps[s] = maps

    # Note that returned `membership` array is for non-overlapping domain decomposition!
    return submeshes, ltog, gtol, subdomain_maps, np.array(membership, dtype = int)