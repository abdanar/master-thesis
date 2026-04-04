import numpy as np
from fem.mesh import Mesh
import pymetis
from collections import defaultdict
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

def _extend_subdomains(subdomain_elements: list, vertex_to_elements: dict, elements: np.ndarray, dim: int, overlap: int):
    n = len(subdomain_elements)
    nelements = elements.shape[0]
    # Precompute boolean masks for each subdomain, where True indicates elements that belong to the subdomain
    masks = []
    for sd in subdomain_elements:
        mask = np.zeros(nelements, dtype = bool)
        idxs = np.array([np.where((elements == e).all(axis=1))[0][0] for e in sd])
        mask[idxs] = True
        masks.append(mask)
    # Iteratively extend each subdomain by one layer
    for layer in range(overlap):
        for j in range(n):
            # Find boundary vertices of current subdomain
            bvertices = _local_boundary_vertices(dim, subdomain_elements[j])
            # Collect candidate element indices that share boundary vertices
            candidate_indices = np.unique(np.concatenate([vertex_to_elements[v] for v in bvertices]))
            # Filter only new elements not already in subdomain
            new_indices = candidate_indices[~masks[j][candidate_indices]]
            # Add new elements to subdomain and update mask
            if new_indices.size > 0:
                masks[j][new_indices] = True
                # Update subdomain elements
                subdomain_elements[j] = elements[masks[j]]
    return subdomain_elements


def decompose(self, n: int, overlap: int = 0, edge_weights = None): # Element-based partitioning
            
    def nodes(elements):
        return np.unique(np.concatenate(elements))

    # Create adjacency list and partition using PyMetis
    adjdict = self.adjacency()
    adjlist = [sorted(list(adjdict[i])) for i in range(len(self.elements))]
    _, membership = pymetis.part_graph(nparts = n, adjacency = adjlist, eweights = edge_weights) # cuts can also be retrieved

    # Extract elements for each subdomain based on the partitioning, note that membership is for non-overlapping partitioning, we will add overlap later
    subdomain_elements = [self.elements[np.asarray(membership) == j] for j in range(n)]

    # Precompute vertex -> elements map
    vertex_to_elements = self.vertex_to_element_map()

    # Extend each subdomain by one overlap layer by adding elements that share at least one boundary vertex with the current subdomain (vertex-based overlap).
    subdomain_elements = _extend_subdomains(subdomain_elements, vertex_to_elements, self.elements, self.dim, overlap)

    # Compute boundary vertices for all subdomains and the global mesh (`vertices` used here means geometric vertices of the mesh, not the `nodes` of the finite element space)
    boundary_vertices = dict()
    boundary_vertices[0] = _local_boundary_vertices(self.dim, self.elements) # whole boundary vertices
    for k in range(1, n + 1):
        boundary_vertices[k] = _local_boundary_vertices(self.dim, subdomain_elements[k-1]) # boundary vertices of the subdomain with domainID = k in global indexing

    # Construct submeshes and mapping between local and global node indices for each subdomain
    submeshes = []
    ltog, gtol = {}, {}
    for j, elements in enumerate(subdomain_elements, start = 1):
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

    subdomain_maps = dict()
    for s in range(1, n + 1):
        maps = defaultdict(list)
        wintersection = set(boundary_vertices[0]) & boundary_vertices[s]
        for w in wintersection: # This is for the boundary nodes of the subdomain that shares with whole boundary nodes
            maps[gtol[s][w]].append((0, w))
        for t in range(1, n + 1):
            if t == s:
                continue
            aintersection = boundary_vertices[s] & set(nodes(subdomain_elements[t-1]))
            for a in aintersection:
                maps[gtol[s][a]].append((t, gtol[t][a]))
        subdomain_maps[s] = maps

    # Note that returned `membership` array is for non-overlapping domain decomposition!
    return submeshes, ltog, gtol, subdomain_maps, np.array(membership, dtype = int)

from collections import defaultdict
import numpy as np

subdomain_maps = {}

# Precompute arrays of global boundary vertices for all subdomains (0 = global boundary)
subdomain_boundary_arrays = [np.array(list(boundary_vertices[s]), dtype=int) for s in range(n + 1)]

# Precompute unique nodes for each subdomain (for intersections with other subdomains)
subdomain_nodes_arrays = [np.unique(sd) for sd in subdomain_elements]

for s in range(1, n + 1):
    maps = defaultdict(list)
    local_s = gtol[s]  # local mapping for subdomain s

    # --- boundary shared with global mesh (domainID = 0)
    wintersection = np.intersect1d(subdomain_boundary_arrays[0], subdomain_boundary_arrays[s], assume_unique=True)
    for w in wintersection:
        maps[local_s[w]].append((0, w))

    # --- boundaries shared with other subdomains
    for t in range(1, n + 1):
        if t == s:
            continue
        nodes_t = subdomain_nodes_arrays[t-1]
        aintersection = np.intersect1d(subdomain_boundary_arrays[s], nodes_t, assume_unique=True)
        local_t = gtol[t]
        for a in aintersection:
            maps[local_s[a]].append((t, local_t[a]))

    subdomain_maps[s] = maps