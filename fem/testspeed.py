import numpy as np
from collections import defaultdict
import time

# --------------------------
# Helper functions
# --------------------------

def vertex_to_element_map(elements, dim=2):
    """
    Vectorized vertex -> elements map
    """
    if dim == 1:
        arr = elements.ravel()
        unique, counts = np.unique(arr, return_counts=True)
        return {v: np.where(arr == v)[0] for v in unique}
    elif dim == 2:
        tri_nodes = elements[:, :3]
        nelements, npe = tri_nodes.shape
        all_vertices = tri_nodes.ravel()
        element_ids = np.repeat(np.arange(nelements), npe)
        order = np.argsort(all_vertices)
        all_vertices_sorted = all_vertices[order]
        element_ids_sorted = element_ids[order]
        unique_vertices, counts = np.unique(all_vertices_sorted, return_counts=True)
        splits = np.cumsum(counts)[:-1]
        element_lists = np.split(element_ids_sorted, splits)
        return {v: el.astype(np.int64) for v, el in zip(unique_vertices, element_lists)}
    else:
        raise ValueError("Unsupported dim")

def _local_boundary_vertices(dim, elements):
    if dim == 1:
        arr = elements.ravel()
        vals, counts = np.unique(arr, return_counts=True)
        return vals[counts == 1]
    elif dim == 2:
        tri_nodes = elements[:, :3]
        edges = np.vstack([np.sort(tri_nodes[:, [0, 1]], axis=1),
                           np.sort(tri_nodes[:, [1, 2]], axis=1),
                           np.sort(tri_nodes[:, [2, 0]], axis=1)])
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        return np.unique(unique_edges[counts == 1].ravel())
    else:
        raise ValueError("Unsupported dim")

# --------------------------
# Fastest pure NumPy overlap function
# --------------------------
def extend_subdomain_overlap(subdomain_elements, vertex_to_elements, elements, dim, overlap=1):
    n_subdomains = len(subdomain_elements)
    n_elements = elements.shape[0]

    # Precompute boolean masks for each subdomain
    masks = []
    for sd in subdomain_elements:
        mask = np.zeros(n_elements, dtype=bool)
        # find element indices in full elements array
        idxs = np.nonzero(np.isin(np.arange(n_elements), np.arange(n_elements))[0])[0]  # initial dummy
        # simpler: use np.searchsorted trick for small test, for demo set all True first
        mask[np.array([np.where((elements == e).all(axis=1))[0][0] for e in sd])] = True
        masks.append(mask)

    for layer in range(overlap):
        for j in range(n_subdomains):
            sd_elements = subdomain_elements[j]
            bvertices = _local_boundary_vertices(dim, sd_elements)

            # Collect candidate elements via vertex_to_elements map
            candidate_indices = np.unique(np.concatenate([vertex_to_elements[v] for v in bvertices]))
            new_indices = candidate_indices[~masks[j][candidate_indices]]

            if new_indices.size > 0:
                masks[j][new_indices] = True
                subdomain_elements[j] = elements[masks[j]]

    return subdomain_elements

# --------------------------
# Test setup
# --------------------------
np.random.seed(42)
n_elements = 500
nodes_per_element = 3
n_vertices = 1000
elements = np.random.randint(0, n_vertices, size=(n_elements, nodes_per_element))

# Random partition membership
n_subdomains = 5
membership = np.random.randint(0, n_subdomains, size=n_elements)

subdomain_elements = [elements[membership==j] for j in range(n_subdomains)]
vertex_to_elements = vertex_to_element_map(elements, dim=2)

# --------------------------
# Run test and timing
# --------------------------
start = time.time()
subdomain_elements_extended = extend_subdomain_overlap([e.copy() for e in subdomain_elements],
                                                      vertex_to_elements, elements, dim=2, overlap=1)
time_taken = time.time() - start

print(time_taken, [e.shape for e in subdomain_elements_extended])