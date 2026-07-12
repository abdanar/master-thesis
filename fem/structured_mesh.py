import numpy as np
from fem.mesh import Mesh
from typing import Literal
from collections import defaultdict
from fem.mesh import DecompositionInfo

# EXPERIMENTAL: DO NOT USE THIS!

# ------------------ Structured mesh generation and decomposition functions --------------------
#
# This file contains functions for generating a structured triangulation of a rectangular domain 
# and decomposing it into subdomains with optional overlap. The main functions are:
#
# - `structured_triangulation`: Generates a structured triangulation of a rectangular domain.
#
# - `decompose_structured_triangulation`: Decomposes a structured triangulation into subdomains 
#    with optional overlap and constructs mappings between local and global node indices.
#    (this function can be thought of as a `decompose` function in `Mesh` class for structured meshes, 
#    where the decomposition is done based on the structured grid rather than a graph-based approach)
#
# Testing and visualization of these functions can be found in `test/test_str.py`.
# ----------------------------------------------------------------------------------------------

def split_1Darray(array: np.ndarray | list, n: int, overlap = 0):
    """
    Split a 1D array into `n` subarrays with optional overlap.

    Parameters
    ----------
    array : np.ndarray or list
        The input array to be split.
    n : int
        The number of subarrays to split into.
    overlap : int, optional
        The number of overlapping elements between adjacent subarrays. Default is 0 (no overlap).
    
    Returns
    -------
    dict
        A dictionary containing the list of subarrays under the key 'arrays' and the list of 
        split indices under the key 'indices'.     

    Example
    -------
    >>> split_1Darray([7, 8, 9, 1, 5], 2, overlap = 1)
    {'arrays': [[7, 8, 9], [9, 1, 5]], 'indices': [3]}
    """
    i = 0
    arrays, indices = [], []
    q, r = divmod(len(array) + overlap*(n - 1), n)
    for k in range(n):
        if k < r:
            j = i + q + 1
        else:
            j = i + q 
        arrays.append(array[i:j])
        indices.append(j) if k != n - 1 else None    
        i = j - overlap
    return {'arrays': arrays, 'indices': indices}

def split_2Darray(array: np.ndarray, nrows: int, ncols: int, overlap: int = 1):
    """
    Split a 2D array into `nrows` x `ncols` subarrays with overlap.

    Parameters
    ----------
    array : np.ndarray
        The input 2D array to be split.
    nrows : int
        The number of subarrays in the row direction.
    ncols : int
        The number of subarrays in the column direction.
    overlap : int, optional
        The number of overlapping elements between adjacent subarrays. 
        Default is 1 (overlap of 1 element).
    
    Returns
    -------
    arrays : list of np.ndarray
        A list containing the subarrays resulting from the split.
    
    Example
    -------
    >>> split_2Darray(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 2, 2, overlap = 1)
    [array([[1, 2], [4, 5]]), 
     array([[2, 3], [5, 6]]), 
     array([[4, 5], [7, 8]]), 
     array([[5, 6], [8, 9]])]
    """
    assert array.ndim == 2, "Input array must be 2D"
    assert nrows > 0 and ncols > 0, "nrows and ncols must be positive integers"
    assert overlap >= 1, "overlap must be a positive integer"
    arrays = []
    rstart, cstart = 0, 0
    qrow, rrow = divmod(array.shape[0] + nrows - 1, nrows)
    qcol, rcol = divmod(array.shape[1] + ncols - 1, ncols)
    for i in range(nrows):
        if i < rrow:
            rend = rstart + qrow + overlap
        else:
            rend = rstart + qrow + overlap - 1
        for j in range(ncols):
            if j < rcol:
                cend = cstart + qcol + overlap
            else:
                cend = cstart + qcol + overlap - 1
            arrays.append(array[rstart:rend, cstart:cend])
            cstart = cend - (2*overlap if overlap > 1 else 1)
        cstart = 0
        rstart = rend - (2*overlap if overlap > 1 else 1)
    return arrays

# (overlap - 1 if j != ncols - 1 else 0) + 1

def structured_triangulation(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int, 
                            order: Literal['C', 'F'] = 'C', diagonal: Literal["main", "anti"] = "main") -> Mesh:
    """
    Generate a structured triangulation of a rectangular domain [`x0`, `x1`] x [`y0`, `y1`]. 
    
    Parameters
    ----------
    x0, x1 : float
        The minimum and maximum x-coordinates of the domain.
    y0, y1 : float
        The minimum and maximum y-coordinates of the domain.
    nx, ny : int
        The number of subdivisions in the x and y directions, respectively.
    order : {'C', 'F'}, optional
        The memory order for the generated arrays. 'C' for row-major (C-style) 
        and 'F' for column-major (Fortran-style). Default is 'C'.
    diagonal : {'main', 'anti'}, optional
        The diagonal direction for splitting each rectangular cell into two triangles.
        - 'main' for the main diagonal (left-bottom to right-top)
        - 'anti' for the anti-diagonal (left-top to right-bottom)
        Default is 'main'.

    Returns
    -------
    Mesh
        A Mesh object containing the vertices and elements of the structured triangulation.
    """
    # Generate structured grid vertices
    x = np.linspace(x0, x1, nx + 1)
    y = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(x, y, indexing = "xy")
    vertices = np.column_stack([X.ravel(order = order), Y.ravel(order = order)])
    # Generate structured cell indices (i,j) for each rectangular element
    I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    # Convert cell indices to global vertex indices
    if order == "C":
        n = nx + 1
        p1 = J * n + I
        p2 = J * n + (I + 1)
        p3 = (J + 1) * n + I
        p4 = (J + 1) * n + (I + 1)
    elif order == "F":
        n = ny + 1
        p1 = I * n + J
        p2 = (I + 1) * n + J
        p3 = I * n + (J + 1)
        p4 = (I + 1) * n + (J + 1)
    else:
        raise ValueError("order must be 'C' or 'F'")
    # Split each rectangular cell into two triangles based on the specified diagonal
    if diagonal == "main":
        t1 = np.stack([p1, p2, p4], axis = -1)
        t2 = np.stack([p1, p4, p3], axis = -1)
    elif diagonal == "anti":
        t1 = np.stack([p1, p2, p3], axis = -1)
        t2 = np.stack([p2, p4, p3], axis = -1)
    else:
        raise ValueError("diagonal must be 'main' or 'anti'")
    # Combine the triangles into a single array
    triangles = np.vstack([t1.reshape(-1, 3), t2.reshape(-1, 3)])
    return Mesh(vertices = vertices, elements = triangles) # degree taken to be 1, be careful when using this function with higher degree elements!

def decompose_structured_triangulation(mesh: Mesh, nx: int, ny: int, order: Literal['C', 'F'] = 'C', nrows: int = 2, ncols: int = 2, overlap: int = 0, version: int = 1):
    """
    Decompose a structured triangulation of a rectangular domain into `nrows` x `ncols` subdomains with optional overlap.
    
    Parameters
    ----------
    nx, ny : int
        The number of subdivisions in the x and y directions, respectively, used to generate the triangulation.
    order : {'C', 'F'}, optional
        The memory order for the input arrays. 'C' for row-major (C-style) 
        and 'F' for column-major (Fortran-style). Default is 'C'.
    nrows : int, optional
        The number of subdomains in the row direction. Default is 2.
    ncols : int, optional
        The number of subdomains in the column direction. Default is 2.
    overlap : int, optional
        Number of layers to extend each subdomain with neighboring elements. 
        Default is 0. (0: nonoverlapping decomposition, 1: one layer of overlap, etc.)
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
    
    Returns
    -------
    submeshes : dict[int, Mesh]
        A dictionary mapping subdomain IDs to their corresponding Mesh objects. The subdomain IDs 
        are integers starting from 1 up to `nrows` x `ncols`. The memory order of the vertices and 
        elements in each submesh is the same as the input mesh.
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
    membership : np.ndarray of int, shape (nelements,)
        Array mapping each triangle in the original mesh to its subdomain index. The subdomain 
        index is an integer from 1 to `nrows` x `ncols` indicating which subdomain the triangle 
        belongs to. It is useful for visualizing the subdomain decomposition on the original mesh.
    """
    assert order in ['C', 'F'], "order must be 'C' or 'F'"
    assert version in [1, 2], "version must be 1 or 2"
    # Extract global elements and vertices from the mesh
    global_elements = mesh.elements
    global_vertices = mesh.vertices
    # Compute the global node indices for each overlapping subdomain
    grid = np.arange((nx + 1) * (ny + 1)).reshape((ny + 1, nx + 1), order = order)
    subgrids_overlap = split_2Darray(grid, nrows, ncols, overlap + 1)
    subnodes_overlap = {i: nodes.ravel(order = order) for i, nodes in enumerate(subgrids_overlap, start = 1)}
    # Construct the membership array for global elements of the non-overlapping subdomains
    subgrids = split_2Darray(grid, nrows, ncols, 1)
    subnodes = {i: nodes.ravel(order = order) for i, nodes in enumerate(subgrids, start = 1)}
    membership = np.zeros(global_elements.shape[0], dtype = int)
    for j, nodes in subnodes.items():
        mask = np.isin(global_elements, nodes).all(axis = 1)
        membership[mask] = j
    # Construct submeshes and mapping between local and global node indices for each subdomain
    submeshes, ltog, gtol = {}, {}, {}
    for i, onodes in subnodes_overlap.items():
        omask = np.isin(global_elements, onodes).all(axis = 1)
        local_indices = np.full(global_vertices.shape[0], -1, dtype = int)
        local_indices[onodes] = np.arange(onodes.size)
        gtol[i], ltog[i] = local_indices, onodes
        elements = local_indices[global_elements[omask]]
        submesh = Mesh(vertices = global_vertices[onodes], elements = elements, dim = mesh.dim, domainID = i, options = mesh.options)
        submesh.degree, submesh.segments, submesh.segment_markers = mesh.degree, None, None
        submeshes[i] = submesh
    # Compute boundary nodes for all subdomains and the global mesh (`nodes` used here means all nodes including edge nodes for higher degree elements, not just corner vertices)
    boundary_nodes = dict()
    boundary_nodes[0] = mesh.boundary_nodes() # whole boundary nodes in global indexing
    for k, submesh in submeshes.items():
        boundary_nodes[k] = ltog[k][submesh.boundary_nodes()] # boundary nodes of subdomain k in global indexing
    # Construct the mapping for each subdomain, which maps each local node index i in subdomain s to a list of tuples (t, lt) where t is either 0 or another subdomain index, and lt is the corresponding local node index in subdomain t. 
    subdomain_maps = {}
    for s in range(1, nrows*ncols + 1):
        maps = defaultdict(list)
        exterior_boundary_g = np.intersect1d(boundary_nodes[0], boundary_nodes[s], assume_unique=True)
        exterior_boundary = gtol[s][exterior_boundary_g]
        for loc, g in zip(exterior_boundary, exterior_boundary_g):
            maps[loc].append((0, g))
        for t in range(1, nrows*ncols + 1):
            if t == s:
                continue
            if version == 1:
                interface_boundary = np.intersect1d(boundary_nodes[s], subnodes_overlap[t], assume_unique=True)
            else:
                interface_boundary = np.intersect1d(boundary_nodes[s], subnodes[t], assume_unique=True)
            local_s = gtol[s][interface_boundary]
            local_t = gtol[t][interface_boundary]
            for ls, lt, g in zip(local_s, local_t, interface_boundary):
                maps[ls].append((t, lt))
        subdomain_maps[s] = maps
    return DecompositionInfo(submeshes = submeshes, ltog = ltog, gtol = gtol, subdomain_maps = subdomain_maps, membership = np.array(membership, dtype = int), version = version)