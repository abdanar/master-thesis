from mesh import Mesh
from refelement import ReferenceElement
from phyelement import PhysicalElement
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
        """
        Initialize a finite element function space over a given mesh.

        Parameters
        ----------
        mesh : Mesh
            The mesh defining the computational domain. If `mesh.domainID == 0`
            (representing the whole domain), it will be upgraded using `self.upgrade(mesh)`.
        domain : str, default 'triangle'
            The type of domain elements. For example, 'triangle' for 2D triangular elements.
        space : str, default 'Lagrange'
            The type of finite element space. Common choices include 'Lagrange'.
        degree : int, default 1
            The polynomial degree of the finite element space.

        Attributes
        ----------
        self.mesh : Mesh
            The (possibly upgraded) mesh associated with this finite element space.
        self.degree : int
            The polynomial degree.
        self.domain : str
            The domain type of elements.
        self.space : str
            The finite element space type.
        self.dim : int
            The spatial dimension of the mesh.

        Notes
        -----
        - The `upgrade(mesh)` method is only called if `mesh.domainID == 0`, i.e., for the
          whole domain. Otherwise, the original mesh is used.
        - After initialization, `self.mesh` is guaranteed to be available and consistent
          for all FEM operations.
        - `self.dim` is taken from the original mesh; it represents the geometric dimension.
        """
        self.degree = degree
        self.domain = domain
        self.space = space
        self.dim = mesh.dim
        self.mesh = self.upgrade(mesh) if mesh.domainID == 0 and mesh.degree == 1 else mesh # upgrade only if the mesh is the given whole domain
        
    def upgrade(self, mesh: Mesh) -> Mesh:
        """
        Upgrade the mesh for higher-degree Lagrange finite elements.
        
        This method generates a new `Mesh` instance with additional nodes
        corresponding to higher-order Lagrange elements. It supports both 
        1D (interval) and 2D (triangular) meshes.

        Parameters
        ----------
        mesh : Mesh, only mesh with degree 1 (mesh.degree = 1)
            The original mesh to upgrade. The method does not modify the input mesh.

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
        - Edge node indices start after the original vertices, i.e., at `mesh.nvertices()`.
        - Interior node indices start after all original vertices and edge nodes, 
            i.e., at `mesh.nvertices() + mesh.nedges() * (degree - 1)`.
        - Dictionaries are used to track unique edge and interior nodes to avoid
          duplicates when elements share edges.
        
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
        if mesh.domainID != 0:
            raise ValueError("Mesh upgrade is only allowed for the whole domain (domainID == 0). Subdomain meshes must be created after upgrading the global mesh.")
        
        if mesh.degree != 1:
            raise ValueError("Mesh upgrade is only allowed for degree 1 Mesh. A mesh that has already been upgraded cannot be upgraded again.")
        
        if self.dim == 1 and self.domain == 'interval':
            updated_vertices = list(mesh.vertices)
            updated_elements = np.zeros((mesh.nelements(), self.degree + 1), dtype = int)
            next_index = mesh.nvertices()
            for i, edge in enumerate(mesh.elements):
                a, b = mesh.vertices[edge]
                interior_nodes = np.linspace(a, b, self.degree + 1)[1:-1]
                interior_indices = []
                for node in interior_nodes:
                    updated_vertices.append(node)
                    interior_indices.append(next_index)
                    next_index += 1
                updated_elements[i] = [edge[0]] + interior_indices + [edge[1]]
            updated_vertices = np.array(updated_vertices).reshape(-1)
        elif self.dim == 2 and self.domain == 'triangle':
            nel = mesh.nelements()
            nvert = mesh.nvertices()
            nedg = mesh.nedges() # <- number of geometric edges
            pedge = 3*(self.degree - 1) # number of edge nodes per triangle

            updated_vertices = np.zeros((nvert + nedg*(self.degree - 1) + nel*(self.degree - 1)*(self.degree - 2)//2, 2))
            updated_elements = np.zeros((nel, (self.degree + 1)*(self.degree + 2)//2), dtype = int) # (self.degree + 1)*(self.degree + 2)//2 is the total number of nodes per triangle
            updated_elements[:, :3] = mesh.elements
            updated_vertices[:nvert, :] = mesh.vertices 

            edge_nodes_dict = defaultdict(int)
            interior_nodes_dict = defaultdict(int)

            edge_count = nvert
            interior_count = nvert + nedg*(self.degree - 1)
            for i, element in enumerate(mesh.elements):
                nodes = PhysicalElement(vertices = mesh.vertices[element], ref_element = ReferenceElement(self.dim, self.domain, self.space, self.degree)).physical_reference_nodes()
                edge_nodes = nodes[3: 3 + pedge, :]
                interior_nodes = nodes[3 + pedge:, :]
                for j, enode in enumerate(edge_nodes):
                    key_node = tuple(np.round(enode, decimals = 12)) # 12 is chosen arbitrarily
                    if key_node not in edge_nodes_dict:
                        edge_nodes_dict[key_node] = edge_count
                        updated_vertices[edge_count] = enode
                        edge_count += 1
                    updated_elements[i, j + 3] = edge_nodes_dict[key_node]
                for k, inode in enumerate(interior_nodes):
                    key_node = tuple(np.round(inode, decimals = 12)) # 12 is chosen arbitrarily
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
        upgraded_mesh = Mesh(vertices = updated_vertices, elements = updated_elements, dim = self.dim, domainID = mesh.domainID, options = mesh.options)
        upgraded_mesh.degree = self.degree
        return upgraded_mesh

    # Following features are expected to be here for FEMSpace
    # - evaluate global FE solution u at a point x; vectorized evaluation at multiple points (for plotting).
    # - project a function f(x) onto the global FEM space.
    # - interpolate a function f(x) onto the global FEM space.
    # - get_physical_element(element_index) → returns PhysicalElement instance for given element index.
    # + get_shape_functions(element_index) → returns shape functions for given element index.

    # change for 1d
    def get_physical_element(self, element_index: int) -> PhysicalElement:
        refelement = ReferenceElement(self.dim, self.domain, self.space, self.degree)
        return PhysicalElement(vertices = self.mesh.vertices[self.mesh.elements[element_index][:3]], ref_element = refelement)

    def get_shape_functions(self, element_index: int) -> dict:
        """
        Return the Lagrange shape functions for a specific mesh element,
        keyed by global node indices.

        This method retrieves the physical element corresponding to the given
        `element_index` and returns a dictionary of callable shape functions
        associated with its nodes. Each function can be evaluated at any point
        in physical coordinates. The dictionary keys are the **global node indices**,
        which makes this convenient for assembling global matrices and vectors.

        Parameters
        ----------
        element_index : int
            Index of the element in the mesh. Must satisfy
            `0 <= element_index < self.mesh.nelements()`.

        Returns
        -------
        phi_dict : dict[int, Callable]
            Dictionary mapping:

            - Key: global node index
            - Value: callable `phi(x_phys)` returning the value of the
              corresponding Lagrange shape function at the physical point `x_phys`.

        Notes
        -----
        - Local-to-global correspondence is determined by the ordering of the
        nodes in `self.mesh.elements[element_index]`.
        - Physical coordinates of nodes are accessed internally via the
        `PhysicalElement` object.
        - Each callable uses a closure to correctly bind its local basis function.
        - This method is a convenient wrapper around
        `get_physical_element(element_index).shape_functions(...)`.

        Example
        -------
            >>> # 2D triangular mesh with 4 vertices and 2 elements
            >>> mesh.vertices
            array([[0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]])
            >>> mesh.elements
            array([[0, 1, 2],
                [1, 3, 2]])
            >>> # Get shape functions for the second element (index 1)
            >>> phi_dict1 = femspace.get_shape_functions(1)
            >>> x_phys = np.array([0.25, 0.25])
            >>> for g, phi_g in phi_dict1.items():
            ...     print(f"global {g}, phi(x_phys) = {phi_g(x_phys)}")
            global 1, phi(x_phys) = 0.1875
            global 3, phi(x_phys) = 0.0625
            global 2, phi(x_phys) = 0.75
        """
        return self.get_physical_element(element_index).shape_functions(self.mesh.elements[element_index])
    
    def find_element_containing(self, x) -> int | None:
        """
        Find the mesh element that contains a given point.

        Depending on the mesh dimension, this method delegates the search to the
        appropriate mesh routine:
        - in 1D, it locates the interval containing the point,
        - in 2D, it locates the triangle containing the point.

        Parameters
        ----------
        x : float or array-like
            Point coordinates. A scalar is expected for 1D meshes, while a
            length-2 array is expected for 2D meshes.

        Returns
        -------
        element_index : int or None
            Index of the element that contains the point. Returns None if the
            point lies outside the mesh.

        Raises
        ------
        ValueError
            If the mesh dimension is not supported. Only 1D and 2D meshes are
            supported.
        """
        if self.dim == 1:
            return self.mesh.locate_interval(x)
        elif self.dim == 2:
            return self.mesh.locate_triangle(x)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}. Only 1D and 2D meshes are supported.")

    def evaluate_solution(self, U: np.ndarray, x) -> float:
        """
        Evaluate the global FEM solution at a given physical point.

        Parameters
        ----------
        U : np.ndarray
            Global FEM coefficient vector, with one entry per global node.
        x : float or np.ndarray
            Physical coordinates of the point to evaluate:
            - 1D interval: scalar or 1-element array
            - 2D triangle: array-like with shape (2,)

        Returns
        -------
        float
            Value of the FEM solution u_h at the point x.

        Notes
        -----
        - Locates the element containing `x` and evaluates the solution
          using the element's shape functions and global coefficients.
        - Works for both 1D and 2D FEM spaces.
        """
        # Find the element containing x
        element_index = self.find_element_containing(x)

        # Get the shape functions for this element
        phi_dict = self.get_shape_functions(element_index)

        # Evaluate u_h(x) = sum(U[g]*phi_g(x))
        return sum(U[g]*phi_g(x) for g, phi_g in phi_dict.items())
    
    def visualize_3d_time_compare(self, u, exact_func, dt, cmap='viridis', nx=100, ny=100):
        """
        Compare numeric FEM solution with exact solution on a fine grid for any element degree.

        Parameters
        ----------
        exact_func : callable
            Exact solution function, exact_func(x, y, t).
        cmap : str
            Colormap for surfaces.
        nx, ny : int
            Resolution of the fine grid for smooth visualization.
        """

        ntime = u.shape[1]

        # Create fine grid
        x_grid = np.linspace(0, 1, nx)
        y_grid = np.linspace(0, 1, ny)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        # Initial time step
        t0 = 0.0

        # Evaluate exact solution
        Z_exact = exact_func(X_grid, Y_grid, t0)
        surf1 = ax1.plot_surface(X_grid, Y_grid, Z_exact, cmap=cmap, edgecolor='none')
        ax1.set_title(f'Exact Solution | t={t0:.3f}')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='u')

        # Evaluate numeric solution on grid using evaluate_solution
        Z_num = np.zeros_like(X_grid)
        for i in range(nx):
            for j in range(ny):
                Z_num[i, j] = self.evaluate_solution(u[:, 0], [X_grid[i, j], Y_grid[i, j]])
        surf2 = ax2.plot_surface(X_grid, Y_grid, Z_num, cmap=cmap, edgecolor='k', linewidth=0.2)
        ax2.set_title(f'Numerical Solution | t={t0:.3f}')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='u')

        # Time slider
        ax_slider = plt.axes((0.25, 0.1, 0.5, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, ntime-1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            t = step*dt

            # Clear axes
            ax1.cla()
            ax2.cla()

            # Exact solution
            Z_exact = exact_func(X_grid, Y_grid, t)
            ax1.plot_surface(X_grid, Y_grid, Z_exact, cmap=cmap, edgecolor='none')
            ax1.set_title(f'Exact Solution | t={t:.3f}')
            ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')

            # Numeric solution interpolated on grid
            Z_num = np.zeros_like(X_grid)
            for i in range(nx):
                for j in range(ny):
                    Z_num[i, j] = self.evaluate_solution(u[:, step], [X_grid[i, j], Y_grid[i, j]])
            ax2.plot_surface(X_grid, Y_grid, Z_num, cmap=cmap, edgecolor='k', linewidth=0.2)
            ax2.set_title(f'Numerical Solution | t={t:.3f}')
            ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')

            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()