import numpy as np
import meshio
import os
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize  # <- correct import
import matplotlib.cm as cm
import fem.mesh as mesh

class MeshVisualizer:
    def __init__(self, meshobj: mesh.Mesh):
        if not isinstance(meshobj, mesh.Mesh):
            raise TypeError("MeshVisualizer expects a Mesh object")
        self.mesh = meshobj

    # Build a color array representing the areas of the triangles in the mesh.
    def carray_areas(self) -> np.ndarray:
        return self.mesh.measures()

    # Build a color array where boundary triangles get color `1` and interior triangles get color `0`.
    def carray_boundary(self) -> np.ndarray:
        colors = np.zeros(self.mesh.elements.shape[0], dtype=int)
        bdtriangles = self.mesh.boundary_elements()
        for i, triangle in enumerate(self.mesh.elements):
            if tuple(triangle) in bdtriangles:
                colors[i] = 1
        return colors
    
    # Build a color array representing the subdomain decomposition of the mesh.
    def carray_decomposition(self, n: int, direction = None) -> np.ndarray:
        _, _, _, membership = self.mesh.decompose(n = n, direction = direction)  # Example with n subdomains
        return np.array(membership)

    def subdomain_boundaries(self, subdomains: list):
        sbd = dict()
        for i in range(len(subdomains)):
            print(subdomains[i].boundary_nodes().shape)
            sbd[i+1] = subdomains[i].boundary_nodes()
        return sbd
    # not correct, change edges
    def bvisualize(self, subdomains: list):

        sb = self.subdomain_boundaries(subdomains)

        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]

        # Create a triangulation
        triang = mtri.Triangulation(x, y, self.mesh.elements)

        fig, ax = plt.subplots()
        ax.triplot(triang, color='lightgray', linewidth=0.5)    

        # Choose a colormap (e.g., tab20 has 20 distinct colors)
        cmap = cm.get_cmap('tab20')  

        # Normalize subdomain IDs to [0,1] for the colormap
        sd_ids = list(sb.keys())
        norm = Normalize(vmin=min(sd_ids), vmax=max(sd_ids))

        for sd_id, edges in sb.items():
            color = cmap(norm(sd_id))  # dynamically assign color
            lc = LineCollection(edges, colors=color, linewidths=2)
            ax.add_collection(lc)

        ax.autoscale()
        ax.set_aspect('equal')
        plt.show()

    def visualize(self, carray: np.ndarray):
        plt.figure(figsize = (6,6), dpi = 150)
        plt.tripcolor(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], triangles = self.mesh.elements, facecolors = carray, edgecolors = "k")
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.show()

class SolutionVisualizer:
    def __init__(self, meshobj: mesh.Mesh, u: np.ndarray, dt=None):

        """
        Parameters
        ----------
        meshobj : Mesh
            Your FEM mesh
        u : np.ndarray
            Solution array, shape (ndofs, ntime)
        dt : float, optional
            Time step size
        """

        self.mesh = meshobj
        self.u = u
        if u.ndim == 1:
            u = u[:, np.newaxis]
        self.u = u
        self.dt = dt
        self.ntime = u.shape[1]

    def plot_iteration_error(self, error_history, logscale=True, marker='o', title = r"Iteration vs $L^{2}$ Error", **kwargs):
        """
        Plot iteration number vs error.

        Parameters
        ----------
        error_history : list or array
            List of error values per iteration.
        logscale : bool, default True
            If True, use semilog-y scale (recommended for convergence plots).
        marker : str, default 'o'
            Marker style for points.
        title : str, default "Iteration vs L2 Error"
            Plot title.
        **kwargs : dict
            Additional keyword arguments to pass to plt.plot / plt.semilogy
            e.g., color='r', linewidth=2, linestyle='--', alpha=0.7
        """

        iterations = range(1, len(error_history) + 1)
        errors = error_history

        plt.figure(figsize=(6,4), dpi = 150)
        if logscale:
            plt.semilogy(iterations, errors, marker = marker, **kwargs)
        else:
            plt.plot(iterations, errors, marker = marker, **kwargs)

        plt.xlabel("Iteration")
        plt.ylabel(r"$\| u_h - u \|_{L^2(\Omega)}$")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize(self, cmap = 'viridis', levels = 50):

        """
        Plot FEM solution for 2D triangular mesh with linear Lagrange elements (degree 1).

        Parameters
        ----------
        cmap : str, optional
            Colormap for contour plot (default 'viridis').
        levels : int, optional
            Number of contour levels (default 50).
        """

        # Use mesh vertices and elements
        vertices = self.mesh.vertices
        elements = self.mesh.elements

        x = vertices[:, 0]
        y = vertices[:, 1]

        # Create a triangulation
        triang = mtri.Triangulation(x, y, elements)

        # Plot filled contour
        plt.figure(figsize=(6,5))
        plt.tricontourf(triang, self.u.ravel(), levels = levels, cmap = cmap)
        plt.triplot(triang, color = 'k', linewidth = 0.5, alpha = 0.3)
        plt.colorbar(label = 'Solution u')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('FEM Solution (Linear Lagrange)')
        plt.show()

    def visualize_3d(self, cmap = 'viridis'):

        """
        Plot FEM solution for 2D triangular mesh as a 3D surface.

        Parameters
        ----------
        cmap : str, optional
            Colormap for surface plot (default 'viridis').
        """

        # Use mesh vertices and elements
        vertices = self.mesh.vertices
        elements = self.mesh.elements

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = self.u.ravel()  # Ensure z is 1D

        # Create a triangulation
        triang = mtri.Triangulation(x, y, elements)

        # Create 3D figure
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_trisurf(triang, z, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Solution u')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title('FEM Solution 3D Surface')

        plt.show()

    def visualize_3d_compare(self, u_exact_func=None, cmap='viridis'):
        """
        Plot FEM solution for 2D triangular mesh as a 3D surface,
        optionally comparing with the exact solution side by side.

        Parameters
        ----------
        u_exact_func : callable, optional
            Function u_exact(x, y) returning the exact solution.
            If provided, a side-by-side comparison will be plotted.
        cmap : str, optional
            Colormap for surface plot (default 'viridis').
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.tri as mtri

        vertices = self.mesh.vertices
        elements = self.mesh.elements

        x = vertices[:, 0]
        y = vertices[:, 1]
        z_fem = self.u.ravel()  # FEM solution

        triang = mtri.Triangulation(x, y, elements)

        if u_exact_func is None:
            # Single plot
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_trisurf(triang, z_fem, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Solution u')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
            ax.set_title('FEM Solution 3D Surface')
            plt.show()
        else:
            # Side-by-side plots
            fig = plt.figure(figsize=(14,6))

            # FEM solution
            ax1 = fig.add_subplot(121, projection='3d')
            surf1 = ax1.plot_trisurf(triang, z_fem, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='u_h')
            ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
            ax1.set_title('FEM Solution')

            # Exact solution
            z_exact = u_exact_func(x, y)
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = ax2.plot_trisurf(triang, z_exact, cmap=cmap, edgecolor='k', linewidth=0.2, antialiased=True)
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='u_exact')
            ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
            ax2.set_title('Exact Solution')

            plt.show()

    def visualize_3d_time(self, cmap = 'viridis'):
        """
        Interactive 3D visualization with time slider.
        """
        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]
        elements = self.mesh.elements

        triang = mtri.Triangulation(x, y, elements)

        fig = plt.figure(figsize = (8,6), dpi = 100)
        ax = fig.add_subplot(111, projection = '3d')
        plt.subplots_adjust(bottom = 0.25)

        # Initial surface
        z = self.u[:, 0]
        surf = ax.plot_trisurf(triang, z, cmap = cmap, edgecolor = 'k', linewidth = 0.2, antialiased = True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(f'FEM Solution 3D Surface | t = 0.0')

        fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 10, label = 'Solution u')

        # Slider axes
        ax_slider = plt.axes((0.25, 0.1, 0.5, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime - 1, valinit = 0, valstep = 1)

        def update(val):
            step = int(time_slider.val)
            ax.clear()
            z = self.u[:, step]
            surf = ax.plot_trisurf(triang, z, cmap = cmap, edgecolor = 'k', linewidth = 0.2, antialiased = True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f'FEM Solution 3D Surface | t = {step*self.dt:.3f}')
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()

    def visualize_3d_time_compare(self, exact_func, cmap='viridis', nx=100, ny=100):
        """
        Compare numeric FEM solution (triangular mesh) with smooth exact solution
        on a fine grid for better visualization.
        
        Parameters:
        -----------
        exact_func : callable
            exact_func(x, y, t)
        cmap : str
            colormap
        nx, ny : int
            resolution of the grid for smooth exact solution
        """
        x_mesh = self.mesh.vertices[:, 0]
        y_mesh = self.mesh.vertices[:, 1]
        elements = self.mesh.elements
        triang = mtri.Triangulation(x_mesh, y_mesh, elements)

        # Create fine grid for smooth exact solution
        x_grid = np.linspace(0, 1, nx)
        y_grid = np.linspace(0, 1, ny)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        # Initial surfaces
        z_exact = exact_func(X_grid, Y_grid, 0.0)
        z_num = self.u[:, 0]

        surf1 = ax1.plot_surface(X_grid, Y_grid, z_exact, cmap=cmap, edgecolor='none')
        ax1.set_title('Exact Solution (Smooth)')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='u')

        surf2 = ax2.plot_trisurf(triang, z_num, cmap=cmap, edgecolor='k', linewidth=0.2)
        ax2.set_title('Numerical Solution')
        ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='u')

        # Slider
        ax_slider = plt.axes((0.25, 0.1, 0.5, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime-1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            t = step*self.dt

            ax1.cla()
            ax2.cla()

            # Smooth exact
            z_exact = exact_func(X_grid, Y_grid, t)
            ax1.plot_surface(X_grid, Y_grid, z_exact, cmap=cmap, edgecolor='none')
            ax1.set_title(f'Exact Solution | t={t:.3f}')
            ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')

            # Numeric
            z_num = self.u[:, step]
            ax2.plot_trisurf(triang, z_num, cmap=cmap, edgecolor='k', linewidth=0.2)
            ax2.set_title(f'Numerical Solution | t={t:.3f}')
            ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')

            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()
    
    def visualize_3d_time_error(self, exact_func, cmap='coolwarm'):
        """
        Visualize nodal FEM error |u_h - u_exact| over time.

        The error is computed at mesh vertices only, which is
        well-defined for Schwarz methods and higher-order FEM.

        Parameters
        ----------
        exact_func : callable
            exact_func(x, y, t)
        cmap : str
            Colormap for error visualization
        """

        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]
        elements = self.mesh.elements
        triang = mtri.Triangulation(x, y, elements)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)

        # Initial error
        t0 = 0.0
        u_exact = exact_func(x, y, t0)
        u_num = self.u[:, 0]
        err = u_num - u_exact

        surf = ax.plot_trisurf(triang, err, cmap=cmap, edgecolor='k', linewidth=0.2)
        ax.set_title(f'Nodal Error | t={t0:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('error')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='u_h - u_exact')

        # Slider
        ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime - 1,
                            valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            t = step * self.dt

            ax.cla()

            u_exact = exact_func(x, y, t)
            u_num = self.u[:, step]
            err = u_num - u_exact

            ax.plot_trisurf(triang, err, cmap=cmap, edgecolor='k', linewidth=0.2)
            ax.set_title(f'Nodal Error | t={t:.3f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('error')

            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()

    def write_vtk(self, filename: str, exact):
        """
        Write FEM solution, exact solution, and error
        for the Poisson problem -Δu = f to a VTK (.vtu) file.
        """

        if not filename.endswith(".vtu"):
            filename += ".vtu"

        # Mesh points
        points = self.mesh.vertices
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0])])

        # Linear triangular elements (P1)
        cells = [("triangle", self.mesh.elements.astype(np.int64))]

        # Numerical solution
        u_h = self.u.ravel()

        # Exact solution at nodes
        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]
        u_ex = exact(x, y)

        # Error
        error = u_h - u_ex

        mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={
                "u_h": u_h,
                "u_exact": u_ex,
                "error": error,
                "abs_error": np.abs(error),
            }
        )

        meshio.write(filename, mesh)

    def write_vtk_time(self, filename: str, step: int = 0):
        """
        Write FEM solution at a given time step to a VTK (.vtu) file
        readable by ParaView.

        Parameters
        ----------
        filename : str
            Output file name (with or without .vtu)
        step : int, optional
            Time step index (default 0)
        """

        if not filename.endswith(".vtu"):
            filename += ".vtu"

        # Mesh data
        points = self.mesh.vertices
        if points.shape[1] == 2:
            # VTK expects 3D points
            points = np.column_stack([points, np.zeros(len(points))])

        cells = [("triangle", self.mesh.elements)]

        # Solution at time step
        u_step = self.u[:, step]

        mesh = meshio.Mesh(
            points=points,
            cells = cells,
            point_data={"u": u_step}
        )

        meshio.write(filename, mesh)

    def write_vtk_time_series(self, exact_func, folder="vtk_output", prefix="solution"):
        """
        Write time-dependent FEM solution, exact solution,
        and nodal error to a VTK time series for ParaView.

        Point data written:
        - u_h      : numerical FEM solution
        - u_exact  : exact solution at mesh vertices
        - error    : u_h - u_exact
        """

        os.makedirs(folder, exist_ok=True)

        # --- Mesh points ---
        points = self.mesh.vertices
        if points.ndim == 2 and points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0])])

        cells = [("triangle", self.mesh.elements)]

        x = self.mesh.vertices[:, 0]
        y = self.mesh.vertices[:, 1]

        pvd_entries = []

        for k in range(self.ntime):
            t = k * self.dt if self.dt is not None else 0.0

            # Numerical solution
            u_h = self.u[:, k]

            # Exact solution at vertices
            u_exact = exact_func(x, y, t)

            # Error
            error = u_h - u_exact

            mesh = meshio.Mesh(
                points=points,
                cells=cells,
                point_data={
                    "u_h": u_h,
                    "u_exact": u_exact,
                    "error": error
                }
            )

            vtu_name = f"{prefix}_{k:04d}.vtu"
            vtu_path = os.path.join(folder, vtu_name)

            meshio.write(vtu_path, mesh)

            pvd_entries.append((vtu_name, t))

        # --- Write PVD file ---
        pvd_path = os.path.join(folder, f"{prefix}.pvd")
        with open(pvd_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write("  <Collection>\n")
            for name, t in pvd_entries:
                f.write(
                    f'    <DataSet timestep="{t}" file="{name}"/>\n'
                )
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")

    def c_write_vtk_time_series(self, exact_func, folder="vtk_output", prefix="solution"):

        os.makedirs(folder, exist_ok=True)

        # --- Mesh points ---
        points = self.mesh.vertices  # all nodes including mid-edge/interior

        # --- Cells (choose type based on degree) ---
        if self.mesh.degree == 1:
            cells = [("triangle", self.mesh.elements)]
        elif self.mesh.degree == 2:
            cells = [("triangle6", self.mesh.elements)]
        elif self.mesh.degree == 3:
            cells = [("triangle10", self.mesh.elements)]
        else:
            raise NotImplementedError(f"Degree {self.mesh.degree} not supported")

        x = points[:, 0]
        y = points[:, 1]

        pvd_entries = []

        for k in range(self.ntime):
            t = k * self.dt if self.dt is not None else 0.0

            # Numerical solution (all DOFs)
            u_h = self.u[:, k]

            # Exact solution at mesh points
            u_exact = exact_func(x, y, t)

            # Error
            error = u_h - u_exact

            mesh = meshio.Mesh(
                points=points,
                cells=cells,
                point_data={
                    "u_h": u_h,
                    "u_exact": u_exact,
                    "error": error
                }
            )

            vtu_name = f"{prefix}_{k:04d}.vtu"
            vtu_path = os.path.join(folder, vtu_name)
            meshio.write(vtu_path, mesh)

            pvd_entries.append((vtu_name, t))

        # --- Write PVD file ---
        pvd_path = os.path.join(folder, f"{prefix}.pvd")
        with open(pvd_path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1">\n')
            f.write("  <Collection>\n")
            for name, t in pvd_entries:
                f.write(f'    <DataSet timestep="{t}" file="{name}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")

    def visualize_1d_time_compare(self, exact_func, nx=200):
        """
        Compare 1D FEM solution with exact solution over time.
        
        exact_func(x, t)
        """

        idx = np.argsort(self.mesh.vertices)
        x_mesh = self.mesh.vertices[idx]
        x_grid = np.linspace(x_mesh.min(), x_mesh.max(), nx)

        fig, ax = plt.subplots(figsize=(8,5))
        plt.subplots_adjust(bottom=0.25)

        # Initial plot
        z_exact = exact_func(x_grid, 0.0)
        z_num = self.u[:, 0][idx]

        line_exact, = ax.plot(x_grid, z_exact, 'r-', label='Exact')
        line_num, = ax.plot(x_mesh, z_num, 'bo-', label='Numerical')
        ax.set_xlabel('x'); ax.set_ylabel('u')
        ax.set_title('Time = 0.0')
        ax.legend()
        
        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        time_slider = Slider(ax_slider, 'Time step', 0, self.ntime-1, valinit=0, valstep=1)

        def update(val):
            step = int(time_slider.val)
            t = step*self.dt
            line_exact.set_ydata(exact_func(x_grid, t))
            line_num.set_ydata(self.u[:, step][idx])
            ax.set_title(f'Time = {t:.3f}')
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()