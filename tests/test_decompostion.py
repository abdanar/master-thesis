from fem.mesh import Mesh
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np

# Square mesh
vert = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh_square = Mesh(vert, options = 'qa0.015')

# Rectangle mesh
rvert = np.array([[0,0],[3,0],[3,1],[0,1]])
mesh_rectangle = Mesh(rvert, options = 'st, dx=0.25, dy=0.25')

# Decompose the square domain into the nonoverlapping subdomains and build a color array representing the subdomain decomposition of the mesh.
subdomains, _, _, membership = mesh_square.decompose(n = 4, overlap = 1)

# Decompose the square domain into the nonoverlapping subdomains and build a color array representing the subdomain decomposition of the mesh.
_, _, _, rmembership = mesh_rectangle.decompose(n = 3, mtype = 'vertical')


def visglobal(mesh: Mesh, carray: np.ndarray, filename: str):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    cmap = plt.get_cmap("Set3")

    ax.tripcolor(
        mesh.vertices[:,0],
        mesh.vertices[:,1],
        triangles=mesh.elements,
        facecolors=carray % cmap.N,  # cycle colors if > cmap.N
        edgecolors='k',
        cmap=cmap
    )

    for i, tri in enumerate(mesh.elements):
        centroid = mesh.vertices[tri].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(i+1),
            color='black',
            fontsize=6,
            ha='center',
            va='center',
            clip_on=True
        )

    ax.scatter(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        s=15,
        c="black",
    )

    for i, (x, y) in enumerate(mesh.vertices):
        ax.text(
            x,
            y,
            str(i + 1),
            color="black",
            fontsize=7,
            ha="right",
            va="bottom",
            clip_on=True
        )

    ax.set_title("Global Mesh")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{filename}_global.svg")
    plt.show()

def visualize(mesh: Mesh, carray: np.ndarray, subdomains: list, filename: str, ncols: int = 4):

    n_sub = len(subdomains)
    nplots = n_sub + 1
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), dpi=200)
    axes = axes.flatten()

    cmap = plt.get_cmap("Set3")

    # ---- Plot global mesh ----
    ax = axes[0]   
    ax.tripcolor(
        mesh.vertices[:,0],
        mesh.vertices[:,1],
        triangles=mesh.elements,
        facecolors=carray % cmap.N,  # cycle colors if > cmap.N
        edgecolors='k',
        cmap=cmap
    )

    # ---- Global triangle numbering ----
    for i, tri in enumerate(mesh.elements):
        centroid = mesh.vertices[tri].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(i+1),
            color='black',
            fontsize=6,
            ha='center',
            va='center',
            clip_on=True
        )

    # ---- Global vertex markers ----
    ax.scatter(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        s=15,
        c="black",
    )

    # ---- Global vertex numbering ----
    for i, (x, y) in enumerate(mesh.vertices):
        ax.text(
            x,
            y,
            str(i + 1),
            color="black",
            fontsize=7,
            ha="right",
            va="bottom",
            clip_on=True
        )

    ax.set_title("Global Mesh")
    ax.set_aspect('equal')

    # ---- Plot each subdomain separately ----
    for i, sub in enumerate(subdomains):
        ax = axes[i+1]
        ax.triplot(
            sub.vertices[:,0],
            sub.vertices[:,1],
            triangles=sub.elements,
            color='k',   # edge color
            linewidth=0.8  # optional, adjust thickness
        )

        # Local triangle numbering
        for j, tri in enumerate(sub.elements):
            centroid = sub.vertices[tri].mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(j+1),
                color='black',
                fontsize=7,
                ha='center',
                va='center',
                clip_on=True
            )

        ax.scatter(
            sub.vertices[:, 0],
            sub.vertices[:, 1],
            s=15,
            c="black",
        )

        # Local vertex numbering
        for k, (x, y) in enumerate(sub.vertices):
            ax.text(
                x,
                y,
                str(k+1),
                color='black',
                fontsize=6,
                ha='right',
                va='bottom',
                clip_on=True
            )

        ax.set_title(f"Subdomain {i+1}")
        ax.set_aspect('equal')

    # Remove extra axes if needed
    for j in range(nplots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{filename}.svg")
    plt.show()

# Visualize square unstructured mesh
visualize(mesh = mesh_square, carray = membership, subdomains = subdomains, filename = 'unstsq')

visglobal(mesh = mesh_square, carray = membership, filename = 'unstsq')

# Visualize rectangle structured mesh
# visualize(mesh = mesh_rectangle, carray = rmembership, filename = 'strec')