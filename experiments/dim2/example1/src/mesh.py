import matplotlib.pyplot as plt
from visualization.visualize import MeshVisualizer
import experiments.dim2.example1.src.setup as setup

# Visualize the original mesh with non-overlapping subdomains
visualizer = MeshVisualizer(setup.mesh2D)
submeshes, ltog, gtol, subdomain_maps, membership = setup.decomposition_info.submeshes, setup.decomposition_info.ltog, setup.decomposition_info.gtol, setup.decomposition_info.subdomain_maps, setup.decomposition_info.membership

# Visualize the non-overlapping mesh decomposition into 4 subdomains
fig, _ = visualizer.plot_mesh(carray = membership, figsize = (6, 6), title = rf"Mesh decomposition into 4 subdomains")

# Visualize the overlapping subdomains
fig_sub, _ = visualizer.plot_subdomains(subdomains = submeshes, membership = membership, figsize = (24, 6), ncols = 5)

# Save the figures
fig.savefig(setup.fig_dir/f"mesh/mesh_4.svg", dpi = 300, bbox_inches = 'tight')
fig_sub.savefig(setup.fig_dir/f"mesh/mesh_4_subdomains.svg", dpi = 300, bbox_inches = 'tight')

plt.show()