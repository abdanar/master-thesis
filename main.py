# main.py
import numpy as np
import mesh
import visualize 

# Create a simple square mesh
vert = np.array([[0,0],[1,0],[1,1],[0,1]])
mesh_obj = mesh.Mesh(vert, options = 'qa0.01')

# Create visualizer and show decomposition into 5 subdomains 
visualizer = visualize.MeshVisualizer(mesh_obj)
visualizer.visualize(visualizer.carray_decomposition(5))

