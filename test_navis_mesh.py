
import navis
import numpy as np
import pandas as pd
import trimesh

# Create a dummy neuron
nodes = pd.DataFrame({
    'node_id': [1, 2, 3, 4],
    'x': [0, 10, 20, 30],
    'y': [0, 0, 0, 0],
    'z': [0, 0, 0, 0],
    'radius': [1, 1, 1, 1],
    'parent_id': [-1, 1, 2, 3]
})
n = navis.TreeNeuron(nodes)

print("Neuron created")

try:
    print("Calling navis.conversion.tree2meshneuron(n)...")
    m = navis.conversion.tree2meshneuron(n)
    tm = m.trimesh
    print("Trimesh created")
    
    # Try to create MeshNeuron from trimesh
    print("Creating MeshNeuron from trimesh...")
    # According to docs/source code, MeshNeuron can take a trimesh object
    m2 = navis.MeshNeuron(tm)
    print("Created MeshNeuron from trimesh:", type(m2))
    print(m2)
    
except Exception as e:
    print(f"Error: {e}")
