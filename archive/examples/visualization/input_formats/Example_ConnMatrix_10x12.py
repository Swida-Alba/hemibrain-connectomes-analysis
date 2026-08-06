import numpy as np
import pandas as pd
from vispath_pkg.vispath import VisualizePath

# Generate a 10x12 connection matrix with random weights
rows = 10
cols = 12
row_names = [f"Neuron_{i+1}" for i in range(rows)]
col_names = [f"Neuron_{j+1}" for j in range(cols)]
np.random.seed(42)
mat = np.random.poisson(lam=2, size=(rows, cols))
mat = mat.astype(float)
mat[mat < 1] = 0  # Make it sparse

conn_df = pd.DataFrame(mat, index=row_names, columns=col_names)
conn_df.to_csv("examples/Example_ConnMatrix_10x12.csv")

# Load and visualize as heatmap
vp = VisualizePath(conn_df)
vp.showfig = True
vp.visualize()
