import sys
from pathlib import Path
import pandas as pd

# Add vispath-subproject to Python path for local development
dir_path = Path(__file__).parent.parent / 'vispath-subproject' / 'src'
if dir_path.exists():
    sys.path.insert(0, str(dir_path))

from vispath_pkg import VisualizePath

# Create a simple 3x3 connection matrix
data = pd.DataFrame(
    [ [0, 5, 2],
      [1, 0, 3],
      [4, 0, 0] ],
    index=['A', 'B', 'C'],
    columns=['A', 'B', 'C']
)
print("Connection matrix:")
print(data)

# Visualize as heatmap
vp = VisualizePath(
    path_file=data,
    output_folder='./output_connection_matrix_heatmap',
    showfig=False
)
vp.visualize_heatmap()
print("\n✓ Heatmap visualization created in ./output_connection_matrix_heatmap/")
