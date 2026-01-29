#!/usr/bin/env python3
"""Test hemisphere suffix is applied to source_df type column."""

import sys
import pandas as pd

sys.path.insert(0, '/Users/apple/Documents/GitHub/hemibrain-connectomes-analysis-v3.1/src')
from coana import FindNeuronConnection

# Create FNC instance with separate_hemispheres=True
fnc = FindNeuronConnection(
    dataset='male-cns:v0.9',
    separate_hemispheres=True
)

# Set source using a simple query
fnc.SetSource(type_name='PPL101')

print("=== SOURCE_DF AFTER SetSource ===")
print(fnc.source_df[['bodyId', 'type', 'instance']].head(10))

# Check if type column has hemisphere suffixes
print("\n=== CHECKING TYPE COLUMN ===")
for idx, row in fnc.source_df.iterrows():
    t = row['type']
    i = row.get('instance', 'N/A')
    print(f"  bodyId={row['bodyId']}, type='{t}', instance='{i}'")
    if isinstance(t, str) and not t.endswith(('_L', '_R', '_U')):
        print(f"    ⚠️ Type '{t}' does NOT have hemisphere suffix!")
    elif isinstance(t, str):
        print(f"    ✓ Type '{t}' has hemisphere suffix")
    else:
        print(f"    ⚠️ Type is not a string: {type(t)}")
