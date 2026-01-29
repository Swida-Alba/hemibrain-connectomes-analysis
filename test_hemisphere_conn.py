#!/usr/bin/env python3
"""Test _apply_hemisphere_suffix_to_conn_df function."""

import pandas as pd
import sys
sys.path.insert(0, 'src')
from coana import FindNeuronConnection

# Create mock connection dataframe similar to output
conn_df = pd.DataFrame({
    'bodyId_pre': ['123', '456'],
    'bodyId_post': ['789', '012'],
    'type_pre': ['AN05B101', 'APL'],  # Without suffix
    'type_post': ['PPL101', 'PPL103'],  # Without suffix (this is the problem)
    'instance_pre': ['AN05B101(dorsal_LAL_layer)_R', 'APL_L'],
    'instance_post': ['PPL101(y1ped)_R', 'PPL103(y2a2a)_L'],  # Has suffix!
    'hemisphere_code_pre': ['R', 'L'],
    'hemisphere_code_post': ['R', 'L'],
})

print('=== BEFORE ===')
print(conn_df[['type_pre', 'type_post', 'instance_post', 'hemisphere_code_post']])

# Create FindNeuronConnection with separate_hemispheres=True
fnc = FindNeuronConnection(
    source_criteria='test',
    target_criteria='test',
    dataset='male-cns:v0.9',
    separate_hemispheres=True,
    verbose=False
)

result = fnc._apply_hemisphere_suffix_to_conn_df(conn_df)
print()
print('=== AFTER ===')
print(result[['type_pre', 'type_post', 'instance_post', 'hemisphere_code_post']])

# Verify
print()
if result['type_pre'].iloc[0] == 'AN05B101_R' and result['type_post'].iloc[0] == 'PPL101_R':
    print('✓ TEST PASSED: Both type_pre and type_post have hemisphere suffixes')
else:
    print('✗ TEST FAILED:')
    print(f'  Expected type_pre: AN05B101_R, got: {result["type_pre"].iloc[0]}')
    print(f'  Expected type_post: PPL101_R, got: {result["type_post"].iloc[0]}')
