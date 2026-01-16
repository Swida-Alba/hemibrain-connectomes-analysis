#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')
from coana import FindNeuronConnection
import polars as pl
import os

fnc = FindNeuronConnection(
    dataset='male-cns:v0.9',
    sourceNeurons=['aMe12'],
    targetNeurons=['PPL101'],
    min_synapse_num=3,
    output_dir='/tmp/test_findallpath',
    verbose_mode='simple'
)

fnc.InitializeNeuronInfo()
fnc.FindAllPath()

print('\nLooking for connection_type.csv...')
for root, dirs, files in os.walk('/tmp/test_findallpath'):
    for f in files:
        if 'connection_type' in f:
            path = os.path.join(root, f)
            print(f'Found: {path}')
            df = pl.read_csv(path)
            print(df.select(['type_pre', 'type_post', 'weight', 'connection_ratio', 'traversal_probability']).head(5))
            
            if len(df) > 0:
                row = df.row(0, named=True)
                ratio = row['connection_ratio']
                trav_prob = row['traversal_probability']
                expected_trav_prob = min(1.0, ratio / 0.3)
                print(f'  connection_ratio = {ratio:.6f}')
                print(f'  traversal_probability = {trav_prob:.6f}')
                print(f'  Expected = {expected_trav_prob:.6f}')
                if abs(trav_prob - expected_trav_prob) < 0.0001:
                    print('  ✅ PASS')
                else:
                    print('  ❌ FAIL')
