#!/usr/bin/env python
"""Quick test to verify connection_ratio and traversal_probability are correctly calculated."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from comparison.comparison_analyzer import ComparisonAnalyzer, ComparisonParameters
import polars as pl

params = ComparisonParameters(
    datasets=['male-cns:v0.9'],
    source_neurons=['aMe12'],
    target_neurons=['PPL101'],
    thresholds=[3],
    comparison_mode='path',
)

analyzer = ComparisonAnalyzer(params)
results = analyzer.run_comparison()

# Get the actual output path from params
output_base = params.full_output_path
edge_file = os.path.join(output_base, 'dataset_data/male-cns_v0_9/minsyn_3/data_details/connection_type.csv')
print(f'\nLooking for edge file at: {edge_file}')
try:
    df = pl.read_csv(edge_file)
    print('=== connection_type.csv ===')
    print(df.select(['type_pre', 'type_post', 'weight', 'connection_ratio', 'traversal_probability']).head(10))
    
    # Find aMe12->KCg-d connection
    aMe12_KCgd = df.filter((pl.col('type_pre') == 'aMe12') & (pl.col('type_post') == 'KCg-d'))
    if len(aMe12_KCgd) > 0:
        row = aMe12_KCgd.row(0, named=True)
        ratio = row['connection_ratio']
        trav_prob = row['traversal_probability']
        expected_trav_prob = min(1.0, ratio / 0.3)
        print(f'\nVerification for aMe12->KCg-d:')
        print(f'  connection_ratio = {ratio:.6f}')
        print(f'  traversal_probability = {trav_prob:.6f}')
        print(f'  Expected (ratio/0.3 capped at 1.0) = {expected_trav_prob:.6f}')
        
        if abs(trav_prob - expected_trav_prob) < 0.0001:
            print('  ✅ traversal_probability is correctly calculated!')
        else:
            print('  ❌ traversal_probability does NOT match expected value!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
