#!/usr/bin/env python
import polars as pl
import os
from pathlib import Path

# Check FAFB connections parquet
# Paths are resolved relative to this script so the check stays portable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
db_path = str(PROJECT_ROOT / 'cache' / 'flywire_FAFB_v783' / 'connections.parquet')
neuron_path = str(PROJECT_ROOT / 'cache' / 'flywire_FAFB_v783' / 'neuron_index.parquet')

print('=== FAFB Connection Cache Analysis ===')
print(f'Connection file: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB')
print(f'Neuron index: {os.path.getsize(neuron_path) / 1024 / 1024:.1f} MB')

# Use lazy scan to check row count without loading into memory
lazy_df = pl.scan_parquet(db_path)
schema = lazy_df.collect_schema()
print(f'Schema: {schema}')

# Get row count efficiently
row_count = lazy_df.select(pl.len()).collect().item()
print(f'Total rows: {row_count:,}')

# Check neuron index
neuron_df = pl.read_parquet(neuron_path)
print(f'Neuron index rows: {len(neuron_df):,}')
print(f'Neuron index columns: {neuron_df.columns}')
