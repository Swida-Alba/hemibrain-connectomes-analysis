import pandas as pd
from comparison.connectivity_profiler import ConnectivityProfiler, ProfilerConfig

def test_scratch(tmp_path):
    p = ConnectivityProfiler(['flywire_FAFB_v783'], config=ProfilerConfig(),
                             cache_dir=str(tmp_path), verbose=False)
    df = pd.DataFrame([('A', 10.0, 1), ('B', 8.0, 2), ('A', 2.0, 3), ('C', 1.0, 4)],
                      columns=['partner_type', 'weight', 'partner_bodyId'])
    # step through the same normalization the source performs
    norm = p._normalize_types_vectorized(df['partner_type'], p.config.fuzzy_match)
    df['partner_type_normalized'] = norm
    print('DTYPES', df.dtypes.to_dict())
    print('INDEX', df.index.tolist())
    print('COL VALUES', df['partner_type_normalized'].tolist())
    print('BLOCK INFO', df._data if hasattr(df, '_data') else '')
    import polars as pl
    pl_df = pl.from_pandas(df[['partner_type_normalized', 'weight']])
    print(pl_df)
