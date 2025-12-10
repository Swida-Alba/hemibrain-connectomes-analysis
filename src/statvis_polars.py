import os
import gc
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

def prepare_connection_data(conn_data, level='type'):
    """
    Pre-process connection data into a Polars DataFrame optimized for joining.
    Aggregates multiple edges between same nodes.
    """
    # Determine source and target columns
    src_col = f'{level}_pre' if f'{level}_pre' in conn_data.columns else 'bodyId_pre'
    tgt_col = f'{level}_post' if f'{level}_post' in conn_data.columns else 'bodyId_post'
    
    # Convert to Polars if needed
    if isinstance(conn_data, pd.DataFrame):
        df = pl.from_pandas(conn_data)
    else:
        df = conn_data
        
    # Cast columns to string for consistent joining
    df = df.with_columns([
        pl.col(src_col).cast(pl.Utf8).alias('src'),
        pl.col(tgt_col).cast(pl.Utf8).alias('tgt')
    ])
    
    # Define aggregations
    aggs = [pl.col('weight').sum().alias('weight')]
    
    if 'traversal_probability' in df.columns:
        aggs.append(pl.col('traversal_probability').mean().alias('traversal_probability'))
    
    if 'connection_ratio' in df.columns:
        aggs.append(pl.col('connection_ratio').mean().alias('connection_ratio'))
        
    if 'nt_type' in df.columns:
        # Custom aggregation for nt_type: unique values joined by |
        # Polars doesn't have a direct 'unique_join' agg, so we might need a custom expression or list
        # For speed, let's just take the first one or list
        # aggs.append(pl.col('nt_type').unique().alias('nt_type_list'))
        pass 

    # Group and aggregate
    df_agg = df.group_by(['src', 'tgt']).agg(aggs)
    
    return df_agg

def process_batch_polars(paths_batch, df_conn, level='type', keyword_in_path_to_remove=None):
    """
    Process a batch of paths using Polars.
    """
    if not paths_batch:
        return pl.DataFrame(), pl.DataFrame()
        
    # 1. Create DataFrame from paths
    # paths_batch is list of lists
    # We want: path_id, node_idx, node
    
    # Create a DataFrame with a single column 'path' containing lists
    # Note: Polars creation from list of lists might infer types. Ensure strings.
    # It's safer to convert all nodes to strings first in Python if they are mixed
    paths_str = [[str(n) for n in p] for p in paths_batch]
    
    df_paths = pl.DataFrame({'path_nodes': paths_str})
    df_paths = df_paths.with_row_index('path_id')
    
    # 2. Explode to get edges
    # We need to create edges (u, v) for each path
    # Strategy: Explode nodes, then shift to get next node
    
    df_exploded = df_paths.explode('path_nodes')
    
    # We need to group by path_id to perform shift operation safely
    # But explode keeps order.
    
    df_edges = df_exploded.with_columns([
        pl.col('path_nodes').alias('src'),
        pl.col('path_nodes').shift(-1).over('path_id').alias('tgt')
    ])
    
    # Filter out the last node which has no target (tgt is null)
    df_edges = df_edges.filter(pl.col('tgt').is_not_null())
    
    # 3. Join with connection data
    # df_conn has ['src', 'tgt', 'weight', 'traversal_probability', ...]
    
    df_joined = df_edges.join(df_conn, on=['src', 'tgt'], how='left')
    
    # Fill missing values (if any edge not found)
    df_joined = df_joined.with_columns([
        pl.col('weight').fill_null(0),
        pl.col('traversal_probability').fill_null(0),
        pl.col('connection_ratio').fill_null(0)
    ])
    
    # 4. Aggregate back to path level
    # We want lists of weights, probs, etc. and summary stats
    
    aggs = [
        pl.col('src').alias('path_nodes_flat'), # We'll reconstruct path string later
        pl.col('weight').alias('weights'),
        pl.col('traversal_probability').alias('probabilities'),
        pl.col('connection_ratio').alias('ratios'),
        
        pl.col('weight').min().alias('min_weight'),
        pl.col('traversal_probability').product().alias('path_prob'),
        pl.col('connection_ratio').min().alias('min_ratio'),
        pl.count('src').alias('length')
    ]
    
    df_results = df_joined.group_by('path_id', maintain_order=True).agg(aggs)
    
    # 5. Filter zero-weight paths (any edge has weight 0)
    # In Polars, we can check if min_weight > 0
    df_results = df_results.filter(pl.col('min_weight') > 0)
    
    if df_results.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    
    # 6. Reconstruct path string and add original path list
    # We need to join back with df_paths to get the original full path list (including last node)
    # because df_edges lost the last node in 'src' column
    
    df_final = df_results.join(df_paths, on='path_id', how='left')
    
    # Create formatted path string "A->B->C"
    # Polars list join
    df_final = df_final.with_columns(
        pl.col('path_nodes').list.join('->').alias('path')
    )
    
    # Convert list columns to string for CSV compatibility
    # Format as "[w1, w2, w3]" to match original statvis output
    df_final = df_final.with_columns([
        (pl.lit("[") + pl.col('weights').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('weights'),
        (pl.lit("[") + pl.col('probabilities').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('probabilities'),
        (pl.lit("[") + pl.col('ratios').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('ratios')
    ])
    
    # Rename path_nodes to path_str (to match statvis output convention)
    # But statvis uses 'path_str' for the list object in pandas.
    # Here we can keep 'path_nodes' as the list column.
    
    # 7. Filter keywords
    excluded = pl.DataFrame()
    if keyword_in_path_to_remove:
        if isinstance(keyword_in_path_to_remove, str):
            keywords = [keyword_in_path_to_remove]
        else:
            keywords = keyword_in_path_to_remove
            
        # Build filter expression
        filter_expr = pl.lit(False)
        for kw in keywords:
            filter_expr = filter_expr | pl.col('path').str.contains(kw, literal=True)
            
        excluded = df_final.filter(filter_expr)
        df_final = df_final.filter(~filter_expr)
        
    # Select and rename columns to match statvis output
    # statvis output: path_str (list), path (str), weights, probabilities, ratios, min_weight, path_prob, min_ratio, length
    
    cols_to_keep = [
        'path', 'weights', 'probabilities', 'ratios', 
        'min_weight', 'path_prob', 'min_ratio', 'length'
    ]
    
    # Note: 'path_nodes' is the list. We can keep it if needed, but CSV writing might stringify it.
    # statvis writes 'path' as string "A->B->C".
    
    return df_final.select(cols_to_keep), excluded.select(cols_to_keep)

def process_paths_streaming(path_gen, conn_data, targets, output_path, 
                          excluded_path=None, real_layer_map=None, 
                          level='type', type_lookup=None, 
                          keyword_in_path_to_remove=None,
                          batch_size=100000):
    """
    Stream paths from generator, process in batches using Polars, and write to CSV.
    Returns total count of saved paths.
    """
    print(f"Optimizing path building: Pre-indexing {len(conn_data)} connections (Polars)...")
    
    # Prepare connection data once
    df_conn = prepare_connection_data(conn_data, level)
    
    batch = []
    total_saved = 0
    total_excluded = 0
    first_batch = True
    
    # Use tqdm for progress bar
    try:
        iterator = tqdm(path_gen, desc=f"Streaming {level}-level paths", unit="path")
    except ImportError:
        iterator = path_gen
        
    for path in iterator:
        batch.append(path)
        
        if len(batch) >= batch_size:
            df_batch, df_excl = process_batch_polars(batch, df_conn, level, keyword_in_path_to_remove)
            
            if not df_batch.is_empty():
                # Write to CSV
                # Polars write_csv doesn't support mode='a' directly in older versions, 
                # but we can use open file handle.
                with open(output_path, 'w' if first_batch else 'a') as f:
                    df_batch.write_csv(f, include_header=first_batch)
                total_saved += len(df_batch)
                
            if excluded_path and not df_excl.is_empty():
                with open(excluded_path, 'w' if first_batch else 'a') as f:
                    df_excl.write_csv(f, include_header=first_batch)
                total_excluded += len(df_excl)
                
            if first_batch:
                first_batch = False
                
            batch = []
            gc.collect()
            
    # Process remaining
    if batch:
        df_batch, df_excl = process_batch_polars(batch, df_conn, level, keyword_in_path_to_remove)
        
        if not df_batch.is_empty():
            with open(output_path, 'w' if first_batch else 'a') as f:
                df_batch.write_csv(f, include_header=first_batch)
            total_saved += len(df_batch)
            
        if excluded_path and not df_excl.is_empty():
            with open(excluded_path, 'w' if first_batch else 'a') as f:
                df_excl.write_csv(f, include_header=first_batch)
            total_excluded += len(df_excl)
            
    return total_saved

def EnrichConnectionTablePolars(conn_table, traversal_probability_threshold=0, dataset=None, script_path=None, target_neurons_df=None, aggregate_method='product', label_mapper=None):
    '''Add traversal probability, connection ratio, and layer information to the connection table using Polars
    
    Parameters
    ----------
    conn_table : DataFrame
        Connection table to enrich
    traversal_probability_threshold : float, optional
        Minimum traversal probability threshold (default: 0)
    dataset : str, optional
        Dataset name (e.g., 'optic-lobe:v1.1') for local dataset lookup
    script_path : str, optional
        Path to script directory containing 'datasets' folder
    target_neurons_df : DataFrame, optional
        Full dataframe of target neurons (with bodyId, type, post columns).
        Used to get correct type-level denominators. If not provided, only
        neurons appearing in connections will be used (less accurate).
    aggregate_method : str, optional
        Method for aggregating type-level traversal probabilities from bodyId level:
        - 'product': compound probability (product of block probs) for paths (default)
        - 'average': weighted average for direct parallel connections
    label_mapper : LabelMapper, optional
        LabelMapper object to standardize types in the local dataset for accurate ratio calculation.
    
    Returns
    -------
    conn_df : DataFrame
        Enriched connection table with bodyId-level metrics
    conn_type : DataFrame
        Type-level aggregation (always based on original type column)
    conn_group : DataFrame or None
        Custom group-level aggregation (only if custom_group columns exist)
    '''
    # Convert to Polars
    if isinstance(conn_table, pd.DataFrame):
        conn_df = pl.from_pandas(conn_table)
    else:
        conn_df = conn_table
        
    # Ensure string types for IDs
    conn_df = conn_df.with_columns([
        pl.col('bodyId_pre').cast(pl.Utf8),
        pl.col('bodyId_post').cast(pl.Utf8)
    ])
    
    # Determine grouping columns (use custom_group if available, otherwise type)
    group_pre = 'custom_group_pre' if 'custom_group_pre' in conn_df.columns else 'type_pre'
    group_post = 'custom_group_post' if 'custom_group_post' in conn_df.columns else 'type_post'
    
    # Try to use local dataset first
    use_local = False
    ndf_complete = None
    if dataset and script_path:
        dataset_clean = dataset.replace(':', '_').replace('.', '_')
        # Prioritize subdirectory structure
        dataset_path = os.path.join(
            script_path,
            'datasets',
            dataset_clean,
            f"{dataset_clean}_allneurons_neuron_df.csv"
        )
        
        # Enhanced dataset discovery logic
        if not os.path.exists(dataset_path):
            # Fallback: Try root datasets folder (legacy)
            legacy_path = os.path.join(
                script_path,
                'datasets',
                f"{dataset_clean}_allneurons_neuron_df.csv"
            )
            if os.path.exists(legacy_path):
                dataset_path = legacy_path
            else:
                # Try globbing for any *_allneurons_neuron_df.csv in subdir
                subdir_path = os.path.join(script_path, 'datasets', dataset_clean)
                if os.path.exists(subdir_path):
                    import glob
                    candidates = glob.glob(os.path.join(subdir_path, "*_allneurons_neuron_df.csv"))
                    if candidates:
                        dataset_path = candidates[0]

        if os.path.exists(dataset_path):
            use_local = True
            # Handle FlyWire/FAFB which might use string bodyIds
            if 'flywire' in dataset.lower() or 'fafb' in dataset.lower():
                ndf_complete = pl.read_csv(dataset_path, infer_schema_length=10000, dtypes={'bodyId': pl.Utf8})
            else:
                ndf_complete = pl.read_csv(dataset_path, infer_schema_length=10000)
                if 'bodyId' in ndf_complete.columns:
                    ndf_complete = ndf_complete.with_columns(pl.col('bodyId').cast(pl.Utf8))
            
            # Apply label mapping to local dataset if provided
            if label_mapper and 'type' in ndf_complete.columns:
                # Convert to pandas for label mapping application (LabelMapper is optimized for scalar lookups)
                ndf_pd = ndf_complete.to_pandas()
                
                # Apply mapping to 'type' column
                # We use get_label which checks all mappings
                # Note: get_label returns str(original_id) if no mapping found, which preserves unmapped types
                ndf_pd['type'] = ndf_pd['type'].apply(
                    lambda x: label_mapper.get_label(dataset, x) if pd.notna(x) else x
                )
                
                # Convert back to Polars
                ndf_complete = pl.from_pandas(ndf_pd)
                
                # Ensure bodyId is string again if lost
                if 'bodyId' in ndf_complete.columns:
                    ndf_complete = ndf_complete.with_columns(pl.col('bodyId').cast(pl.Utf8))
                
                # Update conn_df types to match mapped types in ndf_complete
                # This ensures that aggregation uses the standardized labels
                # Create mapping lookup: bodyId -> mapped_type
                type_map = ndf_complete.select(['bodyId', 'type']).unique(subset=['bodyId'])
                
                # Update type_pre
                conn_df = conn_df.join(
                    type_map.rename({'bodyId': 'bodyId_pre', 'type': 'type_pre_mapped'}),
                    on='bodyId_pre',
                    how='left'
                )
                conn_df = conn_df.with_columns(
                    pl.col('type_pre_mapped').fill_null(pl.col('type_pre')).alias('type_pre')
                ).drop('type_pre_mapped')
                
                # Update type_post
                conn_df = conn_df.join(
                    type_map.rename({'bodyId': 'bodyId_post', 'type': 'type_post_mapped'}),
                    on='bodyId_post',
                    how='left'
                )
                conn_df = conn_df.with_columns(
                    pl.col('type_post_mapped').fill_null(pl.col('type_post')).alias('type_post')
                ).drop('type_post_mapped')
                
                # Update custom_group if present (assume it follows type)
                if 'custom_group_pre' in conn_df.columns:
                    conn_df = conn_df.with_columns(pl.col('type_pre').alias('custom_group_pre'))
                if 'custom_group_post' in conn_df.columns:
                    conn_df = conn_df.with_columns(pl.col('type_post').alias('custom_group_post'))

    # 1. Enrich BodyId Level
    # Need to join 'post' count to conn_df
    
    # Prepare reference dataframe for joining
    ref_df = None
    if ndf_complete is not None:
        ref_df = ndf_complete
    elif target_neurons_df is not None:
        if isinstance(target_neurons_df, pd.DataFrame):
            ref_df = pl.from_pandas(target_neurons_df)
        else:
            ref_df = target_neurons_df
            
    if ref_df is not None:
        # Ensure types
        if 'bodyId' in ref_df.columns:
            ref_df = ref_df.with_columns(pl.col('bodyId').cast(pl.Utf8))
        
        # Join to get 'post'
        # We only need bodyId and post for this step
        if 'post' in ref_df.columns:
            post_lookup = ref_df.select(['bodyId', 'post']).rename({'bodyId': 'bodyId_post'})
            
            # Drop 'post' from conn_df if it exists to avoid collision
            if 'post' in conn_df.columns:
                conn_df = conn_df.drop('post')
                
            conn_df = conn_df.join(post_lookup, on='bodyId_post', how='left')
            conn_df = conn_df.with_columns(pl.col('post').fill_null(0))
        else:
             if 'post' not in conn_df.columns:
                 conn_df = conn_df.with_columns(pl.lit(0).alias('post'))
    else:
        if 'post' not in conn_df.columns:
            conn_df = conn_df.with_columns(pl.lit(0).alias('post'))
        
    # Calculate metrics
    # connection_ratio = weight / post
    conn_df = conn_df.with_columns(
        (pl.col('weight') / pl.col('post')).fill_null(0).alias('connection_ratio')
    )
    
    # traversal_probability = connection_ratio / 0.3 (capped at 1.0)
    conn_df = conn_df.with_columns(
        (pl.col('connection_ratio') / 0.3).clip(0.0, 1.0).alias('traversal_probability')
    )
    
    # block_probability = 1 - traversal_probability
    conn_df = conn_df.with_columns(
        (1.0 - pl.col('traversal_probability')).alias('block_probability')
    )
    
    # Filter by threshold
    if traversal_probability_threshold > 0:
        conn_df = conn_df.filter(pl.col('traversal_probability') >= traversal_probability_threshold)
        
    # 2. Aggregation
    
    # Function to aggregate
    def aggregate_connections(group_pre_col, group_post_col):
        # Sum weights
        agg_df = conn_df.group_by([group_pre_col, group_post_col]).agg([
            pl.col('weight').sum()
        ])
        
        # Calculate Traversal Probability
        if aggregate_method == 'product':
            # Product of block probabilities
            # group_by().agg(pl.col('block_probability').product())
            probs = conn_df.group_by([group_pre_col, group_post_col]).agg(
                pl.col('block_probability').product().alias('block_prob_prod')
            )
            agg_df = agg_df.join(probs, on=[group_pre_col, group_post_col], how='left')
            agg_df = agg_df.with_columns(
                (1.0 - pl.col('block_prob_prod')).alias('traversal_probability')
            ).drop('block_prob_prod')
            
        else: # average
            # Weighted average
            # sum(weight * prob) / sum(weight)
            temp = conn_df.with_columns(
                (pl.col('weight') * pl.col('traversal_probability')).alias('wt_prob')
            )
            probs = temp.group_by([group_pre_col, group_post_col]).agg([
                pl.col('wt_prob').sum(),
                pl.col('weight').sum().alias('weight_sum')
            ])
            probs = probs.with_columns(
                (pl.col('wt_prob') / pl.col('weight_sum')).fill_null(0).alias('traversal_probability')
            )
            agg_df = agg_df.join(probs.select([group_pre_col, group_post_col, 'traversal_probability']), on=[group_pre_col, group_post_col], how='left')

        # Calculate Connection Ratio (Type Level)
        # Need total_post for the group
        # Get total post counts from ref_df (ndf_complete or target_neurons_df)
        
        total_post_df = None
        if ref_df is not None:
            # Check if group column exists in ref_df
            # group_post_col might be 'type_post', but in ref_df it is 'type'
            # or 'custom_group_post' -> 'custom_group'
            ref_group_col = 'type' if group_post_col == 'type_post' else 'custom_group'
            
            if ref_group_col in ref_df.columns:
                total_post_df = ref_df.group_by(ref_group_col).agg(
                    pl.col('post').sum().alias('total_post')
                )
                # Rename for join
                total_post_df = total_post_df.rename({ref_group_col: group_post_col})
        
        if total_post_df is None:
            # Fallback: sum post from connections (less accurate)
            # Note: this sums post of neurons IN connections, not ALL neurons of that type
            # But if we don't have ref_df, it's the best we can do
            total_post_df = conn_df.unique(subset=['bodyId_post']).group_by(group_post_col).agg(
                pl.col('post').sum().alias('total_post')
            )
            
        # Join total_post
        agg_df = agg_df.join(total_post_df, on=group_post_col, how='left')
        
        # Calculate ratio
        agg_df = agg_df.with_columns(
            (pl.col('weight') / pl.col('total_post')).fill_null(0).alias('connection_ratio')
        )
        
        return agg_df

    # Aggregate Type
    conn_type = aggregate_connections('type_pre', 'type_post')
    
    # Aggregate Group
    conn_group = None
    if 'custom_group_pre' in conn_df.columns:
        conn_group = aggregate_connections('custom_group_pre', 'custom_group_post')
        
    # Return Polars DataFrames directly
    return conn_df, conn_type, conn_group

def build_path_dataframe_from_paths(paths, conn_data, targets, real_layer_map=None, level='type', type_lookup=None):
    """
    Build a Polars DataFrame from a list of paths.
    """
    # Prepare connection data
    df_conn = prepare_connection_data(conn_data, level)
    
    # Process all paths
    # We reuse process_batch_polars logic
    # Note: type_lookup is currently ignored in Polars implementation for path string formatting
    # to maintain high performance. Path strings will contain IDs only.
    
    df_final, _ = process_batch_polars(paths, df_conn, level)
    
    return df_final
