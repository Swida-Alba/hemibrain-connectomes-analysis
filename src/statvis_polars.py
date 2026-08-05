import os
import gc
import pandas as pd
import polars as pl
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Module-level cache for the full local neuron table.
#
# EnrichConnectionTablePolars is called once per network layer in
# FindAllPath, and every call used to re-read the (potentially hundreds of
# MB) *_allneurons_neuron_df.csv from disk.  The table is read-only within
# the enrichment pipeline, so a small mtime-keyed cache removes all repeated
# I/O while staying correct if the dataset file is regenerated.
# ---------------------------------------------------------------------------
_NEURON_DF_CACHE = {}  # (dataset_path, mtime_ns) -> pl.DataFrame
_NEURON_DF_CACHE_MAX = 4


def _load_local_neuron_df_cached(dataset_path: str, is_fafb: bool) -> pl.DataFrame:
    """Load the full local neuron CSV once per (path, mtime), cached."""
    try:
        # mtime_ns: getmtime() has only second resolution on some filesystems,
        # so a file regenerated within the same second would hit a stale entry.
        mtime = os.stat(dataset_path).st_mtime_ns
    except OSError:
        mtime = None
    cache_key = (dataset_path, mtime)
    cached = _NEURON_DF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Handle FlyWire/FAFB which might use string bodyIds.
    # NOTE: polars >= 1.0 removed the `dtypes=` kwarg (use schema_overrides).
    if is_fafb:
        ndf = pl.read_csv(dataset_path, infer_schema_length=10000, schema_overrides={'bodyId': pl.Utf8})
    else:
        ndf = pl.read_csv(dataset_path, infer_schema_length=10000)
        if 'bodyId' in ndf.columns:
            ndf = ndf.with_columns(pl.col('bodyId').cast(pl.Utf8))

    if len(_NEURON_DF_CACHE) >= _NEURON_DF_CACHE_MAX:
        _NEURON_DF_CACHE.pop(next(iter(_NEURON_DF_CACHE)))
    _NEURON_DF_CACHE[cache_key] = ndf
    return ndf


def build_bodyid_label_map(label_mapper, dataset: str, neuron_df: pl.DataFrame) -> dict:
    """
    Build a comprehensive bodyId → std_label map from label_mapper.
    
    This implements the user's 6-step label mapping approach:
    Step 3: Convert label_mapper's type/bodyId/instance → std_label map 
            to a complete bodyId → std_label map using the neuron index file.
    
    The label_mapper may contain mappings by:
    - bodyId: Direct bodyId → std_label
    - type: type_name → std_label (need to expand to all bodyIds of that type)
    - instance: instance_name → std_label (need to expand to all bodyIds with that instance)
    
    Args:
        label_mapper: LabelMapper object with source/target/intermediate mappings
        dataset: Dataset name (e.g., 'hemibrain:v1.2.1')
        neuron_df: Polars DataFrame with neuron index (must have 'bodyId', 'type', optionally 'instance')
        
    Returns:
        Dict[str, str]: Mapping of bodyId → std_label
    """
    if label_mapper is None or neuron_df is None or neuron_df.is_empty():
        return {}
    
    bodyid_label_map = {}
    
    # Ensure bodyId is string
    if 'bodyId' in neuron_df.columns:
        neuron_df = neuron_df.with_columns(pl.col('bodyId').cast(pl.Utf8))
    else:
        return {}
    
    # Build lookup dictionaries for efficient mapping
    # type → [bodyIds]
    type_to_bodyids = {}
    if 'type' in neuron_df.columns:
        type_groups = neuron_df.group_by('type').agg(pl.col('bodyId').alias('bodyIds'))
        for row in type_groups.iter_rows(named=True):
            if row['type'] is not None:
                type_to_bodyids[str(row['type'])] = row['bodyIds']
    
    # instance → [bodyIds]
    instance_to_bodyids = {}
    if 'instance' in neuron_df.columns:
        instance_groups = neuron_df.group_by('instance').agg(pl.col('bodyId').alias('bodyIds'))
        for row in instance_groups.iter_rows(named=True):
            if row['instance'] is not None:
                instance_to_bodyids[str(row['instance'])] = row['bodyIds']
    
    # Helper to normalize dataset name for lookups
    def sanitize(name: str) -> str:
        return name.replace(':', '_').replace('.', '_').replace('-', '_')
    
    dataset_sanitized = sanitize(dataset)
    
    # Process all mappings (source, target, intermediate)
    all_mappings = []
    for mapping_dict in [label_mapper._source_mapping, label_mapper._target_mapping, label_mapper._intermediate_mapping]:
        for std_label, ds_dict in mapping_dict.items():
            # Try both original and sanitized dataset names
            neuron_ids = []
            if dataset in ds_dict:
                neuron_ids = ds_dict[dataset]
            elif dataset_sanitized in ds_dict:
                neuron_ids = ds_dict[dataset_sanitized]
            
            for neuron_id in neuron_ids:
                all_mappings.append((str(neuron_id), std_label))
    
    # Process each mapping and expand to bodyIds
    # Precompute the set of existing bodyIds once (O(1) lookups) instead of
    # filtering the whole neuron_df for every mapping (O(M*N) per call).
    bodyid_set = set(neuron_df['bodyId'].to_list())

    for neuron_id, std_label in all_mappings:
        # First, check if neuron_id is a direct bodyId
        # If it matches a bodyId in the neuron_df, map it directly
        if neuron_id in bodyid_set:
            bodyid_label_map[neuron_id] = std_label
            continue
        
        # Check if neuron_id is a type name
        if neuron_id in type_to_bodyids:
            for bid in type_to_bodyids[neuron_id]:
                # Don't overwrite existing mappings (first mapping wins)
                if bid not in bodyid_label_map:
                    bodyid_label_map[bid] = std_label
            continue
        
        # Check if neuron_id is an instance name
        if neuron_id in instance_to_bodyids:
            for bid in instance_to_bodyids[neuron_id]:
                if bid not in bodyid_label_map:
                    bodyid_label_map[bid] = std_label
            continue
        
        # If none of the above, just store the mapping in case it's used directly
        # This handles cases where the ID might be used elsewhere
        bodyid_label_map[neuron_id] = std_label
    
    return bodyid_label_map


def get_classification_map(label_mapper, dataset: str) -> dict:
    """
    Build a map from std_label → classification (source/target/intermediate).
    
    This implements Step 2 of user's approach: remember the classification
    of each label (whether it's a source, target, or intermediate neuron group).
    
    Args:
        label_mapper: LabelMapper object
        dataset: Dataset name
        
    Returns:
        Dict[str, str]: Mapping of std_label → classification
    """
    if label_mapper is None:
        return {}
    
    classification_map = {}
    
    def sanitize(name: str) -> str:
        return name.replace(':', '_').replace('.', '_').replace('-', '_')
    
    dataset_sanitized = sanitize(dataset)
    
    # Process source mappings
    for std_label, ds_dict in label_mapper._source_mapping.items():
        if dataset in ds_dict or dataset_sanitized in ds_dict:
            classification_map[std_label] = 'source'
    
    # Process target mappings (may override source if same label)
    for std_label, ds_dict in label_mapper._target_mapping.items():
        if dataset in ds_dict or dataset_sanitized in ds_dict:
            if std_label in classification_map:
                classification_map[std_label] = 'source+target'
            else:
                classification_map[std_label] = 'target'
    
    # Process intermediate mappings
    for std_label, ds_dict in label_mapper._intermediate_mapping.items():
        if dataset in ds_dict or dataset_sanitized in ds_dict:
            if std_label not in classification_map:
                classification_map[std_label] = 'intermediate'
    
    return classification_map


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
        # For nt_type, take the first value (most edges between same nodes have same NT)
        aggs.append(pl.col('nt_type').first().alias('nt_type'))

    # Group and aggregate
    df_agg = df.group_by(['src', 'tgt']).agg(aggs)
    
    return df_agg

def process_batch_polars(paths_batch, df_conn, level='type', keyword_in_path_to_remove=None,
                         type_to_label_map=None):
    """
    Process a batch of paths using Polars.
    
    Args:
        type_to_label_map: Optional dict mapping original type names to standardized labels
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
    # Check if nt_type exists
    has_nt = 'nt_type' in df_joined.columns
    
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
    
    # Add nt_type aggregation if available
    if has_nt:
        aggs.append(pl.col('nt_type').alias('nt_types'))
    
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
    # Apply type_to_label_map if provided to rename types in output
    if type_to_label_map:
        # Map each node in the path list using type_to_label_map dict
        def map_node(node):
            return type_to_label_map.get(str(node), str(node))
        
        # Apply mapping to path_nodes list
        df_final = df_final.with_columns(
            pl.col('path_nodes').list.eval(
                pl.element().map_elements(map_node, return_dtype=pl.Utf8)
            ).alias('path_nodes_mapped')
        )
        df_final = df_final.with_columns(
            pl.col('path_nodes_mapped').list.join('->').alias('path')
        ).drop('path_nodes_mapped')
    else:
        # Polars list join without mapping
        df_final = df_final.with_columns(
            pl.col('path_nodes').list.join('->').alias('path')
        )
    
    # Convert list columns to string for CSV compatibility
    # Format as "[w1, w2, w3]" to match original statvis output
    list_format_cols = [
        (pl.lit("[") + pl.col('weights').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('weights'),
        (pl.lit("[") + pl.col('probabilities').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('probabilities'),
        (pl.lit("[") + pl.col('ratios').list.eval(pl.element().cast(pl.Utf8)).list.join(', ') + pl.lit("]")).alias('ratios')
    ]
    
    # Add nt_types formatting if available - use quoted strings for proper parsing
    if 'nt_types' in df_final.columns:
        # Format as ["ACH", "GABA"] so ast.literal_eval can parse it
        list_format_cols.append(
            (pl.lit('["') + pl.col('nt_types').list.eval(pl.element().cast(pl.Utf8)).list.join('", "') + pl.lit('"]')).alias('nt_types')
        )
    
    df_final = df_final.with_columns(list_format_cols)
    
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
    
    # Add nt_types if available
    if 'nt_types' in df_final.columns:
        cols_to_keep.append('nt_types')
    
    # Note: 'path_nodes' is the list. We can keep it if needed, but CSV writing might stringify it.
    # statvis writes 'path' as string "A->B->C".
    
    df_final_selected = df_final.select(cols_to_keep)
    
    # Handle excluded DataFrame - ensure it has the same schema even if empty
    if excluded.is_empty():
        excluded_selected = df_final_selected.clear()  # Empty DataFrame with same schema
    else:
        excluded_selected = excluded.select(cols_to_keep)
    
    return df_final_selected, excluded_selected

def process_paths_streaming(path_gen, conn_data, targets, output_path, 
                          excluded_path=None, real_layer_map=None, 
                          level='type', type_lookup=None, 
                          keyword_in_path_to_remove=None,
                          batch_size=100000,
                          verbose=True,
                          type_to_label_map=None):
    """
    Stream paths from generator, process in batches using Polars, and write to CSV.
    Returns total count of saved paths.
    
    OPTIMIZED: Uses buffered batch collection to reduce file I/O overhead.
    Collects 20 batches (~2M paths) before writing to minimize disk I/O.
    
    Args:
        type_to_label_map: Optional dict mapping original type names to standardized labels.
                           Types are fetched using original names but output uses mapped labels
                           for cross-dataset comparison.
    """
    if verbose:
        print(f"Optimizing path building: Pre-indexing {len(conn_data)} connections (Polars)...")
    
    # Prepare connection data once
    df_conn = prepare_connection_data(conn_data, level)
    
    batch = []
    total_saved = 0
    total_excluded = 0
    
    # Collect batches in memory before writing (reduces I/O overhead)
    write_buffer = []
    excl_buffer = []
    write_every_n_batches = 20  # Write every 20 batches (~2M paths) to balance memory vs I/O
    batch_count = 0
    
    # Track if we've written to files yet
    first_write = True
    first_excl_write = True
    
    # Use tqdm for progress bar if verbose
    if verbose:
        try:
            iterator = tqdm(path_gen, desc=f"Streaming {level}-level paths", unit="path")
        except ImportError:
            iterator = path_gen
    else:
        iterator = path_gen
        
    for path in iterator:
        batch.append(path)
        
        if len(batch) >= batch_size:
            df_batch, df_excl = process_batch_polars(batch, df_conn, level, keyword_in_path_to_remove,
                                                      type_to_label_map=type_to_label_map)
            
            if not df_batch.is_empty():
                write_buffer.append(df_batch)
                total_saved += len(df_batch)
                
            if excluded_path and not df_excl.is_empty():
                excl_buffer.append(df_excl)
                total_excluded += len(df_excl)
            
            batch_count += 1
            batch = []
            
            # Write buffered batches periodically
            if batch_count >= write_every_n_batches:
                if write_buffer:
                    _write_buffer_to_csv(write_buffer, output_path, append=not first_write)
                    first_write = False
                    write_buffer = []
                if excl_buffer and excluded_path:
                    _write_buffer_to_csv(excl_buffer, excluded_path, append=not first_excl_write)
                    first_excl_write = False
                    excl_buffer = []
                batch_count = 0
                gc.collect()
            
    # Process remaining paths
    if batch:
        df_batch, df_excl = process_batch_polars(batch, df_conn, level, keyword_in_path_to_remove,
                                                  type_to_label_map=type_to_label_map)
        
        if not df_batch.is_empty():
            write_buffer.append(df_batch)
            total_saved += len(df_batch)
            
        if excluded_path and not df_excl.is_empty():
            excl_buffer.append(df_excl)
            total_excluded += len(df_excl)
    
    # Write any remaining buffered data
    if write_buffer:
        _write_buffer_to_csv(write_buffer, output_path, append=not first_write)
    if excl_buffer and excluded_path:
        _write_buffer_to_csv(excl_buffer, excluded_path, append=not first_excl_write)
            
    return total_saved

def _write_buffer_to_csv(buffer_list, output_path, append=False):
    """
    Helper function to write buffered DataFrames to CSV efficiently.
    Concatenates all DataFrames first, then writes once.
    
    Performance: ~10-20x faster than writing each batch individually.
    """
    if not buffer_list:
        return
        
    # Concatenate all DataFrames in buffer (Polars concat is very fast)
    if len(buffer_list) == 1:
        df_combined = buffer_list[0]
    else:
        df_combined = pl.concat(buffer_list, rechunk=False)  # rechunk=False is faster
    
    # Write to CSV using Polars native I/O (faster than Python file handles)
    if append:
        # For append mode, use file handle with explicit UTF-8 encoding
        # This is critical on Windows where default encoding is often cp1252
        with open(output_path, 'a', encoding='utf-8', buffering=1024*1024) as f:  # 1MB buffer
            df_combined.write_csv(f, include_header=False)
    else:
        # For initial write, use Polars native (faster)
        df_combined.write_csv(output_path)

def EnrichConnectionTablePolars(conn_table, traversal_probability_threshold=0, dataset=None, script_path=None, target_neurons_df=None, aggregate_method='product', label_mapper=None, global_incoming_weights=None, separate_hemispheres=False):
    '''Add traversal probability, connection ratio, and layer information to the connection table using Polars
    
    NOTE: When separate_hemispheres=True, the caller is expected to have already applied
    hemisphere suffixes (_L/_R/_U) to type_pre/type_post columns. This function will
    aggregate by those already-suffixed types. The parameter is accepted for API
    compatibility but does not change the aggregation behavior.
    
    IMPLEMENTS USER's 6-STEP LABEL MAPPING APPROACH:
    Step 1: Fetch neurons using original type/bodyId/instance (done by caller)
    Step 2: Aggregate source/target/intermediate maps from label_mapper
    Step 3: Convert label_mapper's type/bodyId/instance → std_label to complete bodyId → std_label
    Step 4: Aggregate bodyId-level graph using std_label from label_mapper  
    Step 5: Aggregate remaining (unmapped) bodyIds by type
    Step 6: Mark source and target by the classification map
    
    Parameters
    ----------
    conn_table : DataFrame
        Connection table to enrich (bodyId-level)
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
        Accepted for API compatibility (legacy 'product'/'average' aggregation
        of bodyId-level block probabilities). Type-level traversal probability
        is always min(connection_ratio / 0.3, 1), matching coana's
        _apply_type_level_filters() and the pandas EnrichConnectionTable.
    label_mapper : LabelMapper, optional
        LabelMapper object for cross-dataset comparison.
        When provided, aggregation uses std_label for mapped neurons and type for unmapped.
    global_incoming_weights : DataFrame, optional
        Pre-computed total incoming weights for each post-synaptic type.
        Should have columns [type_post, total_incoming_weight].
        If provided, used for calculating GLOBAL type-level ratios.
        If None, local ratios (from provided connections only) are calculated.
    
    Returns
    -------
    conn_df : DataFrame
        Enriched connection table with bodyId-level metrics
    conn_type : DataFrame
        Type-level aggregation (or std_label-level if label_mapper provided)
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
            is_fafb = 'flywire' in dataset.lower() or 'fafb' in dataset.lower()
            ndf_complete = _load_local_neuron_df_cached(dataset_path, is_fafb)
    
    # Step 3: Build complete bodyId → std_label map from label_mapper
    bodyid_label_map = {}
    if label_mapper and ndf_complete is not None:
        bodyid_label_map = build_bodyid_label_map(label_mapper, dataset, ndf_complete)
    
    # Apply bodyId → std_label mapping to connection table
    # For mapped bodyIds: use std_label
    # For unmapped bodyIds: use original type (Step 5)
    # For untyped neurons (empty/null type): use bodyId as fallback
    #
    # IMPORTANT: When separate_hemispheres=True, the type_pre/type_post columns already 
    # have hemisphere suffixes (e.g., "PPL101_L"). The std_label from label_mapper is the
    # base type (e.g., "PPL101"). We need to preserve the hemisphere suffix by extracting
    # it from the type column and appending it to the std_label.
    if bodyid_label_map:
        # Create a Polars-friendly mapping for vectorized lookup
        map_df = pl.DataFrame({
            'bodyId': list(bodyid_label_map.keys()),
            'std_label': list(bodyid_label_map.values())
        })
        
        # Map pre neurons
        conn_df = conn_df.join(
            map_df.rename({'bodyId': 'bodyId_pre', 'std_label': '_mapped_std_label_pre'}),
            on='bodyId_pre',
            how='left'
        )
        
        if separate_hemispheres:
            # When hemisphere separation is enabled, append the hemisphere suffix from type_pre to std_label
            # Extract suffix: if type_pre ends with _L, _R, or _U, extract that suffix
            conn_df = conn_df.with_columns(
                pl.when(pl.col('type_pre').str.ends_with('_L'))
                    .then(pl.lit('_L'))
                    .when(pl.col('type_pre').str.ends_with('_R'))
                    .then(pl.lit('_R'))
                    .when(pl.col('type_pre').str.ends_with('_U'))
                    .then(pl.lit('_U'))
                    .otherwise(pl.lit(''))
                    .alias('_hemi_suffix_pre')
            )
            # Build std_label_pre: mapped_std_label + hemisphere_suffix, else fall back to type_pre, else bodyId
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.when(pl.col('_mapped_std_label_pre').is_not_null())
                        .then(pl.col('_mapped_std_label_pre') + pl.col('_hemi_suffix_pre'))
                        .otherwise(None),
                    pl.when(pl.col('type_pre').is_not_null() & (pl.col('type_pre') != '')).then(pl.col('type_pre')).otherwise(None),
                    pl.col('bodyId_pre')
                ]).alias('std_label_pre')
            )
            conn_df = conn_df.drop('_mapped_std_label_pre', '_hemi_suffix_pre')
        else:
            # No hemisphere separation: use std_label directly
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.col('_mapped_std_label_pre'),
                    pl.when(pl.col('type_pre').is_not_null() & (pl.col('type_pre') != '')).then(pl.col('type_pre')).otherwise(None),
                    pl.col('bodyId_pre')
                ]).alias('std_label_pre')
            )
            conn_df = conn_df.drop('_mapped_std_label_pre')
        
        # Map post neurons
        conn_df = conn_df.join(
            map_df.rename({'bodyId': 'bodyId_post', 'std_label': '_mapped_std_label_post'}),
            on='bodyId_post',
            how='left'
        )
        
        if separate_hemispheres:
            # Extract hemisphere suffix from type_post
            conn_df = conn_df.with_columns(
                pl.when(pl.col('type_post').str.ends_with('_L'))
                    .then(pl.lit('_L'))
                    .when(pl.col('type_post').str.ends_with('_R'))
                    .then(pl.lit('_R'))
                    .when(pl.col('type_post').str.ends_with('_U'))
                    .then(pl.lit('_U'))
                    .otherwise(pl.lit(''))
                    .alias('_hemi_suffix_post')
            )
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.when(pl.col('_mapped_std_label_post').is_not_null())
                        .then(pl.col('_mapped_std_label_post') + pl.col('_hemi_suffix_post'))
                        .otherwise(None),
                    pl.when(pl.col('type_post').is_not_null() & (pl.col('type_post') != '')).then(pl.col('type_post')).otherwise(None),
                    pl.col('bodyId_post')
                ]).alias('std_label_post')
            )
            conn_df = conn_df.drop('_mapped_std_label_post', '_hemi_suffix_post')
        else:
            conn_df = conn_df.with_columns(
                pl.coalesce([
                    pl.col('_mapped_std_label_post'),
                    pl.when(pl.col('type_post').is_not_null() & (pl.col('type_post') != '')).then(pl.col('type_post')).otherwise(None),
                    pl.col('bodyId_post')
                ]).alias('std_label_post')
            )
            conn_df = conn_df.drop('_mapped_std_label_post')
    else:
        # No label_mapper: std_label = type, or bodyId if type is empty/null
        conn_df = conn_df.with_columns([
            pl.coalesce([
                pl.when(pl.col('type_pre').is_not_null() & (pl.col('type_pre') != '')).then(pl.col('type_pre')).otherwise(None),
                pl.col('bodyId_pre')
            ]).alias('std_label_pre'),
            pl.coalesce([
                pl.when(pl.col('type_post').is_not_null() & (pl.col('type_post') != '')).then(pl.col('type_post')).otherwise(None),
                pl.col('bodyId_post')
            ]).alias('std_label_post')
        ])

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
    
    # Check if connection_ratio already exists and has valid values (from coana.py global calculation)
    # If so, preserve it to maintain the correct global ratio calculation
    has_valid_ratio = False
    if 'connection_ratio' in conn_df.columns:
        ratio_stats = conn_df.select([
            pl.col('connection_ratio').is_not_null().any().alias('has_any'),
            (pl.col('connection_ratio') > 0).any().alias('has_positive')
        ]).to_dicts()[0]
        has_valid_ratio = ratio_stats['has_any'] and ratio_stats['has_positive']
    
    if not has_valid_ratio:
        # Only recalculate if ratio doesn't exist or has no valid values
        # NOTE: This local calculation only considers connections in this table,
        # NOT all incoming connections. For accurate global ratios, use coana.py
        total_incoming = conn_df.group_by('bodyId_post').agg(
            pl.col('weight').sum().alias('total_incoming_weight')
        )
        conn_df = conn_df.join(total_incoming, on='bodyId_post', how='left')
            
        # Calculate metrics using LOCAL ratio (not global)
        conn_df = conn_df.with_columns(
            pl.when(pl.col('total_incoming_weight') > 0)
            .then(pl.col('weight') / pl.col('total_incoming_weight'))
            .otherwise(None)
            .alias('connection_ratio')
        )
        
        # Drop temporary column
        conn_df = conn_df.drop('total_incoming_weight')
    
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
        
    # 2. Aggregation (Step 4 & 5: Aggregate by std_label for mapped, type for unmapped)
    # First deduplicate by bodyId pairs to avoid counting same connection multiple times
    cols_to_keep = ['bodyId_pre', 'bodyId_post', 'type_pre', 'type_post', 
                    'std_label_pre', 'std_label_post', 'weight', 'block_probability', 'traversal_probability']
    if 'custom_group_pre' in conn_df.columns:
        cols_to_keep.extend(['custom_group_pre', 'custom_group_post'])
    
    # Check for NT type column - prefer nt_type_pre (presynaptic NT), fallback to nt_type
    nt_col = None
    if 'nt_type_pre' in conn_df.columns:
        nt_col = 'nt_type_pre'
        cols_to_keep.append('nt_type_pre')
    elif 'nt_type' in conn_df.columns:
        nt_col = 'nt_type'
        cols_to_keep.append('nt_type')
    
    # Keep only existing columns
    cols_to_keep = [c for c in cols_to_keep if c in conn_df.columns]
    bodyid_pairs = conn_df.select(cols_to_keep).unique(subset=['bodyId_pre', 'bodyId_post'])
    
    # Rename nt_type_pre to nt_type for consistency in downstream processing
    if nt_col == 'nt_type_pre' and 'nt_type_pre' in bodyid_pairs.columns:
        bodyid_pairs = bodyid_pairs.rename({'nt_type_pre': 'nt_type'})
    
    # Also add std_label to ref_df for total_post calculation
    ref_df_with_labels = None
    if ref_df is not None and bodyid_label_map:
        # Add std_label to ref_df
        map_df = pl.DataFrame({
            'bodyId': list(bodyid_label_map.keys()),
            'std_label': list(bodyid_label_map.values())
        })
        ref_df_with_labels = ref_df.join(map_df, on='bodyId', how='left')
        ref_df_with_labels = ref_df_with_labels.with_columns(
            pl.coalesce([pl.col('std_label'), pl.col('type')]).alias('std_label')
        )
    elif ref_df is not None:
        ref_df_with_labels = ref_df.with_columns(pl.col('type').alias('std_label'))
    
    # Convert global_incoming_weights to Polars if provided
    global_incoming_pl = None
    if global_incoming_weights is not None:
        if isinstance(global_incoming_weights, pd.DataFrame):
            global_incoming_pl = pl.from_pandas(global_incoming_weights)
        else:
            global_incoming_pl = global_incoming_weights
    
    # Check if nt_type exists
    has_nt_type = 'nt_type' in bodyid_pairs.columns
    
    # Function to aggregate
    def aggregate_connections(group_pre_col, group_post_col, ref_group_col=None):
        # Sum weights from deduplicated bodyId pairs
        agg_list = [pl.col('weight').sum()]
        
        # Add nt_type aggregation if available (mode, matching the pandas
        # engine; 'first' would silently pick an arbitrary row on ties)
        if has_nt_type:
            agg_list.append(pl.col('nt_type').mode().first().alias('nt_type'))
        
        agg_df = bodyid_pairs.group_by([group_pre_col, group_post_col]).agg(agg_list)
        
        # Type-level traversal_probability is computed below from the type-level
        # connection_ratio (min(ratio / 0.3, 1)), matching coana's
        # _apply_type_level_filters() and the pandas EnrichConnectionTable.
        # (Legacy product/average aggregation of bodyId-level block
        # probabilities was removed - it was unconditionally overwritten here.)

        # Calculate Connection Ratio (Type Level)
        # Use GLOBAL incoming weights if provided, otherwise fall back to LOCAL calculation
        if global_incoming_pl is not None and group_post_col in ['type_post', 'std_label_post']:
            # Use global incoming weights from the full dataset
            # global_incoming_weights has 'type_post' and 'total_incoming_weight' columns
            
            if group_post_col == 'std_label_post':
                # For std_label aggregation, we need to join bodyid_pairs with global weights
                # first, then aggregate. This handles the case where std_label differs from type.
                
                # Get unique std_label_post -> type_post mappings from bodyid_pairs
                # Since bodyid_pairs still has both type_post and std_label_post
                if 'type_post' in bodyid_pairs.columns:
                    # Build std_label -> sum of global incoming weights
                    # Each std_label_post may map to multiple type_post, so we sum their incoming weights
                    std_label_type_map = bodyid_pairs.select(['std_label_post', 'type_post']).unique()
                    
                    # Add global incoming weight for each type_post (vectorized
                    # join instead of row-wise map_elements with a Python dict)
                    std_label_type_map = std_label_type_map.join(
                        global_incoming_pl.select(['type_post', 'total_incoming_weight']),
                        on='type_post',
                        how='left',
                    ).with_columns(
                        pl.col('total_incoming_weight')
                        .fill_null(0.0)
                        .alias('type_incoming')
                    )
                    
                    # Sum by std_label_post (in case one std_label maps to multiple types)
                    global_incoming_by_std_label = std_label_type_map.group_by('std_label_post').agg(
                        pl.col('type_incoming').sum().alias('total_incoming_weight')
                    )
                    
                    # Join with agg_df
                    agg_df = agg_df.join(global_incoming_by_std_label, on='std_label_post', how='left')
                else:
                    # Fallback: rename type_post to std_label_post (they should be the same)
                    global_incoming_renamed = global_incoming_pl.rename({'type_post': 'std_label_post'})
                    agg_df = agg_df.join(global_incoming_renamed, on='std_label_post', how='left')
            else:
                # Direct join for type_post grouping
                agg_df = agg_df.join(global_incoming_pl, on='type_post', how='left')
            
            # Calculate ratio using GLOBAL denominator
            agg_df = agg_df.with_columns(
                pl.when(pl.col('total_incoming_weight') > 0)
                .then(pl.col('weight') / pl.col('total_incoming_weight'))
                .otherwise(None)
                .alias('connection_ratio')
            )
            if 'total_incoming_weight' in agg_df.columns:
                agg_df = agg_df.drop('total_incoming_weight')
            
            # Recalculate traversal_probability from GLOBAL connection_ratio
            # This ensures type-level traversal_prob matches the global ratio
            # (fill_null(0.0) matches the pandas fillna(0.0) semantics for
            # types with no known incoming connections)
            agg_df = agg_df.with_columns(
                (pl.col('connection_ratio') / 0.3).clip(0.0, 1.0).fill_null(0.0).alias('traversal_probability')
            )
        else:
            # Fall back to LOCAL calculation (only connections in this table)
            total_incoming_df = agg_df.group_by(group_post_col).agg(
                pl.col('weight').sum().alias('total_incoming_weight')
            )
            
            # Join total incoming weight
            agg_df = agg_df.join(total_incoming_df, on=group_post_col, how='left')
            
            # Calculate ratio using LOCAL denominator
            agg_df = agg_df.with_columns(
                pl.when(pl.col('total_incoming_weight') > 0)
                .then(pl.col('weight') / pl.col('total_incoming_weight'))
                .otherwise(None)
                .alias('connection_ratio')
            )
            agg_df = agg_df.drop('total_incoming_weight')
            
            # Recalculate traversal_probability from LOCAL connection_ratio
            agg_df = agg_df.with_columns(
                (pl.col('connection_ratio') / 0.3).clip(0.0, 1.0).fill_null(0.0).alias('traversal_probability')
            )
        
        # block_probability = 1 - traversal_probability (null -> 1.0, matching
        # the pandas fillna(1.0) semantics so both engines emit the same schema)
        agg_df = agg_df.with_columns(
            (1.0 - pl.col('traversal_probability')).fill_null(1.0).alias('block_probability')
        )
        
        return agg_df

    # Aggregate by std_label (Step 4 & 5)
    # This uses std_label for mapped neurons and type for unmapped (since std_label = type for unmapped)
    conn_type = aggregate_connections('std_label_pre', 'std_label_post', ref_group_col='std_label')
    
    # Rename std_label columns to type columns for backward compatibility
    conn_type = conn_type.rename({
        'std_label_pre': 'type_pre',
        'std_label_post': 'type_post'
    })
    
    # Aggregate Group
    conn_group = None
    if 'custom_group_pre' in conn_df.columns:
        conn_group = aggregate_connections('custom_group_pre', 'custom_group_post', ref_group_col='custom_group')
        
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
