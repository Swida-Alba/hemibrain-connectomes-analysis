"""
Cache System v4.0: Pair-Level Unified Database Methods
These methods replace the hash-based query-level caching with a unified database approach.
"""

import pandas as pd
import os
from datetime import datetime
from neuprint import fetch_neurons, NeuronCriteria

class CacheV4Methods:
    """
    Mix-in class for ConnectomeAnalysis with v4.0 pair-level caching methods.
    """
    
    # ============================================================================
    # Core Database Access
    # ============================================================================
    
    def _get_connection_db_path(self):
        '''Get path to unified connection database'''
        return os.path.join(self.cache_folder, 'connections.parquet')
    
    def _get_neuron_index_path(self):
        '''Get path to neuron index (tracks cached neurons)'''
        return os.path.join(self.cache_folder, 'neuron_index.parquet')
    
    def _load_connection_db(self):
        '''
        Load unified connection database.
        Schema: bodyId_pre, bodyId_post, weight, roi (optional), cached_date
        '''
        db_path = self._get_connection_db_path()
        if os.path.exists(db_path):
            try:
                df = pd.read_parquet(db_path)
                return df
            except Exception as e:
                print(f'  ⚠️ Warning: Failed to load connection database: {e}')
                return pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date'])
        return pd.DataFrame(columns=['bodyId_pre', 'bodyId_post', 'weight', 'roi', 'cached_date'])
    
    def _save_connection_db(self, conn_db):
        '''Save unified connection database with compression'''
        db_path = self._get_connection_db_path()
        try:
            conn_db.to_parquet(db_path, index=False, compression='gzip')
        except Exception as e:
            print(f'  ⚠️ Warning: Failed to save connection database: {e}')
    
    def _load_neuron_index(self):
        '''
        Load neuron index - tracks which neurons are fully cached.
        Schema: bodyId, type, instance, downstream_complete, last_fetched, connection_count
        '''
        index_path = self._get_neuron_index_path()
        if os.path.exists(index_path):
            try:
                return pd.read_parquet(index_path)
            except Exception as e:
                print(f'  ⚠️ Warning: Failed to load neuron index: {e}')
                return pd.DataFrame(columns=[
                    'bodyId', 'type', 'instance', 'downstream_complete', 
                    'last_fetched', 'connection_count'
                ])
        return pd.DataFrame(columns=[
            'bodyId', 'type', 'instance', 'downstream_complete',
            'last_fetched', 'connection_count'
        ])
    
    def _save_neuron_index(self, index_df):
        '''Save neuron index with compression'''
        index_path = self._get_neuron_index_path()
        try:
            index_df.to_parquet(index_path, index=False, compression='gzip')
        except Exception as e:
            print(f'  ⚠️ Warning: Failed to save neuron index: {e}')
    
    # ============================================================================
    # Query Resolution Logic
    # ============================================================================
    
    def _query_connection_db(self, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Query unified connection database for specific connections.
        Returns (cached_df, uncached_upstream_ids)
        
        Parameters:
        -----------
        upstream_bodyIds : list
            List of upstream neuron bodyIds to query
        downstream_bodyIds : list or None
            List of downstream neuron bodyIds (None = all downstream)
        
        Returns:
        --------
        tuple: (cached_connections_df, list_of_uncached_upstream_ids)
        '''
        if not self.use_cache:
            return pd.DataFrame(), upstream_bodyIds
        
        conn_db = self._load_connection_db()
        neuron_index = self._load_neuron_index()
        
        if conn_db.empty:
            # No cache yet
            return pd.DataFrame(), upstream_bodyIds
        
        # Separate cached vs uncached neurons
        cached_upstream = []
        uncached_upstream = []
        
        for bodyId in upstream_bodyIds:
            if bodyId in neuron_index['bodyId'].values:
                row = neuron_index[neuron_index['bodyId'] == bodyId].iloc[0]
                
                if downstream_bodyIds is None:
                    # Need all downstream - check if fully cached
                    if row['downstream_complete']:
                        cached_upstream.append(bodyId)
                    else:
                        uncached_upstream.append(bodyId)
                else:
                    # Specific targets - for now, treat as uncached if not fully complete
                    # TODO: Could optimize by checking if specific pairs exist
                    if row['downstream_complete']:
                        cached_upstream.append(bodyId)
                    else:
                        uncached_upstream.append(bodyId)
            else:
                # Not in index = not cached
                uncached_upstream.append(bodyId)
        
        # Retrieve cached connections
        if len(cached_upstream) > 0:
            cached_conn = conn_db[conn_db['bodyId_pre'].isin(cached_upstream)].copy()
            
            # Filter by downstream if specified
            if downstream_bodyIds is not None:
                cached_conn = cached_conn[cached_conn['bodyId_post'].isin(downstream_bodyIds)].copy()
            
            return cached_conn, uncached_upstream
        
        return pd.DataFrame(), uncached_upstream
    
    def _update_connection_db(self, new_connections, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Add new connections to unified database without duplicates.
        Updates neuron index to mark neurons as fully cached (if querying all downstream).
        
        Parameters:
        -----------
        new_connections : pd.DataFrame
            New connections to add (must have bodyId_pre, bodyId_post, weight, optionally roi)
        upstream_bodyIds : list
            List of upstream neurons that were queried
        downstream_bodyIds : list or None
            If None, marks neurons as downstream_complete. If list, doesn't mark as complete.
        '''
        if new_connections.empty:
            return
        
        # Load existing database
        conn_db = self._load_connection_db()
        
        # Prepare new connections
        new_conn = new_connections[['bodyId_pre', 'bodyId_post', 'weight']].copy()
        if 'roi' in new_connections.columns:
            new_conn['roi'] = new_connections['roi']
        else:
            new_conn['roi'] = ''
        
        new_conn['cached_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Merge with existing, removing duplicates (keep existing entries)
        if not conn_db.empty:
            # Remove any new connections that already exist (based on bodyId_pre, bodyId_post, roi)
            merge_cols = ['bodyId_pre', 'bodyId_post', 'roi']
            combined = pd.concat([conn_db, new_conn])
            combined = combined.drop_duplicates(subset=merge_cols, keep='first')
        else:
            combined = new_conn
        
        # Save updated database
        self._save_connection_db(combined)
        
        new_count = len(combined) - len(conn_db)
        if new_count > 0:
            print(f'  💾 Added {new_count} new connections to database (total: {len(combined):,})')
        else:
            print(f'  📂 All connections already in database ({len(conn_db):,} total)')
        
        # Update neuron index
        self._update_neuron_index_after_fetch(new_conn, upstream_bodyIds, downstream_bodyIds)
    
    def _update_neuron_index_after_fetch(self, connections, upstream_bodyIds, downstream_bodyIds=None):
        '''
        Update neuron index after fetching connections.
        Only marks neurons as downstream_complete if we fetched ALL downstream (downstream_bodyIds=None).
        '''
        neuron_index = self._load_neuron_index()
        
        # Get neuron info from complete dataset
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        
        if os.path.exists(dataset_path):
            ndf_complete = pd.read_parquet(dataset_path, header=0, index_col=0)
            neuron_info = ndf_complete[ndf_complete['bodyId'].isin(upstream_bodyIds)][['bodyId', 'type', 'instance']].copy()
        else:
            # Fallback: fetch from API
            try:
                ndf, _ = fetch_neurons(NeuronCriteria(bodyId=upstream_bodyIds))
                neuron_info = ndf[['bodyId', 'type', 'instance']].copy()
            except:
                neuron_info = pd.DataFrame(columns=['bodyId', 'type', 'instance'])
        
        # Count connections per neuron
        if not connections.empty:
            conn_counts = connections.groupby('bodyId_pre').size().reset_index(name='connection_count')
        else:
            conn_counts = pd.DataFrame(columns=['bodyId_pre', 'connection_count'])
        
        # Only mark as downstream_complete if we fetched ALL downstream
        mark_complete = (downstream_bodyIds is None)
        
        for bodyId in upstream_bodyIds:
            neuron_row = neuron_info[neuron_info['bodyId'] == bodyId]
            if not neuron_row.empty:
                neuron_type = neuron_row.iloc[0]['type'] if 'type' in neuron_row.columns else ''
                neuron_instance = neuron_row.iloc[0]['instance'] if 'instance' in neuron_row.columns else ''
            else:
                neuron_type = ''
                neuron_instance = ''
            
            conn_count = conn_counts[conn_counts['bodyId_pre'] == bodyId]['connection_count'].iloc[0] if bodyId in conn_counts['bodyId_pre'].values else 0
            
            if bodyId in neuron_index['bodyId'].values:
                # Update existing entry
                if mark_complete:
                    neuron_index.loc[neuron_index['bodyId'] == bodyId, 'downstream_complete'] = True
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'last_fetched'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                neuron_index.loc[neuron_index['bodyId'] == bodyId, 'connection_count'] = conn_count
            else:
                # Add new entry
                new_entry = pd.DataFrame([{
                    'bodyId': bodyId,
                    'type': neuron_type,
                    'instance': neuron_instance,
                    'downstream_complete': mark_complete,
                    'last_fetched': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'connection_count': conn_count
                }])
                neuron_index = pd.concat([neuron_index, new_entry], ignore_index=True)
        
        self._save_neuron_index(neuron_index)
        
        if mark_complete:
            completed_count = len([b for b in upstream_bodyIds if b in neuron_index[neuron_index['downstream_complete'] == True]['bodyId'].values])
            print(f'  📝 Updated neuron index: {completed_count} neurons marked as complete')
    
    # ============================================================================
    # Enrichment with Type/Instance
    # ============================================================================
    
    def _enrich_connections_with_neuron_info(self, conn_df):
        '''
        Enrich connection dataframe with type and instance from complete local dataset.
        '''
        if conn_df.empty:
            return conn_df
        
        # Get unique bodyIds that need enrichment
        all_bodyids = list(set(conn_df['bodyId_pre'].tolist() + conn_df['bodyId_post'].tolist()))
        
        # Load from complete dataset (includes type=None neurons)
        dataset_path = os.path.join(
            self.script_path,
            'datasets',
            f"{self.dataset.replace(':', '_').replace('.', '_')}_allneurons_neuron_df.csv"
        )
        
        if not os.path.exists(dataset_path):
            # Fallback: try to use standard dataset (may miss type=None neurons)
            print(f'  ⚠️ Warning: Complete dataset not found, using standard dataset')
            print(f'     Some neurons without types may be missing.')
            try:
                import statvis as sv
                neuron_df, _, _, _ = sv.getNeurons(all_bodyids, dataset=self.dataset)
            except:
                neuron_df = pd.DataFrame(columns=['bodyId', 'type', 'instance'])
        else:
            # Load complete dataset from CSV
            ndf_complete = pd.read_csv(dataset_path, header=0, index_col=0)
            # Filter to only neurons we need
            neuron_df = ndf_complete[ndf_complete['bodyId'].isin(all_bodyids)].copy()
        
        neuron_info = neuron_df[['bodyId', 'type', 'instance']].copy()
        
        # Join type and instance for pre-synaptic neurons
        conn_df = conn_df.merge(
            neuron_info.rename(columns={'type': 'type_pre', 'instance': 'instance_pre'}),
            left_on='bodyId_pre',
            right_on='bodyId',
            how='left'
        ).drop(columns=['bodyId'])
        
        # Join type and instance for post-synaptic neurons
        conn_df = conn_df.merge(
            neuron_info.rename(columns={'type': 'type_post', 'instance': 'instance_post'}),
            left_on='bodyId_post',
            right_on='bodyId',
            how='left'
        ).drop(columns=['bodyId'])
        
        return conn_df
    
    # ============================================================================
    # Main Fetch Method (replaces old _fetch_connections_with_cache)
    # ============================================================================
    
    def _fetch_connections_with_cache(self, upstream_bodyIds, downstream_bodyIds=None, 
                                      min_weight=None, min_traversal_prob=None, min_conn_ratio=None):
        '''
        Fetch connections with v4.0 pair-level caching.
        Queries unified database first, only fetches missing neurons from API.
        
        Parameters:
        -----------
        upstream_bodyIds : list
            List of upstream neuron bodyIds
        downstream_bodyIds : list or None
            List of downstream neuron bodyIds (None = all downstream)
        min_weight : int or None
            Minimum synapse count for filtering (uses self.min_synapse_num if None)
        min_traversal_prob : float or None
            Minimum traversal probability for edge filtering (uses self.min_traversal_probability if None)
        min_conn_ratio : float or None
            Minimum connection ratio (weight/post) for edge filtering (uses self.min_ratio if None)
        
        Returns:
        --------
        pd.DataFrame : Connection table filtered by min_weight, min_traversal_prob, and min_conn_ratio
        '''
        if min_weight is None:
            min_weight = self.min_synapse_num
        if min_traversal_prob is None:
            min_traversal_prob = self.min_traversal_probability
        if min_conn_ratio is None:
            min_conn_ratio = self.min_ratio
        
        # Step 1: Query database for cached connections
        cached_conn, uncached_upstream = self._query_connection_db(upstream_bodyIds, downstream_bodyIds)
        
        if not cached_conn.empty:
            print(f'  📂 Found {len(set(upstream_bodyIds) - set(uncached_upstream))}/{len(upstream_bodyIds)} neurons in cache')
            print(f'     Retrieved {len(cached_conn):,} connections from database')
        
        # Step 2: Fetch uncached neurons from API if needed
        api_conn = pd.DataFrame()
        if len(uncached_upstream) > 0:
            print(f'  🌐 Fetching {len(uncached_upstream)} uncached neurons from API (weight ≥ 1)...')
            
            if self.simple_fetch:
                from neuprint import fetch_simple_connections
                upstream_criteria = NeuronCriteria(bodyId=uncached_upstream)
                downstream_criteria = NeuronCriteria(bodyId=downstream_bodyIds) if downstream_bodyIds is not None else None
                api_conn = fetch_simple_connections(
                    upstream_criteria=upstream_criteria,
                    downstream_criteria=downstream_criteria,
                    min_weight=1,  # Always fetch with min_weight=1
                    **self.kwargs_fetch
                )
            else:
                from neuprint import fetch_adjacencies
                import statvis as sv
                neuron_df, roi_conn_df = fetch_adjacencies(
                    sources=uncached_upstream,
                    targets=downstream_bodyIds,
                    min_total_weight=1,  # Always fetch with min_weight=1
                    **self.kwargs_fetch
                )
                api_conn = sv.merge_conn_roi(neuron_df, roi_conn_df)
            
            if not api_conn.empty:
                # Update database with new connections
                self._update_connection_db(api_conn, uncached_upstream, downstream_bodyIds)
        
        # Step 3: Combine cached and API results
        if cached_conn.empty and api_conn.empty:
            return pd.DataFrame()
        
        # Combine results
        combined = pd.concat([cached_conn, api_conn], ignore_index=True) if not cached_conn.empty and not api_conn.empty else (cached_conn if not cached_conn.empty else api_conn)
        
        total_before_filter = len(combined)
        
        # Step 4: Apply filters
        # Filter by weight
        if min_weight > 1:
            combined = combined[combined['weight'] >= min_weight].copy()
        
        # Filter by ratio-based thresholds if needed
        if (min_traversal_prob > 0 or min_conn_ratio > 0) and len(combined) > 0:
            # Get post-synaptic counts
            post_bodyIds = combined['bodyId_post'].unique().tolist()
            post_df, _ = fetch_neurons(NeuronCriteria(bodyId=post_bodyIds))
            post_info = post_df[['bodyId', 'post']].copy()
            post_info.columns = ['bodyId_post', 'post']
            
            # Merge and calculate both ratios
            combined = combined.merge(post_info, how='left', on='bodyId_post')
            combined['connection_ratio'] = combined['weight'] / combined['post']
            combined['traversal_probability'] = combined['connection_ratio'] / 0.3
            combined.loc[combined['traversal_probability'] > 1, 'traversal_probability'] = 1
            
            # Filter by connection ratio
            if min_conn_ratio > 0:
                combined = combined[combined['connection_ratio'] >= min_conn_ratio].copy()
            
            # Filter by traversal probability
            if min_traversal_prob > 0:
                combined = combined[combined['traversal_probability'] >= min_traversal_prob].copy()
            
            # Drop temporary columns
            combined = combined.drop(columns=['post', 'connection_ratio', 'traversal_probability'])
        
        # Step 5: Enrich with type and instance
        combined = self._enrich_connections_with_neuron_info(combined)
        
        # Print filter summary
        if min_weight > 1 or min_traversal_prob > 0 or min_conn_ratio > 0:
            filter_msg = []
            if min_weight > 1:
                filter_msg.append(f'weight ≥ {min_weight}')
            if min_conn_ratio > 0:
                filter_msg.append(f'ratio ≥ {min_conn_ratio}')
            if min_traversal_prob > 0:
                filter_msg.append(f'prob ≥ {min_traversal_prob}')
            
            print(f'     Filtered: {total_before_filter} → {len(combined)} connections ({", ".join(filter_msg)})')
        
        print(f'     Enriched with neuron info from complete local dataset')
        
        return combined
