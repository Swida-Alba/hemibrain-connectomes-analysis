"""
FastGraph - Lightweight Graph Implementation for VisualizePath

A minimal, efficient graph implementation optimized for visualization tasks.
Replaces NetworkX for simple graph operations to reduce dependencies and overhead.

Date: 2025-12
"""

from collections import defaultdict
import numpy as np


class FastGraph:
    """
    A lightweight directed graph implementation for visualization.
    
    Provides a NetworkX-compatible API subset optimized for:
    - Building graphs from edge lists
    - Node/edge attribute storage
    - Graph iteration and filtering
    
    This is a simplified version that doesn't include pathfinding algorithms
    (not needed for visualization).
    
    Attributes
    ----------
    adj : dict
        Adjacency list: {node: {neighbor: weight, ...}, ...}
    node_attrs : dict
        Node attributes: {node: {attr_name: value, ...}, ...}
    edge_attrs : dict
        Edge attributes: {(u, v): {attr_name: value, ...}, ...}
    """
    
    def __init__(self):
        """Initialize an empty directed graph."""
        self.adj = {}  # {u: {v: weight}}
        self.node_attrs = {}  # {node: {attr: value}}
        self.edge_attrs = {}  # {(u, v): {attr: value}}
        self._num_edges = 0

    def add_node(self, n, **attrs):
        """
        Add a single node with optional attributes.
        
        Parameters
        ----------
        n : hashable
            Node identifier
        **attrs : keyword arguments
            Node attributes (e.g., node_type='source', color='#ff0000')
        """
        if n not in self.adj:
            self.adj[n] = {}
        if n not in self.node_attrs:
            self.node_attrs[n] = {}
        self.node_attrs[n].update(attrs)

    def add_edge(self, u, v, weight=1.0, **attrs):
        """
        Add an edge from u to v with optional weight and attributes.
        
        Parameters
        ----------
        u : hashable
            Source node
        v : hashable
            Target node
        weight : float, optional
            Edge weight (default: 1.0). If edge exists, weights are summed.
        **attrs : keyword arguments
            Additional edge attributes (e.g., ratio=0.5, probability=0.8)
        """
        # Ensure nodes exist
        if u not in self.adj:
            self.adj[u] = {}
            self.node_attrs[u] = {}
        if v not in self.adj:
            self.adj[v] = {}
            self.node_attrs[v] = {}
        
        # Add/update edge
        if v not in self.adj[u]:
            self.adj[u][v] = 0.0
            self._num_edges += 1
        self.adj[u][v] += weight
        
        # Store edge attributes
        edge_key = (u, v)
        if edge_key not in self.edge_attrs:
            self.edge_attrs[edge_key] = {}
        self.edge_attrs[edge_key]['weight'] = self.adj[u][v]
        self.edge_attrs[edge_key].update(attrs)

    def number_of_nodes(self):
        """Return the number of nodes in the graph."""
        return len(self.adj)

    def number_of_edges(self):
        """Return the number of edges in the graph."""
        return self._num_edges

    def edges(self, data=False):
        """
        Return edges, optionally with attributes.
        
        Parameters
        ----------
        data : bool, optional
            If True, return (u, v, attr_dict) tuples. Default: False.
            
        Returns
        -------
        iterator
            Edge iterator
        """
        for u in self.adj:
            for v, w in self.adj[u].items():
                if data:
                    attrs = self.edge_attrs.get((u, v), {'weight': w})
                    yield (u, v, attrs)
                else:
                    yield (u, v)

    def __contains__(self, node):
        """Check if node exists in graph."""
        return node in self.adj

    def has_node(self, n):
        """Check if node n exists in graph."""
        return n in self.adj

    def has_edge(self, u, v):
        """Check if edge (u, v) exists in graph."""
        return u in self.adj and v in self.adj[u]

    def neighbors(self, n):
        """Return iterator over neighbors of node n."""
        return iter(self.adj.get(n, {}))

    def successors(self, n):
        """Return iterator over successors of node n (same as neighbors for DiGraph)."""
        return self.neighbors(n)

    def predecessors(self, n):
        """Return iterator over predecessors of node n."""
        for u in self.adj:
            if n in self.adj[u]:
                yield u

    def out_degree(self, n):
        """Return out-degree of node n."""
        return len(self.adj.get(n, {}))

    def in_degree(self, n):
        """Return in-degree of node n."""
        count = 0
        for u in self.adj:
            if n in self.adj[u]:
                count += 1
        return count

    def get_edge_data(self, u, v, default=None):
        """
        Return edge attributes for edge (u, v).
        
        Parameters
        ----------
        u, v : hashable
            Source and target nodes
        default : any, optional
            Value to return if edge doesn't exist
            
        Returns
        -------
        dict or default
            Edge attribute dictionary
        """
        if u in self.adj and v in self.adj[u]:
            return self.edge_attrs.get((u, v), {'weight': self.adj[u][v]})
        return default

    @property
    def _node(self):
        """
        NetworkX-compatible node attribute access.
        Returns a dict-like object for node attributes.
        """
        return self.node_attrs

    @property
    def nodes(self):
        """
        NetworkX-compatible nodes property.
        Returns a NodeView-like object that supports both iteration and attribute access.
        
        Usage:
            - for n in G.nodes: ...  (iteration)
            - for n in G.nodes(): ...  (iteration via call)
            - G.nodes[n]['attr'] = value  (attribute access)
            - G.nodes[n].get('attr', default)  (safe attribute access)
        """
        return _NodeView(self)




    def reverse(self, copy=True):
        """
        Return a graph with all edges reversed.
        
        Parameters
        ----------
        copy : bool, optional
            Ignored (always returns new graph). For API compatibility.
            
        Returns
        -------
        FastGraph
            New graph with reversed edges
        """
        new_G = FastGraph()
        for u in self.adj:
            for v, w in self.adj[u].items():
                new_G.add_edge(v, u, w)
                # Copy edge attributes
                if (u, v) in self.edge_attrs:
                    new_G.edge_attrs[(v, u)] = self.edge_attrs[(u, v)].copy()
        # Copy node attributes
        for n, attrs in self.node_attrs.items():
            new_G.node_attrs[n] = attrs.copy()
        return new_G

    def subgraph(self, nodes):
        """
        Return a subgraph containing only the specified nodes.
        
        Parameters
        ----------
        nodes : iterable
            Nodes to include in subgraph
            
        Returns
        -------
        FastGraph
            New graph with only specified nodes and their internal edges
        """
        nodes = set(nodes)
        new_G = FastGraph()
        for u in nodes:
            if u in self.adj:
                for v, w in self.adj[u].items():
                    if v in nodes:
                        new_G.add_edge(u, v, w)
                        if (u, v) in self.edge_attrs:
                            new_G.edge_attrs[(u, v)] = self.edge_attrs[(u, v)].copy()
        # Copy node attributes for included nodes
        for n in nodes:
            if n in self.node_attrs:
                new_G.node_attrs[n] = self.node_attrs.get(n, {}).copy()
        return new_G

    def copy(self):
        """
        Return a deep copy of the graph.
        
        Returns
        -------
        FastGraph
            New graph with copied data
        """
        new_G = FastGraph()
        for u in self.adj:
            for v, w in self.adj[u].items():
                new_G.add_edge(u, v, w)
        # Copy attributes
        for n, attrs in self.node_attrs.items():
            new_G.node_attrs[n] = attrs.copy()
        for edge, attrs in self.edge_attrs.items():
            new_G.edge_attrs[edge] = attrs.copy()
        return new_G

    def build_from_dataframe(self, df, source_col, target_col, weight_col, **extra_cols):
        """
        Build graph from a DataFrame (modifies graph in place).
        
        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            DataFrame with edge data
        source_col : str
            Column name for source nodes
        target_col : str
            Column name for target nodes
        weight_col : str
            Column name for edge weights
        **extra_cols : str
            Additional column names to store as edge attributes
            e.g., ratio_col='ratio', prob_col='probability'
            
        Returns
        -------
        self : FastGraph
            Returns self for method chaining
        """
        # Check if it's a Polars DataFrame
        is_polars = False
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                is_polars = True
        except ImportError:
            pass

        if is_polars:
            if df.is_empty():
                return self
            # Convert to rows for iteration
            for row in df.iter_rows(named=True):
                u = row[source_col]
                v = row[target_col]
                w = row[weight_col]
                
                # Collect extra attributes
                attrs = {}
                for attr_name, col_name in extra_cols.items():
                    if col_name in row:
                        attrs[attr_name] = row[col_name]
                
                self.add_edge(u, v, w, **attrs)
        else:
            # Pandas implementation
            if df.empty:
                return self
            
            for _, row in df.iterrows():
                u = row[source_col]
                v = row[target_col]
                w = row[weight_col]
                
                # Collect extra attributes
                attrs = {}
                for attr_name, col_name in extra_cols.items():
                    if col_name in df.columns:
                        attrs[attr_name] = row[col_name]
                
                self.add_edge(u, v, w, **attrs)
        
        return self

    def to_dataframe(self, include_attrs=True):
        """
        Convert graph edges to a pandas DataFrame.
        
        Parameters
        ----------
        include_attrs : bool, optional
            If True, include all edge attributes. Default: True.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with columns [source, target, weight, ...]
        """
        import pandas as pd
        
        edges = []
        for u in self.adj:
            for v, w in self.adj[u].items():
                edge_data = {'source': u, 'target': v, 'weight': w}
                if include_attrs:
                    attrs = self.edge_attrs.get((u, v), {})
                    for k, val in attrs.items():
                        if k != 'weight':  # Already included
                            edge_data[k] = val
                edges.append(edge_data)
        return pd.DataFrame(edges)

    def filter_by_weight(self, min_weight):
        """
        Create a new graph with only edges meeting minimum weight threshold.
        
        Parameters
        ----------
        min_weight : float
            Minimum edge weight to include
            
        Returns
        -------
        FastGraph
            New graph with filtered edges
        """
        filtered = FastGraph()
        for u in self.adj:
            for v, w in self.adj[u].items():
                if w >= min_weight:
                    filtered.add_edge(u, v, w)
                    if (u, v) in self.edge_attrs:
                        filtered.edge_attrs[(u, v)] = self.edge_attrs[(u, v)].copy()
        # Copy node attributes for remaining nodes
        for n in filtered.adj:
            if n in self.node_attrs:
                filtered.node_attrs[n] = self.node_attrs[n].copy()
        return filtered

    # =========================================================================
    # Pathfinding Methods (for connectome analysis)
    # =========================================================================

    def aggregate_by_label(self, label_map: dict, return_edge_df: bool = False):
        """
        Aggregate graph nodes by label mapping (e.g., bodyId -> type).
        
        Creates a new graph where nodes are labels (e.g., neuron types) and
        edge weights are the sum of all original edges between nodes with
        those labels.
        
        Parameters
        ----------
        label_map : dict
            Dictionary mapping original node IDs to labels.
            e.g., {bodyId1: 'type_A', bodyId2: 'type_A', bodyId3: 'type_B'}
        return_edge_df : bool, optional
            If True, also return a DataFrame with edge details
                           
        Returns
        -------
        FastGraph
            New graph with aggregated edges by label
        pandas.DataFrame (optional)
            Edge data with columns [type_pre, type_post, weight]
        
        Example
        -------
        >>> G = FastGraph()
        >>> G.add_edge(1, 10, 5)  # bodyId 1 -> 10, weight 5
        >>> G.add_edge(2, 10, 3)  # bodyId 2 -> 10, weight 3
        >>> label_map = {1: 'A', 2: 'A', 10: 'X'}
        >>> G_type = G.aggregate_by_label(label_map)
        >>> # G_type now has edge A -> X with weight 8
        """
        aggregated = FastGraph()
        edge_data = []
        
        # Aggregate edges by label pairs
        label_weights = {}  # (label_pre, label_post) -> total_weight
        
        for u in self.adj:
            label_u = label_map.get(u)
            if label_u is None:
                continue  # Skip nodes without labels
                
            for v, w in self.adj[u].items():
                label_v = label_map.get(v)
                if label_v is None:
                    continue  # Skip nodes without labels
                
                key = (label_u, label_v)
                if key not in label_weights:
                    label_weights[key] = 0.0
                label_weights[key] += w
        
        # Build aggregated graph
        for (label_u, label_v), total_w in label_weights.items():
            aggregated.add_edge(label_u, label_v, total_w)
            if return_edge_df:
                edge_data.append({
                    'type_pre': label_u,
                    'type_post': label_v,
                    'weight': total_w
                })
        
        if return_edge_df:
            import pandas as pd
            return aggregated, pd.DataFrame(edge_data)
        return aggregated

    @classmethod
    def from_dataframe_bodyid(cls, df, pre_col='bodyId_pre', post_col='bodyId_post', 
                               weight_col='weight', type_pre_col='type_pre', 
                               type_post_col='type_post'):
        """
        Build a bodyId-level graph from DataFrame and extract type mapping.
        
        Parameters
        ----------
        df : pandas.DataFrame or polars.DataFrame
            DataFrame with bodyId-level connection data
        pre_col : str
            Column name for presynaptic bodyId
        post_col : str
            Column name for postsynaptic bodyId
        weight_col : str
            Column name for edge weight
        type_pre_col : str
            Column name for presynaptic type
        type_post_col : str
            Column name for postsynaptic type
            
        Returns
        -------
        tuple
            (FastGraph, dict) - bodyId graph and bodyId->type mapping
        """
        import pandas as pd
        
        graph = cls()
        label_map = {}
        
        # Check if Polars
        is_polars = False
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                is_polars = True
        except ImportError:
            pass
        
        if is_polars:
            for row in df.iter_rows(named=True):
                u = row[pre_col]
                v = row[post_col]
                w = row[weight_col]
                graph.add_edge(u, v, w)
                
                # Build label map
                if type_pre_col in row and row[type_pre_col]:
                    label_map[u] = row[type_pre_col]
                if type_post_col in row and row[type_post_col]:
                    label_map[v] = row[type_post_col]
        else:
            for _, row in df.iterrows():
                u = row[pre_col]
                v = row[post_col]
                w = row[weight_col]
                graph.add_edge(u, v, w)
                
                # Build label map
                if type_pre_col in df.columns and pd.notna(row.get(type_pre_col)):
                    label_map[u] = row[type_pre_col]
                if type_post_col in df.columns and pd.notna(row.get(type_post_col)):
                    label_map[v] = row[type_post_col]
        
        return graph, label_map

    def all_simple_paths(self, source, target, cutoff):
        """
        Find all simple paths from source to target with length <= cutoff.
        
        Parameters
        ----------
        source : hashable
            Starting node
        target : hashable
            Ending node
        cutoff : int
            Maximum path length (number of edges)
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        if source not in self.adj:
            return
        
        yield from self._dfs_paths(source, target, cutoff, [source], {source})

    def _dfs_paths(self, u, target, cutoff, path, visited):
        """Internal DFS helper for all_simple_paths."""
        if len(path) > cutoff + 1:
            return
        
        if u == target:
            yield list(path)
            return
            
        if u not in self.adj:
            return

        for v in self.adj[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)
                yield from self._dfs_paths(v, target, cutoff, path, visited)
                path.pop()
                visited.remove(v)

    def find_paths_dfs_backtracking(self, sources, targets, cutoff, verbose=False):
        """
        Standard Backward DFS with backtracking (no memoization).
        Optimized for lowest memory usage at the cost of CPU time.
        Uses Iterative Deepening to find shortest paths first.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        source_set = set(sources)
        R = self.reverse()
        
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        def dfs_recursive(u, target_depth, path, visited):
            current_len = len(path) - 1
            if current_len == target_depth:
                if u in source_set:
                    yield path[::-1]
                return

            if u in R.adj:
                for v in R.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        yield from dfs_recursive(v, target_depth, path, visited)
                        path.pop()
                        visited.remove(v)

        for length in range(1, cutoff + 1):
            iterator = targets
            if verbose:
                iterator = tqdm(targets, desc=f"L{length} Backtracking", leave=False)
            
            for t in iterator:
                yield from dfs_recursive(t, length, [t], {t})

    def find_paths_meet_in_the_middle(self, sources, targets, cutoff, verbose=False):
        """
        Bidirectional DFS (Meet-in-the-middle) to find all paths.
        Optimized for memory by storing paths of length L/2 and streaming the rest.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        from collections import defaultdict
        
        source_set = set(sources)
        target_set = set(targets)
        R = self.reverse()
        
        def simple_dfs_paths(start_node, graph, target_depth, valid_end_nodes=None):
            # In-place backtracking: the previous implementation copied the
            # whole path list (`path + [v]`) on every edge traversal, which is
            # O(depth) allocation per step and dominates runtime on dense
            # graphs. A shared path/visited pair with one copy at the leaf is
            # equivalent and much cheaper.
            path = [start_node]
            visited = {start_node}

            def dfs(u, depth):
                if depth == target_depth:
                    if valid_end_nodes is None or u in valid_end_nodes:
                        yield u, list(path)
                    return
                if u in graph.adj:
                    for v in graph.adj[u]:
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            yield from dfs(v, depth + 1)
                            path.pop()
                            visited.discard(v)

            yield from dfs(start_node, 0)

        def get_reachable_set(start_nodes, graph, depth):
            current = set(start_nodes)
            for i in range(depth):
                next_layer = set()
                if not current: break
                for u in current:
                    if u in graph.adj:
                        next_layer.update(graph.adj[u])
                current = next_layer
            return current

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        for length in range(1, cutoff + 1):
            mid = length // 2
            rem = length - mid
            
            valid_mids = get_reachable_set(target_set, R, rem)
            if not valid_mids:
                continue
                
            fwd_paths_map = defaultdict(list)
            iterator = sources
            if verbose:
                iterator = tqdm(sources, desc=f"L{length} Fwd(L{mid})", leave=False)
            
            for s in iterator:
                if s not in self.adj: continue
                for end_node, path in simple_dfs_paths(s, self, mid, valid_mids):
                    fwd_paths_map[end_node].append(path)
                    
            if not fwd_paths_map:
                continue
                
            valid_ends_for_backward = set(fwd_paths_map.keys())
            iterator = targets
            if verbose:
                iterator = tqdm(targets, desc=f"L{length} Bwd(L{rem})", leave=False)
                
            for t in iterator:
                if t not in R.adj: continue
                for end_node, r_path in simple_dfs_paths(t, R, rem, valid_ends_for_backward):
                    r_path_rev = r_path[::-1]
                    r_set = set(r_path_rev)
                    for f_path in fwd_paths_map[end_node]:
                        if len(set(f_path) & r_set) == 1:
                            combined = f_path + r_path_rev[1:]
                            yield combined

    def find_paths_backward_dp(self, sources, targets, cutoff, verbose=False):
        """
        Backward Reachability (DP) + Forward DFS.
        Computes sets of nodes reachable from targets, then uses guided DFS.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        target_set = set(targets)
        R = self.reverse()
        
        valid_nodes_at_dist = [set() for _ in range(cutoff + 2)]
        valid_nodes_at_dist[0] = target_set
        
        for k in range(1, cutoff + 1):
            prev_set = valid_nodes_at_dist[k-1]
            current_set = valid_nodes_at_dist[k]
            for target_node in prev_set:
                if target_node in R.adj:
                    current_set.update(R.adj[target_node])
        
        def guided_dfs(u, depth, current_path):
            if depth == 0:
                if u in target_set:
                    yield list(current_path)
                return

            if u in self.adj:
                target_set_for_next_step = valid_nodes_at_dist[depth-1]
                for v in self.adj[u]:
                    if v in target_set_for_next_step:
                        if v not in current_path:
                            current_path.append(v)
                            yield from guided_dfs(v, depth - 1, current_path)
                            current_path.pop()

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        for length in range(1, cutoff + 1):
            valid_sources = [s for s in sources if s in valid_nodes_at_dist[length]]
            iterator = valid_sources
            if verbose:
                iterator = tqdm(valid_sources, desc=f"L{length} GuidedDFS", leave=False)
            for s in iterator:
                yield from guided_dfs(s, length, [s])

    def find_paths_memoized_dfs(self, sources, targets, cutoff, direction='forward', verbose=False):
        """
        Memoized DFS with valid successor pruning.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        direction : str
            'forward' or 'backward'
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        if direction == 'backward':
            R = self.reverse()
            for path in R.find_paths_memoized_dfs(targets, sources, cutoff, direction='forward', verbose=verbose):
                yield path[::-1]
            return

        target_set = set(targets)
        valid_successors = {}
        
        def find_valid_successors(u, k):
            state = (u, k)
            if state in valid_successors:
                return valid_successors[state]
            if k == 0:
                return True if u in target_set else None
            successors = []
            if u in self.adj:
                for v in self.adj[u]:
                    if find_valid_successors(v, k-1) is not None:
                        successors.append(v)
            res = successors if successors else None
            valid_successors[state] = res
            return res

        def reconstruct(u, k, path):
            if k == 0:
                if u in target_set:
                    yield list(path)
                return
            succs = valid_successors.get((u, k))
            if not succs: return
            for v in succs:
                if v not in path:
                    path.append(v)
                    yield from reconstruct(v, k-1, path)
                    path.pop()

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        for length in range(1, cutoff + 1):
            iterator = sources
            if verbose:
                iterator = tqdm(sources, desc=f"L{length} BuildMemo", leave=False)
            valid_sources = []
            for s in iterator:
                if find_valid_successors(s, length) is not None:
                    valid_sources.append(s)
            iterator = valid_sources
            if verbose:
                iterator = tqdm(valid_sources, desc=f"L{length} Reconstruct", leave=False)
            for s in iterator:
                yield from reconstruct(s, length, [s])

    def find_paths_bidirectional_bfs(self, sources, targets, cutoff, verbose=False):
        """
        Bidirectional BFS (Layer-based) pathfinding.
        Builds search trees from both sides and finds intersection.
        
        Parameters
        ----------
        sources : iterable
            Source nodes
        targets : iterable
            Target nodes
        cutoff : int
            Maximum path length
        verbose : bool
            Show progress bars
            
        Yields
        ------
        list
            Each path as a list of nodes
        """
        from collections import defaultdict
        
        f_layers = [defaultdict(set) for _ in range(cutoff + 1)]
        for s in sources:
            if s in self.adj:
                f_layers[0][s].add(None)
        
        for d in range(cutoff):
            current_layer = f_layers[d]
            next_layer = f_layers[d+1]
            for u in current_layer:
                if u in self.adj:
                    for v in self.adj[u]:
                        next_layer[v].add(u)
            if not next_layer: break

        b_layers = [defaultdict(set) for _ in range(cutoff + 1)]
        target_set = set(targets)
        R = self.reverse()
        
        for t in targets:
            if t in R.adj:
                b_layers[0][t].add(None)
                
        for d in range(cutoff):
            current_layer = b_layers[d]
            next_layer = b_layers[d+1]
            for u in current_layer:
                if u in R.adj:
                    for v in R.adj[u]:
                        next_layer[v].add(u)
            if not next_layer: break

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        for length in range(1, cutoff + 1):
            mid = length // 2
            rem = length - mid
            
            f_nodes = set(f_layers[mid].keys())
            b_nodes = set(b_layers[rem].keys())
            meet_nodes = f_nodes & b_nodes
            
            if not meet_nodes: continue
            
            def get_fwd_paths(u, depth):
                if depth == 0:
                    yield [u]
                    return
                for parent in f_layers[depth][u]:
                    if parent is not None:
                        for p in get_fwd_paths(parent, depth-1):
                            if u not in p:
                                yield p + [u]

            def get_bwd_paths(u, depth):
                if depth == 0:
                    yield [u]
                    return
                for child in b_layers[depth][u]:
                    if child is not None:
                        for p in get_bwd_paths(child, depth-1):
                            if u not in p:
                                yield [u] + p

            iterator = meet_nodes
            if verbose:
                iterator = tqdm(meet_nodes, desc=f"L{length} Reconstruct", leave=False)

            for u in iterator:
                f_paths = list(get_fwd_paths(u, mid))
                b_paths = list(get_bwd_paths(u, rem))
                for fp in f_paths:
                    for bp in b_paths:
                        if len(set(fp) & set(bp)) == 1:
                            yield fp + bp[1:]

    def __repr__(self):
        return f"FastGraph(nodes={self.number_of_nodes()}, edges={self.number_of_edges()})"


# Compatibility alias


class _NodeView:
    """
    A NetworkX-compatible node view that supports both iteration and dict-like access.
    """
    def __init__(self, graph):
        self._graph = graph
    
    def __iter__(self):
        return iter(self._graph.adj)
    
    def __call__(self, data=False):
        """Called when G.nodes() is used."""
        if data:
            for n in self._graph.adj:
                yield (n, self._graph.node_attrs.get(n, {}))
        else:
            yield from self._graph.adj
    
    def __getitem__(self, node):
        """Allow G.nodes[node] access to node attributes."""
        if node not in self._graph.node_attrs:
            self._graph.node_attrs[node] = {}
        return self._graph.node_attrs[node]
    
    def __setitem__(self, node, attrs):
        """Allow G.nodes[node] = {...} assignment."""
        self._graph.node_attrs[node] = attrs
    
    def __contains__(self, node):
        return node in self._graph.adj
    
    def __len__(self):
        return len(self._graph.adj)
    
    def keys(self):
        return self._graph.adj.keys()
    
    def items(self):
        for n in self._graph.adj:
            yield (n, self._graph.node_attrs.get(n, {}))
    
    def update(self, attrs_dict):
        """Update node attributes from a dict."""
        for node, attrs in attrs_dict.items():
            if node not in self._graph.node_attrs:
                self._graph.node_attrs[node] = {}
            self._graph.node_attrs[node].update(attrs)




DiGraph = FastGraph

