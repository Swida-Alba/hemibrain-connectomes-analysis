from collections import defaultdict, deque

class FastGraph:
    """
    A lightweight graph implementation optimized for pathfinding performance.
    Replaces NetworkX for simple pathfinding tasks to avoid overhead.
    """
    def __init__(self):
        self.adj = {}  # {u: {v: weight}}
        self._num_edges = 0

    def add_edge(self, u, v, weight=1.0):
        if u not in self.adj:
            self.adj[u] = {}
        if v not in self.adj[u]:
            self.adj[u][v] = 0.0
            self._num_edges += 1
        self.adj[u][v] += weight
        
        # Ensure v is in adj to track all nodes
        if v not in self.adj:
            self.adj[v] = {}

    def number_of_nodes(self):
        return len(self.adj)

    def number_of_edges(self):
        return self._num_edges

    def nodes(self):
        return self.adj.keys()
        
    def __contains__(self, node):
        return node in self.adj

    def has_node(self, n):
        return n in self.adj

    def neighbors(self, n):
        return self.adj.get(n, {})

    def reverse(self, copy=True):
        # copy argument is ignored, always returns new graph
        new_G = FastGraph()
        for u in self.adj:
            for v, w in self.adj[u].items():
                new_G.add_edge(v, u, w)
        return new_G

    def subgraph(self, nodes):
        nodes = set(nodes)
        new_G = FastGraph()
        for u in nodes:
            if u in self.adj:
                for v, w in self.adj[u].items():
                    if v in nodes:
                        new_G.add_edge(u, v, w)
        return new_G
    
    def copy(self):
        # Minimal copy
        new_G = FastGraph()
        for u in self.adj:
            for v, w in self.adj[u].items():
                new_G.add_edge(u, v, w)
        return new_G

    def edges(self, data=False):
        for u in self.adj:
            for v, w in self.adj[u].items():
                if data:
                    yield (u, v, {'weight': w})
                else:
                    yield (u, v)

    def build_from_dataframe(self, df, source_col, target_col, weight_col):
        """
        Build graph from DataFrame using vectorized operations.
        Much faster than iterating rows.
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
                return
            # Polars implementation
            grouped = df.group_by([source_col, target_col]).agg(pl.col(weight_col).sum())
            sources = grouped[source_col].to_numpy()
            targets = grouped[target_col].to_numpy()
            weights = grouped[weight_col].to_numpy()
        else:
            if df.empty:
                return
            # Pandas implementation
            # Groupby to sum weights for duplicate edges
            # This handles the logic: if G.has_edge(u,v): weight += w
            grouped = df.groupby([source_col, target_col])[weight_col].sum().reset_index()
            
            # Convert to numpy arrays for faster iteration
            sources = grouped[source_col].values
            targets = grouped[target_col].values
            weights = grouped[weight_col].values
        
        # Bulk update adjacency list
        # While we still iterate, it's over unique edges and without DataFrame overhead
        for u, v, w in zip(sources, targets, weights):
            if u not in self.adj:
                self.adj[u] = {}
            if v not in self.adj:
                self.adj[v] = {}
                
            # We already summed weights in groupby, so just assign or add if called multiple times
            if v in self.adj[u]:
                self.adj[u][v] += w
            else:
                self.adj[u][v] = w
                self._num_edges += 1
                
            # Ensure target node exists in adj
            if v not in self.adj:
                self.adj[v] = {}

    def all_simple_paths(self, source, target, cutoff):
        """
        Find all simple paths from source to target with length <= cutoff.
        Returns a generator of paths (lists of nodes).
        """
        if source not in self.adj:
            return
        
        # Use iterative DFS with state management to avoid recursion limit and overhead
        # Stack stores: (current_node, path_list, visited_set)
        # Actually, for all_simple_paths, we need to backtrack.
        # Recursive generator is often cleanest for this.
        yield from self._dfs(source, target, cutoff, [source], {source})

    def _dfs(self, u, target, cutoff, path, visited):
        if len(path) > cutoff + 1:
            return
        
        if u == target:
            yield list(path)
            return
            
        if u not in self.adj:
            return

        # Iterate over neighbors
        for v in self.adj[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)
                yield from self._dfs(v, target, cutoff, path, visited)
                path.pop()
                visited.remove(v)

    def find_paths_dfs_backtracking(self, sources, targets, cutoff, verbose=False):
        """
        Standard Backward DFS with backtracking (no memoization).
        Optimized for lowest memory usage (no caching) at the cost of CPU time.
        Uses Iterative Deepening to find shortest paths first.
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
        """
        # Ensure sources/targets are sets for fast lookup
        source_set = set(sources)
        target_set = set(targets)
        
        # We need reverse graph for backward traversal
        R = self.reverse()
        
        # Helper: Iterative DFS yielding paths of exact length
        def simple_dfs_paths(start_node, graph, target_depth, valid_end_nodes=None):
            # Stack: (u, path)
            stack = [(start_node, [start_node])]
            
            while stack:
                u, path = stack.pop()
                current_len = len(path) - 1
                
                if current_len == target_depth:
                    if valid_end_nodes is None or u in valid_end_nodes:
                        yield u, path
                    continue
                
                if u in graph.adj:
                    for v in graph.adj[u]:
                        if v not in path: # Cycle check
                            stack.append((v, path + [v]))

        # Helper: Get nodes at exact distance (Set only)
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

        # Progress bar support
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        for length in range(1, cutoff + 1):
            mid = length // 2
            rem = length - mid
            
            # 1. Identify Middle Nodes (Backward Reachability)
            # Find set of nodes reachable from targets in exactly 'rem' steps
            valid_mids = get_reachable_set(target_set, R, rem)
            
            if not valid_mids:
                continue
                
            # 2. Build Forward Paths (Store)
            # Sources -> Mid (length 'mid')
            fwd_paths_map = defaultdict(list)
            
            iterator = sources
            if verbose:
                iterator = tqdm(sources, desc=f"L{length} Fwd(L{mid})", leave=False)
            
            for s in iterator:
                if s not in self.adj: continue
                # DFS from s, depth mid, must end in valid_mids
                for end_node, path in simple_dfs_paths(s, self, mid, valid_mids):
                    fwd_paths_map[end_node].append(path)
                    
            if not fwd_paths_map:
                continue
                
            # 3. Stream Backward Paths & Join
            # Targets -> Mid (length 'rem') in Reverse Graph
            valid_ends_for_backward = set(fwd_paths_map.keys())
            
            iterator = targets
            if verbose:
                iterator = tqdm(targets, desc=f"L{length} Bwd(L{rem})", leave=False)
                
            for t in iterator:
                if t not in R.adj: continue
                for end_node, r_path in simple_dfs_paths(t, R, rem, valid_ends_for_backward):
                    # r_path is [target, ..., mid]
                    # reverse to [mid, ..., target]
                    r_path_rev = r_path[::-1]
                    r_set = set(r_path_rev)
                    
                    # Join
                    for f_path in fwd_paths_map[end_node]:
                        # Cycle check: intersection should be only {mid}
                        if len(set(f_path) & r_set) == 1:
                            combined = f_path + r_path_rev[1:]
                            yield combined

    def find_paths_backward_dp(self, sources, targets, cutoff, verbose=False):
        """
        Backward Reachability (DP) + Forward DFS.
        1. Compute sets of nodes reachable from targets at each distance k.
        2. Forward DFS guided by these sets.
        """
        target_set = set(targets)
        R = self.reverse()
        
        # valid_nodes_at_dist[k] = set of nodes that can reach ANY target in exactly k steps
        valid_nodes_at_dist = [set() for _ in range(cutoff + 2)]
        valid_nodes_at_dist[0] = target_set
        
        # 1. Backward Reachability
        for k in range(1, cutoff + 1):
            prev_set = valid_nodes_at_dist[k-1]
            current_set = valid_nodes_at_dist[k]
            for target_node in prev_set:
                if target_node in R.adj:
                    current_set.update(R.adj[target_node])
        
        # 2. Guided DFS
        def guided_dfs(u, depth, current_path):
            if depth == 0:
                if u in target_set:
                    yield list(current_path)
                return

            if u in self.adj:
                # Look ahead: only visit neighbors that can reach target in (depth-1) steps
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
        Memoized DFS.
        direction='forward': Standard memoization from source.
        direction='backward': Memoization from target (often better for pruning).
        """
        # Note: Full memoization of "all paths" is memory prohibitive.
        # We usually memoize "reachability" or "count", not the paths themselves.
        # If we must return paths, we can memoize the *structure* (DAG) and then expand.
        
        # Here we implement the "Backward Memoized DFS" as requested for comparison.
        # It builds paths from Target -> Source.
        
        if direction == 'backward':
            # Search in Reverse Graph from Targets -> Sources
            R = self.reverse()
            # We just call the same logic but on R, swapping sources/targets
            # But we need to reverse the resulting paths
            for path in R.find_paths_memoized_dfs(targets, sources, cutoff, direction='forward', verbose=verbose):
                yield path[::-1]
            return

        # Forward Memoized DFS
        # memo[(u, k)] = list of paths of length k starting at u and ending at ANY target
        # This is dangerous for memory. We will use the "Valid Successor" approach instead.
        
        target_set = set(targets)
        valid_successors = {} # (u, k) -> list of v
        
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
            # 1. Build Successor Graph
            iterator = sources
            if verbose:
                iterator = tqdm(sources, desc=f"L{length} BuildMemo", leave=False)
            
            valid_sources = []
            for s in iterator:
                if find_valid_successors(s, length) is not None:
                    valid_sources.append(s)
            
            # 2. Reconstruct
            iterator = valid_sources
            if verbose:
                iterator = tqdm(valid_sources, desc=f"L{length} Reconstruct", leave=False)
                
            for s in iterator:
                yield from reconstruct(s, length, [s])

    def find_paths_bidirectional_bfs(self, sources, targets, cutoff, verbose=False):
        """
        Bidirectional BFS (Layer-based).
        Builds search trees from both sides and finds intersection.
        """
        # 1. Forward BFS layers
        # f_layers[d] = {u: {parents}}
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

        # 2. Backward BFS layers (using Reverse graph logic implicitly)
        # b_layers[d] = {u: {children}} (d is distance from target)
        b_layers = [defaultdict(set) for _ in range(cutoff + 1)]
        target_set = set(targets)
        
        # We need reverse graph to find parents of targets efficiently? 
        # Or we can just use the fact that we only care about nodes in f_layers.
        # Actually, standard Bidirectional BFS meets in the middle.
        
        # Let's use the Reverse Graph for clean backward expansion
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
                        next_layer[v].add(u) # v is parent of u in forward graph
            if not next_layer: break

        # 3. Intersect and Reconstruct
        # For each length L, we can meet at any depth i (0..L)
        # But typically we meet at L/2.
        # Here we just iterate all lengths.
        
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, **kwargs): return iterable

        for length in range(1, cutoff + 1):
            mid = length // 2
            rem = length - mid
            
            # Intersection nodes at depth 'mid' from source (which is 'rem' from target)
            f_nodes = set(f_layers[mid].keys())
            b_nodes = set(b_layers[rem].keys())
            meet_nodes = f_nodes & b_nodes
            
            if not meet_nodes: continue
            
            # Reconstruct paths passing through meet_nodes
            # This requires backtracking parents in f_layers and children in b_layers
            
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
                for child in b_layers[depth][u]: # child in b_layers is parent in R, so child in G
                    if child is not None:
                        for p in get_bwd_paths(child, depth-1):
                            if u not in p:
                                yield [u] + p

            iterator = meet_nodes
            if verbose:
                iterator = tqdm(meet_nodes, desc=f"L{length} Reconstruct", leave=False)

            for u in iterator:
                # Generate all forward paths to u
                f_paths = list(get_fwd_paths(u, mid))
                # Generate all backward paths from u
                b_paths = list(get_bwd_paths(u, rem))
                
                for fp in f_paths:
                    for bp in b_paths:
                        # bp starts with u, fp ends with u
                        # Check full cycle
                        if len(set(fp) & set(bp)) == 1:
                            yield fp + bp[1:]

