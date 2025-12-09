
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
        if df.empty:
            return
            
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
