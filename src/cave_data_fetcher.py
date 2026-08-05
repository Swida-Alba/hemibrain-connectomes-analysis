"""
CAVE API Data Fetcher for FlyWire (FAFB) datasets.

This module provides functionality to fetch neuron skeletons, meshes, and synapses
from the CAVE/CloudVolume API for FlyWire datasets.

Key features:
- Fetch meshes via CloudVolume (graphene protocol)
- Generate skeletons from meshes using navis
- Cache results in cache/{dataset}/API_cache/
- Integration with TokenManager for authentication

Usage:
    from cave_data_fetcher import CAVEDataFetcher
    
    fetcher = CAVEDataFetcher(dataset='flywire_FAFB_v783')
    skeleton = fetcher.fetch_skeleton(720575940596125868)
    mesh = fetcher.fetch_mesh(720575940596125868)
    synapses = fetcher.fetch_synapses(720575940596125868)
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import warnings

import pandas as pd

# Suppress warnings during import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import navis

logger = logging.getLogger(__name__)


@dataclass
class CAVEDataFetcher:
    """Fetch neuron data from CAVE/CloudVolume API for FlyWire datasets."""
    
    dataset: str = 'flywire_FAFB_v783'
    """Dataset name (flywire_FAFB_v783 or flywire_BANC_v626)"""
    
    cave_token: str = None
    """CAVE authentication token. If None, reads from token_info_local.txt"""
    
    materialization_version: int = None
    """Materialization version for CAVE queries. None = latest"""
    
    cache_enabled: bool = True
    """Whether to cache fetched data to disk"""
    
    project_root: str = None
    """Project root directory. If None, auto-detected."""
    
    verbose: bool = True
    """Whether to print progress messages"""
    
    # CloudVolume config
    _cv: object = field(default=None, repr=False)
    _cave_client: object = field(default=None, repr=False)
    
    # Dataset-specific settings
    FLYWIRE_FAFB_CONFIG = {
        'datastack': 'flywire_fafb_public',
        'cloudvolume_url': 'graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
        'synapse_table': 'synapses_nt_v1',
        'synapse_view': 'valid_synapses_nt_np_v6',
    }
    
    FLYWIRE_BANC_CONFIG = {
        'datastack': 'brain_and_nerve_cord',
        'cloudvolume_url': None,  # BANC requires special access
        'synapse_table': 'synapses',
    }
    
    def __post_init__(self):
        """Initialize the fetcher with tokens and paths."""
        if self.project_root is None:
            self.project_root = str(Path(__file__).parent.parent)
        
        # Load token if not provided
        if self.cave_token is None:
            self.cave_token = self._load_token('CAVE_TOKEN')
        
        # Create API cache directory
        self._ensure_cache_dir()
    
    def _load_token(self, token_name: str) -> Optional[str]:
        """Load token from token_info_local.txt or token_info.txt"""
        for filename in ['token_info_local.txt', 'token_info.txt']:
            token_path = os.path.join(self.project_root, filename)
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    for line in f:
                        if line.startswith(f'{token_name}='):
                            token = line.split('=', 1)[1].strip().strip("'\"")
                            if token and not token.startswith('YOUR_'):
                                return token
        return None
    
    def _get_config(self) -> dict:
        """Get configuration for current dataset."""
        if 'FAFB' in self.dataset.upper() or 'v783' in self.dataset:
            return self.FLYWIRE_FAFB_CONFIG
        elif 'BANC' in self.dataset.upper():
            return self.FLYWIRE_BANC_CONFIG
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def _ensure_cache_dir(self):
        """Ensure API cache directory exists."""
        cache_dir = self.get_cache_path()
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'skeletons'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, 'meshes'), exist_ok=True)
    
    def get_cache_path(self, subdir: str = '') -> str:
        """Get path to API cache directory.
        
        Returns:
            Path to cache/{dataset}/API_cache/{subdir}
        """
        # Map dataset name to cache folder
        if 'FAFB' in self.dataset.upper() or 'v783' in self.dataset:
            cache_dataset = 'flywire_FAFB_v783'
        elif 'BANC' in self.dataset.upper():
            cache_dataset = 'flywire_BANC_v626'
        else:
            cache_dataset = self.dataset.replace(':', '_').replace('.', '_')
        
        cache_path = os.path.join(self.project_root, 'cache', cache_dataset, 'API_cache')
        if subdir:
            cache_path = os.path.join(cache_path, subdir)
        return cache_path
    
    @property
    def cloudvolume(self):
        """Lazy-load CloudVolume client."""
        if self._cv is None:
            try:
                from cloudvolume import CloudVolume
                config = self._get_config()
                if config['cloudvolume_url'] is None:
                    raise ValueError(f"CloudVolume not available for {self.dataset}")
                
                self._cv = CloudVolume(
                    config['cloudvolume_url'],
                    use_https=True,
                    secrets={'token': self.cave_token}
                )
                if self.verbose:
                    print(f"✓ Connected to CloudVolume: {config['cloudvolume_url'][:50]}...")
            except ImportError:
                raise ImportError("cloudvolume package required. Install with: pip install cloud-volume")
        return self._cv
    
    @property
    def cave_client(self):
        """Lazy-load CAVE client."""
        if self._cave_client is None:
            try:
                from caveclient import CAVEclient
                config = self._get_config()
                self._cave_client = CAVEclient(
                    datastack_name=config['datastack'],
                    auth_token=self.cave_token
                )
                if self.verbose:
                    print(f"✓ Connected to CAVE: {config['datastack']}")
            except ImportError:
                raise ImportError("caveclient package required. Install with: pip install caveclient")
        return self._cave_client
    
    def _get_skeleton_cache_path(self, body_id: int) -> str:
        """Get cache file path for a skeleton."""
        return os.path.join(self.get_cache_path('skeletons'), f'{body_id}.pkl')
    
    def _get_mesh_cache_path(self, body_id: int) -> str:
        """Get cache file path for a mesh."""
        return os.path.join(self.get_cache_path('meshes'), f'{body_id}.pkl')
    
    def _load_from_cache(self, cache_path: str):
        """Load object from cache if exists."""
        if self.cache_enabled and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, obj, cache_path: str):
        """Save object to cache."""
        if self.cache_enabled:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(obj, f)
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def fetch_mesh(self, body_id: int, use_cache: bool = False) -> Optional['navis.MeshNeuron']:
        """Fetch mesh for a neuron from CloudVolume.
        
        Note: Meshes are NOT cached to save disk space. Only simplified skeletons
        are cached (see fetch_skeleton). Each call will fetch the mesh fresh from API.
        
        Parameters
        ----------
        body_id : int
            The root ID of the neuron
        use_cache : bool
            Deprecated - meshes are no longer cached. Parameter kept for compatibility.
            
        Returns
        -------
        navis.MeshNeuron or None
            The mesh neuron, or None if fetch failed
        """
        body_id = int(body_id)
        
        try:
            cv = self.cloudvolume
            mesh = cv.mesh.get(body_id)[body_id]
            
            # Convert to navis MeshNeuron
            mesh_neuron = navis.MeshNeuron(
                mesh,
                id=body_id,
                name=str(body_id),
                units='nm'
            )
            
            if self.verbose:
                print(f"  ✓ Fetched mesh: {body_id} ({len(mesh.vertices)} vertices)")
            
            return mesh_neuron
            
        except Exception as e:
            logger.error(f"Failed to fetch mesh for {body_id}: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch mesh: {body_id}: {e}")
            return None
    
    def fetch_skeleton(self, body_id: int, use_cache: bool = True, 
                       simplify_mesh: float = 0.95) -> Optional['navis.TreeNeuron']:
        """Fetch skeleton for a neuron by skeletonizing the mesh.
        
        Since FlyWire doesn't have L2 cache for pcg_skel, we fetch the mesh
        and skeletonize it using navis.
        
        Parameters
        ----------
        body_id : int
            The root ID of the neuron
        use_cache : bool
            Whether to use cached skeleton if available
        simplify_mesh : float
            Mesh simplification factor before skeletonization (0.0-1.0).
            Higher values remove more faces for faster skeletonization.
            Default 0.95 removes 95% of faces. Cached skeletons use this level.
            
        Returns
        -------
        navis.TreeNeuron or None
            The skeleton, or None if fetch failed
        """
        body_id = int(body_id)
        cache_path = self._get_skeleton_cache_path(body_id)
        
        # Try cache first
        if use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                if self.verbose:
                    print(f"  ✓ Loaded skeleton from API cache: {body_id}")
                return cached
        
        try:
            # Fetch mesh first
            mesh = self.fetch_mesh(body_id, use_cache=use_cache)
            if mesh is None:
                return None
            
            # Simplify mesh for faster skeletonization
            if simplify_mesh > 0:
                import trimesh
                tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                
                # simplify_mesh is the fraction to REMOVE (0.9 = remove 90% = keep 10%)
                # trimesh.simplify_quadric_decimation takes percent= fraction to KEEP
                keep_fraction = 1.0 - simplify_mesh
                target_faces = max(int(len(tm.faces) * keep_fraction), 1000)
                
                try:
                    # Use face_count parameter instead of percent for more predictable results
                    tm_simplified = tm.simplify_quadric_decimation(face_count=target_faces)
                    mesh = navis.MeshNeuron(tm_simplified, id=body_id, name=str(body_id), units='nm')
                    if self.verbose:
                        print(f"  ℹ️  Simplified mesh: {len(tm_simplified.faces)} faces (from {len(tm.faces)})")
                except Exception as e:
                    logger.warning(f"Mesh simplification failed: {e}")
            
            # Skeletonize
            if self.verbose:
                print(f"  ⏳ Skeletonizing mesh for {body_id}...")
            
            skeleton = navis.skeletonize(mesh, method='wavefront')
            skeleton.id = body_id
            skeleton.name = str(body_id)
            skeleton.units = 'nm'
            
            # Save to cache
            self._save_to_cache(skeleton, cache_path)
            
            if self.verbose:
                print(f"  ✓ Generated skeleton: {body_id} ({len(skeleton.nodes)} nodes)")
            
            return skeleton
            
        except Exception as e:
            logger.error(f"Failed to fetch skeleton for {body_id}: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch skeleton: {body_id}: {e}")
            return None
    
    def fetch_skeletons(self, body_ids: List[int], use_cache: bool = True,
                        simplify_mesh: float = 0.95) -> 'navis.NeuronList':
        """Fetch skeletons for multiple neurons.
        
        Parameters
        ----------
        body_ids : list of int
            List of root IDs
        use_cache : bool
            Whether to use cached skeletons if available
        simplify_mesh : float
            Mesh simplification factor (0.0-1.0). Default 0.95 removes 95% of faces.
            Cached skeletons use 0.95 simplification.
            
        Returns
        -------
        navis.NeuronList
            List of skeletons (may be fewer than requested if some fail)
        """
        from tqdm import tqdm
        
        neurons = []
        failed = []
        
        iterator = body_ids
        if self.verbose:
            iterator = tqdm(body_ids, desc="Fetching skeletons")
        
        for bid in iterator:
            skeleton = self.fetch_skeleton(bid, use_cache=use_cache, simplify_mesh=simplify_mesh)
            if skeleton is not None:
                neurons.append(skeleton)
            else:
                failed.append(bid)
        
        if failed and self.verbose:
            print(f"  ⚠️  Failed to fetch {len(failed)}/{len(body_ids)} skeletons")
        
        return navis.NeuronList(neurons)
    
    def fetch_synapses(self, body_id: int, direction: str = 'both') -> Optional[pd.DataFrame]:
        """Fetch synapses for a neuron from CAVE.
        
        Parameters
        ----------
        body_id : int
            The root ID of the neuron
        direction : str
            'pre' for outgoing synapses, 'post' for incoming, 'both' for all
            
        Returns
        -------
        pd.DataFrame or None
            Synapse data with columns: pre_pt_root_id, post_pt_root_id, 
            pre_pt_position, post_pt_position, etc.
        """
        body_id = int(body_id)
        config = self._get_config()
        
        try:
            client = self.cave_client
            
            dfs = []
            
            if direction in ['pre', 'both']:
                # Outgoing synapses (this neuron is presynaptic)
                try:
                    pre_df = client.materialize.query_view(
                        config['synapse_view'],
                        filter_in_dict={'pre_pt_root_id': [body_id]},
                        materialization_version=self.materialization_version
                    )
                    dfs.append(pre_df)
                    if self.verbose:
                        print(f"  ✓ Fetched {len(pre_df)} outgoing synapses")
                except Exception as e:
                    logger.warning(f"Failed to fetch outgoing synapses: {e}")
            
            if direction in ['post', 'both']:
                # Incoming synapses (this neuron is postsynaptic)
                try:
                    post_df = client.materialize.query_view(
                        config['synapse_view'],
                        filter_in_dict={'post_pt_root_id': [body_id]},
                        materialization_version=self.materialization_version
                    )
                    dfs.append(post_df)
                    if self.verbose:
                        print(f"  ✓ Fetched {len(post_df)} incoming synapses")
                except Exception as e:
                    logger.warning(f"Failed to fetch incoming synapses: {e}")
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                # drop_duplicates fails on array columns, use subset of hashable columns
                id_cols = [c for c in ['id', 'pre_pt_root_id', 'post_pt_root_id'] if c in result.columns]
                if id_cols:
                    result = result.drop_duplicates(subset=id_cols)
                return result
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch synapses for {body_id}: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch synapses: {body_id}: {e}")
            return None
    
    def fetch_connections(self, body_ids: List[int], direction: str = 'both', 
                          batch_size: int = 200, show_progress: bool = True) -> Optional[pd.DataFrame]:
        """Fetch connections (aggregated synapse counts) for neurons from CAVE.
        
        This fetches synapses and aggregates them into connection weights.
        For large numbers of neurons, batches requests with a progress bar.
        
        Parameters
        ----------
        body_ids : list of int
            List of root IDs
        direction : str
            'pre' for outgoing connections, 'post' for incoming, 'both' for all
        batch_size : int
            Number of neurons per API request (default 200)
        show_progress : bool
            Whether to show progress bar for batched requests
            
        Returns
        -------
        pd.DataFrame or None
            Connections with columns: pre_pt_root_id, post_pt_root_id, weight
        """
        body_ids = [int(bid) for bid in body_ids]
        config = self._get_config()
        
        try:
            client = self.cave_client
            
            # Batch large requests
            n_neurons = len(body_ids)
            n_batches = (n_neurons + batch_size - 1) // batch_size
            
            # Only show progress bar for multiple batches
            use_progress = show_progress and n_batches > 1 and self.verbose
            
            if use_progress:
                try:
                    from tqdm import tqdm
                    batches = tqdm(
                        range(n_batches), 
                        desc="  🌐 CAVE API", 
                        unit="batch",
                        ncols=80,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    )
                except ImportError:
                    use_progress = False
                    batches = range(n_batches)
            else:
                batches = range(n_batches)
            
            dfs = []
            
            for i in batches:
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_neurons)
                batch_ids = body_ids[batch_start:batch_end]
                
                if direction in ['pre', 'both']:
                    # Outgoing connections
                    try:
                        pre_df = client.materialize.query_view(
                            config['synapse_view'],
                            filter_in_dict={'pre_pt_root_id': batch_ids},
                            materialization_version=self.materialization_version
                        )
                        if not pre_df.empty:
                            dfs.append(pre_df)
                    except Exception as e:
                        logger.warning(f"Failed to fetch outgoing connections for batch {i+1}: {e}")
                
                if direction in ['post', 'both']:
                    # Incoming connections
                    try:
                        post_df = client.materialize.query_view(
                            config['synapse_view'],
                            filter_in_dict={'post_pt_root_id': batch_ids},
                            materialization_version=self.materialization_version
                        )
                        if not post_df.empty:
                            dfs.append(post_df)
                    except Exception as e:
                        logger.warning(f"Failed to fetch incoming connections for batch {i+1}: {e}")
            
            if dfs:
                # Combine and aggregate
                all_synapses = pd.concat(dfs, ignore_index=True)
                
                # Aggregate to connection weights
                connections = all_synapses.groupby(
                    ['pre_pt_root_id', 'post_pt_root_id']
                ).size().reset_index(name='weight')
                
                if self.verbose:
                    print(f"  ✓ Fetched {len(connections):,} unique connections for {n_neurons} neurons")
                
                return connections
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch connections: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch connections: {e}")
            return None
    
    def fetch_neuron_info(self, body_ids: List[int], batch_size: int = 500, 
                          show_progress: bool = True) -> Optional[pd.DataFrame]:
        """Fetch neuron annotations/info from CAVE.
        
        For large numbers of neurons, batches requests with a progress bar.
        
        Parameters
        ----------
        body_ids : list of int
            List of root IDs
        batch_size : int
            Number of neurons per API request (default 500)
        show_progress : bool
            Whether to show progress bar for batched requests
            
        Returns
        -------
        pd.DataFrame or None
            Neuron annotations with columns from hierarchical_neuron_annotations
        """
        body_ids = [int(bid) for bid in body_ids]
        n_neurons = len(body_ids)
        n_batches = (n_neurons + batch_size - 1) // batch_size
        
        try:
            client = self.cave_client
            
            # Only show progress bar for multiple batches
            use_progress = show_progress and n_batches > 1 and self.verbose
            
            if use_progress:
                try:
                    from tqdm import tqdm
                    batches = tqdm(
                        range(n_batches), 
                        desc="  📋 Neuron info", 
                        unit="batch",
                        ncols=80,
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    )
                except ImportError:
                    use_progress = False
                    batches = range(n_batches)
            else:
                batches = range(n_batches)
            
            dfs = []
            for i in batches:
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_neurons)
                batch_ids = body_ids[batch_start:batch_end]
                
                # Query hierarchical annotations for this batch
                df = client.materialize.query_table(
                    'hierarchical_neuron_annotations',
                    filter_in_dict={'pt_root_id': batch_ids},
                    materialization_version=self.materialization_version
                )
                if not df.empty:
                    dfs.append(df)
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                if self.verbose:
                    print(f"  ✓ Fetched annotations for {len(result):,}/{n_neurons:,} neurons")
                return result
            else:
                if self.verbose:
                    print(f"  ℹ️ No annotations found for {n_neurons} neurons")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to fetch neuron info: {e}")
            if self.verbose:
                print(f"  ✗ Failed to fetch neuron info: {e}")
            return None
    
    def clear_cache(self, cache_type: str = 'all'):
        """Clear the API cache.
        
        Parameters
        ----------
        cache_type : str
            'skeletons', 'meshes', or 'all'
        """
        import shutil
        
        if cache_type in ['skeletons', 'all']:
            skel_dir = self.get_cache_path('skeletons')
            if os.path.exists(skel_dir):
                shutil.rmtree(skel_dir)
                os.makedirs(skel_dir)
                print(f"✓ Cleared skeleton cache: {skel_dir}")
        
        if cache_type in ['meshes', 'all']:
            mesh_dir = self.get_cache_path('meshes')
            if os.path.exists(mesh_dir):
                shutil.rmtree(mesh_dir)
                os.makedirs(mesh_dir)
                print(f"✓ Cleared mesh cache: {mesh_dir}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the cache.
        
        Returns
        -------
        dict
            Dictionary with cache statistics
        """
        skel_dir = self.get_cache_path('skeletons')
        mesh_dir = self.get_cache_path('meshes')
        
        skel_count = len([f for f in os.listdir(skel_dir) if f.endswith('.pkl')]) if os.path.exists(skel_dir) else 0
        mesh_count = len([f for f in os.listdir(mesh_dir) if f.endswith('.pkl')]) if os.path.exists(mesh_dir) else 0
        
        # Calculate total size
        def get_dir_size(path):
            total = 0
            if os.path.exists(path):
                for f in os.listdir(path):
                    fp = os.path.join(path, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
            return total
        
        skel_size = get_dir_size(skel_dir)
        mesh_size = get_dir_size(mesh_dir)
        
        return {
            'skeleton_count': skel_count,
            'mesh_count': mesh_count,
            'skeleton_size_mb': round(skel_size / 1024 / 1024, 2),
            'mesh_size_mb': round(mesh_size / 1024 / 1024, 2),
            'total_size_mb': round((skel_size + mesh_size) / 1024 / 1024, 2),
        }


def test_fafb_access():
    """Test FAFB access with a sample neuron."""
    print("=" * 60)
    print("Testing CAVE API Data Fetcher for FlyWire FAFB")
    print("=" * 60)
    
    fetcher = CAVEDataFetcher(dataset='flywire_FAFB_v783')
    
    # Test with a known FAFB neuron
    test_id = 720575940596125868
    
    print(f"\n--- Testing with neuron: {test_id} ---")
    
    # Test mesh fetch
    print("\n1. Fetching mesh...")
    mesh = fetcher.fetch_mesh(test_id)
    if mesh:
        print(f"   Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Test skeleton fetch
    print("\n2. Fetching skeleton (via mesh skeletonization)...")
    skeleton = fetcher.fetch_skeleton(test_id)
    if skeleton:
        print(f"   Skeleton: {len(skeleton.nodes)} nodes")
    
    # Test synapse fetch
    print("\n3. Fetching synapses...")
    synapses = fetcher.fetch_synapses(test_id, direction='pre')
    if synapses is not None:
        print(f"   Synapses: {len(synapses)} outgoing")
    
    # Cache stats
    print("\n4. Cache statistics:")
    stats = fetcher.get_cache_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_fafb_access()
