#!/usr/bin/env python3
"""
3D Neuron Class Visualizer — generates standalone Three.js HTML
visualizations from neuPrint connectomics data.

Usage:
    python generate_visualization.py FB
    python generate_visualization.py EPG MeTu KC
    python generate_visualization.py --all
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import navis
import navis.interfaces.neuprint as neu
from neuprint import (NeuronCriteria as NC, fetch_neurons, fetch_adjacencies,
                      fetch_synapse_connections)
import neuprint
import trimesh as tm_lib
import argparse
import json
import re
import os
import sys
import base64
import time
import functools
from collections import Counter
from pathlib import Path

# Force all print output to flush immediately (needed for Jupyter notebooks)
print = functools.partial(print, flush=True)

# ============================================================
# Constants
# ============================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'Non_Committal'

# Loading screen logo (base64-encoded PNG)
_LOGO_PATH = SCRIPT_DIR / 'Chucky_gold_cropped.png'
if _LOGO_PATH.exists():
    LOADING_LOGO_B64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode('ascii')
else:
    LOADING_LOGO_B64 = ''

LINEWIDTH = 1
ROI_COLOR = (0.85, 0.85, 0.85)
ROI_MESH_OPACITY = 0.08

def _compute_default_camera(norm_params):
    """Compute a default camera looking at the center of the data from the -Z direction.

    After normalization, the data center is at ~(0, 0, 0) and the extent is ~1.0.
    We place the camera back along -Z at ~0.6 to frame the data, looking at center.
    """
    # In normalized coords, data is roughly centered at origin with max extent ~1
    # The center of mass may have a slight offset from origin due to asymmetry
    cx = cy = cz = 0.0
    if norm_params and 'aspect' in norm_params:
        # Small vertical offset to center the data visually
        a = norm_params['aspect']
        cy = -(a.get('y', 1.0) - 1.0) * 0.02  # nudge up slightly if aspect < 1
    return dict(
        eye=dict(x=0.0, y=cy, z=-0.6),
        center=dict(x=0.0, y=cy, z=0.0),
        up=dict(x=0, y=-1, z=0),  # Dorsal-up (-Y)
        projection=dict(type='perspective')
    )

NT_COLORS = {
    'acetylcholine': 'rgb(255,255,255)',
    'gaba':          'rgb(255,40,40)',
    'glutamate':     'rgb(0,220,0)',
    'dopamine':      'rgb(0,180,255)',
    'serotonin':     'rgb(255,220,0)',
    'octopamine':    'rgb(255,0,220)',
}
NT_DEFAULT_COLOR = 'rgb(140,140,140)'

BRAIN_NEUROPILS = [
    'SMP(L)', 'SMP(R)', 'SLP(L)', 'SLP(R)', 'SIP(L)', 'SIP(R)',
    'CRE(L)', 'CRE(R)', 'FB', 'EB', 'PB', 'NO', 'LH(L)', 'LH(R)',
    'AL(L)', 'AL(R)', 'AVLP(L)', 'AVLP(R)', 'PVLP(L)', 'PVLP(R)',
    'GNG', 'LAL(L)', 'LAL(R)', 'IB', 'ICL(L)', 'ICL(R)',
    'ME(L)', 'ME(R)', 'LO(L)', 'LO(R)', 'LOP(L)', 'LOP(R)',
    'aL(L)', 'aL(R)', 'gL(L)', 'gL(R)', 'PLP(L)', 'PLP(R)',
    'SCL(L)', 'SCL(R)', 'CA(L)', 'CA(R)', 'PED(L)', 'PED(R)',
    'WED(L)', 'WED(R)', 'EPA(L)', 'EPA(R)', 'IPS(L)', 'IPS(R)',
]

ALL_PATTERNS = [
    'AOTU', 'Delta', 'Delta7', 'EL', 'EPG', 'ER', 'ExR', 'FB', 'FC', 'KC',
    'LNO|GLNO|LCNO.*', 'MeTu', 'OA', 'PAM', 'PEG', 'PEN', 'PFG',
    'PFL', 'PFN', 'PFR', 'PPL', 'TuBu', 'aMe', 'hDelta', 'vDelta',
]

# Sequential colormaps cycled for auto-assigning continuous (non-divergent) modes
SEQUENTIAL_CMAPS = ['Oranges', 'Purples', 'Greens', 'Blues', 'Reds', 'YlOrBr', 'BuGn', 'PuRd']

META_ROIS = {'CentralBrain', 'CentralBrain-unspecified'}
ANCHOR_MAX_FACES = 200


# ============================================================
# neuPrint Client
# ============================================================

def get_client(server='neuprint-cns.janelia.org', dataset='cns', token=None):
    """Initialize and return a neuPrint client.

    Args:
        server:   neuPrint server URL
        dataset:  Dataset name (e.g. 'cns', 'hemibrain:v1.2.1')
        token:    neuPrint API token. If None, reads from NEUPRINT_TOKEN env var.
                  Get yours at: https://neuprint-cns.janelia.org/ → Account → Auth Token
    """
    if token is None:
        token = os.environ.get('NEUPRINT_TOKEN')
    if not token:
        raise ValueError(
            'No neuPrint token provided. Either:\n'
            '  1. Pass token= to get_client() or generate_visualization()\n'
            '  2. Set the NEUPRINT_TOKEN environment variable\n'
            '  Get your token at: https://neuprint-cns.janelia.org/ → Account → Auth Token'
        )
    return neuprint.Client(server=server, dataset=dataset, token=token)


# ============================================================
# Data Fetching
# ============================================================

def fetch_neuron_data(pattern):
    """Fetch neurons and skeletons matching the regex pattern.

    Returns:
        all_neurons_df, roi_counts_df, neurons_full, type_lookup, nt_lookup
    """
    all_neurons_df, roi_counts_df = fetch_neurons(
        NC(status='Traced', type=pattern))

    # Filter out bare class name (e.g. instance == 'FB' with no suffix)
    class_prefix = re.sub(r'[\^$.*+?\[\](){}|\\]', '', pattern)
    all_neurons_df = all_neurons_df.query('instance != @class_prefix').reset_index(drop=True)

    n_total = len(all_neurons_df)
    print(f'{n_total} neurons ({all_neurons_df["type"].nunique()} types)')
    _B = '\033[1;33m'  # bold yellow
    _R = '\033[0m'     # reset
    if n_total > 500:
        print(f'{_B}🐭 Go talk to Chucky. Seriously, you have time. (~20+ min){_R}')
    elif n_total > 300:
        print(f'{_B}🏃 Do a lap around the building. This is gonna be a while. (~15 min){_R}')
    elif n_total > 150:
        print(f'{_B}☕ Grab a coffee. This\'ll take a few minutes. (~10 min){_R}')
    elif n_total > 50:
        print(f'{_B}📧 Check your email. I\'ll be done when you get back. (~5 min){_R}')
    else:
        print(f'{_B}⚡ Sit tight. (~2 mins) Unless you\'re querying the optic lobe. 💀{_R}')
    print(f'{_B}   Generation is slow, but the HTML it produces loads and runs instantly in the future. So be sure to save it.{_R}')
    if n_total == 0:
        raise RuntimeError(f'No neurons found for pattern: {pattern!r}')

    body_ids = all_neurons_df['bodyId'].tolist()
    print('Fetching neuron skeletons...')
    neurons_full = neu.fetch_skeletons(body_ids, with_synapses=False,
                                       missing_swc='skip')

    type_lookup = dict(zip(all_neurons_df['bodyId'], all_neurons_df['type']))
    if 'predictedNt' in all_neurons_df.columns:
        nt_lookup = dict(zip(all_neurons_df['bodyId'].astype(str),
                             all_neurons_df['predictedNt'].fillna('unknown')))
    else:
        nt_lookup = {}  # Dataset doesn't have predicted neurotransmitter data

    print(f'Fetched {len(neurons_full)} neuron morphologies')
    return all_neurons_df, roi_counts_df, neurons_full, type_lookup, nt_lookup


def _pool_init(script_dir):
    """Initializer for spawned worker processes — ensures they can find our module."""
    import sys
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def _decimate_mesh(verts, faces, max_faces=5000):
    """Decimate a mesh using Open3D quadric edge collapse.

    Produces proper topology-preserving simplification that maintains
    the neuron's anatomical shape.
    """
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Clean up the raw mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    if max_faces and max_faces > 0 and len(mesh.triangles) > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)

    mesh.compute_vertex_normals()

    return (np.asarray(mesh.vertices).astype(np.float32),
            np.asarray(mesh.triangles).astype(np.int32))


def fetch_neuron_meshes(body_ids, max_faces=5000, max_threads=10):
    """Fetch neuron meshes from neuPrint and decimate to target face count.

    Uses vertex clustering for topology-preserving decimation.

    Args:
        body_ids:     List of body IDs
        max_faces:    Max faces per neuron mesh after decimation (default 5000)
        max_threads:  Parallel download threads

    Returns:
        dict: {bodyId_str: {'vertices': np.array (N,3), 'faces': np.array (M,3)}}
    """
    print(f'Fetching neuron meshes ({len(body_ids)} neurons, max {max_faces} faces each)...')
    t0 = time.time()
    mesh_neurons = neu.fetch_mesh_neuron(
        body_ids, lod=None, parallel=True, max_threads=max_threads,
        missing_mesh='skip')

    # Parallel decimation across CPU cores
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    from tqdm import tqdm

    # Detect available cores safely (cpu_count can return None on some systems)
    try:
        n_cpus = multiprocessing.cpu_count() or 1
    except NotImplementedError:
        n_cpus = 1
    n_workers = min(n_cpus, len(mesh_neurons), 8)

    # Prepare args for parallel processing
    mesh_args = [(str(m.id), m.vertices.astype(np.float64), m.faces.astype(np.int64), max_faces)
                 for m in mesh_neurons]

    print(f'  Decimating {len(mesh_args)} meshes ({n_workers} workers)...')
    t1 = time.time()
    result = {}

    if n_workers <= 1 or len(mesh_args) <= 2:
        # Sequential fallback (single core, or very few meshes)
        for bid, v, f, mf in tqdm(mesh_args, desc='Decimating', unit='mesh'):
            dv, df = _decimate_mesh(v, f, mf)
            result[bid] = {'vertices': dv.astype(np.float32), 'faces': df.astype(np.int32)}
    else:
        # Parallel: use 'spawn' start method for cross-platform safety.
        # 'fork' (default on macOS/Linux) can crash with Open3D and other C libraries.
        # 'spawn' works on all platforms (macOS, Linux, Windows).
        try:
            ctx = multiprocessing.get_context('spawn')
            script_dir = str(Path(__file__).resolve().parent)
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=_pool_init, initargs=(script_dir,)
            ) as pool:
                futures = {pool.submit(_decimate_mesh, v, f, mf): bid
                           for bid, v, f, mf in mesh_args}
                with tqdm(total=len(futures), desc=f'Decimating ({n_workers} cores)',
                          unit='mesh') as pbar:
                    for future in as_completed(futures):
                        bid = futures[future]
                        try:
                            dv, df = future.result()
                            result[bid] = {'vertices': dv.astype(np.float32), 'faces': df.astype(np.int32)}
                        except Exception as e:
                            print(f'\n    Warning: decimation failed for {bid}: {e}')
                        pbar.update(1)
        except Exception as e:
            # If parallel fails entirely (rare edge cases), fall back to sequential
            print(f'  Parallel decimation failed ({e}), falling back to sequential...')
            result = {}
            for bid, v, f, mf in tqdm(mesh_args, desc='Decimating', unit='mesh'):
                dv, df = _decimate_mesh(v, f, mf)
                result[bid] = {'vertices': dv.astype(np.float32), 'faces': df.astype(np.int32)}

    elapsed = time.time() - t0
    total_faces = sum(len(v['faces']) for v in result.values())
    total_mb = sum(v['vertices'].nbytes + v['faces'].nbytes for v in result.values()) / 1e6
    print(f'  {len(result)} meshes: {total_faces:,} total faces, {total_mb:.1f} MB, {elapsed:.1f}s')
    return result


def _assign_mesh_face_rois(neuron_meshes, neurons_clipped, roi_clipped,
                           primary_roi, norm_params):
    """Assign each mesh face to an ROI using nearest skeleton point.

    For each neuron mesh, builds a KD-tree from the neuron's per-ROI skeleton
    points, then assigns each face centroid to the nearest skeleton point's ROI.
    Adds 'faceRois' key: list of ROI name per face, and 'roiLookup': sorted
    unique ROI names for compact encoding.
    """
    from scipy.spatial import cKDTree
    t0 = time.time()
    count = 0

    # Build per-neuron skeleton points with ROI labels
    # neurons_clipped: NeuronList clipped to primary ROI
    # roi_clipped: {roi_name: NeuronList} for all other ROIs
    bid_roi_points = {}  # bodyId -> [(roi, points_array), ...]

    # Primary ROI neurons
    for neuron in neurons_clipped:
        bid = str(neuron.id)
        nodes = neuron.nodes
        if nodes is not None and len(nodes) > 0:
            pts = nodes[['x', 'y', 'z']].values
            bid_roi_points.setdefault(bid, []).append((primary_roi, pts))

    # Other ROIs
    for roi_name, neuron_list in roi_clipped.items():
        for neuron in neuron_list:
            bid = str(neuron.id)
            nodes = neuron.nodes
            if nodes is not None and len(nodes) > 0:
                pts = nodes[['x', 'y', 'z']].values
                bid_roi_points.setdefault(bid, []).append((roi_name, pts))

    for bid, mesh_entry in neuron_meshes.items():
        verts = mesh_entry['vertices']
        faces = mesh_entry['faces']

        roi_points = bid_roi_points.get(bid, [])
        if not roi_points:
            mesh_entry['faceRoiIndices'] = [0] * len(faces)
            mesh_entry['roiLookup'] = ['unknown']
            continue

        # Build combined skeleton point array with ROI labels
        all_pts = []
        all_rois = []
        for roi, pts in roi_points:
            all_pts.append(pts)
            all_rois.extend([roi] * len(pts))
        all_pts = np.vstack(all_pts)
        all_rois = np.array(all_rois)

        # KD-tree from skeleton points (in raw coordinates, same as mesh verts)
        tree = cKDTree(all_pts)

        # Face centroids
        centroids = verts[faces].mean(axis=1)  # (M, 3)

        # Find nearest skeleton point for each face centroid
        distances, indices = tree.query(centroids)
        face_rois = all_rois[indices]

        # Distance threshold: faces far from any skeleton point are "unassigned"
        # Use median skeleton edge length * 3 as a generous threshold
        diffs = np.diff(all_pts, axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        dist_thresh = np.median(edge_lengths[edge_lengths > 0]) * 5
        face_rois = np.where(distances <= dist_thresh, face_rois, '_none')

        # Encode as indices into a compact lookup
        unique_rois = sorted(set(face_rois))
        roi_to_idx = {r: i for i, r in enumerate(unique_rois)}
        face_roi_indices = [roi_to_idx[r] for r in face_rois]

        mesh_entry['faceRoiIndices'] = face_roi_indices
        mesh_entry['roiLookup'] = unique_rois
        count += 1

    print(f'  Assigned ROI labels to {count} neuron meshes ({time.time()-t0:.1f}s)')


def fetch_connectivity(body_ids, type_lookup, instance_lookup):
    """Fetch upstream/downstream adjacencies and build connectivity dicts.

    Returns:
        type_upstream, type_downstream, neuron_upstream, neuron_downstream
    """
    print("Fetching upstream connections...")
    t0 = time.time()
    _, upstream_df = fetch_adjacencies(targets=body_ids)
    print(f"  {len(upstream_df)} rows in {time.time()-t0:.1f}s")

    print("Fetching downstream connections...")
    t0 = time.time()
    _, downstream_df = fetch_adjacencies(sources=body_ids)
    print(f"  {len(downstream_df)} rows in {time.time()-t0:.1f}s")

    # Filter meta ROIs
    upstream_df = upstream_df[~upstream_df['roi'].isin(META_ROIS)].copy()
    downstream_df = downstream_df[~downstream_df['roi'].isin(META_ROIS)].copy()

    # Fetch partner metadata
    partner_bids = (set(upstream_df['bodyId_pre'].unique()) |
                    set(downstream_df['bodyId_post'].unique())) - set(body_ids)

    print(f"Fetching metadata for {len(partner_bids)} partner neurons...")
    t0 = time.time()
    partner_type_lookup = {}
    partner_instance_lookup = {}
    if len(partner_bids) > 0:
        partner_bids_list = list(partner_bids)
        batch_size = 5000
        partner_dfs = []
        for i in range(0, len(partner_bids_list), batch_size):
            batch = partner_bids_list[i:i+batch_size]
            pdf, _ = fetch_neurons(NC(bodyId=batch))
            partner_dfs.append(pdf)
        if partner_dfs:
            partner_neurons_df = pd.concat(partner_dfs, ignore_index=True)
            partner_type_lookup = dict(zip(partner_neurons_df['bodyId'],
                                           partner_neurons_df['type'].fillna('unknown')))
            if 'instance' in partner_neurons_df.columns:
                partner_instance_lookup = dict(zip(partner_neurons_df['bodyId'].astype(str),
                                                    partner_neurons_df['instance'].fillna('')))
            else:
                partner_instance_lookup = {}
    print(f"  Done in {time.time()-t0:.1f}s")

    full_type_lookup = {**type_lookup, **partner_type_lookup}
    full_instance_lookup = {**instance_lookup, **partner_instance_lookup}

    # Build TYPE-LEVEL connectivity
    type_upstream = {}
    type_downstream = {}
    for _, row in upstream_df.iterrows():
        our_type = full_type_lookup.get(row['bodyId_post'])
        partner_type = full_type_lookup.get(row['bodyId_pre'], 'unknown')
        roi = str(row['roi'])
        if our_type:
            d = type_upstream.setdefault(str(our_type), {}).setdefault(roi, {})
            d[str(partner_type)] = d.get(str(partner_type), 0) + int(row['weight'])

    for _, row in downstream_df.iterrows():
        our_type = full_type_lookup.get(row['bodyId_pre'])
        partner_type = full_type_lookup.get(row['bodyId_post'], 'unknown')
        roi = str(row['roi'])
        if our_type:
            d = type_downstream.setdefault(str(our_type), {}).setdefault(roi, {})
            d[str(partner_type)] = d.get(str(partner_type), 0) + int(row['weight'])

    # Build NEURON-LEVEL connectivity
    neuron_upstream = {}
    neuron_downstream = {}
    for _, row in upstream_df.iterrows():
        bid = str(int(row['bodyId_post']))
        partner_bid = str(int(row['bodyId_pre']))
        partner_type = str(full_type_lookup.get(row['bodyId_pre'], 'unknown'))
        partner_inst = full_instance_lookup.get(partner_bid, partner_bid)
        roi = str(row['roi'])
        w = int(row['weight'])
        d = neuron_upstream.setdefault(bid, {}).setdefault(roi, {})
        d.setdefault('__types__', {})[partner_type] = d.get('__types__', {}).get(partner_type, 0) + w
        d.setdefault('__instances__', {}).setdefault(partner_type, []).append([partner_inst, partner_bid, w])

    for _, row in downstream_df.iterrows():
        bid = str(int(row['bodyId_pre']))
        partner_bid = str(int(row['bodyId_post']))
        partner_type = str(full_type_lookup.get(row['bodyId_post'], 'unknown'))
        partner_inst = full_instance_lookup.get(partner_bid, partner_bid)
        roi = str(row['roi'])
        w = int(row['weight'])
        d = neuron_downstream.setdefault(bid, {}).setdefault(roi, {})
        d.setdefault('__types__', {})[partner_type] = d.get('__types__', {}).get(partner_type, 0) + w
        d.setdefault('__instances__', {}).setdefault(partner_type, []).append([partner_inst, partner_bid, w])

    print(f"Connectivity: {len(type_upstream)} types upstream, {len(type_downstream)} downstream")
    return type_upstream, type_downstream, neuron_upstream, neuron_downstream


def fetch_synapse_positions(body_ids, type_lookup, partner_type_lookup=None,
                            batch_size=50, synapse_limit=None,
                            max_workers=8):
    """Fetch individual synapse connection positions for neurons in body_ids.

    Uses small parallel batches for maximum throughput against neuPrint.

    Args:
        body_ids:             List of body IDs in our visualization set
        type_lookup:          Dict mapping bodyId → type name (for our neurons)
        partner_type_lookup:  Optional dict for partner bodyId → type name
        batch_size:           Number of body IDs per neuPrint query batch
        synapse_limit:        Max synapses to keep (None = no limit)
        max_workers:          Max parallel neuPrint queries

    Returns:
        syn_df:           DataFrame with synapse positions
        bid_type_map:     Dict mapping all bodyId (str) → type name
    """
    from neuprint import SynapseCriteria as SC
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Fetching individual synapse positions...")
    t0 = time.time()

    # Get client in main thread and pass explicitly to workers
    client = neuprint.default_client()

    # Round-robin deal neurons into buckets for balanced load.
    # Group by type first, then interleave types across buckets so that
    # neurons of the same type (which tend to have similar synapse counts)
    # are spread evenly rather than clumped in one batch.
    from collections import defaultdict
    type_groups = defaultdict(list)
    for bid in body_ids:
        typ = type_lookup.get(bid) or type_lookup.get(int(bid), '')
        type_groups[typ].append(bid)
    # Sort types by group size descending (largest types distributed first)
    sorted_types = sorted(type_groups.keys(), key=lambda t: len(type_groups[t]), reverse=True)
    interleaved = []
    for typ in sorted_types:
        interleaved.extend(type_groups[typ])

    n_batches = max(1, (len(interleaved) + batch_size - 1) // batch_size)
    buckets = [[] for _ in range(n_batches)]
    for i, bid in enumerate(interleaved):
        buckets[i % n_batches].append(bid)

    def _fetch_one(batch, direction):
        """Fetch one batch of synapses."""
        try:
            if direction == 'downstream':
                return fetch_synapse_connections(
                    source_criteria=NC(bodyId=batch),
                    synapse_criteria=SC(primary_only=True),
                    client=client,
                )
            else:
                return fetch_synapse_connections(
                    target_criteria=NC(bodyId=batch),
                    synapse_criteria=SC(primary_only=True),
                    client=client,
                )
        except Exception as e:
            print(f"    {direction} batch failed: {e}")
            return pd.DataFrame()

    # Build all fetch jobs from balanced buckets
    jobs = []
    for batch in buckets:
        jobs.append(('downstream', batch))
        jobs.append(('upstream', batch))

    n_jobs = len(jobs)
    print(f"  {len(interleaved)} neurons → {n_jobs} queries "
          f"(batch={batch_size}, workers={min(max_workers, n_jobs)})")

    # Run all jobs with bounded parallelism
    all_dfs = []
    completed = 0
    total_synapses = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, n_jobs)) as pool:
        futures = {pool.submit(_fetch_one, batch, direction): (direction, len(batch))
                   for direction, batch in jobs}
        for future in as_completed(futures):
            df = future.result()
            completed += 1
            if len(df) > 0:
                all_dfs.append(df)
                total_synapses += len(df)
            print(f"    [{completed}/{n_jobs}] {total_synapses:,} synapses so far "
                  f"({time.time()-t0:.0f}s)")

    if not all_dfs:
        print("  No synapse data found!")
        return pd.DataFrame(), {}

    syn_df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate — same synapse appearing in both upstream & downstream fetch
    before = len(syn_df)
    syn_df = syn_df.drop_duplicates(
        subset=['bodyId_pre', 'bodyId_post', 'x_pre', 'y_pre', 'z_pre',
                'x_post', 'y_post', 'z_post']
    ).reset_index(drop=True)
    if before != len(syn_df):
        print(f"  Deduplicated: {before:,} → {len(syn_df):,} synapses")

    if synapse_limit and len(syn_df) > synapse_limit:
        print(f"  Sampling down to {synapse_limit:,} synapses...")
        syn_df = syn_df.sample(n=synapse_limit, random_state=42).reset_index(drop=True)

    # Build comprehensive bid→type map
    bid_type_map = {str(bid): typ for bid, typ in type_lookup.items()}
    if partner_type_lookup:
        for bid, typ in partner_type_lookup.items():
            bid_type_map.setdefault(str(bid), typ)

    # Find any body IDs still missing type info
    all_syn_bids = set(syn_df['bodyId_pre'].unique()) | set(syn_df['bodyId_post'].unique())
    missing_bids = [b for b in all_syn_bids if str(b) not in bid_type_map]
    if missing_bids:
        print(f"  Fetching type info for {len(missing_bids)} additional partner neurons...")
        for j in range(0, len(missing_bids), 5000):
            batch = missing_bids[j:j+5000]
            try:
                pdf, _ = fetch_neurons(NC(bodyId=batch))
                for _, row in pdf.iterrows():
                    typ = row.get('type') or 'unknown'
                    bid_type_map[str(row['bodyId'])] = typ
            except Exception as e:
                print(f"    Batch failed: {e}")

    elapsed = time.time() - t0
    print(f"  {len(syn_df)} total synapses fetched in {elapsed:.1f}s")
    return syn_df, bid_type_map


def build_synapse_json(syn_df, bid_type_map, norm_params, output_path):
    """Encode synapse data as a compact companion JSON file.

    Normalizes coordinates using the same transform as skeletons,
    quantizes to Int16, and base64 encodes for compactness.

    Args:
        syn_df:       DataFrame from fetch_synapse_positions
        bid_type_map: Dict mapping bodyId (str) → type name
        norm_params:  Dict with cx, cy, cz, dmax
        output_path:  Path to write the JSON file

    Returns:
        Path to written file, or None if no data
    """
    if len(syn_df) == 0:
        print("  No synapse data to write.")
        return None

    synapse_data = _build_synapse_data(syn_df, bid_type_map, norm_params)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(synapse_data, f)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Wrote {output_path} ({size_mb:.1f} MB, {len(syn_df)} synapses)")
    return output_path


def _build_synapse_data(syn_df, bid_type_map, norm_params):
    """Build synapse data dict for embedding or writing to JSON."""
    QUANT_SCALE = 30000
    cx, cy, cz = norm_params['cx'], norm_params['cy'], norm_params['cz']
    dmax = norm_params['dmax']

    def normalize_and_encode_col(values, center):
        arr = np.array(values, dtype=np.float64)
        arr = (arr - center) / dmax
        int16_arr = np.clip(arr * QUANT_SCALE, -32768, 32767).astype(np.int16)
        return base64.b64encode(int16_arr.tobytes()).decode('ascii')

    # Encode bodyIds as base64 int32 arrays (much smaller than JSON int lists)
    pre_arr = np.array(syn_df['bodyId_pre'], dtype=np.int32)
    post_arr = np.array(syn_df['bodyId_post'], dtype=np.int32)
    pre_b64 = base64.b64encode(pre_arr.tobytes()).decode('ascii')
    post_b64 = base64.b64encode(post_arr.tobytes()).decode('ascii')

    # Encode ROIs as lookup table + uint16 index array
    rois_raw = syn_df['roi_pre'].fillna('').tolist()
    unique_rois = sorted(set(rois_raw))
    roi_to_idx = {r: i for i, r in enumerate(unique_rois)}
    roi_indices = np.array([roi_to_idx[r] for r in rois_raw], dtype=np.uint16)
    roi_idx_b64 = base64.b64encode(roi_indices.tobytes()).decode('ascii')

    return {
        'quantScale': QUANT_SCALE,
        'count': len(syn_df),
        'preBidsB64': pre_b64,
        'postBidsB64': post_b64,
        'roiLookup': unique_rois,
        'roiIndicesB64': roi_idx_b64,
        'xPre': normalize_and_encode_col(syn_df['x_pre'], cx),
        'yPre': normalize_and_encode_col(syn_df['y_pre'], cy),
        'zPre': normalize_and_encode_col(syn_df['z_pre'], cz),
        'xPost': normalize_and_encode_col(syn_df['x_post'], cx),
        'yPost': normalize_and_encode_col(syn_df['y_post'], cy),
        'zPost': normalize_and_encode_col(syn_df['z_post'], cz),
        'bidTypeMap': bid_type_map,
    }


def _process_synapse_csvs(synapse_csvs):
    """Read synapse CSVs and return embedded group definitions for the JS viewer.

    Each CSV must have columns: bodyid_pre, bodyid_post, and at least one
    value column (category names or direct CSS color strings).

    Returns list of dicts: {pairs: ["pre|post", ...], color: str, label: str}
    """
    import colorsys
    groups = []
    for csv_path in (synapse_csvs or []):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  Warning: synapse CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        cols_lower = {c.lower(): c for c in df.columns}
        pre_col = cols_lower.get('bodyid_pre') or cols_lower.get('body_id_pre')
        post_col = cols_lower.get('bodyid_post') or cols_lower.get('body_id_post')
        if not pre_col or not post_col:
            print(f"  Warning: {csv_path.name} missing bodyid_pre/bodyid_post columns, skipping")
            continue
        val_cols = [c for c in df.columns if c not in (pre_col, post_col)]
        if not val_cols:
            print(f"  Warning: {csv_path.name} has no value column, skipping")
            continue
        val_col = val_cols[0]
        # Group rows by value
        grouped = df.groupby(val_col)
        unique_vals = sorted(df[val_col].dropna().unique(), key=str)
        n_vals = len(unique_vals)
        # Auto-assign HSL colors (evenly spaced hues)
        auto_colors = {}
        for i, v in enumerate(unique_vals):
            h = i / max(n_vals, 1)
            r, g, b = colorsys.hls_to_rgb(h, 0.55, 0.75)
            auto_colors[str(v)] = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
        for val, sub_df in grouped:
            pairs = []
            for _, row in sub_df.iterrows():
                pre = str(int(row[pre_col]))
                post = str(int(row[post_col]))
                pairs.append(f'{pre}|{post}')
            if pairs:
                groups.append({
                    'pairs': pairs,
                    'color': auto_colors.get(str(val), 'rgb(200,200,200)'),
                    'label': f'CSV: {val}',
                })
        print(f"  Embedded {len(grouped)} synapse groups from {csv_path.name}")
    return groups


# ============================================================
# Color Modes
# ============================================================

def _detect_csv_key_col(df, csv_path):
    """Detect the key column in a color CSV: 'type' or 'bodyid'/'body_id'."""
    cols_lower = {c.lower(): c for c in df.columns}
    if 'type' in cols_lower:
        return cols_lower['type']
    if 'bodyid' in cols_lower:
        return cols_lower['bodyid']
    if 'body_id' in cols_lower:
        return cols_lower['body_id']
    print(f"  Warning: no 'type' or 'bodyid' column in {Path(csv_path).name}, skipping")
    return None


def load_score_modes(continuous_csvs=None, categorical_csvs=None,
                     score_csv=None, modality_csv=None):
    """Load color modes from CSV files.

    Args:
        continuous_csvs:  List of CSV paths. Each CSV must have a 'type' column;
                          every other column becomes a continuous color mode.
                          Divergent data (has negative values) auto-gets 'RdBu';
                          non-divergent data cycles through sequential colormaps.
        categorical_csvs: List of CSV paths. Each CSV must have a 'type' column
                          and one or more category columns. Each category column
                          becomes a color mode where types sharing a value share a color.
        score_csv:        (Legacy) Path to a single continuous CSV.
        modality_csv:     (Legacy) Path to a single continuous CSV.

    Returns:
        Tuple of (continuous_modes, categorical_modes) where each is a dict of:
        {name: {scores/categories: ..., cmap: str, label: str, is_categorical: bool}}
    """
    # Build merged list of continuous CSVs from all input sources
    all_continuous = []
    if continuous_csvs:
        all_continuous.extend(continuous_csvs)
    if score_csv:
        all_continuous.append(score_csv)
    if modality_csv:
        all_continuous.append(modality_csv)

    all_categorical = list(categorical_csvs or [])

    # --- Continuous modes ---
    score_modes = {}
    seq_idx = 0  # cycles through SEQUENTIAL_CMAPS
    for csv_path in all_continuous:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  Warning: continuous CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Detect key column: type (type-level) or bodyid (instance-level)
        key_col = _detect_csv_key_col(df, csv_path)
        if key_col is None:
            continue
        is_instance = key_col != 'type'
        value_cols = [c for c in df.columns if c != key_col]
        for col in value_cols:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) == 0:
                continue
            is_divergent = float(vals.min()) < 0 < float(vals.max())
            if is_divergent:
                cmap = 'RdBu'
            else:
                cmap = SEQUENTIAL_CMAPS[seq_idx % len(SEQUENTIAL_CMAPS)]
                seq_idx += 1
            mode_name = col.replace('_', ' ').title()
            score_modes[mode_name] = {
                'scores': dict(zip(df[key_col].astype(str), df[col])),
                'cmap': cmap,
                'label': mode_name,
                'is_instance_level': is_instance,
            }

    # --- Categorical modes ---
    cat_modes = {}
    for csv_path in all_categorical:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"  Warning: categorical CSV not found: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        key_col = _detect_csv_key_col(df, csv_path)
        if key_col is None:
            continue
        is_instance = key_col != 'type'
        cat_cols = [c for c in df.columns if c != key_col]
        for col in cat_cols:
            mode_name = col.replace('_', ' ').title()
            cat_modes[mode_name] = {
                'categories': dict(zip(df[key_col].astype(str), df[col].astype(str))),
                'label': mode_name,
                'is_categorical': True,
                'is_instance_level': is_instance,
            }

    return score_modes, cat_modes


def build_color_modes(all_types, neurons_full, type_lookup, score_modes,
                      cat_modes=None, default_cmap='RdBu'):
    """Build the list of color mode dicts for JS serialization.

    Returns list of dicts with keys: name, color_dict, is_scalar, cmap_obj, norm, label
    """
    color_modes = []

    # 1. Cell Type (always first)
    n = len(all_types)
    if n <= 20:
        cmap_cat = plt.colormaps['tab20']
        cat_colors = {t: (*cmap_cat(i / max(n - 1, 1))[:3], 1.0) for i, t in enumerate(all_types)}
    else:
        cat_colors = {t: (*plt.colormaps['hsv'](i / n)[:3], 1.0) for i, t in enumerate(all_types)}

    cat_color_dict = {}
    for neuron in neurons_full:
        ntype = type_lookup.get(neuron.id)
        cat_color_dict[neuron.id] = cat_colors.get(ntype, (0.5, 0.5, 0.5, 1.0))
    color_modes.append({
        'name': 'Cell Type', 'color_dict': cat_color_dict,
        'is_scalar': False, 'cmap_obj': None, 'norm': None, 'label': ''
    })

    # 2. Instance mode — unique color per instance (neuron)
    n_neurons = len(neurons_full)
    if n_neurons <= 20:
        inst_cmap = plt.colormaps['tab20']
    else:
        inst_cmap = plt.colormaps['hsv']
    inst_color_dict = {}
    for i, neuron in enumerate(neurons_full):
        inst_color_dict[neuron.id] = (*inst_cmap(i / max(n_neurons - 1, 1))[:3], 1.0)
    color_modes.append({
        'name': 'Instance', 'color_dict': inst_color_dict,
        'is_scalar': False, 'cmap_obj': None, 'norm': None, 'label': ''
    })

    # 3. User-defined categorical modes
    bid_str_set = {str(n.id) for n in neurons_full}
    if cat_modes:
        for mode_name, mode_cfg in cat_modes.items():
            categories = mode_cfg['categories']  # {key: category_value}
            is_instance = mode_cfg.get('is_instance_level', False)
            if is_instance:
                matched = {k: c for k, c in categories.items() if k in bid_str_set}
            else:
                type_set = set(all_types)
                matched = {t: c for t, c in categories.items() if t in type_set}
            if not matched:
                continue
            # Assign colors to unique category values
            unique_cats = sorted(set(matched.values()))
            nc = len(unique_cats)
            if nc <= 20:
                cmc = plt.colormaps['tab20']
            else:
                cmc = plt.colormaps['hsv']
            cat_color_map = {c: (*cmc(i / max(nc - 1, 1))[:3], 1.0)
                             for i, c in enumerate(unique_cats)}
            cd = {}
            for neuron in neurons_full:
                bid = str(neuron.id)
                if is_instance:
                    cat_val = matched.get(bid)
                else:
                    ntype = type_lookup.get(neuron.id)
                    cat_val = matched.get(ntype)
                if cat_val and cat_val in cat_color_map:
                    cd[neuron.id] = cat_color_map[cat_val]
                else:
                    cd[neuron.id] = (0.5, 0.5, 0.5, 0.3) if is_instance else (0.5, 0.5, 0.5, 1.0)
            mode_entry = {
                'name': mode_name, 'color_dict': cd,
                'is_scalar': False, 'cmap_obj': None, 'norm': None,
                'label': mode_cfg.get('label', mode_name)
            }
            if is_instance:
                mode_entry['is_instance_level'] = True
            color_modes.append(mode_entry)

    # 4. Score-based (continuous) modes
    for mode_name, mode_cfg in score_modes.items():
        scores = mode_cfg['scores']
        is_instance = mode_cfg.get('is_instance_level', False)
        if is_instance:
            matched = {}
            for k, s in scores.items():
                if k in bid_str_set:
                    try:
                        matched[k] = float(s)
                    except (ValueError, TypeError):
                        pass
        else:
            type_set = set(all_types)
            matched = {t: s for t, s in scores.items() if t in type_set}
        if not matched:
            continue
        cmap_name = mode_cfg.get('cmap', default_cmap)
        cmap_obj = plt.colormaps[cmap_name]
        vals = np.array(list(matched.values()))
        if vals.min() < 0 < vals.max():
            vmax = np.abs(vals).max()
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            norm = Normalize(vmin=vals.min(), vmax=vals.max())
        cd = {}
        for neuron in neurons_full:
            bid = str(neuron.id)
            if is_instance:
                score = matched.get(bid)
                if score is not None:
                    rgba = cmap_obj(norm(score))
                    cd[neuron.id] = (*rgba[:3], 1.0)
                else:
                    cd[neuron.id] = (0.5, 0.5, 0.5, 0.3)
            else:
                ntype = type_lookup.get(neuron.id)
                score = matched.get(ntype, 0) if ntype else 0
                rgba = cmap_obj(norm(score))
                cd[neuron.id] = (*rgba[:3], 1.0)
        mode_entry = {
            'name': mode_name, 'color_dict': cd, 'is_scalar': True,
            'cmap_obj': cmap_obj, 'norm': norm,
            'label': mode_cfg.get('label', mode_name)
        }
        if is_instance:
            mode_entry['is_instance_level'] = True
        color_modes.append(mode_entry)

    print(f'{len(color_modes)} color mode(s): {[m["name"] for m in color_modes]}')
    return color_modes


def serialize_color_modes(color_modes, all_types, type_lookup, nt_lookup, bid_type_map):
    """Convert Python color_modes to JS-serializable format, including NT mode."""
    js_modes = []

    for mode in color_modes:
        bid_colors = {}
        for bid, rgba in mode['color_dict'].items():
            bid_colors[str(bid)] = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
        tc = {}
        for t in all_types:
            bids_of_type = [bid for bid, typ in type_lookup.items() if typ == t]
            if bids_of_type and bids_of_type[0] in mode['color_dict']:
                rgba = mode['color_dict'][bids_of_type[0]]
                tc[t] = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
        entry = {'name': mode['name'], 'colors': bid_colors, 'type_colors': tc,
                 'is_scalar': mode['is_scalar']}
        if mode.get('is_instance_level'):
            entry['is_instance_level'] = True
        if mode['is_scalar']:
            entry['cmin'] = float(mode['norm'].vmin)
            entry['cmax'] = float(mode['norm'].vmax)
            entry['label'] = mode['label']
            cm = mode['cmap_obj']
            entry['colorscale'] = [
                [round(v, 4), f'rgb({int(cm(v)[0]*255)},{int(cm(v)[1]*255)},{int(cm(v)[2]*255)})']
                for v in np.linspace(0, 1, 64)
            ]
        js_modes.append(entry)

    # Auto-inject "Predicted NT" color mode
    nt_mode = _build_nt_color_mode(nt_lookup, bid_type_map)
    if nt_mode:
        js_modes.insert(2, nt_mode)
        print(f"  Added 'Predicted NT' color mode ({len(nt_lookup)} neurons)")

    return js_modes


def _build_nt_color_mode(bid_nt_map, bid_type_map):
    """Build a 'Predicted NT' color mode from bodyId->NT mapping."""
    if not bid_nt_map:
        return None

    colors = {}
    for bid, nt in bid_nt_map.items():
        colors[str(bid)] = NT_COLORS.get(nt.lower(), NT_DEFAULT_COLOR)

    type_nts = {}
    for bid, nt in bid_nt_map.items():
        typ = bid_type_map.get(str(bid))
        if typ:
            type_nts.setdefault(typ, []).append(nt.lower())

    type_colors = {}
    for typ, nts in type_nts.items():
        majority_nt = Counter(nts).most_common(1)[0][0]
        type_colors[typ] = NT_COLORS.get(majority_nt, NT_DEFAULT_COLOR)

    observed_nts = set(nt.lower() for nt in bid_nt_map.values())
    nt_legend = {}
    for nt_name, color in NT_COLORS.items():
        if nt_name in observed_nts:
            nt_legend[nt_name] = color
    nt_legend['unclear'] = NT_DEFAULT_COLOR

    bid_nts = {str(bid): nt.lower() for bid, nt in bid_nt_map.items()}

    return {
        'name': 'Predicted NT',
        'colors': colors,
        'type_colors': type_colors,
        'is_scalar': False,
        'nt_legend': nt_legend,
        'bid_nts': bid_nts,
    }


# ============================================================
# ROI & Morphology Processing
# ============================================================

def discover_rois(all_neurons_df, roi_counts_df, type_lookup):
    """Discover ROIs and compute synapse counts.

    Returns:
        neuron_rois, roi_neuron_bids, roi_synapse_totals,
        type_roi_synapses, neuron_roi_synapses, instance_lookup
    """
    roi_neuron_bids = {}
    for _, row in all_neurons_df.iterrows():
        bid = row['bodyId']
        all_rois = set()
        if isinstance(row.get('inputRois'), list):
            all_rois.update(row['inputRois'])
        if isinstance(row.get('outputRois'), list):
            all_rois.update(row['outputRois'])
        for roi in all_rois - META_ROIS:
            roi_neuron_bids.setdefault(roi, set()).add(bid)

    roi_neuron_counts = {r: len(bids) for r, bids in roi_neuron_bids.items()}
    neuron_rois = sorted(roi_neuron_counts, key=roi_neuron_counts.get, reverse=True)

    print(f'Neurons innervate {len(neuron_rois)} ROIs')

    # Synapse counts
    rc = roi_counts_df.copy()
    rc['type'] = rc['bodyId'].map(type_lookup)
    rc['total'] = rc['pre'] + rc['post']
    rc = rc[~rc['roi'].isin(META_ROIS)]

    roi_synapse_totals = rc.groupby('roi')['total'].sum().to_dict()

    type_roi_synapses = {}
    for (t, r), g in rc.groupby(['type', 'roi']):
        type_roi_synapses.setdefault(str(t), {})[str(r)] = int(g['total'].sum())

    neuron_roi_synapses = {}
    for (bid, r), g in rc.groupby(['bodyId', 'roi']):
        neuron_roi_synapses.setdefault(str(int(bid)), {})[str(r)] = int(g['total'].sum())

    if 'instance' in all_neurons_df.columns:
        instance_lookup = dict(zip(all_neurons_df['bodyId'].astype(str),
                                   all_neurons_df['instance'].fillna('')))
    else:
        instance_lookup = {str(bid): '' for bid in all_neurons_df['bodyId']}

    return (neuron_rois, roi_neuron_bids, roi_synapse_totals,
            type_roi_synapses, neuron_roi_synapses, instance_lookup)


def fetch_roi_meshes(neuron_rois, default_roi=None):
    """Fetch ROI meshes from neuPrint.

    Returns:
        roi_volumes: {roi_name: navis.Volume}
        primary_roi: str
    """
    roi_volumes = {}
    for name in neuron_rois:
        try:
            vol = neu.fetch_roi(name)
            vol.color = (*ROI_COLOR, 0)
            roi_volumes[name] = vol
        except Exception:
            pass

    print(f'Fetched meshes for {len(roi_volumes)}/{len(neuron_rois)} ROIs')

    if default_roi and default_roi in roi_volumes:
        primary_roi = default_roi
    else:
        primary_roi = neuron_rois[0]
    print(f'Primary ROI: {primary_roi}')
    return roi_volumes, primary_roi


def clip_skeletons(neurons_full, roi_volumes, primary_roi, roi_neuron_bids):
    """Clip neuron skeletons to their respective ROIs.

    Returns:
        neurons_clipped: navis.NeuronList (primary ROI)
        roi_clipped: {roi_name: navis.NeuronList}
    """
    if primary_roi not in roi_volumes:
        return neurons_full, {}

    primary_vol = roi_volumes[primary_roi]
    neurons_clipped = navis.in_volume(neurons_full, primary_vol, mode='IN', inplace=False)
    neurons_clipped = navis.NeuronList([n for n in neurons_clipped if n.n_vertices > 0])
    print(f'Clipped to {primary_roi}: {len(neurons_full)} -> {len(neurons_clipped)} neurons')

    # Pre-build neuron lookup for fast subsetting
    neuron_by_id = {n.id: n for n in neurons_full}

    roi_clipped = {}
    for roi_name, vol in roi_volumes.items():
        if roi_name == primary_roi:
            continue
        bids = roi_neuron_bids.get(roi_name, set())
        if not bids:
            continue
        subset_neurons = [neuron_by_id[b] for b in bids if b in neuron_by_id]
        if not subset_neurons:
            continue
        subset = navis.NeuronList(subset_neurons)
        clipped = navis.in_volume(subset, vol, mode='IN', inplace=False)
        clipped = navis.NeuronList([n for n in clipped if n.n_vertices > 0])
        if len(clipped) > 0:
            roi_clipped[roi_name] = clipped

    print(f'Clipped neurons for {len(roi_clipped)} additional ROIs')
    return neurons_clipped, roi_clipped


# ============================================================
# Direct Skeleton -> Three.js Conversion
# ============================================================

def skeleton_to_segments(neuron):
    """Convert a navis TreeNeuron directly to line segment pairs.

    Walks the node table's parent-child edges and produces a flat
    array of [x0,y0,z0, x1,y1,z1, ...] segment pairs.

    Returns:
        list of floats, int (segments list, segment count)
    """
    nodes = neuron.nodes
    edges = nodes[nodes['parent_id'] >= 0]
    if len(edges) == 0:
        return [], 0

    # Vectorized: build coordinate arrays via merge instead of row iteration
    coords = nodes.set_index('node_id')[['x', 'y', 'z']]
    child_ids = edges['node_id'].values
    parent_ids = edges['parent_id'].values

    # Use numpy indexing via reindex for fast coordinate lookup
    child_xyz = coords.reindex(child_ids).values   # (N, 3)
    parent_xyz = coords.reindex(parent_ids).values  # (N, 3)

    # Filter out any NaN rows (missing parent/child)
    valid = ~(np.isnan(child_xyz[:, 0]) | np.isnan(parent_xyz[:, 0]))
    if not valid.all():
        child_xyz = child_xyz[valid]
        parent_xyz = parent_xyz[valid]

    n_segs = len(child_xyz)
    if n_segs == 0:
        return [], 0

    # Interleave parent,child pairs into flat array: [px,py,pz, cx,cy,cz, ...]
    pairs = np.empty((n_segs, 6), dtype=np.float64)
    pairs[:, 0:3] = parent_xyz
    pairs[:, 3:6] = child_xyz
    # Return raw numpy array — avoid expensive .tolist() conversion here.
    # optimize_bundle will consume these directly as numpy arrays.
    return pairs.ravel(), n_segs


def get_soma_position(neuron):
    """Extract soma position from a TreeNeuron.

    Returns dict {x, y, z} or None.
    """
    if hasattr(neuron, 'soma') and neuron.soma is not None:
        soma_id = neuron.soma
        if hasattr(soma_id, '__iter__') and not isinstance(soma_id, str):
            soma_id = list(soma_id)
            if len(soma_id) > 0:
                soma_id = soma_id[0]
            else:
                return None
        nodes = neuron.nodes
        soma_row = nodes[nodes['node_id'] == soma_id]
        if len(soma_row) > 0:
            return {
                'x': float(soma_row.iloc[0]['x']),
                'y': float(soma_row.iloc[0]['y']),
                'z': float(soma_row.iloc[0]['z']),
            }
    return None


def volume_to_mesh_data(vol, max_faces=50000):
    """Convert a navis Volume / trimesh to vertices/faces flat lists.

    Returns:
        (vertices_flat, faces_flat)
    """
    tm = tm_lib.Trimesh(vertices=vol.vertices, faces=vol.faces)
    if len(tm.faces) > max_faces:
        try:
            tm = tm.simplify_quadric_decimation(max_faces)
        except Exception:
            pass
    verts_flat = tm.vertices.ravel().astype(np.float64)
    faces_flat = tm.faces.ravel().astype(np.int64)
    return verts_flat, faces_flat


def _decimate_anchor(verts, faces, max_faces):
    """Decimate a mesh for anchor use (very low res)."""
    if len(faces) > max_faces:
        step = max(1, len(faces) // max_faces)
        faces = faces[::step]
        used = np.unique(faces.ravel())
        remap = np.full(verts.shape[0], -1, dtype=int)
        remap[used] = np.arange(len(used))
        verts = verts[used]
        faces = remap[faces]
    return verts, faces


# Pre-computed brain bounding box for CNS dataset (fallback if no ROI data available).
_CNS_BRAIN_BBOX = {
    'xmin': 5798, 'xmax': 90473,
    'ymin': 4856, 'ymax': 51895,
    'zmin': 10168, 'zmax': 42568,
}


def build_anchor_bbox(roi_volumes, neurons_full=None, dataset='cns'):
    """Compute brain bounding box for normalization.

    Computes bounds from actual ROI mesh vertices and neuron skeleton coordinates.
    Falls back to pre-computed CNS bounds only if dataset is 'cns' and no ROI data.

    Returns:
        norm_params: dict with cx, cy, cz, dmax, aspect
    """
    # Start from actual data bounds (ROI meshes + neuron skeletons)
    xmin = ymin = zmin = float('inf')
    xmax = ymax = zmax = float('-inf')

    # Include all ROI vertices
    for name, vol in (roi_volumes or {}).items():
        v = vol.vertices
        xmin = min(xmin, v[:, 0].min())
        xmax = max(xmax, v[:, 0].max())
        ymin = min(ymin, v[:, 1].min())
        ymax = max(ymax, v[:, 1].max())
        zmin = min(zmin, v[:, 2].min())
        zmax = max(zmax, v[:, 2].max())

    # Include neuron skeleton bounds if available
    if neurons_full:
        for n in neurons_full:
            if hasattr(n, 'nodes') and len(n.nodes) > 0:
                coords = n.nodes[['x', 'y', 'z']].values
                xmin = min(xmin, coords[:, 0].min())
                xmax = max(xmax, coords[:, 0].max())
                ymin = min(ymin, coords[:, 1].min())
                ymax = max(ymax, coords[:, 1].max())
                zmin = min(zmin, coords[:, 2].min())
                zmax = max(zmax, coords[:, 2].max())

    # If no data found, fall back to CNS hardcoded bbox or zero-centered unit
    if xmin == float('inf'):
        if dataset and dataset.lower() == 'cns':
            xmin, xmax = _CNS_BRAIN_BBOX['xmin'], _CNS_BRAIN_BBOX['xmax']
            ymin, ymax = _CNS_BRAIN_BBOX['ymin'], _CNS_BRAIN_BBOX['ymax']
            zmin, zmax = _CNS_BRAIN_BBOX['zmin'], _CNS_BRAIN_BBOX['zmax']
            print("  Using pre-computed CNS brain bounding box")
        else:
            xmin = ymin = zmin = -1
            xmax = ymax = zmax = 1
            print("  Warning: no ROI or neuron data for bounding box, using unit cube")

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    dmax = max(dx, dy, dz, 1e-9)

    norm_params = {
        'cx': cx, 'cy': cy, 'cz': cz,
        'dx': dx, 'dy': dy, 'dz': dz,
        'dmax': dmax,
        'aspect': {'x': dx/dmax, 'y': dy/dmax, 'z': dz/dmax}
    }
    print(f"Brain bbox: x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}] z[{zmin:.0f},{zmax:.0f}]")
    print(f"Normalization: center=({cx:.0f},{cy:.0f},{cz:.0f}) scale={dmax:.0f}")
    return norm_params


def build_data_bundle(
    neurons_clipped, neurons_full, roi_clipped,
    roi_volumes, type_lookup, nt_lookup,
    all_types, js_color_modes, roi_neuron_bids,
    neuron_rois, primary_roi,
    roi_synapse_totals, type_roi_synapses, neuron_roi_synapses,
    instance_lookup, type_upstream, type_downstream,
    neuron_upstream, neuron_downstream,
    norm_params, regex_term='', neuron_meshes=None,
    _data_source=None, voxel_size_nm=8,
):
    """Build the Three.js data bundle directly from navis objects."""

    bid_type_map = {str(bid): typ for bid, typ in type_lookup.items()}
    type_neurons = {}
    for bid, typ in type_lookup.items():
        type_neurons.setdefault(typ, []).append(str(bid))

    # --- Per-type-per-ROI segments from clipped neurons ---
    print("  Building per-type-per-ROI segments...")
    type_roi_segments = {}
    type_roi_bid_runs = {}

    def _build_type_roi(neuron_list_iter, type_lookup_fn):
        """Build type-ROI segments from neuron list, returning numpy arrays."""
        groups = {}
        for neuron in neuron_list_iter:
            typ = type_lookup_fn(neuron)
            groups.setdefault(typ, []).append(neuron)
        segs_out = {}
        runs_out = {}
        for typ, nlist in groups.items():
            arrays = []
            bid_runs = []
            for neuron in nlist:
                arr, n = skeleton_to_segments(neuron)
                if n > 0:
                    arrays.append(arr)
                    bid_runs.append([str(neuron.id), n])
            if arrays:
                segs_out[typ] = np.concatenate(arrays)
                runs_out[typ] = bid_runs
        return segs_out, runs_out

    # Primary ROI
    segs, runs = _build_type_roi(
        neurons_clipped, lambda n: type_lookup.get(n.id, 'unknown'))
    for typ in segs:
        key = f"{typ}|{primary_roi}"
        type_roi_segments[key] = segs[typ]
        type_roi_bid_runs[key] = runs[typ]

    # Other ROIs
    for roi_name, roi_neurons in roi_clipped.items():
        segs, runs = _build_type_roi(
            roi_neurons, lambda n: type_lookup.get(n.id, 'unknown'))
        for typ in segs:
            key = f"{typ}|{roi_name}"
            type_roi_segments[key] = segs[typ]
            type_roi_bid_runs[key] = runs[typ]

    print(f"    {len(type_roi_segments)} type-ROI segment groups")

    # --- Per-neuron full skeleton segments ---
    print("  Building per-neuron full skeleton segments...")
    neuron_full_segments = {}
    for neuron in neurons_full:
        arr, n_segs = skeleton_to_segments(neuron)
        if n_segs > 0:
            neuron_full_segments[str(neuron.id)] = arr
    print(f"    {len(neuron_full_segments)} neurons with full skeletons")

    # --- Soma positions ---
    print("  Extracting soma positions...")
    neuron_somas = {}
    for neuron in neurons_full:
        pos = get_soma_position(neuron)
        if pos:
            neuron_somas[str(neuron.id)] = pos
    print(f"    {len(neuron_somas)} soma positions found")

    # --- ROI meshes ---
    print("  Extracting ROI meshes...")
    roi_meshes = {}
    for roi_name, vol in roi_volumes.items():
        verts, faces = volume_to_mesh_data(vol)
        if len(verts) > 0:
            roi_meshes[roi_name] = {'vertices': verts, 'faces': faces}
    print(f"    {len(roi_meshes)} ROI meshes extracted")

    # --- ROI bounding boxes ---
    roi_bounds = {}
    for roi_name, vol in roi_volumes.items():
        v = vol.vertices
        roi_bounds[roi_name] = {
            'xmin': float(v[:, 0].min()), 'xmax': float(v[:, 0].max()),
            'ymin': float(v[:, 1].min()), 'ymax': float(v[:, 1].max()),
            'zmin': float(v[:, 2].min()), 'zmax': float(v[:, 2].max()),
        }

    # --- Sidebar ROIs (filtered to those with connectivity) ---
    conn_rois = set()
    for t_data in type_upstream.values():
        conn_rois.update(t_data.keys())
    for t_data in type_downstream.values():
        conn_rois.update(t_data.keys())
    for n_data in neuron_upstream.values():
        conn_rois.update(n_data.keys())
    for n_data in neuron_downstream.values():
        conn_rois.update(n_data.keys())

    sidebar_rois = [primary_roi] + [r for r in neuron_rois
                                     if r != primary_roi and r in conn_rois]

    # --- Type-ROI map ---
    type_roi_map = {}
    for roi, bids in roi_neuron_bids.items():
        for bid in bids:
            t = type_lookup.get(bid)
            if t:
                type_roi_map.setdefault(t, set()).add(roi)
    type_roi_map = {t: sorted(rois) for t, rois in type_roi_map.items()}

    bundle = {
        'typeRoiSegments': type_roi_segments,
        'typeRoiBidRuns': type_roi_bid_runs,
        'neuronFullSegments': neuron_full_segments,
        'neuronSomas': neuron_somas,
        'roiMeshes': roi_meshes,
        'roiBounds': roi_bounds,
        'normParams': norm_params,
        'camera': _compute_default_camera(norm_params),
        'allTypes': all_types,
        'typeNeurons': type_neurons,
        'bidTypeMap': bid_type_map,
        'colorModes': js_color_modes,
        'sidebarRois': sidebar_rois,
        'primaryRoi': primary_roi,
        'typeRoiMap': type_roi_map,
        'roiSynapseTotals': {r: roi_synapse_totals.get(r, 0) for r in sidebar_rois},
        'typeRoiSynapses': type_roi_synapses,
        'neuronRoiSynapses': neuron_roi_synapses,
        'instanceLookup': instance_lookup,
        'typeUpstream': type_upstream,
        'typeDownstream': type_downstream,
        'neuronUpstream': neuron_upstream,
        'neuronDownstream': neuron_downstream,
        'initialLineWidth': LINEWIDTH,
        'regexTerm': regex_term,
        'dataSource': _data_source,
        'voxelSizeNm': voxel_size_nm,
        'axisLabels': {
            'xPos': 'Left', 'xNeg': 'Right',
            'yPos': 'Ventral', 'yNeg': 'Dorsal',
            'zNeg': 'Anterior', 'zPos': 'Posterior',
        },
    }

    # Neuron meshes (optional)
    if neuron_meshes:
        center = np.array([norm_params['cx'], norm_params['cy'], norm_params['cz']])
        dmax = norm_params['dmax']
        mesh_data = {}
        for bid, md in neuron_meshes.items():
            # Normalize vertices to same coordinate space as skeletons
            verts = (md['vertices'].astype(np.float64) - center) / dmax
            faces = md['faces']
            # Flatten and store as base64 for compact encoding
            entry = {
                'v': base64.b64encode(verts.astype(np.float32).tobytes()).decode('ascii'),
                'f': base64.b64encode(faces.astype(np.int32).tobytes()).decode('ascii'),
                'nv': len(verts),
                'nf': len(faces),
            }
            # Per-face ROI assignments (if available)
            if 'faceRoiIndices' in md:
                entry['fri'] = base64.b64encode(
                    np.array(md['faceRoiIndices'], dtype=np.uint8).tobytes()
                ).decode('ascii')
                entry['rl'] = md['roiLookup']
            mesh_data[bid] = entry
        bundle['neuronMeshes'] = mesh_data
        print(f"  {len(mesh_data)} neuron meshes encoded")

    return bundle


# ============================================================
# Bundle Optimization (from extract_and_build.py)
# ============================================================

def optimize_bundle(bundle):
    """Optimize the data bundle using binary Int16 encoding.

    Pre-normalizes coordinates to unit cube, quantizes to Int16 (scale 30000),
    base64 encodes large arrays. ~60% file size reduction.
    """
    norm = bundle['normParams']
    cx, cy, cz = norm['cx'], norm['cy'], norm['cz']
    dmax = norm['dmax']
    QUANT_SCALE = 30000

    def normalize_and_encode(coords):
        if isinstance(coords, np.ndarray):
            arr = coords.astype(np.float64) if coords.dtype != np.float64 else coords.copy()
        else:
            arr = np.array(coords, dtype=np.float64)
        arr[0::3] = (arr[0::3] - cx) / dmax
        arr[1::3] = (arr[1::3] - cy) / dmax
        arr[2::3] = (arr[2::3] - cz) / dmax
        int16_arr = np.clip(arr * QUANT_SCALE, -32768, 32767).astype(np.int16)
        return base64.b64encode(int16_arr.tobytes()).decode('ascii')

    def normalize_val(v, axis):
        centers = {'x': cx, 'y': cy, 'z': cz}
        return (v - centers[axis]) / dmax

    print("  Optimizing typeRoiSegments...")
    opt_segments = {}
    for key, coords in bundle['typeRoiSegments'].items():
        if len(coords) > 0:
            opt_segments[key] = normalize_and_encode(coords)
    bundle['typeRoiSegments'] = opt_segments

    print("  Optimizing neuronFullSegments...")
    opt_full = {}
    for key, coords in bundle['neuronFullSegments'].items():
        if len(coords) > 0:
            opt_full[key] = normalize_and_encode(coords)
    bundle['neuronFullSegments'] = opt_full

    print("  Optimizing roiMeshes...")
    opt_meshes = {}
    for roi_name, mesh_data in bundle['roiMeshes'].items():
        verts = mesh_data['vertices']
        faces = mesh_data['faces']
        v_b64 = normalize_and_encode(verts)
        face_arr = faces if isinstance(faces, np.ndarray) else np.array(faces, dtype=np.int64)
        max_idx = int(face_arr.max()) if len(face_arr) > 0 else 0
        if max_idx < 65536:
            f_arr = face_arr.astype(np.uint16)
            f_dtype = 'u2'
        else:
            f_arr = face_arr.astype(np.uint32)
            f_dtype = 'u4'
        f_b64 = base64.b64encode(f_arr.tobytes()).decode('ascii')
        opt_meshes[roi_name] = {'v': v_b64, 'f': f_b64, 'fd': f_dtype}
    bundle['roiMeshes'] = opt_meshes

    print("  Pre-normalizing somas and roiBounds...")
    for bid, pos in bundle['neuronSomas'].items():
        bundle['neuronSomas'][bid] = {
            'x': normalize_val(pos['x'], 'x'),
            'y': normalize_val(pos['y'], 'y'),
            'z': normalize_val(pos['z'], 'z'),
        }

    for roi, b in bundle['roiBounds'].items():
        bundle['roiBounds'][roi] = {
            'xmin': normalize_val(b['xmin'], 'x'),
            'xmax': normalize_val(b['xmax'], 'x'),
            'ymin': normalize_val(b['ymin'], 'y'),
            'ymax': normalize_val(b['ymax'], 'y'),
            'zmin': normalize_val(b['zmin'], 'z'),
            'zmax': normalize_val(b['zmax'], 'z'),
        }

    bundle['encoding'] = 'binary'
    bundle['quantScale'] = QUANT_SCALE
    return bundle


# ============================================================
# HTML Assembly (from extract_and_build.py)
# ============================================================

_JS_CACHE = {}

def _prepare_js_libs(threejs_path=None, controls_path=None, trackball_path=None):
    """Prepare JS library sources for HTML embedding. Returns (threejs_src, imports_line, controls_adapted, tb_imports_line, trackball_adapted)."""
    if threejs_path is None:
        threejs_path = str(SCRIPT_DIR / 'three.min.js')
    if controls_path is None:
        controls_path = str(SCRIPT_DIR / 'OrbitControls.js')
    if trackball_path is None:
        trackball_path = str(SCRIPT_DIR / 'TrackballControls.js')

    def _read(path):
        if path not in _JS_CACHE:
            with open(path, 'r', encoding='utf-8') as f:
                _JS_CACHE[path] = f.read()
        return _JS_CACHE[path]

    threejs_src = _read(threejs_path)
    controls_src = _read(controls_path)
    trackball_src = _read(trackball_path)

    import_block_end = controls_src.find("from 'three';")
    controls_adapted = controls_src[import_block_end + len("from 'three';"):] if import_block_end > 0 else controls_src
    imported_names = ['EventDispatcher', 'MOUSE', 'Quaternion', 'Spherical', 'TOUCH',
                      'Vector2', 'Vector3', 'Plane', 'Ray', 'MathUtils']
    imports_line = '\n'.join(f'const {name} = THREE.{name};' for name in imported_names)
    controls_adapted = controls_adapted.replace('export { OrbitControls };', 'THREE.OrbitControls = OrbitControls;')

    tb_import_end = trackball_src.find("from 'three';")
    trackball_adapted = trackball_src[tb_import_end + len("from 'three';"):] if tb_import_end > 0 else trackball_src
    tb_imported_names = ['EventDispatcher', 'MathUtils', 'MOUSE', 'Quaternion', 'Vector2', 'Vector3']
    tb_imports_line = '\n'.join(f'const {name} = THREE.{name};' for name in tb_imported_names)
    trackball_adapted = trackball_adapted.replace('export { TrackballControls };', 'THREE.TrackballControls = TrackballControls;')

    return threejs_src, imports_line, controls_adapted, tb_imports_line, trackball_adapted


def build_threejs_html(bundle, output_path, threejs_path=None,
                       controls_path=None, trackball_path=None):
    """Build the standalone Three.js HTML file."""
    if threejs_path is None:
        threejs_path = str(SCRIPT_DIR / 'three.min.js')
    if controls_path is None:
        controls_path = str(SCRIPT_DIR / 'OrbitControls.js')
    if trackball_path is None:
        trackball_path = str(SCRIPT_DIR / 'TrackballControls.js')

    def _read_cached(path):
        if path not in _JS_CACHE:
            with open(path, 'r', encoding='utf-8') as f:
                _JS_CACHE[path] = f.read()
        return _JS_CACHE[path]

    threejs_src = _read_cached(threejs_path)
    controls_src = _read_cached(controls_path)
    trackball_src = _read_cached(trackball_path)

    data_json = json.dumps(bundle, separators=(',', ':'))
    app_js = APPLICATION_JS
    logo_tag = f'<img id="_loadLogo" src="data:image/png;base64,{LOADING_LOGO_B64}">' if LOADING_LOGO_B64 else ''

    # Adapt OrbitControls for inline use
    import_block_end = controls_src.find("from 'three';")
    if import_block_end > 0:
        controls_adapted = controls_src[import_block_end + len("from 'three';"):]
    else:
        controls_adapted = controls_src

    imported_names = ['EventDispatcher', 'MOUSE', 'Quaternion', 'Spherical', 'TOUCH',
                      'Vector2', 'Vector3', 'Plane', 'Ray', 'MathUtils']
    imports_line = '\n'.join(f'const {name} = THREE.{name};' for name in imported_names)
    controls_adapted = controls_adapted.replace(
        'export { OrbitControls };', 'THREE.OrbitControls = OrbitControls;')

    # Adapt TrackballControls
    tb_import_end = trackball_src.find("from 'three';")
    if tb_import_end > 0:
        trackball_adapted = trackball_src[tb_import_end + len("from 'three';"):]
    else:
        trackball_adapted = trackball_src
    tb_imported_names = ['EventDispatcher', 'MathUtils', 'MOUSE', 'Quaternion', 'Vector2', 'Vector3']
    tb_imports_line = '\n'.join(f'const {name} = THREE.{name};' for name in tb_imported_names)
    trackball_adapted = trackball_adapted.replace(
        'export { TrackballControls };', 'THREE.TrackballControls = TrackballControls;')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3D Neuron Visualizer — {bundle.get('regexTerm','')}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #000; overflow: hidden; font-family: sans-serif; color: #fff; }}
[data-tip] {{ position: relative; }}
[data-tip]:hover::after {{ content: attr(data-tip); position: absolute; bottom: calc(100% + 4px); left: 50%; transform: translateX(-50%); background: rgba(30,30,30,0.95); color: #ccc; font-size: 11px; padding: 3px 7px; border-radius: 3px; white-space: nowrap; pointer-events: none; z-index: 10002; border: 1px solid #555; animation: tipFade 0.1s ease-out; }}
@keyframes tipFade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
#_loadScreen {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #000; z-index: 99999; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
#_loadWrap {{ position: relative; width: 200px; height: 200px; }}
#_loadSpinner {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 4px solid #333; border-top-color: rgb(212,160,23); border-radius: 50%; animation: _spin 1s linear infinite; box-sizing: border-box; }}
@keyframes _spin {{ to {{ transform: rotate(360deg); }} }}
#_loadLogo {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 130px; height: auto; object-fit: contain; }}
#_loadLabel {{ margin-top: 16px; color: #888; font-size: 14px; }}
</style>
</head>
<body>
<div id="_loadScreen"><div id="_loadWrap"><div id="_loadSpinner"></div>{logo_tag}</div><div id="_loadLabel">Getting things sorted...</div></div>
<script>
// === THREE.JS LIBRARY ===
{threejs_src}
</script>
<script>
// === ORBIT CONTROLS (adapted for inline use) ===
(function() {{
{imports_line}
{controls_adapted}
}})();
</script>
<script>
// === TRACKBALL CONTROLS (adapted for inline use) ===
(function() {{
{tb_imports_line}
{trackball_adapted}
}})();
</script>
<script>
// === DATA BUNDLE ===
const DATA = {data_json};

// === APPLICATION ===
{app_js}
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


# ============================================================
# Main Pipeline
# ============================================================

def generate_visualization(pattern, continuous_csvs=None, categorical_csvs=None,
                           synapse_csvs=None,
                           output_dir=None, auto_open=False,
                           skip_synapses=False, synapse_limit=None,
                           use_meshes=False, mesh_faces='auto', max_file_mb=500,
                           server='neuprint-cns.janelia.org', dataset='cns',
                           token=None,
                           # Internal / legacy args
                           score_modes=None, _cat_modes=None,
                           score_csv=None, modality_csv=None):
    """Generate a single 3D neuron visualization HTML.

    Args:
        pattern:          Regex pattern for neuron types (e.g. '^FB.*')
        continuous_csvs:  List of CSV paths for continuous color modes.
                          Each CSV: 'type' column + one or more numeric columns.
                          Divergent data (negative values) → RdBu colormap.
                          Non-divergent → auto-assigned sequential colormap.
        categorical_csvs: List of CSV paths for categorical color modes.
                          Each CSV: 'type' column + one or more category columns.
        output_dir:       Output directory (default: Examples/)
        auto_open:        Open in browser when done
        skip_synapses:    Skip fetching individual synapse positions
        synapse_limit:    Max synapses to keep (None = no limit)

    Returns:
        Path to generated HTML file
    """
    t_start = time.time()
    client = get_client(server=server, dataset=dataset, token=token)

    # 1. Fetch neurons & skeletons
    all_neurons_df, roi_counts_df, neurons_full, type_lookup, nt_lookup = \
        fetch_neuron_data(pattern)

    # 2. Build color modes
    if score_modes is None:
        score_modes, cat_modes = load_score_modes(
            continuous_csvs=continuous_csvs,
            categorical_csvs=categorical_csvs,
            score_csv=score_csv, modality_csv=modality_csv,
        )
    else:
        cat_modes = _cat_modes or {}
    all_types = sorted(all_neurons_df['type'].unique())
    color_modes = build_color_modes(all_types, neurons_full, type_lookup,
                                    score_modes, cat_modes=cat_modes)

    # 3. ROI discovery & synapse counts
    (neuron_rois, roi_neuron_bids, roi_synapse_totals,
     type_roi_synapses, neuron_roi_synapses, instance_lookup) = \
        discover_rois(all_neurons_df, roi_counts_df, type_lookup)

    # 4. Connectivity
    body_ids = all_neurons_df['bodyId'].tolist()
    type_upstream, type_downstream, neuron_upstream, neuron_downstream = \
        fetch_connectivity(body_ids, type_lookup, instance_lookup)

    # 5. ROI meshes & clipping
    roi_volumes, primary_roi = fetch_roi_meshes(neuron_rois)
    neurons_clipped, roi_clipped = clip_skeletons(
        neurons_full, roi_volumes, primary_roi, roi_neuron_bids)

    # 6. Compute brain bounding box
    print("Computing brain bounding box...")
    norm_params = build_anchor_bbox(roi_volumes, neurons_full=neurons_full, dataset=dataset)

    # 7. Serialize color modes for JS
    bid_type_map = {str(bid): typ for bid, typ in type_lookup.items()}
    js_color_modes = serialize_color_modes(
        color_modes, all_types, type_lookup, nt_lookup, bid_type_map)

    # 7b. Fetch neuron meshes (optional)
    neuron_meshes = None
    if use_meshes:
        # Handle 'auto' mesh_faces: maximize quality under 1 GB total file size
        actual_mesh_faces = mesh_faces
        if isinstance(mesh_faces, str) and mesh_faces.lower() == 'auto':
            MAX_FILE_BYTES = int(max_file_mb * 1_000_000)
            # Estimate non-mesh HTML size: ~1.5 KB per skeleton segment pair,
            # plus connectivity, ROI meshes, synapses, JS template (~500 KB)
            n_neurons = len(body_ids)
            est_skeleton_bytes = sum(
                n.n_vertices * 6 * 2 for n in neurons_full  # Int16 × 6 coords × base64
            ) * 1.4  # base64 + JSON overhead
            est_overhead = 2_000_000  # JS template + controls + metadata
            est_non_mesh = est_skeleton_bytes + est_overhead
            mesh_budget = MAX_FILE_BYTES - est_non_mesh
            if mesh_budget < 0:
                mesh_budget = 100_000_000  # Minimum 100 MB for meshes
            # Each face ≈ 32 bytes in HTML (12 bytes verts + 12 bytes indices, base64 + overhead)
            bytes_per_face = 32
            faces_per_neuron = int(mesh_budget / (n_neurons * bytes_per_face))
            # Clamp: at least 5000, no upper cap (let the budget decide)
            actual_mesh_faces = max(5000, faces_per_neuron)
            print(f'  AUTO mesh quality: {actual_mesh_faces:,} faces/neuron '
                  f'({n_neurons} neurons, {max_file_mb:.0f} MB file limit)')

        try:
            neuron_meshes = fetch_neuron_meshes(
                body_ids, max_faces=actual_mesh_faces)
            # Assign each mesh face to an ROI using nearest skeleton point
            _assign_mesh_face_rois(neuron_meshes, neurons_clipped, roi_clipped,
                                   primary_roi, norm_params)
        except Exception as e:
            print(f'  Warning: mesh fetching failed ({e}). Continuing without meshes.')
            neuron_meshes = None

    # 8. Fetch dataset metadata for voxel size
    voxel_size_nm = 8  # fallback
    try:
        import neuprint as _neu
        meta = _neu.fetch_meta()
        vs = meta.get('voxelSize')
        if vs:
            if isinstance(vs, (list, tuple)) and len(vs) > 0:
                voxel_size_nm = vs[0]
            elif isinstance(vs, (int, float)):
                voxel_size_nm = vs
            print(f"  Voxel size: {voxel_size_nm} nm (from dataset metadata)")
    except Exception:
        print(f"  Using default voxel size: {voxel_size_nm} nm")

    # 9. Build Three.js data bundle
    print("Building data bundle...")
    regex_term = re.sub(r'[\^$.*+?\[\](){}|\\]', '', pattern)
    bundle = build_data_bundle(
        neurons_clipped, neurons_full, roi_clipped,
        roi_volumes, type_lookup, nt_lookup,
        all_types, js_color_modes, roi_neuron_bids,
        neuron_rois, primary_roi,
        roi_synapse_totals, type_roi_synapses, neuron_roi_synapses,
        instance_lookup, type_upstream, type_downstream,
        neuron_upstream, neuron_downstream,
        norm_params, regex_term=regex_term,
        neuron_meshes=neuron_meshes,
        voxel_size_nm=voxel_size_nm,
        _data_source={'server': server or 'neuprint-cns.janelia.org',
                      'dataset': dataset or 'cns'},
    )

    # 9. Optimize
    print("Optimizing data bundle...")
    bundle = optimize_bundle(bundle)

    # 9b. Embed synapse CSV groups if provided
    if synapse_csvs:
        embedded_syn = _process_synapse_csvs(synapse_csvs)
        if embedded_syn:
            bundle['embeddedSynapseGroups'] = embedded_syn

    # 10. Build HTML immediately (with synapse placeholder)
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    safe_name = re.sub(r'[\^$.*+?\[\](){}|\\]', '', pattern)
    output_path = out_dir / f'{safe_name}_visualization.html'
    synapse_cache = out_dir / f'{safe_name}_synapses.json'

    # If synapse cache exists, embed it directly (fast path)
    if synapse_cache.exists():
        with open(synapse_cache) as f:
            synapse_json_str = f.read()
        bundle['synapseData'] = '__SYNAPSE_PLACEHOLDER__'
        build_threejs_html(bundle, str(output_path))
        # Patch in the real synapse data via string replacement (avoids re-serializing bundle)
        html = output_path.read_text(encoding='utf-8')
        html = html.replace('"__SYNAPSE_PLACEHOLDER__"', synapse_json_str, 1)
        output_path.write_text(html, encoding='utf-8')
        print(f"Using cached synapse data: {synapse_cache.name}")
    else:
        # Write HTML first WITHOUT synapse data (user gets it in ~9s)
        bundle['synapseData'] = None
        build_threejs_html(bundle, str(output_path))

        if not skip_synapses:
            # Now fetch synapses and patch them into the existing HTML
            syn_df, syn_bid_type_map = fetch_synapse_positions(
                body_ids, type_lookup,
                partner_type_lookup=None,
                synapse_limit=synapse_limit,
            )
            if len(syn_df) > 0:
                synapse_data = _build_synapse_data(syn_df, syn_bid_type_map, norm_params)
                synapse_json_str = json.dumps(synapse_data, separators=(',', ':'))
                # Cache for next time
                synapse_cache.write_text(synapse_json_str, encoding='utf-8')
                # Patch into existing HTML (replace "synapseData":null)
                html = output_path.read_text(encoding='utf-8')
                html = html.replace('"synapseData":null', '"synapseData":' + synapse_json_str, 1)
                output_path.write_text(html, encoding='utf-8')
                print(f"  Patched {len(syn_df):,} synapses into HTML (cached to {synapse_cache.name})")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s")

    if auto_open:
        import webbrowser
        webbrowser.open(output_path.resolve().as_uri())

    return output_path


# ============================================================
# CLI
# ============================================================

def pattern_to_regex(pattern):
    """Convert simple pattern to neuPrint regex if needed."""
    if any(c in pattern for c in r'^$.*+?[](){}|\\'):
        return pattern
    return f'^{pattern}.*'


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D neuron visualizations from neuPrint data')
    parser.add_argument('patterns', nargs='*',
        help='Regex patterns (e.g., EPG "^FB.*" "LNO|GLNO|LCNO.*")')
    parser.add_argument('--all', action='store_true',
        help='Generate all standard patterns')
    parser.add_argument('--continuous', nargs='+', default=None,
        help='Continuous color mode CSVs (type + numeric columns)')
    parser.add_argument('--categorical', nargs='+', default=None,
        help='Categorical color mode CSVs (type + category columns)')
    parser.add_argument('--output-dir', type=str, default=None,
        help='Output directory (default: Examples/)')
    parser.add_argument('--open', action='store_true',
        help='Open HTML in browser after generation')
    parser.add_argument('--skip-synapses', action='store_true',
        help='Skip fetching individual synapse positions (faster)')
    parser.add_argument('--synapse-limit', type=int, default=None,
        help='Max number of synapses to keep (default: no limit)')
    parser.add_argument('--use-meshes', action='store_true',
        help='Fetch and embed 3D neuron meshes')
    parser.add_argument('--mesh-faces', default='auto',
        help='Max faces per neuron mesh: "auto" (default), a number, or 0 for no decimation')
    parser.add_argument('--max-file-mb', type=float, default=500,
        help='Target file size in MB when mesh-faces=auto (default: 500)')
    parser.add_argument('--server', default='neuprint-cns.janelia.org',
        help='neuPrint server URL (default: neuprint-cns.janelia.org)')
    parser.add_argument('--dataset', default='cns',
        help='neuPrint dataset (default: cns)')
    parser.add_argument('--token', default=None,
        help='neuPrint API token (or set NEUPRINT_TOKEN env var)')
    args = parser.parse_args()

    # Parse mesh_faces: 'auto', a number, or None/0
    _mf = args.mesh_faces
    if isinstance(_mf, str) and _mf.lower() == 'auto':
        mesh_faces_val = 'auto'
    else:
        try:
            mesh_faces_val = int(_mf)
            if mesh_faces_val <= 0:
                mesh_faces_val = None
        except (ValueError, TypeError):
            mesh_faces_val = 'auto'

    patterns = ALL_PATTERNS if args.all else args.patterns
    if not patterns:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    # Pre-load score modes once
    score_modes, cat_modes = load_score_modes(
        continuous_csvs=args.continuous,
        categorical_csvs=args.categorical,
    )

    results = {}
    for pattern in patterns:
        regex = pattern_to_regex(pattern)
        print(f'\n{"="*60}')
        print(f'  Generating: {pattern}  (regex: {regex})')
        print(f'{"="*60}')
        try:
            path = generate_visualization(
                regex, score_modes=score_modes,
                _cat_modes=cat_modes,
                output_dir=str(output_dir),
                auto_open=args.open and len(patterns) == 1,
                skip_synapses=args.skip_synapses,
                synapse_limit=args.synapse_limit,
                use_meshes=args.use_meshes,
                mesh_faces=mesh_faces_val,
                max_file_mb=args.max_file_mb,
                server=args.server,
                dataset=args.dataset,
                token=args.token)
            results[pattern] = str(path)
            print(f'  OK: {path.name}')
        except Exception as e:
            results[pattern] = None
            import traceback
            traceback.print_exc()
            print(f'  FAILED: {e}')

    if len(patterns) > 1:
        print(f'\n{"="*60}')
        print('  BATCH SUMMARY')
        print(f'{"="*60}')
        for p, result in results.items():
            status = 'OK' if result else 'FAILED'
            print(f'  {status:6s}  {p}')
        failed = sum(1 for s in results.values() if s is None)
        if failed:
            print(f'\n  {failed} of {len(results)} failed')
            sys.exit(1)


# ============================================================
# APPLICATION_JS — Three.js visualization application
# ============================================================

APPLICATION_JS = r"""
// ============================================================
// 3D Neuron Class Visualizer — Three.js Edition
// ============================================================

(function() {
'use strict';

// ---- Constants ----
const SIDEBAR_W = 200;
const TYPE_PANEL_W = 220;
const TOP_BAR_H = 50;
const PANEL_PAD = 8;   // uniform screen-edge padding for all fixed panels (px)
const CONN_PANEL_H = '33vh';
const GOLD = 'rgb(212,160,23)';
const GOLD_HEX = '#d4a017';
const HOVER_CLEAR_MS = 150;

// ---- Light / Dark themes ----
const THEMES = {
    dark: {
        bodyBg:      '#000',
        clearColor:  0x000000,
        panelBg:     'rgba(20,20,20,0.95)',
        stickyBg:    'rgba(20,20,20,0.98)',
        border:      '#444',
        topBorder:   '#333',
        text:        '#fff',
        subText:     '#ccc',
        mutedText:   '#aaa',
        gizmoBg:     'rgba(15,15,15,0.82)',
        gizmoBorder: 'rgba(80,80,80,0.5)',
        labelFront:  '#dddddd',
        labelBack:   '#888888',
        frustumBox:  'rgba(180,180,180,0.55)',
    },
    light: {
        bodyBg:      '#f0f0f0',
        clearColor:  0xf0f0f0,
        panelBg:     'rgba(220,220,220,0.97)',
        stickyBg:    'rgba(215,215,215,0.99)',
        border:      '#bbb',
        topBorder:   '#aaa',
        text:        '#111',
        subText:     '#333',
        mutedText:   '#555',
        gizmoBg:     'rgba(230,230,230,0.88)',
        gizmoBorder: 'rgba(140,140,140,0.6)',
        labelFront:  '#111111',
        labelBack:   '#777777',
        frustumBox:  'rgba(80,80,80,0.45)',
    }
};
let _currentTheme = THEMES.light;

// ---- Brain hull (convex hull of all Drosophila brain ROI meshes, normalized coords) ----
// Union of all 88 anatomical neuropil ROIs across all 24 visualizations (ME, LO, AL, MB lobes, etc.).
// Embedded as base64 Int16 vertices + Uint16 faces (qScale=30000), single shared mesh.
const BRAIN_HULL_V = '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';
const BRAIN_HULL_F = 'AwACAAAABwADAAAABwAAAAIAAgADAAEACgAHAAIABgACAAEAAgAEAAoABwAIAAMABgAEAAIACwABAAMACwADAAgAAQAFAAYABAAJAAoABQABAAsADAAHAAoACgAJAAwADAAIAAcACwASAAUADQAEAAYACQAEABAAFgAIAAwAEAAEAA0ABgAFAA0AEAAOAAkABQAPAA0AFgALAAgAFgAMAAkAEgAqAAUAKgANAA8AKgAPAAUAEAAVAA4AHAAJAA4AEgALABYAEgAWABEAEwAOABUAHAAoAAkAFAAOABMADgAUACAAFAATABgAFgAJACgAGgAOACAAHAAOABcAEwAVACEADgAaABcAGgAcABcAHgAYABMAGAAlABQAFAAlACAAJwAQAA0AEgARACoAIQAZABMAGQAeABMAGwAZACEAGwAdABkAHQAeABkAIAAcABoAHwAdABsAHwAeAB0AIQAVACMAJQAcACAAIwAiACQAJAAlABgAIwAmACIAJgAkACIAJgAjABUAJwANACoAKAAtABYAJgAlACQAFQAQADcALQARABYALAA5ABUALAAVADcAJwAqACkAKQARAC0AJQAmADkAOAA3ABAAEQApACoAOQAcACUANwAvACsALAA3ACsAKAAcADkAJwA4ABAALwAsACsALAAvAC4AMgAsAC4ANwAyADEANwAxAC8ALgAvADIALAAyADAAOQAsADAAFQA5ACYAMgAvADEAMgA5ADAAMgA3ADMAKAA5AC0ANwA0ADMAMwA0ADUANwA2ADQANAA2ADUANAA1ADUALQAnACkANgA5ADMAOQAyADMAOQA2ADcAOAAtADkAOAA5ADcAOAAnADoAOgAnAC0ALQA4ADoAPQA8ADsAOwA8AD8APwA9ADsAPwA8AD4AQQA8AD0APABCAD4APgBDAD8ARABCADwAPQA/AEwAPQBMAEEAQQBEADwAPgBAAEMAQwBAAEcAPgBHAEAAPwBDAEwARABBAEYAQgBNAD4ATABFAEEARgBBAEUARwA+AE0ASQBFAEwASwBEAEYASQBMAEgAUQBMAEMARwBRAEMAQgBEAEoASgBEAE4ARABLAE4AQgBKAE4ATgBLAEYAQgBOAE0ARwBTAFEAUgBQAEgATABPAEgASABPAFIAVQBGAEUATwBMAFEASQBIAFAARgBVAE4ATQBTAEcAUgBPAFAARQBJAFkAVQBFAFkATgBXAE0AUABUAEkATQBXAFMASQBUAFkAUQBWAE8AVQBYAE4AVgBaAE8AWgBQAE8AUQBTAFYAWABVAFcAUABZAFQAVQBdAFcAWABXAE4AVwBWAFMAVgBXAF0AVQBZAF0AWwBZAFAAVgBdAFoAXABQAFoAXQBcAFoAWwBQAFwAXABdAF4AWQBfAF0AXQBfAF4AXgBfAFkAXgBZAGAAWwBgAFkAXgBgAFsAWwBhAF4AXABeAGEAWwBcAGEAZQBmAGIAYwBlAGIAZgBjAGIAZABjAGYAZABmAGcAZwBjAGQAZwBlAGMAcwBoAGUAZQBnAHMAaQBmAGUAZQBoAGwAbwBnAGYAaQBlAGwAZgBqAG0AZgBpAGoAbQBvAGYAawBzAGcAZwBvAGsAcgBrAG8AbgBqAGkAeAByAG8AcABtAGoAeABvAG0AbgBwAGoAbABoAHcAcQBrAHIAaQBsAHQAaAB2AHcAbQBwAIAAbgB/AHAAbAB3AHQAaQB0AG4AfABzAGsAeABtAHoAbQCAAHoAdQBrAHEAdgB7AHcAdwB1AH4AdwCHAHQAdQBxAH4AdQCBAGsAegB5AHgAgQB8AGsAdgBoAHMAcgB+AHEAdQB3AH4AewCFAHcAeAB5AIMAhQCHAHcAfgByAH0AdgCGAHsAggByAHgAhAB6AIAAbgB0AIsAcwCTAHYAgQB1AH4AhAB5AHoAggB9AHIAfACTAHMAfwBuAI8AhQB7AIEAgwCCAHgAewCMAIEAcACJAIAAjQB2AJMAhwCLAHQAfgCFAIEAiwCPAG4AfACRAJMAkQB8AIMAcAB/AI8AkACIAHkAfQCSAH4AiACDAHkAgACJAIQAgwB8AJUAgwCIAJEAhgCMAHsAjQCGAHYAhwCFAJgAjAB8AIEAcACPAIoAiQBwAIoAkAB5AIQAlQCCAIMAmwCEAI4AjgCEAIkAlQB8AIwAigCOAIkAmACFAH4AkgB9AIIAjgCLAIcAjACGAJwAjgCPAIsAjQChAIYAkgCCAJUAjwCOAIoAmwCQAIQAkgCYAH4AlgCUAIgAmACaAIcAlgCIAJcAiACUAJEAkACXAIgAmwCXAJAAmgCOAIcAhgChAJwAjACZAJUAkwCRAJ8AnACZAIwAoACbAI4AlACfAJEAlQCYAJIAlQCZAJgAoACXAJsAmACdAJoAjgCaAKAApQCdAJgAngCXAKAAkwCfAI0AnwCmAI0AmACjAKUAngCUAJYAngCWAJcAoACaAJ0AjQCmAKEAqACfAJQAmQCjAJgAnACjAJkAngCgAKQAoACdAKIAowCnAKUAnQClAKIAqACUAJ4AoQCjAJwApACgAKIApACoAJ4AoQCmAKcApwCjAKEAqACmAJ8AogClAKkApQCnAKkAqQCkAKIApgCoAKkAqACkAKkAqwCmAKkApwCqAKkAqgCnAKwApgCsAKcApgCqAKwAqgCmAK0ArQCmAKsAqwCqAK0AqgCrAK4AqgCuAKkAqQCuAKsAtACxAK8ArwCzALQAsQCzAK8AswCxALAAsQC1ALAAtQCzALAAtAC1ALEAtgC3ALIAtwCzALIAtwC0ALMAswC1ALIAtgC5ALcAwAC5ALYAtAC3ALgAwAC2AMEAvAC3ALkAuwCyALUAwQC2ALIAtQC0AL4AsgC7AL0AwQCyALoAtwC8ALgAvQC6ALIAuADDALQAtADEAL4AtADDAMQAzAC7ALUAvQC7AMkAvwC6AL0AxwC4ALwAzQDKAL0AygC/AL0AuADHAMMAvgDMALUAuQDAAMUAvADFAMcA0AC+AMYAvgDQAMwAwQDCAMAAvgDEAMYAyQDNAL0AyADBALoAwgDFAMAAvAC5AMUAvwDIALoAywDDAMcAwQDIAMIAxADPAMYAuwDMAMkAywDWAMMAywDHAMUA5wDJAM8AzgDKAM0A0gDIAM4AwgDIANcAyAC/AM4AzgC/AMoA2QDRAMsAzwDJAMwA1wDIANIA1QDCANcAyQDnAOAAzQDJAOEA4ADhAMkAxQDVANkA0QDGAMsA2QDLAMUA0QDQAMYAwgDVAMUAzADjAM8AwwDiAMQA4wDWAMsAxgDjAMsA5gDOAM0AxADiAM8A4wDGAM8AwwDWANgA2QDVANIA0wDNAOEA5wDPAO4A4QDNANMA0gDOAOYA2ADtAMMAwwDtAOIA4gDuAM8A1ADaANYA2ADWANoA2QDlANEA0QDMANAA1QDXANIA5QDZAOgA2gDUANwA3ADYANoA3ADUANsA6gDRAOUA3wDUAN0A1ADfANsA2wDfANwA3ADfAN4A2ADcAN4A3QDUAN8A3gDfANgA7wDkAOcAzQDhAOQA7ADfANQA0QDjAMwA5ADmAM0A2ADfAO0A4wDsANYA1gDsANQA0gDoANkA3wDsAO0A6gDjANEA4QDgAOQA5wDkAOAA6gDlAOkA4wDpAOwA5wDuAO8A0gDrAOgA5gDrANIA4wDqAOkA5QDwAOkA7wDxAOQA7gDiAPMA8ADsAOkA8wDiAO0A5gDkAPEA8gDwAOUA8gDlAOgA8QD9AOYA/QDrAOYA6wDyAOgA7gD6AO8A+gDuAPMA+gDxAO8A7QDsAPUA7ADwAPUA7QD1APQA9ADzAO0A/QD7AOsA8ADyAPcA8gDrAPsA9gD3APUA8AD2APUA9wDyAPwA+QD6APMA9gDwAPcA9QD3APgA/AD1APgA9wD8APgA+gD9APEA9QD8APQA/QD6APsA9AD5APMA/gD8APIA9AD8AP4A/wD+APIA+wD/APIA+gD/APsA+gD5AP8A/gD5APQA+QD+AAAB/wAAAf4A/wD5AAABBwEBAQIBCAEBAQcBCAECAQEBAgEDAQcBBwEDAQkBCAEHAQQBAwECAQUBFAERAQYBCQEDAQUBBgEOAQoBBwEdAQQBBAESAQgBBgEQAQsBCwEOAQYBBQEQAQkBBgEbARABBQELARABAgEIAQUBGgEPAQ4BHQEHAQkBDAEXARkBDQEPARoBBQEWAQsBFgEFAQgBDwEjAQ4BCgEUAQYBGwEGAREBFgEIARIBDgELARoBDAEaAQsBEAEfAQkBCwEWAQwBFwEMARYBDgEjAQoBGgETARgBGQEXAR4BHwEdAQkBHQESAQQBEAEcAR8BGgEiARMBGAENARoBGwEVARABLwERARQBFwEWARIBEAEVARwBCgEjARQBIwEPAQ0BGQEwAQwBFwESAR4BDQEYASsBGgEMATABIAEZAR4BLgEeARIBIgEaATABEwE+ARgBIwEvARQBJAEjAQ0BHAE6AR8BOwESAR0BIQETASIBKQEhASIBNwESATsBHAEVATEBNwEuARIBHgEtASABLQEZASABLQEeAS4BKwEyAQ0BHAExAToBLAETASEBGwExARUBOgEdAR8BLwEzAREBJQEhASkBKAE7AR0BJgEhASUBJgElASkBIQEmAScBKQEqASYBJgEqAScBLAEhAScBKgEsAScBIgEsASoBKQEiASoBJAENATIBIgE2ASwBPgErARgBPgETASwBNAEwARkBLwEjASQBLQEuATcBOAEwATQBNAEZAS0BMwEbAREBOQEiATABGwFIATEBHQE6ASgBPwEoAToBNAEtATcBNQEzAS8BNQEvASQBOQE2ASIBMgE8ASQBMAE4ATkBOwEoAT8BQgEbATMBLAE2AT4BPAFHASQBOwE/AT0BQgFIARsBNQEkAUcBNwE7ATQBNgE5AT4BPQE4ATQBRgErAT4BQQE4AT0BKwFGATIBNQFEATMBNAE7AT0BMQFIAToBRQE/AToBQAE9AT8BRgE+ATkBQAFBAT0BUQE8ATIBRQE6AUgBQQFJATgBRQFOAT8BSQE5ATgBQQFAAUoBQQFKAUkBNQFDAUQBRAFDATUBTQFAAT8BRgFRATIBNQFHAUQBQAFNAUoBTgFSAT8BMwFWAUIBSwFEAUcBRgE5AUkBTAFJAUoBUgFNAT8BRAFWATMBRQFIAVYBTQFMAUoBUwFEAUsBUwFWAUQBTwFMAU0BRwFUAUsBRwE8AVEBVwFRAUYBVgFIAUIBVwFGAUkBVAFHAVEBTQFSAVkBTgFFAVUBVgFVAUUBVgFTAVABSQFMAU8BWQFPAU0BUwFWAVABVQFSAU4BUQFXAVQBVwFJAU8BVgFTAUsBXAFSAVUBXAFWAUsBVwFPAVkBVgFcAVUBVAFcAUsBWAFZAVIBVwFZAVgBXAFYAVIBVAFXAVsBVAFbAVwBWgFcAVsBWwFXAVgBWwFdAVoBWwFfAV0BWwFYAV8BYAFcAVoBXQFfAVgBYAFeAVwBYAFaAV4BXgFaAWEBYQFaAV0BWAFeAWEBYQFdAVgBWAFcAV4BZAFoAWIBaAFjAWIBYwFkAWIBaQFjAWgBaQFkAWMBZQFmAWQBZwFmAWUBaQF1AWQBZAF1AWUBZgFoAWQBaQFoAXMBZwFtAWYBawFvAW0BZgFtAXABggFrAWoBaAFmAXMBggFvAWsBbAFtAWcBbQFsAWsBZQF1AXsBbgFwAW8BcAFuAXcBcwFmAXYBbQFvAXABZQGBAWcBcAF2AWYBdgFxAXIBegFqAWsBdAFvAX4BggF+AW8BbwGFAW4BcAFxAXYBegFrAWwBdwFuAXwBewFpAXMBewF1AWkBgQFlAXsBegFsAXkBdwFxAXABbAFnAXgBdgFyAXMBdAGDAW8BagF6AZcBeQFsAXgBbwGDAYUBdwF8AXEBeAFnAYEBcgFxAYQBfgGDAXQBcQF8AX0BigFxAX0BggFqAYgBmAFzAYsBiwFzAXIBfAFuAYUBkwF+AYIBcQGGAYQBeAGHAXkBcwGYAXsBigGGAXEBfgGJAYMBeQGNAXoBlwGIAWoBiwFyAYQBgAF9AXwBjQGXAXoBmQGHAXgBgQF7AX8BfAGsAZ8BewGVAX8BhAGGAYsBiAGPAYIBfAGfAYABfAGFAawBjQF5AYcBmQF4AYEBigGOAYYBkwGJAX4BgAGfAX0BggGPAYwBoAGKAX0BngGBAX8BjgGKAaABjwGQAYwBjwGcAZABogGFAYMBnAGRAZABkQGMAZABjAGRAZQBnAGUAZEBkgGUAZwBkwGoAYkBiwGGAZgBnQGeAX8BkgGhAZQBgQGeAZkBjAGUAaEBoQGSAZYBnAGWAZIBjQGHAZkBoQGWAZwBnQF/AZUBewGYAZUBmwGYAYYBjAGhAYIBfQGmAaABjwGIAZwBmgGOAaABhQGiAawBmwGGAY4BpQGTAYIBiQGiAYMBlQGYAZsBpQGCAaEBmwGaAZUBmwGOAZoBpQGhAZwBlwGuAYgBmgGdAZUBnAGIAaUBogGJAagBngGdAaQBngGjAZkBpwGlAYgBrgGXAY0BqgGeAaQBfQGfAaYBmgGkAZ0BqAGTAbEBpAGaAaABmQGuAY0BngGqAaMBpAGgAakBkwGlAbEBrAGiAagBrgGZAaMBnwGsAaYBrgGnAYgBsQGlAacBqgGkAakBpgGpAaABpwGvAbEBrwGnAa4BsQGzAagBrAGrAaYBqQGmAasBqgGtAaMBtAGqAakBqQGwAbQBqAG3AawBsAGpAasBrQGuAaMBqgGyAa0BsQGvAbMBrQGvAa4BqgG0AbIBtwGoAbMBtQGsAbcBsgGvAa0BrAG1AasBqwG1AbABvAG0AbABtQG8AbABtQG2AbwBvgG3AbMBswGvAb4BvgGvAckBtwG+Ab8BrwGyAckBuAG1AcABtQG3AcABwgG1AbgBtwG4AcABuQG1AcIBwQG5AcIBtwG/AbgBwgG4Ab8BtQG5AcMBugG5AcEBtQHDAbYBuQG2AcMBtgG5AcUBxQG5AboBtgHFAbsBxAG6AcEBuwHGAbYBuwG8AcYBvQG7AcUBugG9AcUBvQG6AccBxgG8AbYBugHEAccBuwHIAbwBuwG9AcgBvQG8AcgByQGyAcoBvAG9AccBtAHKAbIBvAHHAbQBtAHHAcoB1gHYAcsB1gHLAdABzAHLAdgBzAHQAcsBzwHWAdABzAHYAdEBzQHMAdEBzgHMAc0BzwHQAc4BzgHQAcwB0QHYAdIB0wHWAc8B2AHXAdIB1wHVAdIB0wHSAdUB0gHTAc8B1AHTAdUB1QHXAdQB0wHUAdcB0wHXAdYB2AHaAdcB2wHaAdgB2gHcAdcB2QHWAdcB2AHWAdsB1gHZAdsB3AHZAdcB3AHaAeAB2gHbAeAB3AHiAdkB4gHfAdkB2QHdAdsB2wHdAeAB6QHcAeAB4wHcAekB3gHfAeUB3wHnAeUB2QHqAd0B5wHfAeIB3wHqAdkB8gHpAeAB7AHeAeUB4gHcAeMB3gHkAd8B3wHkAeYB3wHmAeoB7AHhAd4B5AHeAeEB5AHhAewB5QHnAfAB3QHqAe8B7wHgAd0B6wHkAewB5wHoAfAB8wHnAeIB8wHiAeMB5wHzAegB8wHjAekB5gHkAesB8QHgAe8B+gHsAeUBBwLlAfAB5gH1AeoB8gHtAekB9QH0AeoB+gHrAewBBwLwAfcB6QHtAfMB/gHwAegB6gH0Ae4B6gHuAe8B4AHxAfIB9AHvAe4B/gH3AfAB9QHmAesB+gHlAQcC7wH0AfEB9QHrAfsB+gH7AesB/gHoAfYB6AHzAfkB9gHoAQYC8wHtAfkBBQLtAfIB9QHxAfQB+AH5Ae0BBQL4Ae0B+QEGAugB8gH4AQIC9wH+AQEC8QH4AfIBAQIHAvcB/AH+AfYB/AH2AQYCAgL4AQUCBQLyAQIC/QHxAfUB8QH9AfgBBwIEAvoB/QH5AfgBCQL6AQQCCQL7AfoBBAIWAgkCBgL/AQAC/wEGAgMC9QEIAv0BCAL1AfsB/gH8ARMCBQL/AQMC+wERAggCAAL8AQYC/QEIAhQCCQIRAvsB/QENAvkBBQIDAgIC/wEVAgACBAIHAhIC/AEVAhMC/gETAhAC+QENAgYCFQL8AQACAQL+ARACBQIPAv8BFAINAv0BAgIDAgUCBwIBAhICBQIDAg8CFQL/AQ8CCwIBAhACAQILAgoCCwIZAgoCGQIBAgoCFAIIAgwCDQIDAgYCCwIQAg4CEAIZAg4CDgIZAgsCEgIWAgQCFwIQAhMCFAIMAhgCAwINAg8CGAIMAhECDQIVAg8CDAIIAhECDQITAhUCAQIZAhICFwIUAhgCFwIZAhACFwITAg0CFgIRAgkCFwINAhQCGAIRAhoCFgISAhwCGAIaAhcCGwIZAhcCGgIRAhYCHAIaAhYCGgIbAhcCEgIZAhsCHAIbAhoCHAISAh0CHgIdAhICGwIfAhICHgISAh8CHAIfAhsCHgIfAiACHAIdAiECHAIeAiICHgIcAiECHgIhAh0CIwIcAiACIAIcAiICIAIiAh4CIAIfAiMCHwIcAiMCJQImAiQCKAIlAiQCJgIoAiQCJwIlAkUCRQIlAigCSQInAkUCJQInAigCJgIlAigCLQIpAioCLQIqAisCKwIsAi4CLgIsAjICMAIvAikCKQIvAioCKgIvAjECKgIxAisCKwIxAi4CKwIuAiwCLAIuAjICRQIoAkcCLQIvAjACLQIwAikCLQIqAi8CKwIxAioCKwIqAi0CLgIxAisCMQIvAioCMwI0AjUCNQI0AjsCNgI4AjMCMwI4AjQCNAI4AjoCNAI6AjsCMwI3AjYCNQI5AjcCNQI3AjMCOwI5AjUCNwI4AjYCOQI4AjcCOQI6AjgCOwI6AjkCPAI9Aj8CPwI9Aj4CPAJAAkECPAJBAj0CPQJBAkICPQJCAkMCPQJDAj4CPgJDAkQCRwIoAicCPwJBAkACPwJAAjwCPwJCAkECPgJDAkICPgJCAj8CRAJDAj4CQQJBAkACQgJCAkECQgJBAkECQwJDAkICQwJCAkICRAJDAkMCJwJJAkcCRQJHAk8CRwJIAkwCTQJIAkYCSAJKAkYCSQJFAlECUQJFAk8CTAJIAk0CSQJIAkcCVQJIAkkCSwJQAk0CTgJVAkkCSwJNAkYCTAJNAlQCSwJGAlACXAJOAkkCTAJUAlMCSQJRAlwCTAJPAkcCTwJMAlICVgJGAkoCUgJRAk8CUwJUAlsCSAJXAkoCRgJWAlACUgJMAlMCUgJZAlECUgJTAloCUAJUAk0CSAJVAlcCVQJOAlgCWwJaAlMCTgJcAlgCWQJcAlECVwJdAkoCSgJdAlYCYgJWAl0CVgJiAlACVwJVAlgCVwJYAl0CUAJfAmMCYwJbAlQCYQJYAlwCUAJjAlQCWwJeAloCWwJjAl4CYgJdAlgCWgJeAmgCXAJZAm0CZQJqAlkCZAJSAloCbQJhAlwCWQJSAm0CXwJQAmYCUAJiAmYCbQJlAlkCbAJeAmACZwJiAlgCUgJhAm0CaAJkAloCYQJSAmQCWAJhAmcCYAJeAm8CaQJlAmQCXgJjAmsCXgJrAm8CagJtAlkCZAJoAmkCbAJoAl4CZwJmAmICcAJkAmUCagJ2Am0CYQJkAnECZQJ2AmoCdgJlAmkCcAJxAmQCZgJ1Al8CZwJuAmYCcwJnAmECYAJ4AmwCdQJ5Al8CcQJzAmECXwJ5AmMCbgJyAmYCawJ4Am8CZgJyAnUCcAJlAm0CZwJzAm4CbQJ2AnACegJpAmgCbwJ4AmACaAJsAncCaQJ6AnYCawJjAnkCdwJ6AmgCbgJzAnQCcQJwAnYCbAJ4AncCcQJ2AnsCcgJuAnQCewJ2AnoCcwJxAnsCdwJ7AnoCfgJ4AmsCeQJ1AoACcwJ7AnQCfgJ8AngCfAJ3AngCfwJyAnQCeQJ+AmsCcgJ/AnUCewJ9AnQCewJ3An0CfQJ3AnwCfwJ0An0CfwJ9AnUCgAJ+AnkCgQKAAnUCgQJ1AoICggJ1An0CggJ9AnwCgAKBAoICfgKCAnwCgAKCAn4CgwKLAooCigKFAoMChQKLAoMCiwKFAocChwKFAoQChQKKAoQCigKHAoQCiwKQAooCigKOAocCiQKbApACiAKGApoCkAKbAogCiwKJApACkgKLAocCjAKaAoYCjgKSAocCjQKGAogCjAKGAo0ClAKKApACjAKNAo8CjQKIApsClQKSAo4ClAKOAooClgKMAo8CjAKWApoCiQKLApsCjQKRAo8CiAKZApACjwKRAo0CmQKIApoCnQKVAo4ClAKdAo4CjQKeApECmAKSApUCqwKeAo0CkQKPAo0CmgKTAqIClQKdApcCqwKNApsCkgKbAosCmQKUApACjwKRApYCmAKVApcCmAKXAqACmgKWApMCmgKiApkCnAKlApcCmAKjApICnQKcApcClwKlAqACkgKfApsCmAKgAqoCpwKSAqMCmwKfAqsCkwKkAqICmAKsAqMCpQKcAqkCngK2ApECmQKiApQCogKkAqYCnAKdAqECqwK2Ap4ClAKiAqEClgKuApMCoQKiAqYClAKhAp0CrAKYAqoCpQKqAqACkgKnAp8CkQKvApYCqQKcAqECsAKjAqwCkwKuAqQCkQK2Aq8CswKlAqkCpgKpAqECsgKmAqQCqgKlArECwQKrAp8CqQKmArcCpQKzArEClgKvAq4CrAKoAq0CtAKkAq4CrAKtArACrQKoArACtAKyAqQCqQK3ArMCxwKjArACwwKuAq8CsgK3AqYCtwK7ArMCxgKsAqoCxgKoAqwCwgK2AqsCqALHArACwQKfArkCvQKuArUCtQKuAsMCtAKuAr0CsgK0AroCxwKnAqMCuwK3AsQCuAKzArsCuAKxArMCwQLCAqsCnwKnArkCuwKxArgCvAKxArsCvwKvArYCvAK7AskCyQKxArwCwAK2AsICvQK1ArQCrwLIAsMCxgKqArECtQLDAr4CwwK0Ar4CtAK1Ar4CyQK7AsQCtgLAAr8CvwLIAq8CsgLKArcCxgLHAqgCxwLbAqcCtwLKAsQCwQK5AsICvwLAAsICwwLPArQCsgK6As8CpwLdArkCxQKxAskCywLIAr8CwgLLAr8CsgLPAsoCugK0As8CxQLGArECzgLDAsgCwwLOAs8CywLCAtECywLOAsgCwgK5AtECygLfAsQC0ALaAswCygLMAt8CxwLGAtsCzwLMAsoCuQLNAtECuQLdAs0CyQLEAuAC0ALNAtoCzgLLAtwCzQLQAtwCpwLbAt0CzwLcAswC3ALQAswC0QLNAtwC0gLTAtYC0gLWAtQC1gLTAtUC1QLTAtIC1QLSAtcC1wLSAtQCywLRAtwC2ALUAtYC2ALZAtQC1wLYAtUC1QLYAtYC1ALYAtcC1ALZAtgCxgLFAt4CzgLcAs8C3wLgAsQC3QLbAs0C2gLfAswCyQLgAsUC3wLaAuQC3gLbAsYCzQLbAtoC3gLFAuAC3wLhAuAC5ALhAt8C4wLaAtsC4gLkAtoC3gLjAtsC2gLoAuIC2gLjAugC5gLjAt4C5gLeAuAC5ALgAuEC5ALiAuAC6ALlAuIC7ALjAuYC4gLmAuAC5gLnAuwC6ALxAukC6ALpAvAC5QLoAvAC5QLwAukC8QLoAuMC8wLlAukC8QLjAukC5QLtAvIC4gLlAvIC4gLyAu0C5QL0Au0C5QLqAvQC5QLrAuoC7ALlAvMC9ALqAu0C6gLrAvUC5QLsAusC7ALzAuMC4wLzAukC7QLqAvUC6wLtAvUC6wLsAvYC6wL2Au0C7QL2AucC9gLsAucC7QL3AuIC7QLvAvcC7gLiAvcC9wLvAu4C7wL4Au4C+ALvAu0C+ALtAucC+ALnAu4C7gL5AuIC+QLmAuIC5gL5Au4C5wLmAu4C+wL8Av8C+wIBA/wCAQMGA/oCAQP7Av4C+gIAAwUD+wL/AgcDAQP+AgYDBwP+AvsCCAP/AgID/AICA/8C/QIFAwsD+gIGAwADBwP/AggD+gIdAwEDCgP8AgED/AIKAwID/gIHAwwDBgMDAwAD/QIdAwUDBAP9AgsDBQMdA/oCCQMFAwADAAMDAwkDHwP+AgwD/QIEAx0DHwMkA/4CBQMJAwsDCwMOAwQDMQMSAwIDBgP+AiQDJQMHAwgDDgMLAyADBwMlAwwDAgMKAzEDFAMOAyADDQMQAxQDDQMZAxADBgMkAwMDMQNDAxIDEgMPAwIDBAMOAxADEAMOAxQDFQMNAxQDJgMRAxADJQMIAwIDIwMZAw0DEAMRAwQDHQMKAwEDJgMQAxkDCgMTAzEDJwMTAxEDAgMPAyUDGgMPAxIDKQMNAxUDHAMKAx0DCgMcAxMDBAMRAx0DFgMXAxYDFgMWAxYDFwMWAxgDFgMYAxYDGwMeAxsDGwMbAxsDHgMbAxsDGwMbAxsDEQMcAx0DIQMVAxQDEQMTAxwDIwMNAykDKAMPAxoDCQMgAwsDEgMoAxoDJwMRAyYDQgMkAx8DIwMiAxkDFAMgAzgDKAMlAw8DIQMUAzUDMwMMAyUDPgNCAx8DNgMmAxkDMwMfAwwDJAMqAwMDJgM2AzIDMAMVAyEDIAMJAzgDIQM1AywDMwM+Ax8DOgMUAzgDOQMxAxMDIgM2AxkDQwMoAxIDPQMnAyYDKwMhAywDMAMhAysDFAM6AzUDRAMJAwMDLAMtAysDIgMjAykDEwMnAzkDLAM0Ay4DLQMsAy4DNQM0AywDNAMtAy4DLQM0AysDJQMoAzMDLQM0Ay8DNAMtAy8DKwM0AzADMgM9AyYDMgM2A00DSgM2AyIDQgMqAyQDFQMwAz8DQAMzAygDNAM7AzADIgMpA0oDNgNKA00DFQM/AykDOwM0AzUDQAM3AzMDQwMxAzkDPQM5AycDMAM7Az8DMwM8Az4DAwMqA0QDCQNEAzgDQwNAAygDPAMzAzcDNQM6AzsDQgNBAyoDOwM6Az8DOANLAzoDPANFAz4DQANGAzcDNwNGAzwDQwM5Az0DRQNCAz4DSgMpAz8DKgNBA0QDPANGA0UDRANLAzgDOgNLAz8DRgNAA0MDQQNCA0UDQQNHA0QDQQNFA0cDRQNGA0gDRgNDAz0DMgNQAz0DSgM/A08DSANGAz0DPQNMA0gDRwNFA0kDRQNIA0kDTAM9A1ADSQNIA0wDUANNA0oDSwNEA0cDTgNPAz8DRwNJA1IDUAMyA00DUgNLA0cDUgNJA0wDSwNOAz8DUANKA1EDSgNPA1EDTgNRA08DUQNLA1IDUQNOA0sDUANSA0wDUANTA1IDUwNRA1IDUQNTA1ADVgNUA1gDXgNZA1cDVgNdA1QDXgNXA1UDVgNbA10DVANgA2QDWgNXA1kDaANYA1QDVQNcA14DWgNjA1cDWgNZA18DWANwA1YDVwNjA1UDaANsA1gDXgNlA1kDVgNnA1sDZwNWA3ADYANUA10DWwNyA10DaANUA2QDXwNZA3wDbANwA1gDbgNhA10DXQNyA24DYANjA1oDYwNiA1UDcgNbA2oDZgNeA1wDZANgA1oDVQNiA1wDdwNsA2gDcQNiA2MDXwNkA1oDXQNhA2ADZwN1A1sDfANZA2UDXwN6A2QDaQNqA1sDdQNpA1sDYANhA2MDdQNqA2kDawNqA3UDZAN2A2gDdwNoA3YDagNrA3UDcQNjA2EDZQODA3wDZgN+A14DcQNhA20DegN2A2QDXANzA2YDcwNcA2IDfgNlA14DdwNwA2wDZQN+A4MDcgNvA24DggNiA3EDeANqA3UDZgNzA38DcgNqA3QDcAN5A2cDeANyA3QDagN4A3QDcwNiA4IDcAN3A3kDbQOCA3EDeQN1A2cDhgN2A3oDbQNhA28DXwN8A4QDhAN6A18DbwNhA24DZgN/A34DcgN4A28DewN4A3UDfQNtA4sDeQN7A3UDbwOLA20DjwN3A3YDgANzA4IDggNtA30DkgOGA3oDfAODA4UDkgN6A4QDbwN4A5EDhQOEA3wDfwOAA34DeQN3A4gDfwNzA4ADewN5A40DjwOIA3cDkQN4A3sDjwN2A4YDkAOCA30DjgOBA3kDgAOHA34DhwODA34DiAOOA3kDggOKA4ADiQN9A4sDewONA5EDkgOEA4UDkQOLA28DgQONA3kDhgOSA4kDigOCA5ADjAODA4cDgAOMA4cDiQOQA30DgwOMA4UDjwOGA4kDjgONA4EDkAOJA5IDjwOJA4sDigOMA4ADkwOSA4UDjAOTA4UDkQONA44DjAOKA5MDkgOKA5ADjgOIA48DiwORA48DjwORA44DkwOKA5IDlQOeA5QDngOVA5YDlgOhA54DngOVA5QDoQOWA50DmQOVA54DlgOVA5kDngOYA5kDngOcA5oDlwOYA5oDmgOYA54DmQOYA5cDnQOWA5oDnAOfA5sDnAOdA5oDnwOdA5sDnQOcA5sDowOgA6QDnwOcA54DoQOdA6UDoQOfA54DqgOgA6MDnwOlA50DqwOhA6UDpgOpA6MDowOpA6oDqwOoA6EDpQOmA6sDnwOiA6UDpgOlA6kDoQOiA58DpwOhA6gDpwOiA6EDtAOgA6oDtAOkA6ADqwOsA6gDqQOlA6IDtAOjA6QDqAOuA6cDrgOoA60DqQOiA6oDqwOmA7MDowO4A6YDowOyA7gDrQOoA6wDtwOjA7QDowO3A7IDsAO0A6oDogOwA6oDrwOrA7MDrwOsA6sDrQO1A6cDtQOiA6cDrgOtA6cDpgO4A7MDsQOtA6wDuQOwA6IDsgO6A7gDrQOxA7YDtQO5A6IDrwOxA6wDtwO6A7IDtAOwA7kDuQO3A7QDtgOxA70DrQO2A8ADuAPDA7MDwgOzA8MDtQOtA8ADsQO8A70DwwO4A8EDrwO7A7EDwgOvA7MDsQO7A7wDvAO+A70DvAO7A74DuQO1A8ADvwO9A74DvgO7A78DvQO/A7sDxgO6A7cDuQPGA7cDxgPBA7gDtgPPA8ADvQPEA7YDrwPLA7sDugPGA7gDvQO7A8QDywPEA7sDwAPKA7kDxAPFA7YDtgPFA88DygPGA7kDwgPDA8EDxwPEA8sDyQPBA8YDyAPLA68DwgPIA68DxAPHA8UD0APOA8ADyQPCA8EDzwPQA8ADwAPOA8oD1APIA8IDzAPIA9QDxQPHA88DzQPLA9YDzQPHA8sDyQPUA8IDyAPMA9QDygPJA8YDywPIA9YDzwPHA9ED1APJA9MDzgPXA8oD0APPA9ED0QPHA80DzgPQA9ED0gPKA9cD0gPTA8oD1QPRA80D0wPJA8oD2APWA8gD2APIA9QDzgPZA9cDzQPWA9UDzgPRA9kD0QPVA9kD2APUA9MD1QPWA9gD0gPaA9MD2QPVA9gD3QPYA9MD2QPdA9cD2gPSA9cD3QPZA9gD0wPaA90D1wPdA9oD3APdA9oD3QPbA9oD3gPaA9sD2wPcA94D3APaA94D2wPfA9wD2wPdA98D3APfA90D4QP1A+AD4QPgA/ID4APwA/ID8gPtA+ED4gPhA+0D5QPhA+ID4wPhA+QD4QPjA/UD4gPpA+UD5AP1A+MD7wPhA+UD7wPkA+ED4gPtA+cD5APvA+YD6QPvA+UD4gPnA/ED5wPtA+gD9QPkA+YD7wP1A+YD7wPpA+wD6QPiA+oD4gPxA+oD6APxA+cD8gPwA/MD7APpA/ED8QPpA+oD6APtA+sD9QPvA+4D8QPvA+wD8QPoA+sD7QPxA+sD8QP1A+4D7wPxA+4D7QPyA/ED8gPzA/gD8wPwA/8D4AP2A/AD9AP2A+AD9QP5A+AD9APgA/kD8gP1A/ED8gP4A/sD9gP/A/ADAwT/A/YD+APzA/8D8gP8A/UD9AP5A/YD/wP3A/gD/wP4A/cD/AP5A/UD8gP7A/wD/wP6A/gD+wP4AwEE/AP2A/kDAQT4A/oD/AP7AwAE/QMDBPYD9gP8A/0D+gP/AwMEAQT6AwQE/QMCBAME+wMBBP4DAAT+AwEE/AMABP0D/gMABPsDBAQCBAEEBgQEBAMEAgQABAEE+gMDBAQEAAQCBP0DAgQEBAMEAwQEBAcEBgQDBAUECQQDBAcEBwQEBAgEBwQFBAkEBgQIBAQEBQQDBAkEBQQHBAgEBgQFBAgECgQLBAwECgQMBA4ECgQRBAsEDgQNBAoECwQTBAwEDAQTBBIEDQQRBAoEDQQPBBEEDgQMBBAEFgQNBA4EEgQQBAwEFgQPBA0EEQQPBAsEFwQTBAsEDwQXBAsEEwQYBBIEEwQXBBQEFAQYBBMEFAQXBBgEEgQVBBAEDgQQBBYEEgQYBBUEGgQPBBYEGAQbBBUEGQQQBBUEDwQaBBcEGAQXBBoEGwQZBBUEGQQWBBAEFgQbBBoEGAQaBBsEGQQcBBYEGwQcBBkEHAQbBBYEHgQrBB0EHgQlBCsEJQQeBB0EHwQlBB0EHQQkBB8EJAQlBB8EJAQnBCEEIwQgBCQEIQQjBCQEJgQkBCAEIwQhBCcEIwQmBCAEJgQnBCIEJgQjBCcEKAQiBCcEJgQiBCgEJAQdBCcEJwQdBC4EJgQoBCoELgQdBCsEJwQuBCgEKgQkBCYEJQQkBCoEKgQrBCUEKAQuBCwELQQsBCkELAQuBCkELwQyBCkEKQQyBC0EKwQqBC4ELQQxBCwEMQQzBCwEKAQsBDMEOAQpBC4EKQQ1BC8EKgQ6BC4EKgQ0BDoEKQQ4BDUEMAQ0BCoEMwQqBCgEKgRDBDAEKgQzBEMEOgQ4BC4ELwQ1BDIEMQQtBDgEQwRQBDAEMgQ4BC0EOAREBDEENQQ4BDIEOQQ6BDQERAQzBDEENAQ3BDkEPQQ3BD4ENwQ9BDkENARCBDcERAQ2BDMEQwQzBDYEPAQ5BD0EOgRABDgEPQQ+BDwEOQQ8BD8EOQQ/BDoEPgQ3BEIEQAREBDgESARQBEMEQAQ6BD8EMARQBDQESwRDBDYEQgRKBD4ENgRRBDsEOwRLBDYEPgRMBDwEWAREBEAENARQBFUENgREBEEEPwQ8BEwENARVBEIEWARUBEQEUQQ2BEEEPgRKBEwEPwRYBEAESARDBEcEQQRFBEYERwRDBEsETARTBD8EPwRTBFgERQRBBEQETQRRBEEETQRBBEYESgRcBEwEOwRRBEkESQRLBDsESgRCBFUESQRRBEsEUgRIBEcESwRRBE8ETwROBEsETgRPBFEEZgRFBEQETgRRBE0ERQRaBEYEUARIBFYETQRGBFoESARSBFYEVARmBEQERwRLBE4EWwRUBFgEVwRKBFUEUwRMBFwETgRSBEcESgRXBFwEVQRQBFYEUgROBFkEWARTBFwEWQRWBFIEXgROBE0EXwRbBFgETgReBFkEWwRrBFQEVQRWBFkERQRmBF0EVwRVBGAEVQRZBGAEZwRNBFoEZwReBE0EWgRFBF0EWARcBF8EWQReBGAEXQRnBFoEYQRcBFcEZARXBGAEVARrBGYEZQRrBFsEXARhBF8EVwRkBGEEbARgBF4EYQRoBF8EYwRlBFsEZARgBGwEWwRfBGMEbAReBGcEawRiBGYEYgRdBGYEYQRkBGkEbQRkBGwEagRhBGkEXwRoBGoEXwRqBGMEZARtBGkEaARhBGoEcARiBGsEYgRnBF0EagRpBGUEagRlBGMEbQRlBGkEZQRsBGsEcgRiBHAEbARwBGsEYgRyBGcEZwRvBGwEcARxBHIEcgRvBGcEbQRzBGUEZQRzBGwEbARzBG0EbAR0BHAEcAR0BG4EdARsBG4EdQRwBG4EbAR1BG4EbARwBHUEcARsBHYEcQRwBHYEcQR2BG8EbwR2BGwEdwRxBG8EcQR3BHIEcgR3BG8EhASHBIAEhASABHgEgASEBHgEgAR/BHkEeQSEBIAEfwSEBHkEhAR/BHoEiQSEBHoEiQR6BH8EiQR/BHsEfwSABHsEgASJBHsEiQSABHwEfgR8BIAEfgSJBHwEggR+BH0EfgSABH0EggR9BIAEggSJBH4EggSKBIkEggSBBIoEhASDBIcEggSABIEEgASGBIEEhASJBIMEiwSBBIYEhQSHBIgEhgSABIcEhwSDBIgEjASFBIgEhgSHBIUEjASIBI0EjgSGBIUEhQSMBI4EiASDBI0EiQSKBJAEjASNBJMEiQSPBIMEiwSGBI4EiQSQBI8EjwSSBIMEigSBBJAEkQSBBIsEjASTBJoEkQSQBIEEiwSOBJEEogSaBJMEjQSUBJMEgwSVBI0EkgSVBIMEjwSQBJEEpASOBIwEkwSUBKIEjQSVBJQEkQSOBJwElwSWBJsEnASZBJEEpASMBJoEmASSBI8ElwSUBJYEpAScBI4ElgSUBJUEkgSWBJUEoASbBJYEmASPBJEElgSSBJgElASdBKIEmQSlBJEEmASRBKUEmASgBJYEpgSUBJcEnwSlBJkEpQSnBJgEpQSfBKkElwSbBKsEmASsBKAEnwSZBLcEoQSZBJwEoASrBJsErASYBKcEqwSmBJcEoASsBJ4EtwSZBKEEsAShBJwEsAScBKQEpgSjBJQEnQSUBKMEnQSjBKYEoASeBKsEsgSkBJoErgSlBKkEswSyBJoEogSdBLYEogSzBJoEtQS3BKEEpQS5BKcEoQSwBLUEuQSlBK4EqASpBJ8EqASfBLcEsASkBLIEnQSmBKoEqgS2BJ0EngSsBK0EpgSrBK0EsASzBLUErwSwBLIErQSxBKYEqwSeBK0EqgSmBLEEsgSwBK8EpwS0BKwEsASyBLMEvASsBLQEogS2BLMEtASnBLkEtQSzBLYEtgSqBLEEvwS1BLYEqASuBKkExASoBLcEsQS/BLYEqATEBK4EsQS9BL8EvAStBKwEugS1BLgEuAS1BL8EuQTHBLQEtwS1BLoEsQTCBL0ErgS7BLkExAS3BLoEvwTBBLgEwQS6BLgEtAS+BLwEtATHBL4ErgTEBLsEvwS9BMAErQTDBLEErQS8BMMEsQTDBMIExQTBBL8EuQTBBMcExAS6BMgEuQS7BMEEvQTCBMAEvwTABMUExgTBBLsEvgTDBLwEwATCBMcEvgTCBMMEyATGBLsEyAS7BMQEvgTHBMIEwQTIBLoEwATHBMUEygTIBMEEyQTKBMEEywTHBMEExQTHBMsEwQTFBMsEwQTMBMkEzATBBMYExgTJBMwEygTJBM0EyQTGBM0EygTNBMYEzgTKBMYEyATKBM4ExgTIBM4EzwTTBNAE0ATRBM8E0QTTBM8E0QTQBNUE1ATTBNIE0wTRBNIE4gTVBNAE3QTSBNEE1ATWBNME1ATgBNYE3wTRBNUE1gTlBNME5QTiBNAE4gTXBNUE0ATTBOUE1QTaBN8E4ATUBNIE4ATSBN0E2ATVBNcE2ATaBNUE1wTaBNgE3wTdBNEE2QTXBNsE2QTaBNcE2wTaBNkE3ATaBNsE3AThBNoE3wTaBOEE2wThBNwE2wTXBN4E4QTbBN4E1wThBN4E4wTXBOIE1wTjBOcE4ATdBOQE5wThBNcE1gTgBOQE3wToBN0E5ATlBNYE4QTnBN8E5QTpBOIE3QToBOQE4gTmBOME5gTiBOkE5ATpBOUE5gTnBOME5gTpBPUE9QT2BOYE5ATqBO0E6QTkBO0E5AToBOoE5gTrBOcE6wTfBOcE5gT2BOwE5gTsBO8E9wTpBO0E7wTrBOYE7wTsBO4E6QT3BPAE6AT0BOoE8gTpBPAE7QTqBPQE7gTsBPEEAgXoBN8E8QTsBPYE9gTuBPEE8gT1BOkE7gT2BPME+ATuBPME9gT4BPME9AToBAIF7wTuBPgE6wQCBd8E+QTyBPoE7QT0BPcE9QTyBPkE9QT/BPYE+gTyBPAE/QT/BPUE9AQCBQQF8AT3BP4E+QT9BPUE9wT0BPwE+wTrBO8EAAX5BPoE/wT4BPYEBAX8BPQE/gT3BPwEAQX6BPAE7wT4BBMF+QQJBf0EAwX/BP0E/QQJBQMF/gQBBfAE+AT/BBMF+gQBBQAFEwX/BAMFAQX+BAoFAgXrBPsEAAUMBfkEEAUKBf4E/gT8BBAFDAUJBfkE/AQEBRAFBAUCBREFAwUJBQ8FBgUABQUFDAULBQkFDAUABQYFAAUHBQUFAAULBQcFBQUMBQYFAAUBBQoF+wQVBQIFDAUFBQgFBQULBQgFCwUFBQcFEQUCBRUFDAUIBQsFEgUPBQkFCgULBQAFCwUSBQkFDQUKBRAF7wQTBRUF7wQVBfsEBAUXBRAFDgUKBQ0FDQUaBQ4FCwUKBQ4FFwUEBREFEgULBQ4FEwUDBRIFEAUYBQ0FEgUDBQ8FGgUNBRgFEwUWBRUFEgUOBRQFFAURBRUFGgUbBQ4FEgUWBRMFFAUVBRIFEgUVBRYFEQUUBRcFFwUYBRAFGwUUBQ4FGQUaBRgFGgUZBRgFGgUcBRsFHAUYBRcFGwUdBRQFGgUYBRwFFwUUBR0FHQUcBRcFHAUeBRsFGwUeBR0FHQUeBRwFIwUfBSAFHwUoBSAFIwUgBSgFIQUfBSMFJQUfBSEFJQUhBSMFHwUlBSgFKAUrBSMFKQUlBSMFIgUkBSwFJgUkBSIFIwUrBSkFJgUiBScFLAUxBSIFJQUuBSgFLgUlBTMFLQUlBSkFLAUkBSYFMQUnBSIFKQUqBScFKwUqBSkFKAUuBSsFJQUtBTMFKwU2BSoFJwUqBS8FLQUpBScFJwUxBS0FKwUuBTAFLwU9BScFJgUnBT0FMwUtBTEFMAU2BSsFNwUuBTMFOwUvBSoFMgUqBTYFNwUwBS4FPAUsBSYFMQUsBTUFPAU1BSwFOwUqBTIFMAU3BTQFMQU1BTMFNwU6BTQFNAU4BTAFOAU2BTAFOQU8BSYFPQU5BSYFPQUvBTsFQQUyBTYFQQU2BTgFMwU1BTcFNAU/BTgFPgU5BT0FPwU0BUUFMgU9BTsFQAU0BToFOgU3BUcFOQU+BTwFPgU1BTwFPQVEBT4FNQVHBTcFNAVABUUFPwVDBTgFPQUyBUYFRwU1BUIFQwVBBTgFQgU1BT4FSgVBBUMFRAVCBT4FRgUyBUEFRgVEBT0FRQVOBT8FSgVGBUEFQwU/BUkFSQVKBUMFTgVJBT8FTAVEBUYFSgVMBUYFRQVABU4FRwVCBUgFTwVABToFSwVKBUkFOgVHBU8FTAVCBUQFQAVPBU4FQgVMBUgFUAVJBU4FSwVJBVIFTQVKBUsFUgVNBUsFSgVNBUwFSQVQBVIFUQVPBUcFTwVRBU4FWAVUBUgFTAVYBUgFVAVHBUgFTQVSBVMFRwVUBVEFUwVSBVAFTgVRBVAFUwVQBVEFWAVMBU0FVQVNBVMFVAVWBVEFTQVVBVgFVQVXBVgFUwVRBVYFUwVXBVUFVwVTBVYFVgVUBVgFVgVYBVcFWQVYBVcFWgVXBVgFWgVYBVwFWwVaBVkFWgVcBVkFWAVZBVwFVwVaBV0FWQVaBVsFWQVXBV0FWQVdBVoFXgVgBV8FYgVgBV4FXgVfBWIFYQVgBWIFZAVfBWAFYAVjBWQFYgVlBWEFXwVmBWIFZAVoBV8FYgVmBWUFYwVgBWcFZAVqBWgFaAVmBV8FYQVnBWAFYwVqBWQFbQVmBWgFawVnBWEFZwVuBWMFbgVqBWMFZwVpBXAFfQVlBWYFawVpBWcFbAVhBWUFaAVqBW8FfQVsBWUFbgVnBXAFbAVrBWEFbQV9BWYFdAVuBXAFbQVoBW8FbwVqBXEFcAVpBXcFagVuBXEFbQVzBX0FcQVuBXIFdAVyBW4FdwVpBXYFeQV0BXAFbwVzBW0FdwV5BXAFaQVrBXYFfgVyBXQFcQV8BW8FegV+BXQFfQV1BWwFbwV4BXMFdQVrBWwFfAV4BW8FfgV/BXIFdgVrBXUFdgV1BXcFhgVzBXgFfAVxBXIFcwWGBX0FegV0BXkFfAWABXgFdQV7BXcFcgV/BXwFhAV/BX4FgwV7BXUFgQV6BXkFgQV5BXcFdQV9BYMFdwV7BYEFgQV7BYMFeAWABYUFhwWEBX4FggV6BYEFggWBBYMFfQWIBYMFfAV/BYQFgAV8BZIFeAWFBYYFggWMBXoFegWHBX4FiAV9BYYFgwWJBYIFgwWIBYkFkgV8BYQFjAWCBaMFhQWABYoFgAWSBYoFkwWCBYkFlQV6BYwFlAWGBYUFhQWKBYsFiwWUBYUFmQWJBYgFigWNBYsFiwWNBY4FigWOBY0FlAWKBZIFiwWRBZQFowWCBZMFiwWOBZEFkQWOBY8FjgWKBY8FkQWPBYoFlAWYBYYFiQWbBZMFkQWKBZAFkQWQBZQFigWUBZAFiAWGBZgFlQWHBXoFmQWIBZYFiQWZBZsFlQWMBZcFiAWYBZYFhAWaBZIFlAWSBZ8FhwWaBYQFlwWMBaMFpQWbBZkFlgWYBaUFlgWlBZkFlAWcBZgFnAWUBZ8FlQWaBYcFmgWfBZIFmwWlBagFnQWaBZUFnQWVBZcFqAWeBZsFnwWhBZwFoAWTBZsFnQWXBaYFowWTBaAFpAWlBZgFmwWeBaAFnAWhBacFnAWkBZgFpAWiBaUFnAWnBaQFsQWoBaUFowWmBZcFoQWfBZoFogWxBaUFngWuBaAFoAWpBaMFnQWrBZoFnQWmBawFmgWrBbAFqgWpBaAFrgWqBaAFrAWmBaMFmgWwBaEFnQWtBasFrAWjBakFpwWhBaQFsQWvBagFqwWtBaoFqwWqBa8FnQWsBa0FsAWrBbEFqwWvBbEFuQWiBaQFqAWuBZ4FpAWhBbkFrwWqBa4FoQWwBbkFrwWuBagFsgWxBaIFrQWpBaoFsAWxBbQFrQWsBakFsAW0BbkFswWyBaIFsQW9BbQFuQWzBaIFvQWxBbIFtQW7BbIFtwWyBbYFtQWyBbsFtgWyBbsFuwW3BbYFvQWyBbcFtwW7BbgFtwW4BboFuwW6BbgFugW9BbcFsgWzBbsFvQW6BbwFswW5BbQFugW7BbwFswW0Bb0FuwW9BbwFuwWzBb0FvgWzBb0FvQWzBb4FxAXDBb8FwAW/BcMFwwXBBcAFyAW/BcAFwQXCBcAFyAXABcIFxgXEBb8FvwXIBcUFxgW/BcUFwQXDBckFwQXPBcIFyAXGBcUFyQXDBcQFxAXGBcoFxgXIBccFyAXWBccF1gXGBccFzwXMBcIFzQXGBdYFyQXEBcoFxgXLBcoFzAXIBcIFywXGBc0FzgXPBcEFwQXJBc4F0AXMBc8F0QXJBcoFzQXaBcsFzwXOBdIF0gXQBc8FywXYBcoF2AXZBcoFywXaBdMF2AXLBdMF3gXJBdEFyQXeBc4FzgXeBdUFzgXVBdIF0QXKBdkF1gXIBfgF0wXfBdQF+AXIBcwF1AXfBdMFzQXWBdoF2AXTBd8F2wXMBdAF3AXQBdIF3gXRBegF2AXiBdkF3AXSBdUF0QXZBd0F3AXbBdAF1QXeBeAF3QXXBdEF2gXfBdMF2wX4BcwF+AXhBdYF4QXaBdYF1QXgBdwF0QXXBegF3wXiBdgF3wXaBeMF5QXdBdkF3QXmBdcF4QXjBdoF6AXgBd4F2QXiBeUF2wXcBeQF3AXgBeQF3wXjBeIF3QXnBeYF6AX3BeAF3QXlBecF5gXoBdcF+AXbBeQF5AXgBewF4wXlBeIF7AXgBfcF+wXnBeUF5gXnBfMF+wXlBf8F5gX5BegF/QXjBeEF+AXtBeEF6QXzBecF6QXnBeoF5gXzBfkF6QXwBe8F8AXpBeoF5wXwBeoF6QXvBfMF8gXvBfAF5QXrBf8F5QXjBf0F8wXvBe4F9gXuBfEF8QXuBe8F8gXxBe8F9wXoBfkF9gXzBe4F8gXwBecF9gXxBfQF8QXyBfQF9gX1BfMF8gX2BfQF+gXzBfUF9gX6BfUF9gXyBfoFBAbkBewF+AXkBfwFAgbrBeUF5wX6BfIF8wX6BfkF/QUCBuUF4QXtBf0F5wX7BfoF/gX3BfkF/AXkBQQG/gUIBvcF+gX7BQEG/gX5BfoF/AXtBfgF/wXrBQIGAQb7Bf8F9wUIBuwF7QUABv0F+gUBBv4FAAbtBfwF7AUIBgQGBQb+BQEG/QUABgIGAQYABvwFBAYFBvwFAgYBBv8FAwYIBv4F/AUFBgEGAgYABgEG/gUGBgMGBQYGBv4FBAYIBgUGAwYGBgkGCAYDBgcGCQYHBgMGCQYIBgcGCAYMBgUGCAYJBgwGBQYMBgYGCgYMBgkGBgYNBgkGCgYJBg0GCgYNBgsGBgYLBg0GCwYOBgoGCwYGBg8GDAYKBg4GDAYOBgsGDAYLBg8GBgYMBg8GEgYTBhEGEAYQBhAGEAYQBhAGEAYQBhAGEAYQBhAGEwYVBhEGEgYRBhUGEgYVBhQGFwYTBhIGFgYUBhUGGwYSBhQGGQYVBhMGFQYZBhYGGwYXBhIGGAYUBhYGFwYaBhMGGQYcBhYGGwYUBhgGGQYlBhwGEwYeBhkGHAYaBhcGFwYgBhwGIAYXBhsGHgYhBhkGHAYmBhYGGAYWBiYGGwYYBiYGJQYaBhwGJgYcBiAGEwYvBh4GHwYdBiIGGQYhBiUGLwYTBhoGKAYkBh8GHQYfBiMGHwYiBigGIQYeBh0GHQYjBiEGLwYaBiUGIwYuBiEGLgYlBiEGKAYtBiQGIAYnBhsGGwYnBiAGIAYbBiYGJAYjBh8GLwYlBi4GLQYpBiQGHQYsBiIGKQYtBioGKQYqBiQGHQYeBiwGKgYjBiQGLQYrBioGKwYtBjUGNQYqBisGNQYjBioGKAYiBjYGMAYtBigGLwYsBh4GKAY2BjAGNQYuBiMGLgYsBi8GNAYtBjEGMAYxBi0GLQY0BjUGMgYxBjAGNAYxBjIGMgYwBjMGNAYyBjMGMAY0BjMGMAY2BjcGOAYiBiwGMAY3BjQGOAYsBi4GOAY2BiIGOAYuBjUGNwY1BjQGNgY5BjcGOgY4BjUGNQY7BjoGNwY7BjUGOQY2BjgGNwY5Bj4GOwY3Bj4GOQY7Bj4GOQY8Bj8GPwY7BjkGPQY/BjwGOwY/BjoGOgY/Bj0GPAY5BkAGPQY8BkEGOAY8BkAGOQY4BkAGOgY9BkEGPAY6BkEGPAY4BjoGQwZEBkIGRwZDBkIGRAZHBkIGQwZHBkYGRAZDBkUGRgZLBkMGTwZEBkkGTwZHBkQGRQZDBk4GSAZGBkcGSwZGBkgGTgZDBkoGRQZRBkQGTwZJBk0GRQZOBkwGTwZIBkcGSwZKBkMGUQZJBkQGSwZSBkoGUQZFBkwGUgZLBkgGVwZNBkkGTQZTBk8GTAZOBlAGSQZRBlUGUgZIBk8GSgZYBk4GTAZgBlEGVwZTBk0GUAZWBkwGVgZgBkwGVQZRBlQGVAZRBmAGWgZTBlsGUAZOBlgGVAZgBlUGWAZKBlIGVwZZBlMGXgZSBk8GXwZTBlkGXAZQBlgGWgZPBlMGUAZdBlYGUwZfBlsGUgZeBmEGYQZYBlIGWQZXBmQGWgZeBk8GXQZQBlwGVQZXBkkGXgZaBlsGXgZbBmMGVwZoBmQGYAZWBmoGWAZiBlwGWwZnBmMGVQZoBlcGXwZZBmcGZwZbBl8GcQZVBmAGaAZZBmQGVgZdBmoGYwZmBl4GWAZhBmIGZgZhBl4GZQZZBmgGYAZqBnEGYwZnBmYGZgZnBlkGagZdBmkGVQZxBmgGZQZmBlkGbAZcBmIGawZiBmEGZgZlBm0GXQZcBmwGbAZpBl0GbAZuBmkGewZsBmIGawZhBmYGbQZrBmYGbQZvBmsGagZ1BnEGZQZoBnAGaQZ1BmoGcgZsBnsGcwZtBmUGcwZ0Bm8GcwZvBm0GbAZyBm4GawZvBnwGcwZlBn4GawZ7BmIGcgZ2Bm4GfQZ8Bm8GdgZ3Bm4GcgZ3BnYGbwZ0Bn0GbgZ3BngGcgZ4BncGbgZ4BnsGcgZ7BngGcQaABmgGgQZuBnoGewZ6Bm4GaQZuBnkGewaBBnoGZQZwBn4GgAZwBmgGaQZ5BnUGcQZ1BoIGdAZ8Bn0GfwZzBn4GhQZ7BmsGggaABnEGfwZ0BnMGbgaBBnkGewaFBoEGfwZ+Bo0GawZ8BoUGfwaMBnQGfgZwBo0GcAaABoMGjAZ8BnQGjQZwBoQGeQaVBnUGfAaGBoUGlQaCBnUGfAaMBpAGgQaVBnkGhgZ8BpAGlAaHBn8GlAZ/Bo0GggaKBoAGhAZwBoMGiQaBBoUGggaVBooGjAZ/BocGiQaFBoYGiQaVBoEGgAaKBosGhAaDBo8GiwaPBoAGiAaEBoMGhAaIBoMGjwaDBoAGkwaNBoQGlgaGBpAGkgaOBowGjAaOBpAGhAaPBpMGjAaHBpIGkgaKBo4GiwaUBo0GiwaNBpMGigaSBosGkgaUBosGkQaOBooGkwaPBosGlAaSBocGigaVBpEGmAaQBo4GiQaGBpYGlQaYBpEGkAaYBpYGkQaYBo4GiQaYBpUGlwaYBpYGlwaWBpgGmAaJBpYGtgabBpkGoAabBrYGmQaeBpoGowaZBpoGpwaZBpsGngaZBqcGngajBpoGswajBpwGswaZBqMGnQazBpwGowadBpwGtQaeBqcGogajBp4GswadBp8GtwazBp8GrwabBqAGnga1BqIGqQajBqIGnQajBqEGpQadBqEGnQalBp8GpQa3Bp8GtgavBqAGmwavBqcGqQaiBrUGpQajBqQGowalBqEGtga1BqYGrwa2BqYGowapBqoGrQajBqoGowatBqQGrQalBqQGtQavBqYGtQanBq8GqQa1BqgGqQatBqoGpQatBqsGpQarBqwGsQalBqwGtQayBqgGsgapBqgGqwatBq4GsQarBq4GqwaxBqwGtwalBrEGrQapBrIGrQayBrAGtAatBrAGtAaxBq0GrQaxBq4Gsga0BrAGsga1BrQGtwaxBrQGswa3BpkGtAa1BroGtwa4BpkGtQa2BroGtAa6BrcGvga6BrYGmQa4BrYGuAbGBrYGvga2BrsGtgbGBrsGwQa3BroGtwa8BrgGvga9BroGuQbGBrgGywa9Br4GwQa8BrcGvwbLBr4GvAbHBrgG0Qa5BrgGvAbOBscGxAa/Br4GxAa8Br8G0Qa4BscGxAbOBrwGywa8BsEGvwa8BssGwAa6Br0GvQbQBsAGvga7BsIGxAa7BsMGuwbEBsIGxAa+BsIGuwbEBsMGxAa7BsUGuwbEBsUGxAa7BsgGwAbKBroGuwbMBsgGxAbIBswG0QbdBrkGuwbGBs8GwAbQBskG1QbHBs4GzAbOBsQGugbKBuQGxga5Bt0GwQa6BuQGzwbMBrsGwAbNBsoGywbBBtIGywbWBr0GzQbABskG0Aa9BtYGywbSBtYG2QbRBscG5AbSBsEGzwbGBtsG2wbOBswGzQbYBtMG2AbNBskG3QbbBsYGygbNBuQG0wbYBtQG2AbJBtAG2wbMBs8G1QbOBtsG5AbNBtoG1AbeBtMG1AbYBt4G2QbdBtEG3wbZBscG1gbSBgEH0wYEB80G1gYBB9AG1wbHBtUG1wbfBscG0Ab1BtgG1wbVBtwG4wbXBtwG4QbkBtoGBAfTBt4G1QbbBuIG3gbYBvUG4gbcBtUG3AbhBuMGzQYEB9oG5AYBB9IG2QYGB90GAgf1BtAG2wbdBuAG4gYDB9wG4QbcBgcH4gbbBuUG7AbhBtoG5AbhBgcHAgfQBgEH9QYCB94G2wbgBuUG5AYKBwEHBAfeBgIH5wbmBusG6wbmBuoG4QbsBuMG6QbqBuYG5wbpBuYGBgfgBt0G7QboBvEG8QboBvAG7gbqBukG7gbpBucG6wbqBu4G5wbrBu4G/wbfBtcG7wbwBugG7QbvBugG2gYEBwUH8wbwBu8G8wbvBu0G8QbwBvMG7QbxBvMG9Ab0BvIG8gb0BvQG9gb2BvMG8wb2BvYG9wb4BvQG+Qb3BvQG+Qb0BvQG9Ab0BvgG9Ab4BvkG+gb6BvYG+wb6BvYG+wb2BvYG9gb2BvoG9gb6Bv0G9gb9BvsG9wb8BvgG+Qb4BvwG+Qb8BvcG/Qb9BvoG/Qb6BvoG/Qb6BvsG+wb9Bv0G/wb8BvcG9wb8Bv8GAwfiBgAH/gb9Bv0G/Qb9Bv4G/wbXBuMGCAffBv8G4gblBgAHAwcHB9wGBQfsBtoG2QbfBgYH7Ab/BuMG5AYHBwoH7AYIB/8G5QYJBwAHCQcDBwAHCAcGB98GDAcFBwQHEQcDBwkHBgcPB+AGEQcHBwMHAgcBBwoHAgcMBwQHBwcRBwoHBQcLB+wG7AYLBwgH5QbgBgkH4AYQBwkHBQcMBwsHCgcMBwIHFAcLBwwHEQcJBxAHEQcNBwoHCAcSBwYHEgcOBwYHDwcGBw4HDwcQB+AGEgcPBw4HDQcMBwoHCwcSBwgHEQcTBw0HDAcNBxMHFgcSBwsHDwcVBxAHDAcTBxQHEAcVBxEHFQcTBxEHCwcUBxYHDwcSBxUHFAcTBxYHFQcSBxYHFQcWBxcHFgcTBxcHFQcXBxMHJAcgBxgHIAclBxgHJAcYByUHJQcgBxkHJQcZByAHIAclBxoHGgclByAHJQcgBxsHIAcfBxsHJQcbBx8HHwchBxwHJgccByEHHwccByYHJgchBx0HIgcdByEHIgcmBx0HHgcmByIHIgcjBx4HIwcmBx4HHwcgByEHJgclBx8HIQcgByIHIgcnBygHIgcgByoHIgcoByMHLQcpBywHLAcpBzAHJAcqByAHJQcmBzEHIwcoBzcHMwckByUHMAc3BywHKAcnBysHMActBzgHIwcxByYHKwc2BygHNwcoBywHMAcyBy0HLgctBywHKgc8ByIHIwc3BzEHLgcoBzYHNgcrB0AHKAcuBywHKwcnB0AHMgcvBy0HOQcqByQHKQcyBzAHJwciBzwHLwcpBy0HLwcyBykHSwcqBzkHQAcnBz0HMwclBzEHLgc2BzQHNQc4By0HJAczBzoHPAc9BycHOAc7BzAHLgc0BzUHLgc1By0HKgdLBzwHNwcwBzsHQAc0BzYHPQc8Bz4HQQc4BzUHNAdDBzUHRQc6BzMHOQckB0IHSwc5Bz8HMQdFBzMHRwc7BzgHQwdBBzUHMQc3B0QHQwc0B0AHOwdEBzcHRwc4B0EHQgckBzoHRQcxB0QHPQdPB0AHQgc/BzkHTwdGB0AHRwdEBzsHRgdDB0AHSQc6B0UHVQdJB0UHTwc9Bz4HRgdHB0MHVAc6B0kHQwdHB0EHTgdFB0QHSAdCBzoHTAc8B0sHUAc+BzwHTAdQBzwHUQdHB0YHTgdEB0cHTwc+B1AHQgdSBz8HWAdOB0cHTQdIBzoHQgdIB1YHWQdIB00HSgdKB0oHSgdKB0oHVAdZBzoHSgdKB0oHSgdKB0oHTgdcB0UHRgdPB1sHUQdYB0cHWwdRB0YHTQc6B1kHTAdLB1oHVQdFB1wHVAdJB1cHWQdWB0gHUQdTB1gHUgdLBz8HWwdTB1EHVQdXB0kHVgdSB0IHUgdaB0sHWwdPB1AHVgdeB1IHXQdTB1sHWAdcB04HXAdTB10HWAdTB1wHVwdVB2EHYgdUB1cHVgdZB2AHUgdeB2QHXwdQB0wHUgdkB1oHXQdbB1AHYwdZB1QHXAdhB1UHVgdkB14HYwdgB1kHYQdcB10HYQdiB1cHXwdMB1oHYwdUB2YHZwdUB2IHVgdgB2QHXwddB1AHZwdmB1QHXwdaB2UHYwdmB2AHXwdtB10HYQddB20HYgdhB2kHbgdkB2AHaQdhB20HYAdmB24HZAdlB1oHbQdfB2UHZwdiB2oHYgdpB2oHbgdlB2QHZwdoB2YHZwdrB2gHawdnB2oHaAduB2YHcAdqB2kHbwdtB2UHbQdsB2kHawdqB3AHbwdlB24HcgdrB3AHawdyB2gHbAdtB28HaQdsB3AHbgdoB3EHaAdyB3EHbgdxB28HcgdwB3QHeAdsB28HdAdzB3IHcwd0B3cHcQdyB3MHcQd4B28HbAd1B3AHdAd1B3YHeAd1B2wHcAd1B3QHdAd2B3cHeAdxB3cHcQdzB3cHdQd4B3kHeAd2B3kHdQd5B3YHeAd6B3YHdwd2B3oHdwd6B3gHfQd8B4MHewd7B3sHewd7B3sHewd7B3sHewd7B3sHfAd9B34Hfgd9B4MHggeDB3wHjgeCB38HfweCB4oHhgeBB4AHfweQB5IHigeQB38HfweAB44HjgePB4IHfgeIB3wHigeCB3wHhgeAB38Hkwd+B4MHgAeBB44Hiwd+B4QHiAd+B4cHfgeIB4UHiAd+B4UHfgeLB4kHiAd+B4kHfgeIB4cHkgeXB38HiweIB4kHggeTB4MHiweEB4wHhAd+B5MHiweMB5EHiAeLB5EHhAeRB40HkQeMB4QHhAeTB5EHhAeNB5EHlAeKB3wHfAeIB5EHfweXB4YHjweOB5kHkQeTB3wHkweVB3wHlQeUB3wHkAeKB5QHhgeiB4EHmAePB5kHlQeTB4IHkAehB5IHogeWB4EHjweYB4IHmQeOB4EHlQebB5QHlgeaB4EHmAeVB4IHmweVB5wHoQeQB5QHhgeXB6IHkgemB5cHlgeiB54HmgeZB4EHlQeYB58HmwetB5QHnAeVB58HoQeUB60HmQedB5gHoAeZB5oHpQeiB5cHuAeaB5YHpAecB58HmwecB6QHpgelB5cHkgehB6YHoAedB5kHnweYB50HlgeeB7gHoAeaB7gHpAetB5sHoQejB6YHnwedB6oHpAefB6oHoQetB6MHpAeqB6sHvQe4B54HvQeeB6IHvQeiB6UHnQegB6oHuAeyB6AHtQemB6cHpwemB6MHpge1B6gHpwejB7UHqQemB6gHqQeoB7UHvQelB7kHowetB7UHsge8B6AHvgeuB7gHwQeyB7gHqQelB6YHqgfIB6sHoAe8B6oHvAfIB6oHvgesB64Htwe4B64HuAe3B8EHrQekB7oHvge2B6wHuge1B60HrweuB6wHugekB8gHtgevB6wHrgewB7cHpAerB8gHvge4B8IHrwe0B64Hrge0B7EHuweuB7EHsAeuB7sHuwe3B7AHugepB7UHtAevB7MHsQe0B7sHpQepB78Hrwe2B7MHtge0B7MHtge7B7QHuwfBB7cHsgfAB7wHtge+B7sHvwepB7oHvgfCB7sHuwfCB8EHwQfKB7IHwAeyB8oHpQe/B7kHwge4B70HuQfGB70HvQfEB8IHyAe8B8MHwAfDB7wHywe6B8gHvwe6B8UHxQe6B8sHvQfGB8QHwwfAB8oHwgfEB9AHwQfCB8cHwQfHB8oHxAfGB8kHuQe/B88HxAfJB9AHywfIB8MH0AfJB8YHwgfQB8cHzAfKB8cHzgfDB8oH0AfSB8cH0QfFB8sH0QfLB8MHzQfDB84H0gfMB8cHzQfRB8MHzwfGB7kHzgfKB8wHzwe/B9MH0we/B8UHxQfRB9MHzgfMB9QH1QfNB84H0QfNB9UHzwfQB8YH2gfMB9IH1AfMB9oH2AfRB9cH1gfQB88H1QfXB9EH0AfWB9IHzgfUB9UH2gfSB9YH0QfbB9MH1gfPB9MH0wfcB9YH2wfRB9gH1gfcB9oH1wfZB9gH1wfVB9QH2gfXB9QH0wfbB9wH2gfdB9cH2QfXB90H2AfZB90H3QfaB9wH3gfYB90H3AfbB94H2AfeB9sH3AfeB98H3wfdB9wH3wfeB90HKwgqCOYHKwjmB+AH4gcrCOAHKgj2B+EH5gcqCOEH5gfiB+AH9gfmB+EH+AfiB+YHKwjiB/gHLggrCPgH8AfmB+UH4wfjB+MH4wfjB+MH5AfkB+gH5AfkB+QH5gf2B+UH9gfwB+UH5gfwB/gH4wfjB+MH4wfjB+MH5wfnB+cH5wfpB+cH5AfkB+gH5AfkB+QH9gcqCOsH8Af2B+8H5wfnB+cH6QfnB+cH6gfqB+oH6gfqB+oHKgjuB+sH7gf2B+sH9gfuByII7AfqB+oH6gfqB+wH7QftB/IH7QftB+0H9AcuCPgHLgj0B/MH9gf3B+8H9wfwB+8H8Af3B/gH8Qf5B/wH7QftB/IH7QftB+0H9AcfCPMHHwguCPMHKggvCP4H7gcqCP4H9Qf1B/8H9Qf1B/UHIgj3B/YH+Af3ByII+gf5B/EH+wf7B/sH+wf7B/sH/Af5B/oH/Qf9B/0H/Qf9B/0HHwj0B/gHLwjuB/4H9Qf1B/8H9Qf1B/UH+wf7B/sH+wf7B/sHAAgACAAIAAgACAAIBggICAEICAgOCAEI/Qf9B/0H/Qf9B/0HAggCCAIIAggCCAII7gcvCAoIFgjuBwoIBAgDCBAIBQgFCAUIBQgFCAUIAAgACAAIAAgACAAIBwgHCAcIBwgHCAwIAggCCAIIAggCCAIICQgJCAkICQgJCAkILwgWCAoIBQgFCAUIBQgFCAUICwgLCAsICwgLCAsIBwgHCAcIDAgHCAcIDQgNCA0IDQgNCA0IDggICAYIIwj4BxQICQgJCAkICQgJCAkIFggvCBUI7gcWCCIIAwgECA8IEAgDCA8IEQgRCBEIEQgRCBEICwgLCAsICwgLCAsIEggSCBIIEggSCBIIDQgNCA0IDQgNCA0IEQgRCBEIEQgRCBEIEggSCBIIEggSCBIIGggXCBMIIwgUCBgIIQgWCBUIFwgaCBkI+AccCBQIFAgjCBgIIwgfCPgHLwghCBUIFgghCCIIGggTCBkIFAgcCB0IIwgUCB0IHwgjCB4IIQgvCCAIGwgcCPgHHAgjCB0IIwgwCB4IMAgfCB4ILwgsCCAILAghCCAIIQgpCCIIIggbCPgHIQgsCCQIKQghCCQILAgpCCQIKQgoCCUILQgbCCIIGwgtCCYIHAgbCCYIMAgjCBwIKAgpCCcIKQgtCCIILQgcCCYIKQgsCCcILAgoCCcIKAgtCCUIKAgsCC0ILQgpCCUIMAguCB8IMAgcCC0ILQgsCDAIKggyCDEIKgguCDIIKwguCCoIMQgzCCoILwgqCDMILwgzCCwIMAgsCDcIMgguCDAINwgsCDMIMQgyCDYIMggwCD0IMQg2CDUISAgyCD0INQgzCDEIMgg0CDYIOAg3CDMISAg0CDIIMAg3CDoINghHCDUISAg9CDkIOgg3CEAINAg7CDYINwg5CEQIOQg3CDgIMwhYCDgIMwg1CFUIRAhACDcIMAg6CD0IOwg8CD8IOwg/CDwIPwhCCDsIOQg4CEgIOwg+CD8IOwg0CD4INAg/CD4ISAg4CFIIQgg2CDsIPwg0CEEIVQhYCDMIPwhBCEIIQghHCDYIQghBCDQIQgg0CEMINAhNCEMIRghVCDUIQwhNCEIINAhICE0INQhHCEYIRQg5CD0IRQg9CDoIOghACEoISQhACEQISQhRCEAIUQhKCEAIOQhOCEQISghRCEsIRwhTCEYIRQhOCDkIRAhOCEkIOghKCFAITQhICFIIUgg4CE8IRwhCCF8IUAhFCDoISwhRCEwIWwhRCEkITAhRCF0ISwhQCEoIQghNCFIITwg4CFgIXwhCCFIISQhOCFsIRQhQCE4ISwhMCFQISwhUCFkIUwhHCF8IWwhhCFEITwhYCGIITAheCFQIUAhLCFkIXghMCF0IVwhUCF4IXAhUCFcIVAhcCFYITwhaCFIIXAhZCFYIWQhUCFYIVQhGCGMIXghcCFcIZghOCFAITghmCFsIYwhYCFUIYghYCGgIZAhgCFsIUQhhCF0IWwhgCGEIUwhjCEYIWQhmCFAIXAhmCFkIXwhnCFMIWghfCFIIYAhkCGIIYAhlCGEIXQhhCF4IWAhjCGgIYghoCGAIXghmCFwIZwhjCFMIaQhPCGIIXwhaCGcIZghkCFsIYAhoCGUIYQhqCF4IXghqCGYIaQhiCGQIaAhrCGUIZQhqCGEIbwhjCGcIbAhaCE8IbwhrCGgIZwhaCGwIaQhsCE8IZghpCGQIbwhoCGMIaQhmCGoIZwhsCG0IaghuCGkIbghqCGUIbwhtCGsIZwhtCG8IbghsCGkIZQhrCG4IbghtCGwIawhtCHEIcQhuCGsIcAhuCHEIbQhuCHMIcQhtCHQIbQhyCHQIcghtCHMIcghzCG4IcAhxCHQIdAhyCHAIcghuCHAIcQhwCHUIcAhxCHUIeQh2CHcIeQh3CHYIeAh3CHkIeQh7CHgIdwh4CHsIegh5CHcIegh9CHkIewh5CH0Iegh3CHsIfgh6CHsIfQh8CHsIfgh/CHoIgQh8CH0Ifgh7CHwIegiGCH0IgQiACHwIegh/CIIIegiCCIYIjgiBCH0Ijgh9CIYIgAh+CHwIhQiCCH8IfgiDCH8IfgiACIMIgQiNCIAIgQiOCJ4IfwiDCIUIigiICIkIgwiACIcIiAiECIkIhwiACI0IkQiICIoIgQieCI0IhgicCI4IkAiHCI0IkQiLCIgInAiGCIIIjQieCJIInwiCCIUIiwiMCIgIlAiECIgIkQiYCIsIjQiRCJAIhAiUCJcIlwiUCI8IjwiWCJcIlgiUCJUIlAiWCI8IngiZCJIIkwiTCJMIkwiTCJMIlAijCJUIkwiTCJMIkwiTCJMImwijCJQIowiWCJUIlwiWCKMIgwihCKYIpgiFCIMIhwiQCKQIjAiiCIgIiAidCJQInQibCJQIqAiKCIkInAiCCLkInwi5CIIInQijCJsInQiICKIIiwiYCIwIowiECJcIkgiRCI0ImgiiCIwIqAigCIoIigigCKgIpAiDCIcIqAiRCIoIhQilCJ8IkgiYCJEIiQiqCKgIuQiuCJwIjgicCJ4IjAiYCKcImgiMCKcItAilCIUIsgidCKIImQiYCJIImgi4CKIIsgijCJ0IsAinCJwIpAihCIMIpwiZCJ4InAinCJ4Ipgi0CIUItgimCKEIkAiRCLMImQinCJgIsAicCK4IowipCIQIswikCJAIqQiJCIQIsAisCKsIsgipCKMIqQiqCIkIkQioCLMIqwisCLUIrgi1CLAIqwi1CK0ItQisCLAIvwinCLAIsQirCK8Iqwi1CK8IrQi1CKsIsQivCLUIqwixCLAIuAiyCKIIvwiwCLEIoQikCLYIpwi4CJoItAimCMAIvgizCKgIvQikCLMIrgi5CLsItQiuCLsIvwi4CKcIvAipCLIItwi0CMAIsQi1CL8Iugi2CKQItgjACKYIqgi+CKgInwjBCLkItAjBCKUInwilCMEIvQizCL4Isgi4CMYItgi6CMAIpAi9CLoIwwi1CLsIvAiyCMYIqgjECL4ItQjDCL8ItwjBCLQIqQi8CKoIugi9CMIIvAjGCMUIwQi3CMkItwjACMgIxQi3CMgIvgjECL0Ixgi4CL8IxAjCCL0IuQjBCLsIwwjGCL8Iwwi7CMEIvAjECKoIwgjACLoIyQi3CMUIyAjACMIIyQjGCMMIxgjJCMcIwwjBCMkIxQjHCMkIyAjCCMQIxgjHCMUIyAi8CMUIvAjICMoIyAjECMoIygjECLwIzQjPCMsIywjPCMwIywjMCM0IzQjMCNEIzgjRCMwI0QjQCM0I1gjPCM0IzgjSCNEIzQjQCNUIzQjVCNYI1QjQCN4IzAjPCNMI2wjSCM4I0AjRCNII1AjTCM8I0gjYCNAIzwjZCNQI1wjWCNUIzwjWCNkIzAjTCOMI0gjbCNgI4wjTCNQI2wjOCMwI2wjMCOMI1AjZCNwI1QjeCNcI3gjQCNgI3gjaCNcI1gjXCNkI1wjdCNkI2QjdCN8I1wjgCN0I2gjgCNcI2QjfCOQI3AjZCOQI7QjaCN4I3AjhCNQI4AjaCOUI5QjrCOAI3QjgCOkI4QjjCNQI2gjnCOUI6gjeCNgI5AjhCNwI3QjvCN8I2wjjCNgI4wjqCNgI5wjiCOUI3QjpCO8I5wjlCOII5wjmCOUI6QjgCOsI3wjyCOQI6wjlCOYI6wjmCOcI6AjnCNoI6gjtCN4I8AjkCPII2gjtCOgI5AjwCOEI4wjhCPcI6gjjCPEI9wjxCOMI5wjoCOsI6gjsCO4I7QjqCO4I7AjtCO4I6wjoCPQI9AjpCOsI7wjpCPQI3wjvCPII8wjvCPQI6Aj1CPQI8Qj3CPsI7wjzCPII8Aj3COEI6AjtCPUI7AjqCPEI9gjwCPII9gj3CPAI9QjtCOwI8gjzCPYI+Aj1COwI/wjxCPsI9Aj1CPoI+gj1CPgI+Qj4CP8I+gj4CPkI9Aj6CPMI8wj6CPYI9wj2CPsI8Qj4COwI+AjxCP8I+gj5CPYI/Aj5CP8I+wj2CP0I/Qj2CPwIAAn2CPkI9ggACfkI9gj5CAEJAQn8CPYI+Qj8CAEJ/Aj+CAIJ/AgCCf0I/gj8CAMJAgn+CP0I/wj+CAMJ/Aj/CAMJBAn7CP0IBAn9CP4I/gj/CAQJ+wgECf8I+wj/CAUJBQn/CPsICQkICQYJBgkHCQkJCAkHCQYJCAkKCQcJBwkOCQkJCQkLCQgJBwkKCREJDQkKCQgJDwkOCQcJDgkMCQkJEQkKCQ0JDAkLCQkJEAkOCQ8JCAkLCQ0JEQkPCQcJEQkNCRIJCwkSCQ0JGwkOCRAJDAkOCS4JIgkSCQsJFgkUCRMJFgkTCRwJDgkbCS4JFAkXCRMJEAkPCRgJDAkuCTQJDAkiCQsJHwkWCRwJFQkTCRcJEwkkCRwJEQklCQ8JHQkUCRYJEgkeCREJFAkZCRcJFAkgCRkJHwkdCRYJIAkXCRkJJgkXCRoJFwkgCRoJIAkmCRoJEAkqCRsJHgklCREJKgkQCSkJJgkVCRcJFAkdCSAJIwkkCRMJDwklCRgJEwkhCSMJFQkwCRMJEgkkCR4JLAkwCRUJNAkiCQwJJAkSCT0JJwkcCSQJHQkmCSAJHgkkCSMJEwkwCSEJIwkhCR4JEgkiCTEJGAkpCRAJMAkoCSEJIQklCR4JJgkdCS8JNAkyCSIJFQkmCSwJMQk9CRIJIQkoCSUJOgkfCRwJJgkvCSwJGAklCS0JKQkYCS0JJwk4CRwJHwlACR0JKwkxCSIJHAk4CToJLQklCSgJKwkiCTYJLQk3CSkJNAk1CTIJQAkvCR0JNgkiCTIJOAknCUIJLgkbCTQJQgknCSQJMAktCSgJKgkzCRsJJAk9CUIJGwkzCTQJKgkpCVAJMAksCS0JQAkfCToJLwk+CSwJOwk9CTEJNwk5CSkJMQkrCTsJQAk6CUsJQQk3CS0JLQksCUEJKwk2CTsJOQk3CUcJUAkzCSoJRwk3CUEJKQk5CVAJPQk7CUMJNAkzCTwJQAlFCS8JNQlICTIJOgk4CUYJQQlMCUcJLwlFCT4JMglICTYJSAk7CTYJNAk8CTUJNQlKCUgJLAlMCUEJQwlCCT0JRAlDCTsJRglLCToJTAksCT4JUAk8CTMJQwlECUIJOAlCCUYJSAlECTsJTglACUsJSQk1CTwJSQk/CTUJPwlKCTUJTglFCUAJTAk+CUUJRQk/CUkJRglCCUQJSgk/CUUJRwlQCTkJSglPCUgJTQlJCTwJTglLCUYJRAlICU8JRglECU4JTAlNCUcJTQk8CVAJTQlMCUkJSQlMCUUJTwlOCUQJUQlQCUcJTQlRCUcJTQlQCVEJUglKCUUJTglSCUUJTwlKCVIJTglPCVIJUwl0CXsJewlWCVQJewlVCVMJVQl0CVMJVQlUCVYJVQl7CVQJagl0CVUJaglVCVYJcwl0CVcJVglxCVgJYAlWCVgJZgl0CWoJZglXCXQJcwlXCWYJcQlgCVgJVglgCWoJWQlZCVkJWQlaCVkJWwlzCWYJXAlgCXEJWQlZCVkJWglZCVkJZQlmCWoJbwlbCWYJbwlzCVsJcQlfCVwJXwlgCVwJYgldCWEJYgleCV0JYgljCV4JZwlvCWYJaAlgCV8JaQlgCWgJYAlpCWoJYgliCWEJZAliCWIJYwliCWQJawllCWoJZgllCWcJbwlnCWUJaQloCV8JawlqCXIJZQlrCXUJbwllCWwJbQlpCV8JcglqCWkJawlyCW4JdQlrCW4JZQl1CWwJbwlsCXUJcgltCV8JcglpCW0Jcgl1CW4JfAl2CXAJeQl8CXAJdgl5CXAJVgl7CXEJcQl7CV8Jbwl1CXMJeAl0CXMJcwl1CXgJXwl7CXoJeAl/CXQJcglfCXoJdQlyCXoJegl9CXUJfwl7CXQJdwl5CXYJeAl1CX0Jegl7CYAJfAl+CXYJegmCCX0JgQl8CXkJgwl3CXYJgQl5CYQJhwl5CXcJfAmGCX4JfAmBCYYJegmACYIJeQmHCZUJeAmWCX8JgAl7CYUJdgl+CZAJfwmKCXsJigmFCXsJfQmCCYgJfQmRCXgJgwl2CZAJkQmMCXgJhwl3CYMJggmACYkJfwmWCYoJlQmECXkJhgmQCX4JkQl9CYgJgQmJCYYJgQmECYkJjAmWCXgJggmJCYsJjAmRCaMJgAmFCYkJhgmJCZwJjQmJCYQJggmOCYgJngmVCYcJiQmNCYsJkQmICY8JjgmCCYsJlQmdCYQJlgmkCYoJnAmQCYYJgwmQCZcJgwmXCaAJjQmdCYsJgwmzCYcJhAmdCY0JlgmMCaMJlgmTCaQJiwmdCY4JmwmFCYoJswmDCaAJjwmICY4JkAmYCZcJmgmYCZAJsgmjCZEJnAmoCZAJhQmbCZIJhQmSCYkJlAmTCZYJkgmcCYkJkwmZCaQJtwmOCaYJswmsCYcJjwmyCZEJpAmbCYoJjgmdCaYJlQmeCaoJswmgCa8JmQmTCaUJpQmTCZQJnwmYCZoJkAmoCZoJlwmvCaAJuQmVCaoJrgmOCbcJrAmeCYcJoQmyCY8JmAmfCZcJrgmPCY4JqAmfCZoJownACZYJqQmoCZwJkgmpCZwJogmmCZ0JnQmVCbkJnwmnCZcJmwmkCasJnwmoCacJkgmbCasJlAmtCaUJmQm2CaQJwAmUCZYJrgnWCY8JoQmPCdYJtgmrCaQJlwmnCbsJpQm2CZkJkgmrCakJtAmmCaIJswmxCawJrwmXCbsJuQm0CZ0JqAmpCcwJuQmqCZ4JuwmnCbAJogmdCbQJqQmrCcIJvAmeCawJsAmoCcwJqwm2Cb8JtgmlCa0JpwmoCbAJlAnACb0JlAm9Ca0JwAmjCbIJngm8CbkJwgnMCakJwwm3CaYJtQmsCbEJrwmxCbMJuQm8CcYJpgm0CcMJugmsCbUJrgm3Cb4JrAm6CbwJwQnWCa4J3Am2Ca0Jqwm/CcIJwwm+CbcJygmyCaEJrQm9CdwJsAnNCbsJrgm+CdoJuwnvCa8JuAm6CcUJygnACbIJ7wmxCa8JuQnUCbQJuQnGCcQJxwnPCcEJ3wmhCdYJvwm2CdwJyAnNCckJvgnDCdUJxQm6CbgJuQnECdcJywnMCcIJugm1CbwJywnCCb8JzQnTCbsJvQnACdsJsQnQCbUJzQnICdMJ1QnRCb4JwQnPCccJ5gnECcYJwQmuCdoJ2wneCb0J0QnSCb4JuwnTCe8J1gnBCdoJyQnTCcgJzQmwCcwJ3Am9CdgJwAnKCdsJxgm8CeYJ2Am9Cd4J1AnOCbQJ1AnDCbQJ1wnUCbkJ7wnQCbEJ5gm8Ce4J1Am0Cc4J1QnSCdEJwwnUCdUJzQndCckJvgnSCeEJzQnMCdkJ1wnECeYJ0wnJCd0J7QncCeAJoQnfCeoJ4gnMCcsJ1QnUCdcJ6AncCdgJ5wnjCdkJ7gm8CbUJzgnkCcsJygmhCeoJ3wnWCdoJvgnhCdoJtQnQCSMK0gnVCeEJ4wnrCdkJ5wnZCcwJ3AnoCeAJ2QnsCc0J1AnkCc4JCAq/Ce0J7AndCc0J5AnOCcsJzgnkCdQJ3gnbCdgJ4gnLCeQJ3AntCb8J7wnTCekJ6QnTCd0JvwkICssJ2QnrCewJCAriCcsJ4gnkCcsJ5QnbCcoJ7gnyCeYJSQrfCdoJIwruCbUJHArqCd8J5gnyCdcJ2wkGCtgJ4gklCswJ5wnMCSUK2AkGCugJ5QkGCtsJ3QnsCekJ4QnVCTUK4wnnCesJ6AntCeAJ6wnnCRsK2gnhCUkK6AkGCvMJJQoRCucJNQrVCfEJ8QnVCdcJHwrtCegJKgrxCdcJ7gn0CfIJ7wnpCSIK7AkYCukJKgrXCfIJNQpJCuEJ6wnwCewJ4gkICiUKEQobCucJAQr0Ce4JIgrQCe8JygnqCSAK6wkbCvAJFwroCfMJ9An1CfIJ/AnyCfUJIArlCcoJKArtCR8KHQoqCvIJ9wn8CfYJ/An1CfYJFwofCugJ+AkiCukJ9Qn0CfoJ9gn1CfcJ9Qn5CfcJ+Qn8CfcJ8gn8CfsJ9An/CfoJ/wn1CfoJ+Qn1CQAKAAr8CfkJ/AkNCvsJDQryCfsJ8gkNCv0JFwrzCQYK7AnwCRgKHArfCUkK/wn0CQEKAAr1Cf8JDQr8CQAK/QkNCv4J/gnyCf0JHQryCf4JKAoICu0J/wkNCgAKIgojCtAJ/wkBCgIK/gkNCgMKHQr+CQMKAQoECgIKAQoKCgQKBAr/CQIK/wkECgUKBQoNCv8JDQodCgMK7gk/CgEKBAoKCgcKBAoNCgUKCQoHCgoKDAoECgcKEwoNCgQKFQoNChMKGArwCRsKCwoKCg4KCgoLCgkKCQoLCgwKCQoMCgcKBAoMChMKFwoGCuUJGgoKCgEKCgoaCg8KFAoKCg8KCgoSCg4KCgoUChIKEgoLCg4KDAoSChMKDAoLChIKDQoVChAKEAodCg0KGgoUCg8KEAoVCh0KIQoSChQKEgohChYKFQoSChYKEgoVChMK+AnpCTIKIQoUChoKIQoVChYKKgoVChkKKgodChUKFQohChkKGQohCioKIQoaCgEKMgrpCRgKIAorCuUJJQobChEKPwohCgEKIgr4CV0KPwruCSMKGwowChgK8Qk0CjUKPwoqCiEKKwogCiQKJwoICh8KHgosCjoKCAonCiUKMQoeCiYKMAobCiUKIwoiCkUKSQpZChwKNArxCSoKLAoeCjwKHgo6CiUKHwomCh4KHwoICk4KMQozCh4K5Qk4ChcKJQopCh4KHwoeCikKOgowCiUKMwovCjcKFwpACh8K+AkuCl0KKgpDCjQKNwo8CjMKJAotCisKUAokCiAKHgozCjYKQAomCh8KTgooCh8KKwo4CuUJMgouCvgJIApaClAKLwozCjkKUAotCiQKJwpOCggKLAo8CkgK6glaCiAKPAoeCjYKggpaCuoJPgoyChgKOgosCkgKKAo9CggKJwopCiUKYAojCkUKMQo5CjMKQAoXCkoKQgpdCi4KMgpCCi4KggrqCRwKggocClkKTQo0CkMKHwpOCicKMAp+ChgKQwoRCk0KNApNCj0KUQovCjkKMwovClEKSgoXCkYKeAooCk4KaQo7CjcKVApGChcKVgpDCioKOAorCkAKKQpOCh8KcAo0Cj0KNwpBCjwKPgpMCjIKMwppCi8KUgpACisKXQqFCiIKTgonChEKUwoYCn4KEQonCk0KOwpBCjcKOQoxCksKNgozClEKTQonCggKOQpECk8KNwovCmkKIgqFCkUKPQpNCggKPQooCnAKRAo5CksKOgpMCjAKTgopCicKcAo1CjQKUgorCi0KVAoXCjgKUwo+ChgKOwpHCkEKPAppCjMKMApMCj4KQQpICjwKXgpCCjIKMgpMCl4KgwpHCjsKUQo5Ck8KfQpOChEKeApwCigKNgppCjwKJgpLCjEKfQoRCkMKQwpWCn0KPwojCmAKRApXCk8KXwprCkEKeQpJCjUKOApACkoKRApLCmEKOgpICmsKUQppCjYKawpMCjoKWwotClAKawpvCkwKfgowCj4KYgpAClIKYgomCkAKVQotClsKQQprCkgKRAplClcKRwpfCkEKfAprCl8KLQpoClIKWApCCmYKTApvCl4KUwp+CmcKnQo/CmAKYQplCkQKJgpiCksKXQpCClgKVgoqCmcKhwpnCioKPwqHCioKLQpVCmgKNQpwCnkKXApbClAKUAp7ClwKewpbClwKYgphCksKewpjClsKYgpSCmgKWwp7CmQKewpbCmMKZgpqClgKWwqiClUKewpbCmQKnwp8Cl8KWwp7Cm4Kcgp1CmYKVQphCmgKaAphCmIKXQpYCmwKbQo+ClMKQgpeCmYKjQpZCkkKiQpaCoIKdQqACmYKcQpdCmwKTgp9CpUKkApgCkUKnwpfCkcKaQp6CjsKawp8CnIKUQpPCo8KagpmCoAKagpxCmwKWApqCmwKegqDCjsKcgpmCmsKZgpvCmsKewpQCloKPgptCn4KcQpqCnMKjQpJCnkKewqiCm4KYQqRCmUKOAqpClQKfwqVCk4KRQqFCpAKbgqiClsKcQpqCnQKcwpqCnEKbwpmCl4KcAp4CogKTgqVCngKagqWCnQKlgpxCnQKcQq4Cl0KlQp/Ck4KYQpVCpEKTwqGCo8KXQq4CoUKcAp4CnkKVApKCkYKqQo4CkoKcAqICngKnQqHCj8KhAp2CncKVQqiCo4KeQp4CpUKjgqRClUKagqACpYKiAp3CnYKfQpWCmcKkwp7CloKdwqBCooKWgqJCpMKVAqpCo0KmQp2Co8KhAqPCnYKhAp6CmkKdwqICn8KmwqCClkKcgqUCnUKZwqHClMKegqnCoMKdwqKCoQKhwptClMKRwqDCp8KmQqICnYKjQp5ClQKfAqfCpQKmgqGCk8KVwqaCk8KfAqUCnIKjgqiCowKdQqXCoAKhgqZCo8KgQp3Cn8KjQqbClkKoAqZCoYKSgpUCqgKZwqVCn0KiAqkCn8KZQqeClcKoQqKCoEKngplCpEKkgqECmkKcQqWCrgKpwp6CoQKkgpRCo8KiAqZCqQKYAqQCrIKUQqSCmkKlwqWCoAKhAqKCpgKhQq+CpAKbQpnCn4KjgqMCpEKqgqJCoIKZwqlCpUKkQqMCq4KqQpKCqgKoQqYCooKrwqDCqcKeQqVCqUKYAqyCp0KowqgCoYKmAqcCoQKlwp1CpQKqgqCCpsKkQqzCp4KVAp5CqgKogp7Cq4KhAqcCqcKjwqECpIKeQqlCqgKmwqNCr8KpAqLCpkKmgqjCoYKmQqLCqQKgQp/CqEKjAqiCq4KmQqgCqQKVwqsCpoKsQq4CpYKhQq4Cr4KuQqHCp0KnwqDCrUKkwquCnsKZwptCrYKsQqWCpcKrgqzCpEKsAp/CqQKoQp/CrAKrAqkCqAKsgqQCr4KqQq/Co0KbQqmCrYKbQqHCrkKVwqeCqwKrwqcCpgKoAqaCqwKowqaCqAKpAq7CrAKtQqvCpgKqgqTCokKsQqXCpQKbQq5CqYKvAqUCp8KqwqwCq0KpwqcCq8KtQqDCq8KoQq0CpgKtgqlCmcKswq6Cp4Kngq6CsQKsAq7Cq0KsQqUCrcKnQqyCrkKoQqwCrQKpQrACqgKtAqwCqsKtgqmCrkKrgq6CrMKvAq3CpQKtQqYCrQKrQq7CqsKqAq/CqkKngrdCqwK3QqeCsQKsQq3CrgKrgqTCr0KwAq/CqgK1gqfCrUKtwq+CrgKrAq7CqQKwQqrCrsKkwqqCr0KugquCr0KxQq1CrQK3gq7CqwKugrHCsQK1grMCp8KwwrBCrsKrArdCt4KxgqyCr4K3grDCrsKwwrACqUKxQqrCsMKqwrFCrQKwQrDCqsKtwrCCr4KygqqCpsKqgrRCr0KywqlCrYKwgrJCr4KvwrACtAKmwq/CsoK0gq5CrIKvgrJCsYKywq2CrkKwwqlCssKtwq8CsgK0grLCrkKvArNCsgK0QqqCsoKxgrSCrIKnwrMCrwKvwrQCsoKxQrWCrUKwgrPCskKxwq6Cs4K0wq6Cr0K0QrTCr0KzArNCrwK3grACsMKxQrDCssKyArCCrcKywrVCsUKzgq6CtMK2wrQCsAKyArPCsIK0graCssK2grVCssKxwrdCsQKxgrJCtcK2QrHCs4KzQrMCtQKzQrUCsgKyArUCs8KyQrPCtcK3grbCsAK2wriCtAK0AriCsoK1ArMCtgK2ArWCsUK3ArdCscK3ArHCtkK2QrOCtMK0wrRCtkK3ArbCt4KzArWCtgK3grdCtwK1ArXCs8K1QrYCsUK0grGCtcK3wrRCsoK2ArVCuAK3wrZCtEK2grSCuEK4wrKCuIK0grXCuEK1wrUCtgK3ArZCt8K3wrKCuMK4grbCuMK1wrYCuAK2wrcCuMK5ArhCtcK1wrgCuQK3wrjCtwK2grhCtUK5ArgCuUK1QrlCuAK1QrkCuUK5ArVCuYK5grVCuEK5ArmCuEK6wroCucK5wrqCusK6ArqCucK6groCukK6ArtCukK6grpCu0K6wrtCugK7grrCuoK7QrsCuoK7grqCuwK6wr3Cu0K6wruCu8K8ArvCu4K8QruCuwK9grvCvAK7ArzCvEK9QrrCu8K9Qr3CusK9gr8Cu8K9QrwCu4K8wrsCu0K+QryCvQK9QruCvEKAwvtCvcK9QrxCgML9grwCvgK9wr1CgML+wr4CvAK9AryCvgK9Ar4CvsK/QryCvkK7wr8CvUK8Ar1Cv8K7QoDCwIL+AoGC/YKAgvzCu0K9Qr8Cv8K+QoBC/4KAQv5CvoK+Qr0CvoKAQv6CvQKBgv8CvYK8goGC/gKCQvyCv0K8woDC/EK+Qr+CgEL+QoBCwALAQv5CgAL8Ar/CvsKAQv0Cv8KBQv9CvkK+wr/CvQK8goJCwYL/wr8CgYLAwvzCgIL+QoBCwULBQsEC/0KCQv9CgQLCQsECwULBQsBCwcLCQsICwYLBQsHCwkLCAv/CgYLAQsJCwcLCwsBC/8KCAsLC/8KCgsBCwsLAQsMCwkLCQsMCwgLAQsKCwwLCwsMCw4LCwsICwwLDAsLCw4LCwsMCw8LDAsNCw8LDAsKCw0LDQsLCw8LDQsKCxALCwsNCxALCgsLCxALFAsSCxELHAsUCxELEgscCxELHAsSCxMLFQsTCxILFQscCxMLFQsSCxQLHAseCxQLFQsUCxcLGQsVCxcLGAsVCxYLFAseCxcLGAscCxULFgsVCxkLGAsmCxwLGgsXCyILIgsXCx4LGwsWCxkLGgsZCxcLGwsYCxYLGAsbCyYLIAseCxwLGQsaCygLGwsZCyULGQsoCysLIAsdCx4LIQsdCyALJQsZCysLPgscCyYLJwsaCyILIQsfCx4LIQseCx0LHgsfCyMLHwshCyMLJQsmCxsLIwsiCx4LGgsnCyQLHAs+CyALJwsiCyMLIwsqCycLJAsoCxoLJAsnCyoLIQsgCz4LKAstCysLKAskCy0LJAsqCykLIwsvCyoLKQsqCzgLJQssCyYLOQslCy4LNgstCyQLOQssCyULJQsrCy4LNQsrCy0LLws6CyoLPgsmCywLOAskCykLIwshCy8LOAsqCzoLNQstCzMLLgsrCzULJAs4CzYLNQszCzcLMQstCzYLLQsxCzALMAszCy0LLQszCzILMwstCzILNQs3Cy4LMAsxCzMLMQs0CzMLLwshC0cLMQszCzQLQgssCzkLIQs+C0cLNgs4CzsLLwtHCzoLNgs7CzELPQs5Cy4LOgtHCzgLRQs8Cy4LLgs8Cz0LNwszC0YLPgtLC0cLPAs/Cz0LQAs/CzwLOws4C0QLQAs8C0ULRQs/C0ALPwtFC0ELPQs/C0ELRQs9C0ELRgszCzELRwtDCzgLOwtEC0MLRQsuCzcLOQs9C0oLOAtDC0QLRQs3C0YLRQtKCz0LOwtGCzELQgs5C0oLOwtDC0gLSws+CywLQwtHC0gLOwtNC0YLSwtOC0cLRgtKC0ULSAtHC04LTQs7C0gLSwssC0wLRgtJC0oLTAtCC0oLTAssC0ILTQtJC0YLTgtMC0kLSQtPC04LTQtIC04LTQtQC0kLTAtKC0kLTgtLC0wLUQtOC08LUAtNC1MLTQtRC1MLTgtSC00LTQtSC1ELUQtSC04LUQtQC1MLVAtQC1ELUAtUC08LUAtPC1ULSQtQC1ULTwtUC1ELSQtVC08LWgtXC1YLWQtaC1YLVwtZC1YLWQtXC1gLXgtaC1kLWAtXC1wLWgtbC1cLWwtdC1cLYgtaC14LXQtcC1cLXgtZC1gLWAtcC18LYgtbC1oLXgtjC2ILZgtcC10LYAtdC1sLXgtYC2QLWwtiC2ELawtbC2ELZgtdC2ALXwtkC1gLZQtoC2kLYAtnC2YLWwtrC2ALZAtjC14LYwtsC2ILaAtlC2cLaAtgC2sLaAtnC2ALYQtiC2sLfgtnC2ULaAtqC2kLYwtkC20LZgt5C1wLXwttC2QLZwt5C2YLXwtcC3QLaQt+C2ULcAttC18LYwttC2wLeQt0C1wLYgtyC2sLbgtiC2wLewt+C2kLegt5C2cLYgtuC3ILXwt0C28LcAtfC28LcAtvC3QLagtoC3wLawtyC2gLcAt0C3ELcAtxC3QLbQtwC4cLcAt0C3MLdgtwC3MLdAt2C3MLfwt7C2kLdQt0C3gLdQt2C3QLlgtsC20Lagt/C2kLZwt+C3oLdAt5C3gLfAtoC3ILfwtqC3wLdQt4C3YLeAt3C3YLeAt2C3cLfAtyC4ELcAt2C3gLfQt6C34LhwuGC20LeQt6C4ALeAt5C4ALhwtwC3gLhguWC20Lgwt8C4ELfQuAC3oLgAuHC3gLhAt9C34LcgtuC4ELewt/C4ULewuFC34LgQtuC4ILfAuKC38LfQuOC4ALfAuDC4oLhAt+C4ULbAuJC24LgwuBC4ILgguPC4MLiQuCC24LlguJC2wLiAuFC38LjguHC4ALiQuPC4ILiAt/C5ALfwuKC5ALhAuOC30LjQuHC44LiguDC48LiAuMC4ULhQuXC4QLjguEC5cLlguGC5ULlQuGC4cLiguRC5kLjAuIC4sLjAuIC5ALjwuRC4oLiwuIC4wLjAuUC4ULjAuQC5ILlwuFC5QLoAuJC5YLkQuPC6ALlwuNC44LiQugC48LkwuQC4oLkguQC5MLlwuYC40LlQuHC40LjAuSC5QLjQuYC5ULlQugC5YLkwuKC5kLlAuSC5wLnAuXC5QLmAueC5ULmQudC5MLnguYC5cLkgudC5wLmwuZC5ELlQueC6ALnguXC5wLmQufC50LnQuSC5MLmgufC5kLnwuaC5sLmguZC5sLoAubC5ELnguhC6ALoQueC5wLnQuhC5wLnwubC50LoQudC5sLmwugC6ILoguhC5sLogugC6ELpAulC6MLpQukC6MLpgulC6gLrAukC6ULpAuoC6ULpQunC6wLrAurC6QLpwulC6YLqwuoC6QLpwumC7ILrAutC6sLswunC7cLqwuuC6gLqAuwC6YLrAunC7MLqAuvC7ALsAuyC6YLqgutC6wLrguvC6gLswuqC6wLrQuuC6sLtAutC6oLqQuqC7MLsQuqC6kLsQupC7gLswu4C6kLtwunC7ILtAuqC7ELrgu9C68Lswu3C7ULrgutC70LwQutC7QLwAuyC7ALwAu2C7ILyQuzC7ULtgu3C7ILyAu0C7ELrQu5C70LuAuzC8kLtwu2C8oLtAvIC8ELuQutC8ELuAvEC7ELrwvAC7ALyAuxC74LxAu+C7ELugu2C8ALwQvLC7kLvAu2C7oLuwu2C7wLuwu/C7YLugu7C7wLygu1C7cLugvCC7sLvwu7C8ILtgu/C8YLvwvCC7oLvwu6C8ULwwvBC8gLxwu9C7kLugvGC8ULvwvFC8YLwAvGC7oLtgvGC8oLuAvJC8QLywvBC8MLygvJC7ULuQvLC8cLxgvAC8oLrwvNC8ALywvDC9ALvgvEC84LwwvRC9ALxwuvC70LyAvRC8MLvgvTC8gL2QvNC68LrwvHC9kL2gvIC9MLywvQC8wLzAvHC8sL3QvEC8kLygvUC8kL0QvIC9oL1QvNC9kLyQvUC90LvgvOC9cL1gvUC8oLzwvQC9EL2QvHC8wLygvAC9YL1wvTC74L3wvOC8QLzAvSC9kL3QvfC8QL2AvXC84LzAvQC9IL2gvTC9cL0AvPC9IL1gvAC80L1gvdC9QL3gvWC80L0gvVC9kL1wvbC9oL2wvXC9gL3AvSC88L4wvaC9sL2AvOC9sLzQvVC94LzgvfC9sLzwvRC9wL0QvaC+IL2wvfC+QL3wvdC9YL0gvcC+AL4wviC9oL4QvfC9YL4QvWC94L4AvVC9IL5QvcC9EL3wvhC+QL4wvbC+QL0QviC+YL5QvRC+YL5gvpC+UL1QvgC94L4wvvC+IL8AvhC94L4wvkC+sL8gveC+AL8AvkC+EL7wvjC+sL8AvrC+QL8gvwC94L4gvvC+YL6QvmC+gL5wvoC+YL4AvcC/IL6QvoC+cL6QvnC+oL5QvpC+wL5QvyC9wL7wvnC+YL7gvpC+oL7gvqC+cL6QvtC+wL6QvuC+0L7QvuC+wL6wvwC/EL7wvrC/EL7AvuC+cL5QvsC+cL5wvvC+UL8gvxC/AL5QvvC/EL8gvzC/EL8wvyC+UL5QvxC/ML9Qv0C/wL+Qv0C/UL/Av0C/oL+gv0C/sL9Av3C/YL9Av2C/sL9wv0C/kL+wv2C/cL+wv4C/cL+wv3C/gL/Av9C/UL/Qv+C/UL9Qv+C/kL6AbtBvEG6AbxBvAG8AbvBugG7wbtBugG8AbzBu8G7wbzBu0G8AbxBvMG8QbtBvMG9gb2BvMG9gbzBvYG+wv3C/kL+gb6BvYG+gb7BvYG9gb7BvYG9gb2BvoG+gb2Bv0G/Qb2BvsG/Qb9BvoG+gb9BvoG+gb9BvsG/Qb7Bv0G/Qb+Bv0G/Qb9Bv4G/gv7C/kL+wv+C/oL+gsBDPwLAQz9C/wL/QsADP4L/gsBDPoLAAz/C/4LAAz9C/8L/wv9CwEM/wsBDP4LBAwCDAUMBQwCDAEMAgwFDAEMAwwFDAIMBgwEDAUMBwwCDAQMCAwCDAcMCAwHDAQMAwwCDAgMBgwFDAwMDQwIDAQMCQwDDAgMBQwDDAwMCQwIDA0MDQwEDAYMDAwDDAkMFAwMDA8MFwwTDAYMFwwGDAwMCgwJDBgMCQwLDBgMCQwNDAsMDAwJDBkMCQwKDBkMCwwKDBgMCgwMDBkMIwwLDA0MCwwjDBoMGgwKDAsMDAwKDBoMDwwMDBoMDgwPDBoMDgwaDBsMDwwODBsMDQwQDCMMDwwbDBwMEQwgDB0MDQwGDCIMDwwcDB4MDwweDBIMEgweDB8MIAwRDB8MEQwSDB8MEQwdDCEMEAwNDBUMFQwNDCIMBgwTDCIMIQwSDBEMEAwUDCMMEAwVDCQMEwwVDCIMDwwSDCEMJAwUDBAMFAwPDCEMIwwUDCEMFgwkDBUMFwwkDBYMFgwVDCUMFQwTDCUMFAwkDBcMFgwTDCYMEwwWDCUMFwwWDCYMEwwXDCYMFAwXDAwMKgwoDCcMJwwoDC4MKQwqDCcMKQwnDC4MKQwuDC0MKgwsDCgMKwwpDDMMKgw1DCwMLwwxDCkMMQwqDCkMLwwpDCsMKgwxDDUMKQwtDDMMKAw0DC4MLAwwDCgMOwwvDCsMMAw0DCgMKwwzDD0MLgwyDC0MMgwzDC0MPQw7DCsMNAwwDDgMNQw2DCwMNwwwDCwMOAwuDDQMQgwyDC4MOQwvDDsMNgw3DCwMOAxCDC4MRwwxDC8MMgw6DDMMPQwzDDoMMAw8DDgMSgw1DDEMSgwxDEcMLww5DEcMMAw3DEYMRgw8DDAMOAxMDEIMNQxBDDYMQww5DDsMNgxEDDcMSgxBDDUMRQxKDEcMRAw2DD4MTQxDDDsMPQxNDDsMSAw/DEAMOAw8DFwMUAw8DEYMQAxEDD4MNgxJDD4MSwxIDEAMPwxEDEAMRAxUDDcMSgxJDEEMUwxRDD0MTAw4DFwMQgxXDDIMOgwyDFUMSQxKDFgMTgxLDEAMVwxCDEwMPQw6DFMMVAxGDDcMawxFDEcMPAxQDFwMUQxNDD0MRAw/DGQMTwxHDDkMNgxBDEkMTgxADD4MRQxYDEoMawxYDEUMOQxDDE8MSww/DEgMPwxLDFIMPwxSDGQMTwxrDEcMQwxNDF4MRAxkDFQMSQxWDD4MTwxDDF4MSwxODFYMVgxODD4MMgxXDFUMZwxSDEsMUww6DG4MSQxbDFYMaAxVDFcMXgxNDFEMVQxuDDoMWAxbDEkMawxsDFgMUwxZDFEMcAxPDF4MYQxMDFwMUQxZDFoMUAxmDFwMbgxiDFMMXwxRDFoMYQxcDGYMWQxfDFoMWQxfDF0MXwxZDF0MZQxSDGcMUQxfDGIMZQxkDFIMXwxZDGMMYwxZDFMMYgxfDGMMdAxYDGwMTAxhDFcMUwxiDGMMTwxwDGsMVgxgDEsMdAxbDFgMaAxXDGEMUQxiDF4MZwxLDGAMcgxQDEYMUAxxDGYMaAxuDFUMbQxpDGoMbgxeDGIMZgxxDGEMagxoDGEMYAxWDG8MYQxtDGoMaQxwDGoMewxWDFsMagxwDGgMZwxgDGUMdgxpDG0McwxsDGsMbgxoDHAMVAxkDHoMcwxwDGkMcwx0DGwMawxwDHMMcAxeDG4MewxvDFYMYAx1DGUMcQxQDHIMcQxtDGEMegxyDFQMcgxGDFQMbQxyDHYMcgxtDHEMeQxbDHQMeQx7DFsMdwxpDHYMcgx6DHYMfQxkDGUMcwxpDHcMeAx1DGAMeQx0DHcMdgx8DHcMbwx4DGAMegxkDH0Megx8DHYMdwx8DHkMdwx0DHMMZQx1DH0MeAxvDHsMewx5DH4MeAx7DH4MfAx+DHkMdQx4DH0MeAx+DH0Mfgx8DHoMfQx+DHoMfwyCDIAMgAyBDH8MgQyCDH8MggyHDIAMhAyDDIAMgAyDDIEMhgyCDIEMjAyEDIAMhwyMDIAMgwyEDIUMhwyCDI8MhAyDDIUMggyIDI8MiAyCDIYMiQyBDIMMigyDDIQMiQyGDIEMiQyDDIoMjAyKDIQMhgyODIgMiQyNDIYMigyLDIkMjwyZDIcMjQyJDIsMjwyIDJEMjQyODIYMjAyHDJkMiAyODJEMigyMDJkMmQyLDIoMjQyLDJQMkQycDI8MjQySDI4MkgyUDI4MjQyUDJIMkwyRDJAMkwyVDJEMkQybDJAMkAyVDJMMlQyQDJYMjgybDJEMnAyVDJYMkQyVDJwMnAyWDJAMkAyXDJgMlAyLDJkMkAyYDJwMkAybDJcMlwycDJgMlwybDJoMnAyXDJoMmwycDJoMmwyODJQMmwyUDJkMjwycDJkMnAybDJ0MnQyZDJwMmwyZDJ0MqwyhDJ4MowyrDJ4MoQyjDJ4MqwyjDJ8MowyrDJ8MqwyjDKAMowyiDKAMogyrDKAMoQykDKMMogylDKgMpQyiDKMMpQymDKgMpQyqDKYMpwylDKMMowykDK4MrgynDKMMqgyoDKYMpwyqDKUMoQyvDKQMogysDKsMpAypDK4MrwypDKQMogywDKwMsAyiDKgMqgynDK0MrQynDK4MqAyqDLIMoQyrDLcMrwyhDLcMqQyvDLQMtAyuDKkMqwysDLcMqgytDLIMsgywDKgMuQyyDK0MrQyuDLEMsQyuDLQMrAywDLIMrQyxDLkMrwy6DLQMrAy8DLcMuAyxDLQMtAy6DLMMugy0DLMMtAy6DLUMuQysDLIMtwy6DK8Mugy4DLUMtQy4DLQMtgy4DLQMtAy4DLYMuQy8DKwMugyxDLgMugy3DLsMuwyxDLoMvAy7DLcMuwy5DLEMuwy8DL0MvAy5DL0Muwy9DLkMvgzADL8MwgzADL4MvwzCDL4MvwzADMEMyAzADMIMxAzBDMAMyAzEDMAMxwy/DMEMxQy/DMcMvwzFDMMMwwzFDMcMyAzGDMQMvwzDDMIMygzBDMQMwgzDDMgMygzHDMEMxwzKDMMMzgzEDMYMyAzPDMYMygzNDMMMywzLDMsMywzLDMwMxAzODMoMywzLDMsMzAzLDMsMyAzDDM0MyAzJDM8MyAzNDMkMzgzNDMoMzQzPDMkMzgzGDM8MzgzPDNAMzQzODNAMzQzQDM8M0QzSDNUM1QzSDNYM1wzRDNUM0gzRDNcM0gzWDNQM1AzWDNIM1wzSDNMM1wzTDNIM0AzXDNUM1wzWDNIM1wzQDNkM1QzWDNAM1gzZDNAM2QzYDNcM1gzaDNkM1gzXDNoM2gzXDN0M1wzYDN0M2AzaDN0M3gzbDN8M2wzfDN8M3wzbDN4M4AzcDOIM3AzgDOEM4QzhDNwM2gzYDOQM3AziDOIM4gzcDOMM3AzhDOMM2gzkDNkM2QzkDNgM6gwWDeUM5wwWDeYMFg3nDOUM5wzqDOUM6gznDBQNFg3sDOYM7AznDOYM7AwWDesM5wzsDBQN7wzoDO4M7wzpDOgM6QzvDPAMFg0VDfIM9AwWDfIMFg30DOsM9AzsDOsM8wztDPMM8wzzDO0M7wzvDO4M8QzvDO8M8AzvDPEMFQ30DPIMFA3sDPQM8wzzDPMM8wzzDPMMGA3qDBQNFA30DBUN9Qz1DPUM9Qz1DPUM9Qz1DPUM9Qz1DPUM9gz2DPYM9gz2DPYM9wz3DPcM9wz3DPcM9gz2DPYM9gz2DPYMFQ0bDRQN+Qz4DPgM+Qz5DPgM+Qz5DPwM9wz3DPcM9wz3DPcM+gz6DPoM+gz6DPoM+Qz4DPsM/Az5DPsM/Az5DPwM/gz9DAEN/gz+DP0M/gz+DAINBQ3/DP8MBQ0ADf8MAA0FDQYN+gz6DPoM+gz6DPoM/gz+DAENAw3+DP4MAg3+DAMNBA0EDQQNBA0EDQgNBQ3/DAUNBw0FDQUNBg0FDQcNBA0EDQQNCA0EDQQNCQ0JDQkNCQ0JDQkNCQ0JDQkNCQ0JDQkNFw0YDRQNCw0KDQsNCw0LDQoNCw0LDQsNCw0LDQsNDA0MDQwNDA0MDQwNEQ0NDQ0NEQ0ODQ0NDg0RDRANDA0MDQwNDA0MDQwNDw0PDQ8NDw0PDQ8NEQ0NDRANDw0PDQ8NDw0PDQ8NEg0SDRINEg0SDRINEg0SDRMNEg0SDRMNFA0bDRcNFg3qDBgNGg0VDRYNGA0aDRYNFQ0aDRsNIA0YDRcNGQ0XDR0NFw0bDR0NHQ0bDR4NGA0cDRoNFw0iDSANGw0hDR4NIg0XDRkNGw0pDSENHw0ZDR0NHg0hDSoNGA0gDSQNGw0aDSgNGA0kDSUNHA0jDRoNLg0dDR4NHg0nDS4NIA0iDSQNJQ0cDRgNKQ0mDSENNw0eDSoNGg0jDSgNHQ0uDR8NHw0iDRkNJw0eDTcNKA0pDRsNUw0fDS4NKg0hDSYNLA0kDSINOA0cDSUNPA0lDSQNLw0jDRwNJA0sDTwNLA0iDUENKQ0rDSYNUw0tDR8NHw00DSINKA0jDS8NOA0vDRwNOA0lDTwNKA0yDSkNKw0wDSYNNA1BDSINKQ1jDSsNNA0fDS0Nqw0uDScNKg0mDTANKw1jDTANYw0pDUMNKA0vDUYNKA0zDTINKg01DTcNMg0+DSkNMA18DSoNqw0nDXcNQQ1EDSwNLw0xDToNKA1GDTMNKQ0+DUMNKg18DTUNdw0nDTcNLA1EDTkNLw07DTENOA07DS8NMA1jDTYNNA0tDbYNPg0yDTMNqw1TDS4Nkg1jDUMNOQ08DSwNLw06DUYNkg1DDT4NMw1GDT4NPA05DYMNPQ05DUQNsg1BDTQNMQ1CDToNRg06DWsNOw19DTENPA2DDTgNNQ13DTcNPQ1WDTkNNA22DboNPQ1EDT8NRA1ADT8NQA09DT8NSg1FDT0NQA1KDT0NRA1KDUANRw09DUUNRw1UDT0NSA1NDUcNRw1NDVQNSQ1NDUgNSg1RDU4NTg1RDVANSg1SDVENRA1SDUoNvQ2SDT4NeA05DVYNTQ1MDVQNVA1MDUsNTQ1JDUwNTw1ODVANUA1VDU8NUQ1VDVANUg1VDVENfA21DTUNOw04DYMNVg1YDVcNPQ1LDVYNVA1LDT0NXQ1VDVINRA1dDVINRA1qDV0Nrg02DWMNXw1WDVcNWA1fDVcNWA1WDUsNWQ1gDVkNWQ1ZDVkNYg1hDVoNWw1iDVoNXA1iDVsNXA1qDWINXQ1qDVwNtQ13DTUNeA1WDV4NVg1fDV4NWQ1gDVkNWQ1ZDVkNcA1hDWINag1wDWINXw14DV4NXw1YDUsNaA1mDWQNZQ1oDWQNZw1nDWcNZg1oDWkNZw1nDWcNZQ1pDWgNkg2uDWMNMQ20DUINXw1LDXgNZw1nDWcNZw1nDWcNQQ1qDUQNNg2vDTANuw2rDXcNaw06DUINSw1xDXgNbA1sDWwNbA1sDWwNdA1zDW0Nbg10DW0Nbw10DW4NcA11DW8Nbw11DXQNcA16DXUNag16DXANeA1xDXINSw1yDXENbA1sDWwNbA1sDWwNeQ1zDXQNdQ15DXQNeg15DXUNdg16DWoNeg1+DXkNeg12DX4NNg2uDa8New17DXsNew17DXsNdg1/DX4NQQ12DWoNug2yDTQNtA0xDX0NeA2BDYANew17DXsNew17DXsNkQ1+DX8Nfw12DZENhA14DYANgQ2EDYANgQ14DXINgg2GDYUNgg2HDYYNdg2aDZENeA2EDTkNhg2HDYUNig2IDYkNiQ2LDYoNiA2LDYkNQQ2aDXYNhA2BDXINuA0tDVMNcg2TDYQNjA2TDXINjQ2UDY0Njg2NDY0Njg2VDY0Njw2XDZYNkA2XDY8NkA2YDZcNkA2ZDZgNkQ2ZDZANkQ2aDZkNlQ2UDY0NoA2WDZcNmA2gDZcNmQ2gDZgNUw2rDbgNhA2TDYwNmQ2aDaANuA22DS0NjA2hDYQNmw2hDYwNnA2jDaINpg2lDZ0Nng2mDZ0Nnw2mDZ4Nnw2nDaYNoA2nDZ8NoA2aDacNhA23DTkNpA2iDaMNnA2kDaMNqA2lDaYNpw2oDaYNhA2hDZsNmw2pDYQNmw2qDakNqA2qDZsNqA2aDaoNpw2aDagNqQ2qDawNtQ28DXcNhA2pDawNqg2tDawNqg2aDa0NrA23DYQNrQ2aDawNmg23DawNrw18DTANqw27DbANfQ07DbQNtw2yDbMNQQ23DZoNvA27DXcNtA07DbENqw2wDbsNtA3DDUINsw3CDbcNsw2yDboNtQ18Da8NuA2rDbsNsQ25DbQNuQ3DDbQNPg1GDb0Ngw05Db8Nkg2vDa4NQQ2yDbcNrw3BDbUNsQ07DcoNaw1CDcMNug22DcQNRg1rDc0NuA3EDbYNvw3CDbMNvg2zDboNgw3KDTsNkg3FDa8NxQ2SDcANuw28DcQNvQ3ADZINsw2+DccNgw2/DcsNzQ29DUYNrw3FDcENug3EDb4NxA28DcENwg05DbcNwQ28DbUNvw05DcINxA24DbsNyw3KDYMNww3PDWsNvg3EDcENvw2zDccNuQ2xDcMNyg3MDbENwA3GDcUNvg3BDcUNxQ3GDb4NvQ3ODcANxw2+DcYNwA3HDcYNyQ2xDcwNyQ3DDbENzQ3ODb0NyA2/DccNyA3ADc4NyA3HDcANzw3DDckNvw3IDcsNzw3NDWsNyg3SDcwNzg3NDdANyQ3SDc8NyA3ODcsNyQ3MDdINzw3SDdANzw3QDc0Nyg3RDdIN0g3ODdANyw3ODdINyw3RDcoN0Q3LDdIN3A3eDdYN3g3bDdMN3g3TDdQN1g3eDdQN0w3bDdUN1g3UDdMN2w3YDdUN2A3TDdUN1g3TDdgN3A3WDdcN1g3YDdcN3A3XDdgN3A3YDdkN2A3dDdkN3Q3cDdkN2g3bDd4N2A3bDdoN2A3aDd0N3g3fDdoN3A3dDeIN3g3cDeEN2g3fDekN4A3kDewN3Q3jDeIN2g3mDd0N5g3aDekN4g3hDdwN4A3oDeQN3g3hDd8N7A3rDeAN7w3jDd0N5w3oDeAN3w3oDecN6w3nDeAN4Q3lDd8N5w3pDd8N5g3vDd0N3w3lDegN7Q3kDegN9Q3hDeIN4Q31DeoN5g3pDfEN4Q3qDeUN7A33DesN5Q3tDegN7w3mDfEN8Q34De8N7g3yDfgN+A3xDe4N4g3jDQEO5w3rDfcN6Q3nDfMN8w3xDekN5Q3qDfAN9g3yDe4N5A35DewN+Q33DewN+A0FDu8N9A32De4N8A3yDfYN8w3uDfEN7w0EDuMN5w33DfMN7Q35DeQN+A3yDfsN8w30De4NAQ7jDQQOAw7yDfANBQ4EDu8N9Q3yDQMO6g31DQMO8A3qDQMO7Q3lDQIO9Q3iDQEOBQ74DfsNAA7zDfcN9A36DfYN+g3lDfANAA76DfMN8w36DfQN+Q0ADvcN9Q37DfIN8A32DfoN+g0CDuUNCQ71DQEO/w39DfwN/A39Df4N/w38Df4N7Q0IDvkN/w3+Df0NAg76DQAOBA4GDgEO9Q0JDvsNBQ4KDgQOCQ4FDvsNBg4JDgEOBA4HDgYOAg4IDu0NCA4ADvkNCA4CDgAOBA4KDgcOBg4LDgkOBQ4ZDgoOBQ4JDhkODA4GDgcOBg4MDg0ODQ4MDgcODQ4HDhoOCg4ZDhAODQ4ODgYODQ4PDg4OBg4ODg8ODQ4RDg8OCw4GDhoOBg4PDhoODw4RDhoODQ4aDhEOFw4SDgoOEw4HDgoOFA4HDhMOFg4HDhQOEg4UDgoOCg4UDhMOFw4UDhIOFg4aDgcOFA4XDhUOFg4UDhUOFQ4XDhYOFw4KDhAOCw4ZDgkOGA4XDhAOGA4WDhcOGQ4LDhoOGA4QDhkOGg4WDhgOGQ4aDhgOHw4dDhsOHg4fDhsOHQ4eDhsOHg4dDhwOHQ4eDhwOIw4eDh0OHg4jDh8OHQ4lDiMOJQ4dDh8OJA4hDiAOIg4hDiQOIQ4vDiAOHw4jDiAOIg4mDiEOIw4kDiAOJw4hDiYONA4lDh8OLw4hDicOJg4oDicOJg4pDigOKg4nDigOKg4oDikOKw4qDikOJg4rDikOKg4rDiYOIw4lDjYOKg4mDiwOJg4iDi4OMg4iDiQOKg4uDicOMg4wDiIOIA43Dh8OLg4qDiwOLA4mDi4OLQ4uDiYOJg4uDi0OLw4nDi4OLg4iDjAOLw4uDiQOLg4yDiQOHw43DjQOMg4uDjAOJA4jDjYOJQ4xDjMOMQ41DjMONQ4lDjMOJA42Di8OJQ40DjEOLw43DiAONw4vDjYOOA41DjEONA44DjEONA43DjkONA41DjgONw47DjkOOw40DjkOOg41DjQONA47DjoONg4+DjcOOg47DjUOPg42DiUONQ4+DiUONQ47DjwOPQ43Dj4OOw43Dj0OPg41DjwOPw4+DjwOPw5GDj0OPw49DkMOPg5DDj0OPQ5EDjsOPQ48DkQOPQ5GDjwORQ5BDkAOQA5BDkIOQg5CDkAOQw4+Dj8ORA48DjsOQA5FDkUORQ5ADkAOQA5CDkAOPA5GDj8ORw55DnYOfA5JDkcOSQ55DkcOeg52DnkOZg5SDkgOSA53DksOfA6DDkkOZg5IDkoOSA5LDkoOSw5mDkoOew55DkkOTA5PDk4OTQ5MDk4OTg5ODk0OTg5ODk8OUA5UDlAOUQ5QDlAOUQ5RDlAOSA54DncOfA5HDn4OSw5TDmYOhg5HDnYOdw5TDksOVA5UDlAOUA5VDlQOVQ5QDlEOVg5YDlYOVw5WDlYOVw5aDlYOew5JDoUOWQ5YDlYOVg5bDlkOWw5WDloOXA5cDlwOXA5cDlwOXQ5hDl0OXg5dDl0OXg5iDl0OUg5+DkcOUg5mDn4OXw5fDl8OXw5fDl8OYA5cDlwOXA5gDlwOYw5hDl0OYw5dDmIOaQ5nDmQOZQ5pDmQOZQ5qDmkOZg5TDn0OXw5fDl8OXw5fDl8OaA5nDmkOaQ5pDmgOaQ5pDmoObg5sDmsOaw5vDm4ObQ5sDm4Obg5uDm0Obg5uDm8OdA5zDnAOcQ50DnAOcg5yDnIOcg5yDnIOdQ5zDnQOdQ50DnEOcg5yDnIOcg5yDnIOeA5IDlIOfQ5+DmYOUw54Dn0Odw54DlMOUg5HDoYOhQ5JDogOgA54DlIOgw6IDkkOgA5SDoYOhg52DnoOfA6JDoMOfQ5/Dn4OeQ57Do0OjQ56DnkOiQ58Dn4OgQ6EDoIOeA6ADocOig59DngOhA6GDnoOhA56Do0Ohw6KDngOhg6EDoEOfQ6KDowOiQ6IDoMOjA5/Dn0OkA6NDnsOhg6HDoAOgQ6CDokOkA57DoUOhg6BDocOfw6LDn4OiA6ODoUOig6HDoEOfg6BDokOhA6ODoIOiw6BDn4Ojg6EDo0Ogg6IDokOgg6ODogOiw6MDooOfw6MDosOiw6KDoEOjQ6QDo8OhQ6ODpAOjw6QDpMOkQ6ODo0OkQ6NDo8OkA6VDpMOjg6VDpAOkw6SDo8OkQ6VDo4OlA6PDpkOlA6RDo8OmQ6RDpQOmQ6PDpIOkQ6ZDpgOkg6TDpYOkw6cDpYOnA6SDpYOlQ6XDpMOmA6ZDpoOmw6RDpgOlQ6RDpsOmg6bDpgOlw6cDpMOmw6XDpUOmQ6SDpsOkg6cDpsOmw6aDpkOnQ6cDpcOlw6bDp0Omw6cDp0Onw6iDp4Ong6iDqAOpQ6gDqIOtA6xDqEOow6gDqUOnw6eDrEOng6gDrEOsQ6gDqEOnw6tDqIOrA6gDqMOsQ6uDp8Oog6mDqUOog6nDqYOpw6iDqQOpA6iDqcOpw6iDq0Oow6lDqoOpw6lDqYOpQ6nDqgOrQ6gDqwOrA6lDqgOrA6oDqcOuA6hDqAOpw6tDqwOqg6lDqkOpQ6sDqkOrA6qDqkOqg6sDqsOqw6jDqoOrA6jDqsOrQ64DqAOsA60Dq8OrQ6fDq4Orw60DrAOtA6wDrEOrQ6uDrgOsA60DrMOsg6uDrEOtQ6xDrAOsA6yDrUOsg6xDrUOtA63DrMOoQ64DrQOtw6yDrAOtw6wDrMOsg6xDrYOtg6xDrIOuA65DrQOuQ63DrQOrg6yDrgOsg65DrgOuQ6yDrsOug66DroOug66DroOug66DroOug66DroOuw6yDrcOuQ67DrcOvA6/DsAOvQ6/DrwOwA69DrwOvg6/Dr0OwQ6/Dr4OvQ7ADr4OwQ6+DsAOvw7BDsMOvw7FDsAOvw7DDsUOxQ7HDsAOwA7HDsQOzA7DDsEOwA7MDsEO3A7EDscOwg7IDskOyA7CDsYOwA7EDswOwg7JDtAO0A7KDsIOzQ7IDsYOyg7GDsIOxQ7VDscOyQ7IDs0Ozw7YDtYO1Q7cDscO0Q7FDsMO2w7mDsYOyQ7rDtAOxg7KDtsO2g7DDswOxQ7kDtUO5A7FDtEO2g7MDvYOyw7VDtIO1A7eDvcO1A7TDt4O1A7ODtMO3Q7YDs8O1Q7dDtwO5g7gDsYO2Q7dDs8OzQ7fDskO2g7RDsMO9g7MDsQOyw7SDtUOzQ7GDuAO3w7rDskO4w7gDtcO5A7SDtUO1A7lDs4O1A73DuUO0Q7aDvAO4A7fDs0O2A4DD9YO7A7PDtYO3A7zDsQO1w7gDuYO4g73Dt4O2Q7PDuwO2w7hDuYO6g7QDusO6A7bDsoO6A7KDtAO7Q7cDt0O3A7tDt0O4g7hDtsO3g7mDuEO3A7dDtkO8w7cDtkO4w7pDuAO/A7VDtIO5Q73Dv4O7A7WDgMP8A7kDtEO0A7qDugO3g7hDuIOCg/YDt0O/A7SDu8O2w7oDuIO5A7vDtIO1Q7yDt0O/A7yDtUO5A7wDvkO4w77DukOAw/YDgoP9g7EDvMO7g7nDtkO4w4AD/sOBg/ZDuwO2Q7nDvMO1w4AD+MO9A73DuIO7w7kDvUO8Q7XDuYO4g7oDvoO5g7eDvEO9Q7kDvkO0w7ODg4P9w7/Dv4O3w4CD+sO3w7gDgIP0w4OD94OBg/uDtkO3g4FD/EOCg/dDvIO6Q74DuAOCw/7DgAPAg/gDvgOzg7lDg4PAQ/2DvMOAA/XDvEODg8FD94O6A7qDvoODg/lDv4OAQ/aDvYO+g70DuIO9Q4PD+8O7g4GDwgP2g4UD/AO6g7rDgcP+A7pDv0O7A4DDwYPAQ/zDucOFQ/1DvkO6Q77Dv0O/A4KD/IODw/1DhUPAg8ED+sO9w4jD/8O9A4ZD/cO2g4BDxQPDg/+DhcPFA/5DvAO7w4PD/wOBA8CD/gOBA8HD+sO+w4MD/0OIw/3DhkP+w4LDwkP+g7qDgcPBA/4DicPBQ8AD/EO/w4XD/4OHA/5DhQPCg8RDwMPEQ8GDwMP+g4HDx0P/A4PDyoPCg/8DioPHA8VD/kO9A76Dh0PGw8ZD/QO+A79Dg0PJA8ADwUP/w4TDxcPCA8gD+4O/w4jDxYP5w4iDwEPEA8GDxEPCQ8MD/sOGw/0Dh0PAA8kDwsP5w7uDhgPDg8eDwUP+A4NDycP/w4WDxMPPA8RDwoPHw/uDiAPLg8HDwQP/Q4MDw0PEg8QDxEPDA8JDwsPIg/nDhgPGA/uDh8PDw8VDysPDA8LDyQPFA8BDyEPHg8kDwUPPA8KDyoPLw8IDwYPKw8qDw8PEg9WDxAPHg8ODxcPFg8jDyUPHA8UDzgPSQ8YDx8PKw8VDxwPEw8aDx4PFg8yDxMPAQ8iDyEPFw8TDx4PPA8SDxEPDA8mDw0PDQ8mDycPJQ8jDygPVg8SDzwPEw86DxoPMQ8oDxkPKA8jDxkPBg8QDy8PMg8WDyUPKQ80DwgPEw8yDzoPBA8nDy4PCA80DyAPQA8ZDxsPCA8tDykPCA8vDy0PNQ8gDzQPIA87Dx8PHQ8+D0UPOA8UDyEPRQ8bDx0PKQ8yDyUPMQ8ZD0APMw8pDy0PJQ8oD00PGw9FDywPVg8vDxAPKQ8zDzAPTQ8pDyUPQg8mDwwPKQ8wDzIPPg8dDwcPLQ9ODzMPTQ80DykPIA81Dz0PPg8HDy4POw8gDz0PMw82DzAPHA84DysPOw9JDx8PGw8sD0APMQ83DygPIg8YD0kPTQ8oD1QPMQ9AD0oPOQ80D00PVg9GDy8PNw9UDygPKg8rD0EPHg9QDyQPJA9QDwwPQg8MD1APOg8yD08POA9BDysPQQ88DyoPJw8mD0IPNA9SDzUPPA9zD1YPRw9QDx4PMg8wD08PJw9CDy4PPQ81Dz8PSQ9TDyIPcg9GD1YPMA82D08PLw9ODy0PRg9ODy8PIQ8iD1MPVw8/DzUPGg9rDx4PQA9RD0oPMw9ODzYPSw8sD0gPRQ9IDywPWQ9FDz4PSg9RDzEPNw8xD0wPQQ9DD0QPLA9LD0APLg9ZDz4PPA9BD2QPdQ86D08PTA8xD14PWg9ND1QPGg86D2sPHg9rD3gPRw8eD3gPQw9BDzgPUg9uDzUPXg8xD1EPTQ9qDzkPVw89Dz8PRQ9ZD0gPOA8hD0MPSQ87D1UPZA9zDzwPTg9vDzYPNA85D1IPdQ9rDzoPTQ9aD2oPUw9JD1UPNw9MD2APTg9GD2cPQg9ZDy4PgA9kD0EPIQ9sD0MPIQ9TD2wPcw9yD1YPOw89D1cPUA9ZD0IPSw9RD0APcg9nD0YPNQ9uD1cPeg9aD1QPTw82D3QPUA99D1kPRA+AD0EPWg93D2oPSA9RD0sPZQ9UDzcPXg9mD1wPXA9MD14PbA9ED0MPUQ9ID14PWw9MD1wPWw9gD0wPZg9bD1wPag9tDzkPXw9gD1sPNw9gD2UPXw9bD2YPTw90D3UPRw9oD1APYg9gD18PbQ9SDzkPZg9iD18Pbg9SD20PYw9gD2IPVQ9XD2kPVw9VDzsPVw9uD2kPcQ9gD2MPdw9aD3APag93D3kPYw9iD3EPbw92DzYPXQ92D2EPdg9dD1gPcQ9iD2YPNg92D3QPbA+AD0QPZQ96D1QPSA9ZD30Pdg9dD2EPXg9xD2YPVQ9sD1MPaQ9uD20PZw9vD04PXQ92D1gPeA9oD0cPcA9aD3oPbQ9qD3kPUA9oD30Peg+DD1oPZQ9gD3EPXg+ND3EPgw96D1oPjQ9eD0gPiw9sD1UPdg9vD1gPfA9kD4APew9wD3oPfg9tD3kPbw92D1gPjA+PD2sPeQ93D3APjw94D2sPaw91D4IPaw+CD4wPcw9kD38PZw9yD4YPcQ96D2UPgg90D3YPZw92D28Phw95D3APfg9pD20PeA+BD2gPZw9wD3YPew+CD3YPdg9wD3sPew96D4MPVQ9pD4sPZA98D38PZw+HD3APdA+CD3UPfQ+ND0gPfg95D4cPiA95D4cPig+DD3oPbA+JD4APcQ+KD3oPkw+ND30Pcg9zD4gPjw+BD3gPiA+HD3kPgA+JD3wPcg+ID4YPhw+ID34PgQ99D2gPcw+ED4gPiQ9sD4sPhg+HD2cPcw9/D4QPjA+CD4UPiA+OD34PhQ97D4MPlA+FD4MPhg+ID4cPkQ+ED38PlA+DD4oPgQ+TD30PjA+QD48Pfg+LD2kPgQ+PD5IPgg97D4UPcQ+ND4oPlA+KD40PiA+ED44Pfw98D5EPhQ+QD4wPkQ98D4kPiw9+D44PiQ+LD44PjQ+TD5QPkg+TD4EPjw+QD5IPlQ+TD5IPjg+ED5EPhQ+UD5APiQ+OD5EPlQ+SD5YPkg+QD5QPmg+UD5MPmg+ZD5QPlg+SD5wPkg+UD5sPkg+bD5kPnA+SD5kPlA+ZD5sPlg+cD5kPlg+iD5UPng+dD5cPnQ+XD5cPmA+VD6IPlw+fD54Plw+XD6APlg+ZD6IPlw+hD58Plw+gD6EPow+VD5gPmA+iD5oPmQ+aD6IPlQ+jD5oPmA+aD6MPmg+YD6QPmA+aD6QPlQ+lD5MPlQ+aD6UPmg+TD6UPpw+mD6gPqA+mD6cPqQ+nD6gPqg+oD6cPqg+rD6gPqg+nD6sPpw+pD6sPqA+rD6kPsA+sD60PrA+wD7UPqQ+vD7EPqQ+xD68PrQ+sD64Ptw+wD60Psw+tD64PsA+3D7EPrA+1D64Prg+yD7MPtw+0D7EPsA+xD7cPrg+zD7IPrQ+zD7cPrw+xD7YPrw+2D7EPsA+5D7UPug+xD7QPtw+zD7QPug+2D7EPtg+3D7EPrg+7D7MPtw+2D7oPtQ+4D64PsA+3D7sPuw+5D7APrg+4D7UPuw+uD7UPtA+zD7oPsw+7D7oPug+7D7cPtQ+5D7sPwA+/D7wPvQ+8D78PwA+8D70PwA/MD78Pvw/MD74PwA+9D9APvg/MD78PzA/AD80PvQ/FD9APxQ+9D78PwQ/FD78P0Q/MD80Pvw/HD8IPxQ/BD8IPwQ+/D8IPxQ/CD8MPxw+/D8wPww/CD8UPyQ/ED8IPxA/FD8IPyQ/FD8QP0A/ND8APxQ/LD9APxQ/JD8YPyw/FD8YPyw/GD8kPyQ/ID8sPxw/JD8IPyA/HD8sPyQ/HD8gPyg/OD8sPyw/HD8oPzg/KD8cPxw/SD84P0g/QD8sP0g/LD88Pzg/PD8sPzw/OD9IP0g/HD8wPzQ/UD9EP0g/MD9EPzQ/QD9QP2A/QD9IP0w/SD9EP1Q/UD9AP0Q/UD9MP2A/VD9AP0w/UD9YP0w/YD9IP1Q/WD9QP1Q/XD9YP2w/VD9gP1Q/bD9cP2Q/YD9MP2Q/TD9YP2g/YD9kP2g/ZD9cP2Q/WD9cP2A/aD9sP1w/bD9oP2g/bD9wP2w/aD9wP2g/bD90P2g/dD9sP5Q/gD94P4A/hD94P4Q/lD94P4A/fD+EP4A/hD98P4Q/gD+QP4Q/iD+UP5Q/kD+AP4g/hD+MP4Q/iD+MP5A/lD+YP5Q/xD+YP5A/mD+sP6g/hD+QP8A/lD+IP7Q/wD+IP5A/nD+gP6A/qD+QP7Q/iD+EP5w/qD+gP5w/kD+sP5w/rD+kP6g/nD+kP6w/qD+kP6w/uD+wP7A/qD+sP5g/uD+sP7A/uD+oP7g/mD+8P5Q/wD/EP5g/zD+8P6g/vD/MP7g/vD+oP8Q/zD+YP8g/tD+EP8g/hD+oP6g/zD/IP8Q/wD/QP9A/4D/EP9A/wD+0P8g/zD/UP8Q/3D/MP8Q/4D/cP9Q/2D/IP9Q/zD/cP9A/tD/IP9g/0D/IP9w/6D/UP9A/2D/sP9A/5D/gP+Q/0D/sP+Q/8D/gP+g/3D/sP9g/1D/sP+A/8D/cP+Q/7D/0P+w/1D/oP+w/3D/wP/A/5D/0P/Q/7D/wP/A/9D/4P/w/8D/4P/g/8D/8PABD8D/4PABD+D/0P/Q/8DwAQAxABEAIQAxACEAEQBxADEAIQBRACEAMQBBAFEAMQCBACEAUQAhAIEAcQBBADEAcQBRAEEAcQCRAGEA8QCxAPEAYQCBAFEBAQCxAGEAoQCBAQEAcQBRAHEBAQChAGEAkQDxANEAkQDBAKEAgQCBAOEAwQChAJEAgQEBAIEAkQCBAQEA4QCRANEBAQDRAUEBAQEhAMEA4QDBASEAoQDRAPEBMQEBAUEA4QFRALEAoQFRAPEAsQEhAOEBQQERAPEBUQExAUEA0QChAWEBUQERAVEA8QFhAKEBIQDxAVEBMQFBATEBcQFBAWEBIQFRAYEBMQFxATEBgQFRAXEBgQFxAVEBkQFhAZEBUQFhAXEBkQFxAWEBoQFxAaEBQQFBAaEBYQKBAhEB4QIRAbEB4QKBBqECEQKBAeECAQHRApEBwQHBApEB8QHxApECIQIBAsECgQIhApECQQIxAdEBwQNRAcEB8QLBAqECgQIBAeECUQIhAkEC0QGxAlEB4QRhAfECIQIhArEEYQGxAnEC4QIhAtEDEQJxAbEFMQNRAjEBwQGxAuECUQKhAsEDAQMRAtECYQUxAbEEsQLhA6ECUQRhA1EB8QIBA0ECwQTBArECIQNxAwECwQMRAmEDkQIxBCEB0QJxBDEC4QIRBLEBsQKBAvEGoQKBAqEC8QJRA0ECAQNBA3ECwQTBAiEDEQZBBPECkQKRBmECQQMBA3EDsQJhA4EDkQOhAzECUQShA0ECUQNxBHEDsQQBA5EDgQMhBJECcQUxAyECcQHRBkECkQSxAhEGoQXBA1EEYQNhAvECoQJBB1EC0QNxBfEEcQYBAxEDkQQBA4EDwQJxBJEEMQcRA+ECMQdRAmEC0QkxA4ECYQIxBiEEIQNxA0EF8QTBAxEHwQOBA9EDwQPBA9EFUQMhBBEEkQNhAqEDAQJBBmEHUQRxA/EDsQPhBiECMQUxBjEDIQcRAjEDUQQBBgEDkQSRCBEEMQLhCIEDoQJRAzEEoQTxBGECsQMBA7EFAQRxBRED8QRRBNEFcQRBBFEFcQXBBxEDUQKRBPEGYQPBBVEEAQXRBAEFUQQxCIEC4QRBBOEEUQYBB8EDEQQRBWEEkQQhBZEB0QMBB6EDYQUhCYEEgQMhBuEEEQMhBjEG4QVBBOEEQQWhBHEF8QWBA7ED8QURBHEFoQSBCbEFIQTRBSEFcQMxA6EFwQVhCBEEkQVxBUEEQQMxBcEEoQZhBPECsQUBB6EDAQOxBsEFAQWBBsEDsQWBA/EFEQYBBAEJQQoBCbEEgQmxBXEFIQWRBbEB0QOhCIEGcQWxBkEB0QShBcEGUQZhArEEwQdRCTECYQQRCOEFYQUhBtEJgQYRBNEEUQQxCBEIgQWRBhEEUQcBBLEGoQOhBnEFwQeRAvEDYQfhBfEDQQTBB0EGYQWRBCEGEQUxBwEGMQWxBZEEUQRRBOEFsQOBCTEJ4QlBBAEF0QWxBOEF4QaRBREFoQlxBdEFUQYRB9EE0QTRB9EFIQiBBiED4QcBBTEEsQZBBbEF4QThBUEF4QXhBUEHIQXhBvEGQQRhBPEFwQShB+EDQQkRBQEIwQQhCHEGEQiBA+EHEQeBBcEE8QbxBeEHIQTxBkEG8QjBBQEGwQnhA9EDgQgRBWEKEQUhB9EG0QaBBPEG8QaBBvEGsQXxB+EHQQXxB0EFoQjRBsEFgQqBCgEEgQlRBUEFcQeRBqEC8QeBBPEGgQShBlEH4QkRB6EFAQaRCAEFEQgBBYEFEQjxCOEEEQQhBiEIcQkBBiEIgQchCCEG8QaBBrEGYQcxCCEGsQaxCCEHYQWhB0EJYQgBCDEFgQlRBXEJsQahB5EHcQXBB4EGUQbxBzEGsQbxCCEHMQaBBmEHQQdRBmEGsQghB7EHYQdRBrEHYQfBCWEEwQZRB4EH4QdBBMEJYQlxBVED0QVhCOEKEQbRB9EJoQihBcEHgQZxCKEFwQehB5EDYQfhBoEHQQfxB2EHsQmBCoEEgQmhCYEG0QgRCQEIgQrBBwEGoQcRBcEIoQeRB6EIsQaBB+EHgQaRBaEJYQdRB2EH8QhhB/EHsQjRBYEIMQtBCAEGkQlxA9ELkQoRCcEIEQfRBhEKsQfxCGEIUQhRCGEIQQpRChEI4QiRBxEIoQiBCJEGcQrBBqEHcQeBBcEIoQgBCvEIMQZxCJEIoQnRBsEI0QkxB1EH8QnxC0EGkQkxB/EIUQhxCmEGEQgRCcEJAQiBCnEIkQVBC2EHIQjBBsEJ0QnxBpEJYQpBCWEHwQuxCNEIMQkxCFELwQhRCEELwQYxBwEKwQjBCSEJEQYBCqEHwQXRCXEJQQiRCIEHEQmxCtEJUQlBCqEGAQqxCaEH0QYhCmEIcQehCyEIsQvBCeEJMQPRCeELkQoRDAEJwQmxCpEK0QlRCuEFQQvhDBEHIQqhCkEHwQvBCwEJ4QyRClEI4QqBCiEKAQnBBiEJAQpxCIEIkQthC+EHIQkhCMEJ0QsxCdEI0QuxC3EI0QtBC4EIAQjxDJEI4QmhCZEJgQzhBuEGMQmxCgEKkQghByEMEQlhCkEJ8QphCrEGEQrBDOEGMQtRB3EHkQehCRELIQ0xCkEKoQeRCLELUQrxDkEIMQYhCcEKYQoBCiEKkQghDHEHsQnxCkELQQuBDIEIAQmBDYEKgQmRCaEKsQVBCuELYQoxCREJIQoxCSEL8QxhCSEJ0QexDHEIYQhhDhEAIRsBC5EJ4QmRCrELEQrhCtEMUQrhCVEK0QxhC/EJIQyBCvEIAQAhGEEIYQvBCEEAIRoRDcEMAQQRBuEL0QqxCmEMQQnBDAEKYQpBDTELQQqRDLEK0QthCuEMUQthDFELoQ1xDGEJ0QwxCXELkQyRDlEKUQ2BCZELEQyxDeEK0QthC6EL4QxxDhEIYQwhC4ELQQuBD1EMgQvBAGEbAQoRClENwQtRCsEHcQoxCyEJEQwRDHEIIQsxCNELcQCRG7EIMQzRCPEEEQvRDNEEEQrBC1EN8QwRC+EMwQsxC3EOMQ0xDCELQQBhG5ELAQxBDSEKsQ2hDOEKwQlBDCEKoQmRDYEJgQsRCrEMoQ2BCxEMoQtRCLENsQxRDuELoQshCjEOAQCRG3ELsQwBDEEKYQ0hDKEKsQvRBuEM4QrRDeEMUQxxDBEMwQohDzEKkQqRDzEMsQ8RC+ELoQxhDVEL8QxxDMENAQ5BCvEMgQlBCXENQQohDmENkQzxDgEKMQvxDPEKMQsxDwEJ0Q0xCqEMIQxxDQEOEQwhCUENQQ9RC4EMIQlxDDENQQwxC5EAYRwBDnEMQQ3xDaEKwQshDgEIsQ0RDGENcQ1xCdEPAQ7xDxELoQxhDRENUQ5BDIEPQQ5RDiEKUQwBDcEOcQ3BClEOIQ2BDmEKgQ3hDLENYQvhDxEMwQCRGDEOQQChEGEbwQvRDOEOgQ3xDdENoQyxASEdYQ4BD6EIsQ0BDyEOEQ9hDJEI8Q2RDzEKIQABESEcsQ2xDfELUQ3hDuEMUQ+hDbEIsQzBDpENAQBRECEeEQqBDmEKIQyxDzEAAR7hDvELoQ4xC3EAkRwhDUEPUQ3BAQEecQ2RDYEMoQ2hDoEM4QEhHsENYQ6RDyENAQ1BDDEP4QxBDnENIQzxADEeAQyBD1EPQQ5hDYENkQyhDSENkQ7BDqENYQ1hDqEN4QHBEDEc8QvxDrEM8QBBHrENUQ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aBGbEX4RnhF4EVgRnhFYEWoRixFhEZARghGKEXcRgxGCEWkRaBF+EU4RlhGdEWYRnhFqEZERlBGBEYARShGJEXwRghGDEYoRgxFpEWYRnRGDEWYRfhGlEW4RpRF/EXkRiBF9EX8RoBFrEY8RrRF6EVwRoBGpEWsRqRFyEWsRfBGbEWgRbhGlEXkRdBGnEYkRrRFcEZIRchGaEYQRlxGUEVQRjBGFEW8RpBF8EYkRexGNEacRkBF6Ea0RdxGKEZURnBGKEYMRnBGDEZ0RjhGLEZARmxF8EaQRkRFlEaMRoxFlEYsRkxGGEYURdxGVEZkRoRGWEZ4RtxF/EaURiBF/EbcRjhGjEYsRiBF1EX0RqRGaEXIRkxGSEYYRmBGTEYURdhGZEY0RdRGIEY8RqhGtEZIRlRGKEZwRlhGhEZ0RnhGREaMRnxGeEaMRvBGQEa0RtBGEEZoRpBGrEZsRoRGcEZ0RjxGzEaARfhGbEaURoRGeEZ8RohGfEaMRmBGFEYwRpBGJEacRsBGZEZURphGcEaERvBGOEZARtBGaEakRrhGMEYcRvRGBEZQRrhGHEYERqhGSEZMRoRGoEaYRqBGhEZ8RshGIEbcRsxGIEbIRsxGPEYgRthGnEY0RrBGlEZsRlxHDEZQRpxGvEaQRsRGcEaYRrBGbEasRuxGoEZ8RuxGfEaIRrhGBEb0RmRG2EY0RnBGxEZURsxGyEbgRoBG6EakRhBG0EZcRlRGxEbARrBG3EaURwBGrEaQRtRGsEasRuRGjEY4RsxG6EaARuhG0EakRtBHDEZcRvhGYEYwRrxGnEbYRuRGiEaMRwBGkEa8R0xGoEbsRuRGOEbwRsBHFEZkRqhG8Ea0RwxG9EZQRtRGrEcARqBHTEaYRrhG+EYwRxhGyEbcRyRG0EboRmRHFEbYR0hGwEbERohG5EbsRvxG4EbIRzhG5EbwRsxHJEboRwhG9EcMRthHEEa8RtRHAEcERwRGsEbURxhG3EawRvxGzEbgRkxGYEc8RwxG0EckRxxG+Ea4RphHSEbERphHTEdgRxhGsEcERxhG/EbIRvhHPEZgRxxGuEb0RxBG2EcURyBHAEa8RzxGqEZMRxBHIEa8RzBGzEb8RvRHCEccRxRGwEdIRzBHJEbMRyRHCEcMRyhHAEcgR2BHSEaYRyhHBEcARwRHKEcYRxhHLEb8R3RGqEc8RwhHJEcwRxBHFEdARvxHLEcwRvBGqEc4RyBHNEcoRxhHKEcsR0xG7EbkRzxG+EccRxBHQEc0RxBHNEcgRzhGqEd0R2hHCEcwR1xHCEdoRxxHCEdcRxRHSEdARyxHKEdkR0RHLEdkR7BG5Ec4RzBHLEdERzxHHEdcR2RHNEdYRyhHNEdkR1BHPEdcRzRHQEdURzRHVEdYRzBHREdoR1BHdEc8R0hHVEdAR6RHYEdMR6RHTEbkR4BHREdkR3xHUEdcR2hHfEdcR2BHVEdIR1hHgEdkR4BHhEdER0RHhEdwR0RHcEdoR1RHYEeMR1hHVEdsR1RHjEdsR2xHgEdYR6RG5EewRzhHdEeIR3hHtEdQR7RHdEdQR3xHaEdwR3hHUEd8R4hHdEe0R3BHlEd8R5RHeEd8R2BHoEeMR5BHbEeMR2xHkEeYR4BHbEeYR4BHmEeUR4BHlEeER4RHlEdwR4hHsEc4R5hHeEeUR5xHtEd4R4xHoEeoR6BHYEekR5BHjEeoR6xHtEecR5BHqEecR5hHkEecR6hHrEecR5hHnEd4R6hHoEewR7BHiEeoR6BHpEewR6hHiEesR6xHiEe0R7xHyEe4R7xHuEfER8hEFEu4R/BHyEe8R8RHuEfYR8RH2EfAR8BH1EfER9hH6EfAR9BH1EfARBRL2Ee4R9BHzEfUR+hH0EfAR9RHzEfcRJhLxEfURFxL1EfcRJhLvEfERJhL1EQESFxIBEvURJhIeEu8RCRL3EfMR/BHvER4SIRL+EfoR+BEXEvcR+REAEgwSBBIFEvIR9xEJEvgR+REMEg0S9BEkEvMR+RH7EQASDRIGEvkRDRIMEv0R8xEkEgkSAhL/EQoSABIDEgwSChL/ERQS/xECEgMSABL/EQMS+xH5EQ4S8hH8EQQSBBIHEggSBRIEEggSQRL6EfYRQRIhEvoR+xH/EQASFxL4EQkSIRIiEv4R/xH7EQsS/hEkEvoRFBL/EQsS+xEOEgsSDBIDEgISERILEhYSCxIOEhsSChIYEgISDBISEv0RKxIEEvwRGxIWEgsS+REGEg4SDBICEhISBhINEh8S/REaEg0SKxL8ER4SAhIPEhISFRIREhASBhI2Eg4SHxINEjMSGhIzEg0SERI4EgsSGRITEhUSFRI4EhESRRIYEgoSDxIdEhISERIWEhASMBIXEgkSNhIcEg4SQBISEh0S/RESEkASGhL9EUASOBIUEgsSChIUElISKxIgEgQS/hFqEiQSAhIYEg8SExJLEhUSFRIQEhkSJBIwEgkSBBIgEgcS/hEiEnESLhIeEiYS9BH6ESQSFhIbEkQSTRIPEnkSJRJAEh0SBRIIEiASMBImEgESFxIwEgESRRIKElISXhIiEiESHBIbEg4SNhIGEh8SNxIPEj8SDxI3Eh0SJRIdEjcSOxIaEkASBxIgEggSFhJMEhASWRIcEjYSdxJDEiMSGRIQEpMSOBIVEiwSPBIuEiYSXhIhEkESXhJmEiISFRJLEiwSLRIjEkMSFBI4EkgSNRInEi8SSBJSEhQSZRIYEkUSGBI9Eg8SMhIoElQSMhJUEikSMhIpEjkSTRI/Eg8SLxInEjwSMBI8EiYSMRI1EjQSNRIxEj4SNBIoEjESMRIoEjISVBJhEikScBITEhkSNRIvEjQSMBJTEjQSUxIoEjQSUxJUEigSYRJWEikSKRJWEjkSMxJVEh8SGhJaEjMSUBIrEh4SLRJDEi4SLRIuEjwSJxItEjwSOBJREkgSdRIwEiQSLxI8EjASNBIvEjASRBIbEhwSBRJHEvYRQhJQEh4ScRIiEmYSHhIuEkMSFhJEEnUSJBJrEnUSPRIYEmUSMRJGEj4SRhIxEjISHxJVEjYSORJWEjoSVhJXEmgSOhJWEmgSOhJoEioSaBJXEmMSaBJPEioSaBJjEkoSRxIFEiAS9hFHEkESdxJCEh4SQxJ3Eh4SZhKFEnESJBJqEmsSZBIcElkSMhI5EkYSYhI/Ek0SWhJVEjMSOxJAEiUSIBIrElAShxIjEpYScRJqEv4RLBJREjgSJxKWEi0SZBJcEhwSSRI5EjoSRhI5EkkSdRJTEjASRBIcElwSiBI1Ej4SSRI6Ek4SJRI3El0SThI6EioSThIqEk8SUxJkElQSNhJVElkSPxJiEjcSOxJaEhoSThJYEkkSiRI7EiUSOxJtEloSTxJYEk4SQhJvElASdxIjEmkSNxJiEl0SWBJPElsSeBIsEksSlhIjEi0SFhJqEkwSPRJ5Eg8SeRJsEk0SiRIlEl0SWxJPEmgSkhJpEiMSoxKiEicSURJSEkgSchJhElQSlBKGEkYSchJWEmESXxJXElYSbRI7EokSXhJBEkcSExKaEksSVRJUElkSuRKUEkkSYBJKEmMSUBJHEiASRBJcEnUS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jROVE4UTgxORE5YThhN9E48TjxN9E4cTkBOPE4cThBOjE5cTjBOCE5UTixOGE48TiROaE6MTkBOHE4oTnBOWE5ETgROIE54TlBOKE4wToRNqE5cTkxNqE6ETlBOME5UTmRN0E6oToBORE4sTixOPE6ATkhOQE4oTkhOKE5QTmBOTE6ETlhOeE4gToBOPE5ATjhOaE4kTkhOUE50TnROUE5sTmxOUE5UTjRObE5UTjROZE6UTlhOcE54ToBOQE5ITqhOlE5kTmxONE6UToBOcE5ETrxOXE6MTnBOgE58TrhOOE4ETnBOfE54TjhOuE5oTohOgE6cToBOSE6cTpxOSE50TpBOpE5sTqROdE5sToBOiE58TnhOuE4ETqROnE50TpROkE5sTqhOsE6UTnhOfE60TnxOmE60TnhOtE64TphOfE6ITphOiE6gTqBOiE6cToROqE5gTpBOlE6wTtBOjE5oToxO0E68ToROXE68TsxOsE6oTqBOnE7ATsBOnE6kTqxOpE6QTsxOqE6ETqxOkE6wTsBOpE6sTrxOzE6ETphOoE7ITshOoE7ATqxOsE7MTphOyE60TrhOtE7kTtBOaE64TuROtE7ETrROyE7ETtROyE7ATtROwE6sTtROrE7MTtxO1E7MTrxO3E7MTuBOxE7ITrhO5E7QTuBOyE7UTthOxE7gTrxO0E7cTuBO1E7cTuROxE7YTuhO3E7QTuBO3E7oTtBO5E7oTuhO2E7gTuRO2E7oTvRPEE7sTxBO8E7sTvBO9E7sTvxO9E7wTvBPEE8ITvRO/E8ATvxPJE74TvBPCE78TwxO+E8cTxBO9E8wTvxO+E8ATwBPBE70TxBPNE8ITvxPCE8gTwRPME70TvhPJE8cTwRPAE88T0BPEE8wTwxPFE8YTyBPJE78TwxPLE8UTwxPHE8sTxhO+E8MTzxPAE8oTvhPKE8ATxRPLE9ITxRPSE8YTzhPHE8kT2RPME8ET0hPXE8YTwhPNE8gT1BPEE9AT0xPGE9cTxhPTE74T1RPOE8kTzRPEE9QT1RPJE8gTxxPOE+sT0RPLE8cTGhTVE8gT0hMOFNcT2xO+E9MTvhPbE8oTyBPNExoUxxPrE9ETwRPPE9YT1hPZE8ETGhTNExcUzRPUExcU1BPQEw0U2RPQE8wTyxPYE9IT0hPYExAUDhTSExAUDRTQE9kTFxTUE9oT0RPYE8sT6xMSFNET1RMWFM4TzhP/E+sT4BPUEw0U1BPgE9wTzxPKE9YT2hPUE9wT3BPgE90T3RPaE9wT5RPVExoU3hPdE+AT4BPhE94T4RPdE94T3RPhE98T3xPaE90T4RPiE98T3xPiE+MT4xPaE98T5hPgEw0U5BPhE+YT5hPhE+AT5BPiE+ET5BPjE+ITDRTwE+YT5hPwE+wT5hPsE+cT5hPnE+QT0xPXEwoUBRQSFOsT5BPnE/ET8RPjE+QT4xPxE+kT7RPjE+kT4xPtE9oT7BPwE+4T5xP5E+gT5xPsE/kT8RPtE+kT+hPaE+0T+RPuE/AT+RPsE+4T+RPxE+gT6BPxE+cT6xP/E+8T8hPtE/ET+hPtE/IT6hMFFOsT9xPqE+sT9xPrE+8T/xP3E+8T8BMNFPMT9BPwE/MT+RPwE/QT+xPxE/kT8RP6E/IT6hP2E/UT6hP3E/YT9xP/E/gT8xP5E/QT+xP5E/0T+hPxE/sTBxTqE/UT9hMHFPUTBxT2E/wT9hP3E/wTCRT3E/gT/xMJFPgT/RP5E/MT+xP9E/oTBRTqE/4T6hMAFP4TABQFFP4T6hMDFAAU6hMHFAMU9xMHFPwTBBTzEw0UARTzEwQU/RPzEwEUAhT9EwQUAhT6E/0TCRQHFPcTBBT9EwEUBBT6EwIUEhQiFNETAxQHFAAU2hP6ExcUMBQIFNkTBRQAFAYUEhQFFAYUABQSFAYU/xPOEwkUDhQKFNcTEBTYEx8UBBQXFPoTABQHFAkUCRQSFAAUGRQiFBIU1RPlExYUERQIFAsUCBQPFAwUCBQMFNkTDxQIFBUUDxTZEwwUFxQmFBoUFRQIFBEUDhQpFAoU2xPWE8oTCxQIFDAU2xPTExwUFhQJFM4TChQcFNMTMBQUFAsUFBQRFAsU2RMPFA0UFRQRFBQU1hPbEysUExQPFBUUGBQUFDAUMBTZE9YTFBQTFBUUJxQaFCYUJhQXFAQUFBQYFBMUGBQPFBMUJhQEFA0U0RMfFNgTGBQwFA8UJxTlExoUEhQJFBkUIhQ4FNETHxQbFBAUEBQpFA4U0RM4FB8UGxQdFBAUKBQQFB0UGxQoFB0UDxQqFA0UHxQeFBsUGxQgFCgUIBQbFB4UHxQgFB4ULhQJFBYUKhQmFA0UHxQhFCAUIBQkFCgUJBQgFCEU5RMuFBYUKBQzFBAUIRQfFCUUJBQlFCMUIRQlFCQUKBQkFCMUKBQjFCUUHBQrFNsTHxQoFCUUCRQyFBkUIhQsFDgUKxQwFNYTLBQiFBkUEBQzFCkUHBQKFCsUKBQfFDMUJxQtFOUTMBQqFA8UNhQmFC8ULxQmFCoUJxQmFDYUKRQ0FAoULRQnFDYULhTlEy0UCRQuFDIUMBQrFDEUQRQvFCoUMBRBFCoUMhQ+FBkUNRQtFDYUMxQ0FCkUNRQuFC0UOBQ6FB8UMRQrFD0UNxQsFBkUKxQKFD0UMxQfFDoULBQ3FDgUMRRBFDAUOxQ+FDIULhQ7FDIUNxQ6FDgUORQKFDQUPRQKFDkUOxQuFDUUPBQ2FC8UMxQ5FDQUNxQZFD4UNhQ8FDUUQBQxFD0UQRQ8FC8UNxRFFDoUOhREFDMUMxREFDkUQhQ3FD4UPBQ7FDUURRQ3FEIUQBQ9FEYURhQ9FDkUPhQ7FEMUQBQ/FDEUPxRBFDEUQxQ7FDwUQhQ+FEMUPBRBFD8UOhRIFEQUPxRDFDwURBRGFDkURRRIFDoUQhRDFEUUQBRDFD8UQBRGFEMURxRFFEMURxRDFEYURBRIFEYURhRIFEkUSRRHFEYUSRRKFEcURRRJFEgUSxRFFEcURRRLFEkUShRJFEwUSRRLFEwUShRMFEsUSxROFEoUTRRHFEoUShROFE0USxRNFE4UTRRLFEcUTxRjFFMUUxRRFE8UYxRPFFEUbhRQFFYUUBRZFFIUUBRYFFkUVhRQFFQUUxRVFFEUWBRQFG4UVxRRFGsUURRXFGcUVhRfFG4UWhRjFFEUURRVFGsUYxRqFFMUUhRbFFAUYBRWFFQUWxRUFFAUYxReFGoUWhRRFGcUWxRSFFkUVRRTFH4UVhRgFF8UVBRbFHEUWRRpFFsUXhRtFGoUdRRUFHEUXhRjFFwUbBRpFFkUahR+FFMUbBRiFFwUWBRsFFkUZxRXFHIUaxRyFFcUWBRdFGwUbxRpFGwUXRRiFGwUYxRwFFwUYxRaFHAUZBRaFGcUYBRUFGEUaxRlFHIUXhRcFGIUfhR0FFUUYhRmFF4UYhRdFGYUdBRrFFUUYRRUFHUUaxR5FGUUWhRkFHAUgRR2FF8UeRRyFGUUgRRfFGAUbRReFGYUbxRsFHAUaBRYFG4UbBRcFHAUaRRxFFsUchRkFGcUfBR5FGsUbxRxFGkUdhRuFF8UfBRrFHQUbRR+FGoUchR9FGQUXRR4FGYUcxRzFHMUcxRzFHMUcxRzFHMUcxRzFHMUeBRdFHoUgRRgFGEUjRRtFGYUbxR3FHEUcBR7FG8UXRRYFGgUbxR7FHcUgBRoFG4UfxRkFH0UfBR0FHkUgxRhFHUUcRSRFHUUfhRtFIQUfxRwFGQUfhSEFHQUbhR2FIAUfxR7FHAUZhR4FHoUhRR9FHIUjRSHFG0UfxRxFHcUkRRxFH8UhRRyFHkUehRdFGgUehSNFGYUYRSDFIEUixR3FHsUfxSLFHsUjxR5FHQUkRSDFHUUixR/FHcUjhSCFIEUhxSEFG0UhRR/FH0UghR2FIEUgxSOFIEUhhR6FGgUeRSPFIUUgBR2FIIUiRSCFI4UhxSMFIQUkRSQFIMUgBSGFGgUhBSKFHQUihSPFHQUjhSDFJAUfxSLFJEUhhSAFIkUiRSAFIIUhBSMFIoUhRSIFH8UfxSIFIsUhhSNFHoUiRSTFIYUiRSOFJQUmBSQFJEUhRSPFIgUkhSMFIcUixSYFJEUlBSOFJAUjRSSFIcUjRSGFJMUjRSTFJIUiRSUFJMUlhSMFJIUjBSXFIoUjBSWFJcUixSIFJcUiBSPFIoUkhSTFJUUkxSUFJsUihSXFIgUixSbFJgUmBSUFJAUmxSUFJgUlhSSFJUUixSXFJsUlRSZFJYUmxSXFJ0UnRSXFJYUnRSWFJkUnBSTFJsUlRSTFJwUlRScFJkUnBSbFJ0UnBSaFJkUmhScFJ8UmRSaFKAUnhSZFKAUnhSgFJoUmRShFJ0UoRSZFJ4UnhSaFKIUoRSfFJ0UnhSfFKEUmhSfFKIUnxSeFKIUnRSfFKMUnRSjFJwUnxScFKMUpBSmFKcUpxSlFKQUpRSmFKQUphSlFKoUqBSlFKcUqRSqFKUUqRSlFKgUpxSpFKgUpxSmFKkUphSqFKkUrBSvFKsUrBSrFK4UrRSsFK4UrxSyFKsUrRSuFKwUsRSvFKwUshSuFKsUrBSuFLAUrxSxFLIUshSxFKwUsBSyFKwUshSwFK4UsxS0FLYUsxS1FLQUthS1FLMUtRS3FLQUtxS6FLQUtBS6FLYUuRS1FLYUuBS6FLcUuRS2FLoUuxS1FLkUuhS4FMIUuhTDFLkUxBTDFLoUuxS+FLUUvBS4FLcUvBS9FLgUvxS3FLUUvhS/FLUUuBS9FMAUuhTCFMQUvBS3FL8UwBTCFLgUvhTBFL8UuRTDFMcUxhS5FMcUuRTGFLsUwhTAFNUUwhTFFMQUuxTJFL4UuxTGFMkUvxTQFLwUyhS9FLwUzhTFFMIUwBTIFNUUxxTDFMQUwhTVFM4U0BTKFLwUxBTNFMcU0xTQFL8UwRTTFL8UvRTRFMAUxxTNFNgU4xS+FMkUyhTMFL0UxBTFFM0UxhTSFMkUvRTMFNEU0RTLFMAUyBTAFMsUxxTWFNQUxxTUFMYU2RTBFL4U2BTWFMcU3BS+FOMUvhTcFNkU4xTJFOAU0hTGFNQUyRTSFOAU2xTYFM0U4hTKFNAUwRTZFNMU0hTUFOEUzxTOFNUUzxTVFNoU0xTXFNAU1RTIFMsU0RTMFN4U4hTQFNcU0xTZFNcU4hTMFMoU7RTNFMUU6xTgFNIU7RTFFM4U2hTkFM8U2RTdFNcU5hTVFMsUAxXWFNgUzhTPFBwV2hTVFOYU2BTxFAMVzBTiFN4U4RTvFNIU2BTbFOcU8RTYFOcUyxTRFN8U6hTiFNcU5RTRFN4U5BQNFc8U7RTOFBwV5xTbFBMV7RT/FM0U3xTmFMsU1xTdFOoU4RTUFBEV+hTgFOsU2xTNFP8U1hQDFfcU1hQRFdQU6xTSFO8U8BTeFOIU9hTlFN4U2hTmFAYV9RTZFNwU4hTqFPAU4RQVFe8U4xTgFOgU/xTtFDUV2RT1FN0UChXhFBEVBhX4FNoU3BTjFOgU2hT4FOQU4BT6FOgU9hTeFPAU0RTlFN8U7hTqFN0UHBXPFA4V3xTpFOYU1hT3FPIU1hTyFBEV6BT1FNwU6BT6FOwU+hToFOwU5RT2FPMU7hTdFPUUCRXwFOoU9BT6FOsUFRXhFAoVGBXoFPoU3xTlFPMU3xQaFekUPRXrFO8UzxQNFQ4VFRU9Fe8U6hTuFAkV6xQ9FfQUHRX2FPAUMRXkFPgU+BT5FOQU+BTkFPkUOBUGFeYU9RT+FO4UMRUNFeQUDxXuFP4U/hQBFQ8V+xQBFf4U/BQCFfsU+xQCFQEV/RQHFfwU/BQHFQIVBxX9FPUU/RT+FPUU2xT/FBMVAhUAFQEVBxUAFQIVABUHFQEVGBX6FBIVIxX/FDUVDxUBFQQVBxUEFQEV9hQ6FfMU7RQcFQgVBRUPFQQVBBUHFQUVBhUbFfgUGxUiFfgUFhXuFA8V9RQMFQcVCRUdFfAUBRULFQ8VChUyFRUVPRUVFSUVPRX6FPQUIRUbFQYVAxVPFfcUCxUQFQ8VCxUFFRAVDBXoFBgVNBUNFTEVDRUfFQ4V7RQIFTUVEBUUFQ8VBRUUFRAVBxUMFQUVOBUhFQYVFhUPFRQVERXyFAoVDBX1FOgUExX/FCMVFBUFFRYVPRUSFfoU8hQXFQoV3xTzFBoVDBUWFQUVMxUcFQ4VIBUJFe4UFhUgFe4UTxUDFTAVGhUpFekU6RQ4FeYUNBUfFQ0VAxXxFDAVPRU8FRIVKRUZFekUGRU5FekU6RQ5FTgVHRU2FfYUHhUJFTsVOhX2FDYVDhUfFTMVIBU7FQkV+BQiFTEVCRUeFR0V5xRQFfEU8RRQFTAVSxUVFTIVSxUlFRUVPxU3FfMU8xQkFRoVOhU/FfMUGhUkFSkVNxUoFfMU8xQoFSQVJhUrFSoVKxUmFScVKxUnFSwVLBUnFSgVLBUoFTcVKRUkFS4VKxUtFSoVLBUtFSsVNxUvFSwVLBUvFS0VHRUeFTYV5xQTFU4VKRUuFTcVNxUuFS8VNRVEFSMVFxUyFQoVNxVZFSkVRxUYFRIVIRU4FUEVNBUxFUwVRRUgFRYVPhVFFRYVQhUjFUQVQBUXFfIUDBU+FRYVORVBFTgVSxUyFRcVNxU/FVkVRRU7FSAVQhUTFSMVThVQFecUSxUXFUAVRxUMFRgVShUcFTMVNRUIFUQVSxVXFSUVTBUxFSIVCBUcFUoVHxVDFTMVHxU0FVoVRxU+FQwV8hT3FE8VOhVGFT8VMxVDFUoVExVCFU4VVxU9FSUVSBUSFTwVOhU2FUYVVhVBFTkVSBVHFRIVIRVRFRsVGxVJFSIVIhVJFUwVVxU8FT0VKRVUFRkVURVJFRsVHxVaFUMVVxVIFTwVRRVNFTsVMBVQFU8VUxVFFT4VHhVVFTYVUhVNFUUVTxVAFfIUTBVaFTQVShVEFQgVRxVTFT4VWRVUFSkVYBVAFU8VGRVUFTkVIRVBFVEVWhVYFUMVRBVOFUIVVxVLFUAVVhVRFUEVOxVVFR4VUhVFFVMVVRU7FU0VRhU2FVUVRhVZFT8VWxVHFUgVUxVHFVsVRBVfFU4VTxVQFWAVORVdFVYVURVhFUkVThVfFVAVTBViFVoVWRVeFVQVXRU5FVQVUhVcFU0VXBVSFVMVYhVMFUkVWBVKFUMVVxVAFWAVWRVGFV4VTRVmFVUVURVWFWEVZBVNFVwVXhVGFVUVXRVhFVYVZhVNFWQVXxVEFUoVVxVbFUgVYhVJFWEVXBVTFVsVYhVYFVoVZRVbFVcVVBVeFV0VXxVKFWMVYxVgFVAVZRVXFWAVYxVKFVgVXxVjFVAVYBVjFWUVZRVcFVsVahVVFWYVVRVqFV4VYhVrFVgVYhVhFWsVaBVkFVwVYxVoFWUVYRVtFWsVahVmFWwVaBVcFWUVWBVrFWMVYRVdFV4VaRVhFV4VaxVnFWMVaBVjFWcVYRVpFW0VZxVkFWgVZxVmFWQVaRVeFWoVbRVqFWwVaRVqFW0VaxVtFWwVaxVsFWcVbBVmFWcVbBVuFW0VbBVtFW4VbxVwFXMVdhVwFXEVcxVyFW8VchVwFW8VcRVwFXIVchVzFXEVcxV2FXEVcBV5FXQVcBV0FXMVeRVwFXYVdBV1FXMVdhVzFXoVdBV3FXUVdRV6FXMVdxV4FXUVfxV3FXQVdhV6FXsVdxWBFXgVfBV4FYEVeRV+FXQVdRV4FX0VehV1FX0VeRV2FcwVdxV/FYEVexV6FX0VfxV0FX4VdhV7FcgVfRV4FXwVfRV8FYIVexV9FdAVzBV+FXkVyBV7FdAVfhXTFX8VdhXIFcwVixWCFXwVhRV8FYEV0hWBFX8VfxXSFYAVfxWAFdIVghXQFX0VfBWFFYMViRV8FYMVhRWJFYMVyxWCFYsVhRWGFYQViRWFFYQVhhWlFYQVhBWlFYkVpRWGFYcVhhWFFYcViBWlFYUVhRWlFYcVfBWJFaUVhRWlFYgVzRV+FY4VihV+FcwV1xXTFX4VfhWMFY4VihWMFX4VjBWKFY0V1RWLFXwVihWMFY0V1RV8FaUVjBWPFY4VjBWRFY8VkRWMFZAVihWQFYwVihWSFZAVkRWOFY8VkhWRFZAVkhWKFa0VihXMFa0VjhWRFZMVkhWtFZEVrRWTFZEVmBWOFZMVlBWYFZMVlRWaFZQVlBWaFZgVlhWaFZUVlhWbFZoVnhWcFZcVzRXXFX4VoBXNFY4VjhWmFaAVphWOFacVmBWnFY4VmhWZFZgVmhWbFZkVnRWcFZ4VnhWfFZ0VlxWfFZ4VphXNFaAVmRWpFZgVoRWpFZkVohWqFaEVoRWqFakVoxWqFaIVpBWqFaMVpBWrFaoVrRWyFa8VzBWyFa0VmBWoFacVqRWoFZgVqRWsFagVqhWsFakVqxWsFaoVrhWtFa8VrxWwFa4VshW3Fa8VrxW3FbAVphWnFc0VsRWxFbEVsRWxFbEVshWzFbcVpRWFFdEVsRWxFbEVsRWxFbEVhRWBFdEVsxW+FbcVvhWzFbQVsxWyFbQVtRW4FbUVthW1FbUVthW2FbUVuhW9FbwVtxW+FcQVshW+FbQVuBW4FbUVtRW5FbgVthW5FbUVuxW6FbwVvBW9FbsVshXEFb4VshW/FcQV0xXSFX8VwBXDFcIVwRXAFcIVwhXDFcEVpxXOFc0VxBXFFacVpxXFFc4VxBXGFcUVxBXHFcYVxBW/FccVxRXGFckVxxXKFcYVxxW/FcoVzhXFFckVxhXOFckVzhXGFcoVyhW/Fc4V1RXLFYsV0hXRFYEV1BWCFcsV0BXYFcgVvxWyFcwV0BWCFdQVzxWlFdEV1xXOFb8VzhXXFc0VyxXVFdQVyBW/FcwV0hXTFdsV0RXSFdYVvxXIFdcVzxXVFaUVzxXcFdUV2BXQFdQV2xXWFdIV2hXPFdEV1BXVFdwV1hXaFdEV0xXXFdkV3RXYFdQV2RXbFdMV2RXXFcgV3BXdFdQV3xXIFdgVzxXaFdwV3hXbFdkV2RXIFd8V2xXeFdYV3xXYFd0V1hXcFdoV3BXWFd4V3BXeFeAV3hXZFd8V3BXgFd0V3RXmFd8V4hXeFd8V4BXeFeIV4xXmFd0V4hXhFeAV4BXhFekV5xXdFeAV3RXnFeAV3RXgFekV4RXiFegV3RXpFeMV4xXpFeEV4RXoFeMV4hXjFegV4xXiFeoV4hXkFeoV5BXiFesV5BXjFeoV5RXrFeIV6xXjFeQV4xXrFeUV5RXmFeMV5RXiFewV4hXmFewV5hXlFewV5hXiFe0V3xXmFe0V4hXfFe0V3xXuFeIV7hXfFeIV8RXvFfMV7xXxFfUV9BX3FfIV8hX2FfQV9RX0FfYV9RX4Fe8V+hX5FfQV9BXwFfcV+hX0FfUV7xX4FfMV9xX2FfIV8BX0FfkV9RXxFfoV8RXzFfoV8BX7FfcV+hX1FfYV9RX6FfgV9xUFFvYVAhb6FfYVAhb2FQMW+BX6FfMV9xX7FQUWDxb7FfAV9hUFFgMW+RUPFvAVDxb5FQEWARb5FfoVBBb9FQYW/RX+FQYWABb+FQQWARb6FQIW/RUEFv4VDxYRFvsVABYEFv8V/xUUFgAW+xUNFgUWBRYNFgMWBxYUFv8V+xUHFg0WBxYLFhQW/BX/FQQWABYZFgkWDRYIFgMWABYUFhkW+xURFgcWCBYCFgMWEBYEFgYWDhb+FQAWDhYAFgoWBxb/FfwVBBYMFvwVFhYSFgIWARYCFhcWABYJFgoWBhb+FQ4WCxYHFhEWFhYCFggWAhYSFhcWHRYNFgcWCxYTFhQW/BUdFgcWCBYEFhAWERYPFhUWEBYGFg4WFxYPFgEWCBYMFgQWGBYIFg0WHhYJFhkWDBYdFvwVGxYUFhMWDRYVFhgWDRYRFhUWDhYcFhAWFRYPFhcWDBYIFhgWCBYQFhYWERYTFgsWDhYKFgkWEhYWFhoWDhYJFh4WEhYaFhcWHBYWFhAWDRYdFhEWFhYcFhgWGxYTFh8WGBYcFgwWGBYVFhYWERYdFhMWGhYWFhUWFBYgFhkWHRYMFh8WDhYeFhwWFBYbFiAWHhYZFiAWHxYMFhwWExYmFh8WFRYXFhoWExYdFiYWKxYcFh4WJRYfFhwWIBYhFh4WIxYcFiIWIxYlFhwWKhYeFiEWJhYbFh8WKhYrFh4WKxYiFhwWKhYhFicWHRYfFiUWIBYpFiEWHRYkFiYWHRYlFiQWKRYnFiEWGxYoFiAWGxYmFigWKxYjFiIWJxYpFioWIxYwFiUWKhY0FisWLRYpFiAWKBYtFiAWNBYuFisWLhYxFisWJhYvFigWNBYsFi4WLxYmFiQWKRY0FioWKxYyFiMWKBYvFi0WKRYtFjQWMRYyFisWNhYvFiQWNhYkFiUWLhYsFjQWMBYjFjIWLxY0Fi0WLxY2FjgWNhYlFjAWNBY1Fi4WMRYuFjUWORY3FjYWLxY4FjQWMBYyFjUWNRYyFjEWNhYzFjgWMBY1FjYWNBY4FjUWNxYzFjYWMxY3FjkWORY2FjUWMxY5FjUWMxY1FjgWOhZCFjsWQRZCFjoWOxZBFjoWPBZBFjsWOxZAFjwWPBY9FkEWOxZFFj8WPxZAFjsWQBY9FjwWPRZAFj4WQRY9Fj4WQBZBFj4WQBY/FkUWQBZDFkEWRhZCFkEWQxZAFkUWQRZDFkYWRhZHFkQWRxZGFkMWSRZDFkUWRhZWFkIWSBY7FkIWTRZEFkcWRRY7FlcWQhZWFkgWRRZXFkkWTRZPFkQWVhZGFkQWTxZLFkQWQxZJFkoWYBZNFkcWOxZMFlcWVxZhFkkWZhZDFkoWTBY7FlsWUxZNFmAWSBZWFkoWVhZmFkoWRBZmFlYWThZXFlAWVxZOFlEWZhZHFkMWVBZTFmAWYRZXFlEWThZhFlEWWxZKFkkWSxZPFlIWVxZaFlAWWhZOFlAWSxZSFl8WTRZjFk8WTxZfFlIWVRZTFlQWWBZNFlMWVRZYFlMWSBZbFjsWWBZVFlQWTBZbFloWWBZUFmAWWRZNFlgWYxZNFlkWTBZaFlcWWBZgFlkWWRZgFmMWWxZIFkoWYRZOFl4WRBZLFmYWThZaFl0WWhZOFl0WThZaFlwWYhZOFlwWThZiFl4WXhZiFmEWWhZiFlwWRxZjFmAWWhZbFmIWZhZjFkcWYhZJFmEWWxZJFmIWZRZPFmMWZRZfFk8WSxZfFmQWYxZmFmQWZRZjFmgWZBZmFksWYxZkFmgWXxZpFmQWZRZpFl8WZRZkFmkWZRZqFmQWZxZkFmoWZxZoFmQWZRZnFmoWaBZnFmsWZxZlFmsWZRZoFmsWbRZwFm4WbhZsFm0WbxZtFmwWbxZwFm0WbxZzFnAWeBZzFm8WbBZuFnUWbBZ1FnIWbxZsFnYWdhZsFnIWeBZvFnEWcBZ3Fm4WcBZzFncWdxZ1Fm4WbxZ2FnEWdxZzFnQWdBaEFncWdRZ6FnIWcxZ4Fn8WeRZ4FnEWdBZzFn8WfRZ2FnIWehZ9FnIWdRZ3FnsWhBZ0Fn8WcRZ2FnkWdxaEFnsWhRZ2FoIWeRZ/FngWeRZ2FoUWeRaAFn8WghZ2Fn0WfBZ9FnoWfBZ6Fn4WfhZ9FnwWehZ1FoEWexaBFnUWgBZ5FoUWfhaCFn0WhRaHFoAWhBaDFnsWehaGFn4WhBZ/FocWehaBFoYWgRZ7FoMWfxaAFocWgxaEFocWixaHFoUWfhaGFokWihaCFn4WghaKFoUWgxaGFoEWihaLFoUWixaGFocWiRaGFosWfhaJFooWiRaIFooWiBaJFooWgxaHFoYWihaJFowWiRaLFowWixaKFowWjRaYFpQWlBaRFo4WjxaUFo4WlBaPFo0WjxaYFo0WkBaUFpcWkRaUFpAWkRaPFo4WkRaQFpcWkhaPFpEWkhaYFo8WkhaRFpcWlxaYFpIWkxaYFpcWkxaVFpgWlxaVFpMWmBaWFpQWlhaYFpwWlxaUFpYWmRaVFpcWmBaaFpwWmRaXFpsWmBaVFqIWohaaFpgWlRaZFqIWlxaWFqEWnRacFpoWoRabFpcWohafFpoWnBaeFpYWnhahFpYWnRaaFp8WoxaiFpkWnRaeFpwWoxaZFpsWoRagFpsWoxafFqIWnxaeFp0WoxabFqAWoBahFqMWnhajFqEWnhafFqMWpBaeFp8WnhakFp8WuRalFrcWtBa3FqUWuRa0FqUWpha0FrkWqBa0FqYWqBamFrkWqBa5FqkWtBaoFqcWqBazFqcWsxa0FqcWsxaoFqkWsxapFrkWqhazFrkWqxaqFrkWqxazFqoWuRatFqwWrBarFrkWqxasFq0WqxatFq4WrhatFrYWrxauFrYWrxa2FrEWrxarFq4WsBavFrEWsBarFq8WqxawFrEWsRa2FrIWqxaxFrIWtha1FrIWtRarFrIWqxa0FrMWuRa2Fq0WtRa0FqsWtxa0FrUWuRa1FrYWtxa1FrgWvha4FrUWuha4Fr4Wtxa4FrwWtRa5Fr4WwRa3FrwWuBa6FrwWvBa/FsEWxha5FrcWuRa7Fr4WwRbGFrcWuxa5FsYWvxa8FsMWuha+FsIWvRa8FsAWwBa8FroWvBa9FsMWxxa9FsAWwBa6FsIWxRbDFr0WvhbVFsIWxBbMFsEWwRa/FsQWvxbMFsQWxhbBFsgWvRbHFsoWvRbKFsUWuxbVFr4WwxbFFu0WwxbvFr8W9BbCFtUWvxbvFswWxRbKFskWyhbFFskW6RbVFrsWxhbPFrsWyBbBFswWxRbLFuwWyxbFFsoW7BbLFsoWzRbPFsYWwBbuFscW5RbNFsYWzxbNFs4W7xbDFu0WzRbSFs4W0hbPFs4W6Ra7Fs8WzRbQFtIW0hbQFtEW0hbTFs8WzxbUFukW7hbAFsIW0BbNFtYW0BbZFtEW2RbSFtEW0hbZFtMW0xbZFtcW1xbPFtMWzxbXFtQW1BbXFukWzRbZFtYWzRblFtkW2RbQFtYW1xbZFtoW1xbbFukWxRbYFu0W2RblFtoW2xbXFtoWxhbdFuUW5RbbFtoWxhbIFtwW3BbdFsYW2xblFuEW5RbgFuEW2xbhFukW3RbjFuUW5RbjFt4W4BblFt8W4BbpFuEW4xbdFuIW3hbjFuQW5BblFt4W5RbkFt8W5BbgFt8W5BbdFtwW3RbkFuIW5BbjFuIW4BbnFukW5BbmFuAW5xbgFuYW5hboFucW5BboFuYW5xboFuoW3BbnFuoW3BbpFucW5BbrFugW3BbVFukW5BbcFusW6xbcFugW6BbcFuoW7hbKFscW7BbYFsUW7RbxFu8Wwhb0Fu4W8RbMFu8W2BbwFu0WyBbMFvIW8xbVFtwW1RbzFvQW7BbwFtgW7BbKFu4W8hbcFsgW9hbwFuwW7hb2FuwW9xbuFvQW7RbwFvUW7Rb1FvEWzBbxFvIW8Bb2FvUW8hbzFtwW9hbuFvcW8xb4FvQW9xb0FvgW8Rb3FvIW9hbxFvUW8xbyFvgW8Rb2FvcW9xb4FvIW+xb3FvkW9xb6FvkW9xb2FvoW+Rb6FvYW9hb3FvsW9hb7FvkW/Rb+Fv8WARf9Fv8W/hYBF/8WARf+Fv0W/RYDFwAX/RYAF/wW/RYBFwMXARf9FvwWBRcBF/wWCRf8FgAXCRcEF/wWChcAFwMXABcCFwkXChcCFwAXBRf8FgQXARcGFwMXCxcFFwQXBRcIFwEXBBcJFwwXCBcGFwEXAhcKFwcXBhcKFwMXCRcCFwcXDBcLFwQXBhcIFwoXCRcTFwwXChcNFwcXDRcKFxIXBRcLFxQXEBcLFwwXERcLFxAXCxcRFxQXFhcTFwkXBxcOFwkXDxcKFwgXDxcIFwUXDRcOFwcXFBcPFwUXDxcSFwoXCRcOFxYXEBcMFxMXFBcgFw8XEhcOFw0XFBcRFxAXHxcOFxIXFRcTFxYXDxceFxIXFhcOFx8XHBcUFxAXHxcSFx4XFhcfFxcXFxcVFxYXIxcQFxMXFRcXFxgXGBcZFxUXDxcgFx4XFxcZFxgXExcVFxsXIRcTFxsXGxcVFxkXIBcUFxwXHBcQFyMXGhcTFyEXHxcjFyQXIRcfFyQXIRcXFx8XGRcXFxsXHBcjFx8XIhcTFxoXHhccFx8XFxchFxsXExciFx0XHRckFxMXHBceFyAXIRciFxoXHRciFyQXJBciFyEXJBcjFyUXExclFyMXJBclFxMXKBctFyYXJhctFycXKBcmFycXKRcnFy0XKBcnFysXKxcnFyoXKRc2FycXLBcoFysXNhcqFycXLRcoFy8XKBcsFy8XNBcsFysXLRcuFykXNhc5FyoXLRcvFzMXPRcqFzkXNRctFzMXKRc7FzYXKRcuFzsXMhcrFyoXNhc+FzkXPRdJFyoXMBcxFzUXNRcxFy0XLRdBFy4XLBc0Fy8XMBc6FzEXSRcrFzIXKxdIFzQXKhdJFzIXNRc4FzAXMRc6FzsXOxc6FzYXMxcvF0MXMBc4FzoXQBc1FzMXLRcxF0EXNhc6Fz4XQBczF0MXORc+FzcXPhdEFzcXNxdEFzkXOBc1F0AXLxc8F0MXPxdDFzwXLxc0Fz8XLxc/FzwXKxdJF0gXRBc9FzkXOhdHFz4XOxdGFzEXQRcxF04XQRc7Fy4XOBdAF0IXShc/FzQXThcxF0YXTBdJFz0XOBdCF0cXRxdCF0AXRhc7F0EXTBc9F0QXOhc4F0cXPxdKF0MXRxdEFz4XRBdNF0wXShc0F0gXRRdBF04XTRdEF0cXTBdIF0kXRhdBF0UXThdKF0UXRRdKF0gXRRdIF0wXSxdAF0MXTBdNF0UXSxdHF0AXRRdNF0YXThdLF0MXThdDF0oXTRdHF0sXThdGF0sXRhdNF0sXVxdiF1oXTxdiF1cXXxdiF08XURdQF1MXURdRF1AXVRdSF1EXURdSF1EXVRdPF1IXVRdfF08XVBdRF1MXVBdVF1EXVBdfF1UXVhdfF1QXWhdUF1cXWBdUF1oXWBdWF1QXVhdYF1kXXxdWF1kXWxdYF1oXWBdfF1kXWhdhF1sXXRdbF2EXXRdcF1sXXBdYF1sXXRdgF1wXYBdYF1wXXxdYF2AXXRdhF14XYBddF14XYBdeF2EXZhdfF2AXYxdgF2EXXxdmF2IXaxdjF2EXYBdjF2YXaxdhF1oXdRdiF2YXYhdrF1oXaRdlF2YXZRd1F2YXZRdpF2gXYhd1F2cXYxdkF2YXaRdmF2QXZRdqF3UXdxdpF2QXYxdrF3EXbhdlF2gXZBdjF3EXdRdkF3EXdRdxF2cXYhdnF2sXahd2F3UXbhdqF2UXbhdyF2oXbhdoF2wXahdyF3YXaxdnF3EXbBdoF3AXaRdzF2gXbBdvF24XdxdzF2kXdBdvF2wXdBduF28XbBdwF3QXdBdwF2gXaBdtF3QXdBdtF24XbRdyF24XdRd2F2QXbRdoF3MXZBd2F3cXdhd6F3cXcxd7F20XeRdyF20XbRd7F3gXexdzF3cXdhdyF3kXdhd5F3oXfBd5F20XbRd4F3wXeBd5F3wXeRd4F30Xexd9F3gXfRd6F3kXehd9F3sXehd7F34Xexd6F34Xehd/F3cXdxd/F3sXfxd6F3sXgBeBF4IXgBeEF4EXhBeAF4IXgReDF4IXgReGF4MXhxeFF4IXhReEF4IXgheDF4cXhBeGF4EXhxeMF4UXixeEF4UXiheDF4YXhReMF40XhBeLF4kXjReLF4UXhBeJF4YXgxeIF4cXiheQF4MXhheJF4oXjBeRF40XhxeOF4wXgxeQF4gXkheKF4kXkReMF44XiReLF48XhxeIF44XkReOF5MXlReOF4gXkxeOF5sXlhePF4sXiBeQF5cXrReWF4sXmBetF4sXmReQF4oXlxeVF4gXkBeUF5cXnBeJF48XmBeLF40XmheXF5QXkxeeF5EXmReKF5IXkBeaF5QXjheVF5sXkBeZF5oXnxeNF5EXnReVF5cXiRecF5IXnxeYF40XnReXF5oXrBeWF60XlRegF5sXmxeeF5MXqxeVF50XlhecF48XqxelF5UXlRelF6EXohegF5UXnBenF5IXkReeF58XoReiF5UXoxecF5YXpxeZF5IXnReqF6sXoxeWF6wXqhedF5oXpBeYF58XoxenF5wXnxeeF6YXmReqF5oXpBe2F5gXoxeoF6cXqBejF64XoRezF6IXrReYF6kXpxexF5kXqReYF7YXpxeoF64XphekF58XnhewF6YXoBeeF5sXmRexF6oXoBewF54Xqxe7F6UXpRezF6EXoxesF64XrxeqF7EXqxeqF7sXpxeuF7IXuhekF6YXvBesF60XsRenF7IXuhemF7AXrhesF7wXuxeqF68XtRe7F68XrRepF7gXoBe6F7AXthe4F6kXvReiF7MXohe9F6AXuBe2F7QXsheuF7wXtBe2F7gXrxexF7UXtRexF7IXuRe9F7MXvxekF7oXthekF78XwRetF7gXsxelF7sXwRe8F60Xsxe3F7kXsxe5F7cXvhe7F7UXtReyF7wXvhe1F7wXwRe+F7wXoBfGF7oXvxe4F7YXxhe/F7oXwhe4F78XxhegF70Xxxe7F74XsxfAF7kXuxfAF7MXwxe+F8EXvRe5F8AXxxfAF7sXxBfHF74XyBe/F8YXxRfBF7gXvhfDF8QXvxfLF8IXxRe4F8IXvRfAF8YXwBfHF8YXxRfDF8EXyxfFF8IXyBfLF78XxhfHF8QXxBfDF8kXyxfDF8UXxBfJF8YXxhfJF8gXwxfLF8oXyRfDF80XwxfKF80XyhfJF80XzhfJF8oXyRfOF8gXzBfOF8oXyBfOF8wXzxfIF8wXzxfMF8oXyhfLF88XyBfPF8sX0Bf4F/MX8xfRF9AX+BfQF9EX0hf4F9EX1BfRF9MX0RfUF9IX0xfRF/IX0hfUF9wX8hfUF9MX1hfSF9UX2RfSF9YX+BfSF9kX3BfUF/IX1hfWF9UX1xfWF9YX2BfZF9cX1xfZF9YX+BfZF9gX2xfaF98X2xfbF9oX2xfhF9sX5BfdF+MX4xfdF9wX5BfeF90X5BflF94X4BfbF98X4hfbF+AX4RfbF+IX9Rf4F9gX8hfjF9wX7BfkF+MX5RfkF+wX9RfYF+8X6xfmF+kX6xfnF+YX7RfnF+sX7RfoF+cX7hfoF+0X7xfoF+4X6BfvF9gX4xfyF+wX6hfrF+kX7BftF+oX6hftF+sX7BfuF+0X7BfvF+4X9RfvF+wX8hf0F+wX8Bf1F+wX8BfsF/QX8Bf0F/EX9RfwF/EX9Bf1F/EX9BfyF/cX8hfRF/cX8xf2F9EX/BfzF/gX/xf1F/QX9xf/F/QX9Rf/F/gX8xf5F/YX0Rf2F/oX+hf3F9EX9hf5F/sX8xf8F/kX+hf2F/sX/xcDGPgX/Rf/F/cX+BcDGPwX/Bf+F/kX+xcBGPoX+hf9F/cXARj7F/kXARj9F/oX/xcCGAMYBBj/FwAY/hf8FwMY+Rf+FwEYBBgCGP8XABj/F/0X/RcEGAAYBBgDGAIYARgDGP0XBBj9FwMYARj+FwUYBRj+FwMYAxgBGAUYCRgGGAgYChgIGAYYCRgHGAYYBhgHGAoYChgJGAgYCRgKGAcYCxgWGA0YDxgWGAsYDBgSGA4YDhgVGAwYFhgQGA4YFhgPGBAYERgLGA0YDxgTGBAYERgPGAsYFRgOGBAYGRgQGBMYGRgVGBAYFBgNGBYYDBgVGBIYDRgUGBEYFhgOGBIYDxgqGBMYFhgPGBQYKhgfGBMYExgfGBkYHxgjGBkYFBgPGBEYGxgeGBcYEhgVGCUYIBgSGCUYHBgYGBoYHRgaGBgYHBgbGBcYFhgqGA8YKRgYGBwYGxgcGBoYIhgVGCgYGRgoGBUYJRgVGCIYEhggGBYYKRgdGBgYFhggGCoYFxgeGCQYHBgtGCkYIxgoGBkYKBgjGCIYJRgnGCAYKRg0GB0YHBgXGC0YJBgeGCEYJhgtGBcYHRg0GCIYGhgdGCwYLBgbGBoYKxgfGCoYMRgXGCQYFxgxGCYYIhgjGB0YKxgjGB8YLBgeGBsYHhgsGCEYIRgsGDAYLRgvGCkYKhguGDIYLBgrGDAYNBgpGC8YJxguGCAYHRgzGCwYIBguGCoYKhgyGCsYNRgnGCUYJBghGDAYNhgdGCMYNhgzGB0YKxgzGCMYMRgkGDwYIhgjGDUYKxgsGDMYOhgtGCYYJhgxGDwYLRg5GC8YIxgiGDYYOhgmGDwYNRglGCIYMxg1GCMYMBg8GCQYOhg5GC0YIhg0GDYYOxg8GDAYNhg7GDMYMhguGCsYORhAGC8YLxg4GDQYMxg3GDUYMBgrGDMYMxg7GDAYNRg3GCcYKxguGDMYOxg2GDgYNhg0GDgYMxguGDcYQBg4GC8YPRg5GDoYSRg4GDYYJxg3GC4YPhg5GD0YPxg9GDoYOxhCGDwYNhg4GEkYPBhDGDoYOBhAGEkYQhhDGDwYQRg+GD0YPRg/GEEYOxg4GFIYPhhPGDkYORhPGEAYPxg6GEMYRBhIGEEYOBhJGFIYUhhCGDsYRxhAGEsYRBhBGD8YRRg/GEMYQhhGGEMYQRhIGD4YQBhPGEsYQBhHGEkYUhhGGEIYPxhIGEQYRRhIGD8YRRhDGEoYTRhPGD4YTRg+GEgYVRhKGFMYShhDGFMYThhKGFUYVxhJGEcYShhMGEUYWRhDGEYYRhhSGFkYTRhIGEUYVxhSGEkYQxhZGFMYSxhXGEcYSxhPGFEYTBhKGE4YTRhFGEwYTxhNGEwYVhhSGFcYURhPGEwYWxhMGE4YWxhYGEwYWxhOGFUYSxhRGFcYWhhUGFAYUhhbGFkYVhhbGFIYURhaGFcYWBhRGEwYWRhVGFMYUBhWGFoYUBhUGFYYVxhaGFYYWRhbGFUYVBhbGFYYURhYGFoYWBhbGFQYWBhUGFoYnQ+XD5cPlw+fD50Plw+XD6APlw+gD58PYBhfGFwYYBhdGF8YYhhdGGAYXhhiGGAYYBhcGGEYYRhcGGUYYBhhGGUYaBhcGF8YZhhdGGIYYBhpGF4YZhhfGF0YZRhpGGAYXhhrGGMYYhheGGMYYxhrGGIYZRhcGGQYaBhkGFwYXxhmGGcYaxhmGGIYZBhoGGUYaRhsGF4YZxhoGF8YaBhnGGYYaBhqGGUYbBhrGF4YZRhqGGkYahhoGGYYZhhtGGoYZhhrGG0YaRhqGGwYaxhsGG0YbBhqGG4YbRhsGG4YbRhuGGoYdxhzGG8YcBh3GG8YcxhwGG8YdxhwGHEYcBhzGHQYcBh6GHEYehh3GHEYehhwGHQYchhyGHIYchhyGHIYchhyGHIYchhyGHIYdhh6GHQYcxh2GHQYdhhzGHUYcxh4GHUYeBh2GHUYehh2GHgYdxh7GHMYehh4GHkYfxh7GHcYdxh6GH0Ycxh7GHgYeRh8GHoYdxiAGH8Ydxh9GIAYeRh4GHsYgxh6GHwYfRh6GIMYfBh5GIEYgRiIGHwYeRh7GIIYeRiCGH4YfhiBGHkYfxiAGIcYgxiJGH0Ygxh8GIgYiRiAGH0YgRh+GIwYgRiMGIgYhRiHGIAYfhiCGJMYfxiEGHsYgBiKGIUYexiEGJIYgxiIGJAYghh7GIYYhxiPGH8YgBiJGIoYghiGGJMYhBh/GI8YnBiIGIwYgxiOGIkYexiSGJgYmRiPGIcYmBiGGHsYhRiUGIcYhxiUGJkYhRiKGJYYmxiMGH4YjhiDGJAYmBiLGIYYjRiJGI4YmxicGIwYnBiQGIgYkxibGH4YihiJGJYYhBiXGJIYjRiOGJUYmRiRGI8YlBiFGJYYhhidGJMYjRiWGIkYnBiqGJAYkBikGI4YjxiRGIQYmRitGJEYnhiYGJIYtxicGJsYlBiWGJ8YqBiNGJUYqRiZGJQYpxiXGIQYnhiSGJcYnhiLGJgYnRiGGIsYlBifGKAYlRiOGKQYoxibGJMYrRjJGJEYtxiqGJwYlhiNGJ8YtxibGJoYoxiaGJsYlxinGLAYnhilGIsYrBikGJAYkBiqGKwYnRijGJMYshiaGKMYqRihGJkYoRitGJkYphieGJcYixilGJ0YnhimGKIYnxi1GKAYqxifGI0YnhiiGKUYqBiVGLYYthiVGKQYqxi1GJ8YsRiXGLAYoBipGJQYlxiuGKYYwhidGKUYqxiNGKgYtBiEGJEYsxi3GJoYsxiaGLIYrhiXGLEYuhijGJ0YvhiiGKYYhBi0GKcYvRirGKgYuBilGKIYyRjBGJEYtBiwGKcYtxivGKoYrBiqGK8Yuhi5GKMYoRjAGK0YuBjCGKUYqxi/GLUYuhidGMIYoRipGLsYkRjBGLQYvhi4GKIYwxiwGLQYrxjZGKwYoxi5GLIYsBjcGLEYsRjKGK4YrRjAGMkYoBi7GKkYvRioGOAYqxi9GL8YrxjiGNkYqBi2GOAYvxjUGLUYthikGNgYsRjEGMoYrhjKGKYYpBisGNgYtRjHGKAY2RjYGKwYuxjAGKEYxxi7GKAYsRjcGMUYvBivGLcY5RizGLIYuBi+GMYYphjdGL4YwRjNGLQYxxjMGLsYxBixGMUY8hi3GLMY2xi2GNgYvBjIGK8YwxjcGLAY0hjCGLgYxxi1GNQYuBjGGNIYyhjdGKYYuxjLGMAYvRjfGL8YrxjIGOIY+hi8GLcYthjbGOAY8hj6GLcYBBnKGMQYuhjCGNoYuxjMGMsYwxi0GM0YvhjdGOoY5RiyGLkYxBgHGQQZwxgNGdwY6xi9GOAY5RjyGLMY1Bj+GMcY3hjnGMAYwBjnGMkYxRgHGcQYyhgEGd0Y4RjlGLkYwRjJGPkY3xi9GOsY0hjGGNMYzBjXGMsY2BjZGOMYARnQGMYY0xjGGM8YwhjSGNoYxRjkGAcZxhjQGM4YxhjOGM8Y7xjHGP4YzhjQGM8YzxjQGNMY3hjAGMsY0BjSGNMY0hjQGNEYyRjnGPkY0BgBGdYY0BjWGNEY1hjSGNEYxhi+GAEZ0hjWGNUYDxnDGM0Y7xjMGMcY1hgBGdUY1RgBGdIY3BjkGMUY2hjSGOkYARm+GOoY7xjoGMwYvBgaGcgY/hjUGL8YuRi6GOEYwxgPGQ0ZvxjfGOYY6xjgGPQY4hjIGAwZERnkGNwY4hgIGdkY1xjMGBgZzBjoGBgZGhm8GPoY3RgEGQIZHxngGNsY3hjLGAMZ+xjoGO8Y4xjZGAgZuhj9GOEY7RjyGOUYDRkRGdwY8BjnGN4YvxjmGP4YCBniGAwZAxnLGNcYwRjuGM0Y3hjsGPAYDBnIGBoZ2xjYGAAZDxnNGO4YAxnsGN4Y7hjBGPkYuhjaGP0Y4BgfGfQYAhkeGd0Y7RjlGOEYARnpGNIY3RgeGeoY8RjvGP4Y7RjhGPwY/xj6GPIYGBkDGdcY9xjwGPMY8BjsGPMY8Bj3GPYY7Bj3GPMY9hj1GPAY+Bj2GPcY+Bj1GPYY5xgFGfkY8BgFGecYBRnwGPUY9Rj3GAUZ9Rj4GPcY7BgFGfcY/xjyGO0YDhnoGPsY/Rj8GOEYGxnmGN8Y5BgWGQcZCBkZGeMYARklGekY/hgbGfEY/xgaGfoY7xgSGfsY5hgbGf4Y6xj0GN8Y4xgAGdgYBhkHGSYZHhkUGeoYBxkGGQQZAxkYGSIZ6BgOGRgZ/xjtGBUZBRnsGBAZ+xgSGQ4ZBRkQGfkYARnqGBQZERkWGeQYGRkAGeMYGhkjGQwZ7hj5GBAZ8RgSGe8Y2xgAGR8ZLBn/GBUZHBn8GP0Y/BgKGQkZCRntGPwY/BgcGQsZChn8GAsZChntGAkZFhkmGQcZ2hjpGC0ZHBkKGQsZLRnpGCUZ7RgKGRUZ2hgtGSgZNhkRGQ0Z7hgQGS4ZAxkdGewYDRkPGTYZKhkbGd8YDhkkGRgZJBkOGRIZIBksGRUZOhkPGe4YExkKGRwZExkVGQoZHBkVGRMZ9BgqGd8YKBn9GNoYIhkdGQMZFRkcGRcZFxkcGSAZIBkVGRcZLhk6Ge4YHxkhGfQYARkUGSUZBBkGGQIZLBkaGf8YKRkkGRIZKRkSGfEYOhk2GQ8ZERk3GRYZKxkIGQwZIxkaGSwZIhkYGSQZKBkcGf0Y7BgdGRAZGRlBGQAZIxkrGQwZGxkzGfEYJBkpGTUZNBkiGSQZIhkvGR0ZAhkGGUQZIxksGUYZGRkIGSsZRxkUGR4ZNhk3GREZMxkbGSoZHRkvGRAZNBkvGSIZIRkqGfQYJxkhGR8ZMxkpGfEYOBlEGQYZRRklGRQZKBkgGRwZNRk0GSQZORkpGTMZIRkwGSoZRRktGSUZIxlGGSsZEBk7GS4ZJxkfGQAZPBkvGTQZMRkoGS0ZOxkQGS8ZOhkuGTsZNRkpGTkZKhkyGTMZMBkyGSoZRhksGT0ZMBkzGTIZJhk4GQYZQBkmGRYZQRknGQAZRRkUGUcZPRksGSAZSBlBGRkZSBkZGSsZPRkgGSgZQBk+GSYZKxlGGUgZNRlDGTQZLxk8GTsZQxk1GTkZKBkxGT0ZNxlAGRYZQhk3GTYZPBk0GUMZHhkCGVgZORkzGUkZOhlTGTYZPBlDGT8ZOBkmGT4ZHhlYGV4ZMRktGUUZOhk7GVMZORlJGUMZIRknGVAZTRlGGT0ZOxk8GT8ZRBlYGQIZSxk9GTEZTxkzGTAZIRlQGTAZQhk2GVMZQxlJGUwZPxlOGTsZHhleGUcZTxlJGTMZQxlMGT8ZQhlAGTcZSxkxGUUZSxlNGT0ZUhknGUEZQRlKGVIZTRlIGUYZShlIGU0ZUxk7GU4ZTBlJGVQZSxlFGVoZPxlMGVQZVBlJGU8ZJxlSGVAZQRlIGUoZVBlPGVUZUBlPGTAZVBlOGT8ZVhlPGVAZSxlXGUoZURlbGUAZSxlKGU0ZQhlRGUAZUxlOGVEZVRlPGVYZUBlSGV0ZXRlSGVcZVxlSGUoZWRk4GT4ZRRlHGVoZUxlRGUIZVhlQGV0ZRxleGVoZXBlLGVoZWRk+GUAZThlUGVsZThlbGVEZVhldGWAZYRlVGVYZXxlUGVUZVhlgGWEZVxlLGVwZWxlZGUAZRBk4GWAZRBlgGVgZXRlXGVwZVBlfGVsZOBlZGWMZXhlcGVoZOBljGWAZVRlhGV8ZWBlcGV4ZXxlZGVsZWBlgGVwZYBldGVwZWRliGWMZYRliGV8ZYRlgGWMZWRlfGWIZZBljGWIZZBliGWYZYhlhGWYZYRlkGWYZaBlkGWEZYxlkGWcZYxlnGWUZZBllGWcZZRlkGWgZZRloGWEZYxllGWEZfBlrGWkZaRlqGXwZaxlqGWkZaxluGWoZchluGWsZbhlyGXMZbRlwGXEZdhlvGXEZgBl8GWoZcBmAGWoZghlsGW8ZdRluGXcZbRlxGXkZbxlsGYcZdxluGXMZexmHGWwZdhmCGW8ZdRlqGW4ZehlyGWsZeRmAGW0ZcBltGYAZeBlxGW8ZfRlwGWoZbxmHGXQZcRlwGXYZbxl0GXgZehlrGX8ZfBl/GWsZhxl4GXQZkRlzGXIZfhl3GXMZcRl4GXkZexlsGYMZeRmBGYAZghmDGWwZgRl2GXAZeRl4GYEZkRlyGYoZdRl3GXwZgRlwGX0ZfRlqGXUZiRmKGXIZiRlyGXoZgBl1GXwZjRl4GYcZhBmBGX0Zehl/GX4ZeBmCGYEZdhmBGYIZfhl8GXcZfhl/GXwZiBl7GYUZhBmAGYEZeBmNGYIZgBmEGXUZhBl9GXUZghmNGYMZgxmFGXsZixl+GXMZfhmJGXoZihmPGZEZhhmFGZIZkhmFGYMZiBmHGXsZixlzGZEZixmJGX4ZlhmPGZMZlBmHGYgZkRmPGZYZjRmHGZQZhRmMGYgZiBmMGZAZlRmGGZIZhRmQGYwZhRmGGY4ZgxmNGZIZkBmOGYYZhRmOGZAZlRmQGYYZkBmUGYgZihmJGY8ZlBmSGY0ZkxmXGZYZjxmXGZMZixmRGZYZlBmQGZUZlRmSGZQZiRmXGY8ZlxmJGYsZlxmLGZYZixmYGZcZlxmYGYsZmhmZGZwZmxmcGZkZmhmbGZkZmxmaGZ0ZnBmdGZoZnRmcGZsZpRmjGaAZoBmhGaUZnhmhGaAZoxmhGZ8ZoRmeGZ8ZnxmgGaMZohmhGaMZoxmlGacZpBmlGaIZohmlGaEZpRmmGacZpRmwGaYZpxmrGaMZphmqGacZqxmnGa8ZrhmlGaQZqxmtGaMZtBmsGagZqhm0GagZrRmiGaMZpRmuGbMZpRmzGakZpRmpGbAZqhmvGacZsRmoGawZphmwGaoZsRmqGagZsRmsGbgZqxmpGbMZrxmqGbEZqxmvGakZqhmwGbcZpBmiGa0Ztxm0GaoZrBm0GbIZrxmwGakZrRmuGaQZuRmtGasZuRmrGbMZuhmvGbEZrBmyGbUZsxmuGbkZrBm1GbgZtRmyGbgZuBm2GbEZrxm3GbAZtBm7GbIZuBm6GbEZthm4GbEZrRm5Ga4Ztxm7GbQZshm6GbgZuhm3Ga8Zshm7GboZvBm7GbcZuxm8GboZtxm6GbwZvhnDGb0Zwxm/Gb0Zvxm+Gb0ZwBnDGb4ZvxnDGcEZwxnFGcEZvxnNGb4ZwxnEGcUZvhnCGcAZ0RnAGcIZvhnRGcIZyhm/GcEZzhnNGb8ZyBnHGcYZwxnAGcQZxRnIGcYZxRnGGckZxRnEGcgZzhm/GcoZyhnBGckZwBnLGcQZxBnTGcgZyRnBGcUZ1BnHGcgZxxnZGcYZ0RnLGcAZxxnUGdkZzBnRGb4ZyxnKGcQZvhnNGcwZ0xnUGcgZyxnOGcoZxBnKGckZxBnJGdUZ0BnUGdMZyxnSGc4ZyxnRGdIZ0BnPGdQZzBnNGdIZxhnVGckZzRnOGdIZ1RnTGcQZ0RnMGdIZ1hnGGdkZ1BnPGdwZ1hnVGcYZ1BnXGdkZ0BnYGc8Z2xnTGdUZ3BnPGdgZ2hnZGdcZ1BnaGdcZ2hnWGdkZ2BnQGdMZ3BnaGdQZ0xnbGdgZ1RnWGdsZ3RnWGdoZ3BnYGd0Z3RnaGdwZ3RnYGdsZ2xnWGd0Z2xneGd0Z2xndGd4Z5BngGd8Z4BnkGd8Z5BnlGeAZ4BnlGegZ4BniGeQZ4RnpGeAZ4BnjGeIZ4xnkGeIZ5hngGekZ4xngGeYZ4RngGegZ5hnpGeMZ4RnoGekZ5xnnGecZ5xnnGecZ5xnnGecZ5xnnGecZ6BnlGeQZ5BnjGegZ6BnjGeoZ6BnqGekZ4xnpGeoZ6xnyGewZ7BntGesZ6xntGfIZ8hn7GewZ7Bn2Ge0Z7Rn2Ge4Z+xnxGewZ7xn9Ge4Z8RnwGewZ7RnuGfgZ9hnvGe4Z7BnwGfMZ9BnsGfMZ7Bn0GfYZ+hnyGe0Z7hn9GQca8xn1GfQZ7Rn4GfcZ+BnuGQca+hntGfcZ8xnwGQka+hn3GfgZ/xn9Ge8Z+Bn5GfoZARr6GfkZ9hn0GfUZ7xn2GfUZ+BkHGg8aCxrwGfEZ+xn+GfEZ9Rn/Ge8Z+BkBGvkZ8Rn+GQMaCRr1GfMZ+hn7GfIZ+Bn8GQEa+BkPGvwZDxoBGvwZ8BkLGgkaCRr/GfUZ+hkOGvsZAxr+GQAa8RkDGgoa+xkOGv4Z/RkMGgca/hkDGgAaAxr+GQIaChoLGvEZAxoCGgQa/hkEGgIa/Rn/GQwaBhoDGgUaAxoEGgUaBBr+GQYaChoDGggaAxoGGggaBBoGGgUaERoJGgsa/hkOGgYaChoIGgYaEhoBGg8aEBr/GQkaDhr6GQEaDhoKGgYaBxoMGg0aDRoVGgcaFRoNGgwaEBoMGv8ZDhoUGgoaDxoHGhUaChoRGgsaERoQGgkaFBoOGgEaExoRGgoaDxoVGhIaExoQGhEaChoUGhMaEhoUGgEaFRoMGhAaEhoWGhQaExoUGhYaFhoSGhUaFhoYGhMaExoYGhAaEBoXGhUaGBoXGhAaGBoWGhoaGRoYGhoaGRoaGhcaFxoaGhUaFRoaGhYaGxoYGhkaGRoXGhsaGxoXGhgaHBolGi4aJholGhwaLhomGhwaJhouGh0aHRouGiMaIxomGh0aHholGiYaHhonGiUaNhouGiUaHxokGiAaLhohGiMaJhojGiQaIhoeGiYaMBogGiQaMxoeGiIaHxomGiQaIRouGjEaMxonGh4aHxoiGiYaMBoiGh8aIxohGioaORokGiMaHxogGjAaKBojGioaKRohGjEaLRohGikaKhohGi0aKBorGiMaMRotGikaKhotGjUaKBo1GisaNRojGisaMRosGi0aMRo4GiwaLBo4Gi0aKho1GigaNRo5GiMaLRo4Gi8aNRotGi8aMBokGjkaOBo1Gi8aIho0GjMaLho4GjEaJxo2GiUaNBoiGjAaOho4Gi4aMhonGjwaOhouGjYaPBonGjMaMxo0Gj8aMho2GicaORo3GjAaQRowGjcaMBpBGjQaPxo8GjMaNhoyGjoaOBo7GjUaOxo4GjoaORo1GjsaQRo/GjQaMho8GkAaOhpAGjsaQBo6GjIaORpBGjcaPBo9GkAaPBo/Gj0aQRo5GjsaQRo+Gj8aPRpEGkAaQxo9Gj8aQxo/Gj4aOxpSGkEaQxo+GlEaQhpAGkkaQBpCGksaSxo7GkAaPhpBGlIaVRpAGkQaSRpAGlUaRBpJGlUaWRpEGj0aRRpJGlYaVhpJGkQaVhpEGlkaSRpFGlgaRRpWGlcaWBpKGkkaRRpKGlgaRRpXGkYaSxpCGkcaRxpCGl0aQhpIGl0aWRpFGloaWhpFGkYaWhpGGlsaWxpGGlwaRhpXGlwaSxpHGl8aSBpHGl0aQhpJGkgaXhpJGkoaShpFGlkaPRpNGlkaSBpLGl8aRxpIGl8aSRpMGmIaXhpMGkkaShpMGl4aYhpIGkkaTRpKGlkaTRo9GmAaYxpMGkoaShpNGmAaSxpIGmIaShpDGmMaQxpgGj0aSxphGjsaSxpMGmEaThpjGkMaQxpKGmAaYRpMGjsaTBpLGmIaYxpOGmUaTBpjGmQaZhpOGkMaThpmGmUaOxpMGlIaUhpMGmQaURpmGkMaTxpoGlAaUBpoGmYaURpQGmYaUhpkGlMaUxpkGmcaaBpPGmkaaRpPGlAaaRpQGmoaahpQGlEaUxpnGmoaUhpTGmoaUhpqGlQaVBpqGlEaVBpRGmsaUhpUGmwaVBo+GmwaPhpUGmsaPhprGlEaPhpSGmwabhpvGm0abhptGm8acBpyGnEachpwGm8achpwGnEacBpyGm8acxpwGnIachpwGnMadRp2GnQadxp1GnQadxp0GnYadRp3GnYaeBp5GnoaeBp6GnkaeRp6Gnsaexp6Gn8afBp/Gnoaehp/GnwafRp6GnkaeRp7Gn8afhp/GnoafRp+GnoafRp5Gn4aeRqAGn4afxp+GoAaeRp/GoAahhqFGoEaghqGGoEaghqBGoUaiBqFGoYahBqDGocaghqFGo8aihqHGoMajhqKGoMahBqUGoMahxqLGoQaiRqOGoMahRqIGo0ahhqcGogalBqJGoMaihqMGocahRqNGo8ajxqNGqYaixqGGoIakRqMGooaihqOGqkaqBqOGokaoBqLGoIarRqNGogalhqCGo8aiBqcGpMakRqsGowahhqhGpwaoBqEGosakBqTGpwaoBqCGpcalhqXGoIakRqKGqkalBqEGqAalRqWGo8anhqVGo8alhqVGpIamhqWGpIalxqWGpgalRqaGpIalhqaGp0anhqXGpgamRqsGpEaohqZGpEalhqeGpgalxqeGpsarBqZGqIaohqRGqkanxqQGpwanhqaGpUamhqeGp0anhqWGp0anhqgGpsaoBqXGpsarBqqGowaixqhGoYarRqnGo0akxqtGogaoRqLGocajhqkGqkakBqfGpMaoRqfGpwajhqoGqMasBqeGo8aqxqOGqMalBqxGokaqxqjGqgaphqNGq4aohqpGqwapRqkGo4ajhqrGqUaoRqHGowaoRqvGp8apBqlGqsaqxqoGrEarhqNGqcasBqPGqYalBqgGrEakxqfGq0aiRqxGqgasBqgGp4arRqfGq8atRqhGowaqhq1GowarxqhGrUarhq0GqYasxqqGqwasBqxGqAarhqnGrQarBqpGrMasBqmGrQaqhqzGrUapxqtGrQasRqwGrIapBqrGqkaqxqzGqkarRqvGrQaqxqxGrMashqwGrQarxq1GrQasRqyGrgatBq1GrIauBq2GrEatRqzGrsasRq7GrMathq7GrEatxq7GrYauRqyGrUatRq7GrkauBq7Grcatxq2GroauBq3GroauBq6GrYauBq5GrsauBqyGrkawxq8Gr0awBq9GrwawxrAGrwavxrFGr4avRrCGr8awxq+GsQavxrIGsUawhq9GsAavhrFGsYawxq9Gr8avhrMGt8avhrfGsQawRrIGsIawhrIGr8azhrCGsAawBrDGsQavxq+GsMazhrAGskazBq+GsYawhrOGtEaxRrLGsYaxxrFGsgaxxrNGsUazRrLGsUayRrAGsQa0RrBGsIawRrKGsga3BrOGskaxBrWGska0RrSGsEayBrKGscaxBrfGtYazxrOGtwa0hrKGsEayxrhGsYayhrNGscazhrPGtEa0BrfGswaxhrQGswa3BrJGtYa1RrNGsoa1xrVGsoayxrUGtga0RrTGtIazRrgGssa4BrUGssaxhrdGtAa2xrcGtYazRrVGtkayhrSGtca7hrWGt8a6BrGGuEa1xrmGtUa1BrgGtga2BrhGssa5xrWGu4a5hraGtUa1xrSGtMa6BreGsYa1RraGtka2RrgGs0a6xrTGtEa0Br+Gt8a3RrGGt4a5BrPGtwa5hrZGtoa3BrbGuQa0BrdGv4a0xrtGtca4BriGtga1xrtGuYa2BriGuEa5BrvGs8a7RrpGuYa4BrZGuMazxrrGtEa5hrjGtka6BrhGvga4BrjGuIa/hrdGgMb8xrbGtYa8xrWGuca6hrnGu4a7xrrGs8a8xrkGtsa5RrTGusaCRveGuga6hryGuca5hrpGvQa5hrsGuMa7BrmGvQa5RrtGtMa/hoVG98aAxvdGt4aFRvuGt8a5RoFG+0a+BrhGuIa8hrzGuca7RoFG+ka4xr4GuIa+BoJG+ga5BrzGvEa8BryGuoa8RrvGuQaCRsDG94a6hruGgQb8Br6GvIa/Rr4GuMaFRsEG+4aBRv0Guka7Br5GuMa+Rr9GuMa+RrsGvQaEhvrGu8a8BrqGvUa9RoEG/Aa6hoEG/Ua9hrwGgQbDRv0GgUb9xrwGvYaAxsVG/4aBhvwGvcaBhv6GvAa8RoTG+8a9hoEG/ca9xoEGwYb+BoMGwkb/xr8Gvsa+xr8Gvwa/BoBG/waABv8Gv8aAhv8GgAbAhsBG/waDxsNGwUb5RoIGwUb9BoRG/kaERv0Gg0bBhsEG/oaDhsIG+Ua+RoRG/0aExsSG+8aGhv6Ggcb+hoEGwcbBBsaGwcbChsFGwgb/RoRGxAbCRsXGwMbDxsFGwobDBv4Gv0aCBsPGwob5RrrGhIbDxsIGwsbExvxGvMaDBv9GhAbDxsOGw0bCBsOGwsbDhsPGwsbFBsQGxEbGRsDGxcbFRsaGwQb5RoSGw4bHxvyGvoaDBsQGwkbEBsXGwkbDRsWGxEbFBsRGxYbHxv6GhobDRsOGxIbDRsSGxMbEBsUGxcb8xryGh8bGRsVGwMbHxsbG/MaExsYGw0bFBsWGxwbFBscGxcbFhsNGxgbGxsTG/MaIBsaGxUbGRsXGxwbGRsgGxUbGBsTGxsbGBsbGxYbGxscGxYbGRscGx0bHRsgGxkbHRsbGx8bHhsfGxobHRscGxsbHRsfGyEbHhsdGyEbHxseGyEbIBsdGyIbHRseGyIbIBsiGx4bIBsjGxobIxsgGx4bGhsjGx4bJBslGykbJBsnGyUbKRsnGyQbJhssGyUbJRssGykbKBslGycbKBsnGyobKxsmGyUbJhsuGywbLBstGykbKBsrGyUbJxspGy8bLxsqGycbLBs1Gy0bKRswGy8bLhs1GywbKRstGzAbJhsrGy4bKhsxGygbKxszGy4bLRs1GzIbKBs6GysbMRs6GygbLxswGzkbNxszGysbOBsqGy8bMBstGzIbKhs7GzEbOxsqGzgbLxs5GzgbMBsyGzYbMxs0Gy4bNBs1Gy4bMBs2GzkbNxs+GzMbOhs3GysbORs2G0IbNxtFGz4bMxs9GzQbMhs8GzYbNRs0G0AbMhs1GzwbMRtDGzobOxs/GzEbPxs7GzgbRxszGz4bQxsxGz8bPRtBGzQbQBs8GzUbPBtCGzYbRxs9GzMbOBtEGz8bNBtBG0AbOhtFGzcbTxs4GzkbQhs8G00bQhtKGzkbRBs4G08bRRs6G0MbShtPGzkbRRtHGz4bRBtPG0wbPBtAG1MbShtCG1IbPRtIG0EbVBs/G0QbRhtQG0MbRhtDGz8bSRtGGz8bSxtSG0IbUBtFG0MbQBtBG1MbQhtNG0sbRBtMG1QbRxtIGz0bPBtRG00bVRtJGz8bVRs/G1QbPBtOG1EbUxtOGzwbRxtFG1YbVxtFG1AbXxtTG0EbWBtIG0cbSBtvG0EbbxtfG0EbShtqG08bVBtMG2UbWRtLG00bZRtMG08bXxtaG1MbTRtRG1sbTRtbG1kbWRtSG0sbRxtWG1gbVxtkG0UbUxtwG04bZBtWG0UbXRtGG0kbUxtaG3AbURtmG1sbVBtlG1wbShtSG2obThtmG1EbSBtYG3MbWxtsG1kbZRtPG2obZhtsG1sbRhtdG1AbVRtdG0kbYhtfG28bcBteG04bahtSG2gbVRtUG3gbZBtQG10bXBt4G1QbUBtkG1cbXxthG1obbRtSG1kbZxtdG1UbZhtOG2kbdhtkG10bYBtzG1gbZBtgG1YbbBt1G1kbdxtSG20bYxtlG2obVhtgG1gbVRt4G2cbSBtzG28bUht3G2gbWhthG3AbXxtiG2EbThteG2kbbxt5G2IbahtoG2sbaRteG4AbZhtyG2wbYRt6G3AbdhuCG2QbdxtrG2gbdBtkG4IbZRtxG1wbXhtwG4AbZBt+G2AbYRtiG3obfhtkG3QbWRt1G20bXBtxG3gbbhtlG2MbextvG3MbchtmG2kbcBt6G4AbZxt2G10bcRtlG24bbBtyG3UbahtrG2Mbght2G4MbbxuFG3kbfRtxG24bbxt7G4UbhBt2G2cbhBtnG3gbeht/G4AbchuIG3Ubhxt9G24bkBtrG3cbihtuG2MbfBtzG2AbfBtgG34bkBuMG2sbeRuFG2IbkxtpG4AbaxuKG2MbYhuLG3obiRttG3UbfBt7G3MbaRuIG3IbeBtxG5EbgRt+G3QbehuLG38bgRt0G4Ibgxt2G4QbkRuEG3gbkxuAG44bbRuQG3cbaxuMG4obihuHG24biBuJG3UbjxuLG2IbfhuGG3wbhRuPG2IbiRuQG20bfRuHG6Ubhht+G4EbhBuRG58biBtpG5gbkRtxG5YbghuDG40bfxuOG4AbaRuTG5gboRt9G6UbcRt9G6EbhBuSG4MbnxuSG4QbihuMG5UbjRuUG4IbiBueG4kbhxuKG6MbhRt7G6cblBuBG4IboRuWG3EbgxuSG5objhusG5MbuhuMG5Abnhu6G5AbfBuZG3sbjxuiG4sbmhuNG4MbfxuLG6YbnhuQG4kbmxufG5EbhxuXG6UbhhuZG3wbhhuBG50brBuYG5MboxuXG4cbjxuFG6IbhhudG5wbfxumG44boBuWG6EbexuZG6cbrBuOG6YbmxuRG5YbiBuYG54boBubG5YbpxuiG4UbgRuUG50bnBuZG4YbohumG4sblRujG4obnBuyG5kbuhuVG4wbnRuUG7kbpBuYG6wbkhufG7cblRu6G8IboxuVG8IbnBudG7MbmhuSG64bpxuZG8kbmRuyG6obqRueG5gbyRuZG6objRu0G5Qbrhu0G5obohunG7YbuhueG6kbmxugG78bpRuXG8AbnBvEG7IbxBucG7MbpBupG5gbmxu3G58boxvCG7sboRulG6gbrBumG60btBuNG5obtxuuG5IbuRuUG+8blxujG7sbpRvAG6gbohu2G6YbpxurG7Ubpxu1G6sbpxu1G7YbsRubG7gbuBubG78brxu3G5sbrxuxG7cbmxuxG68buxvaG5cbsRuwG7cbsRu3G7AboRvDG6AbtBvvG5QbtxuxG7gb7RudG7kbwxuhG6gbqhuyG8YbthvqG6YbpBusG70brRumG+obvBu7G8IbshvEG+sbwxvBG6AboBvBG78brRu+G6wb6xvGG7IbnRvtG7MbqRvCG7obwBuXG9obpBviG6kbyRu1G6cbvBvCG+QbrRvKG74bqRvHG8Ib2hu7G7wbyBuoG8AbuBu/G/ob6hu2G8UbxBuzG+wbzBu0G64buRvvG+0bqBvIG8MbuBvzG7cbtRvFG7Yb6hvKG60b+hu/G8EbrBu+G/EbrBvxG70brhu3G8wbsxvtG+wbqRviG8cbtxvzG8wbChzvG7QbuBv6G/MbyRvuG7UbzBsKHLQb6xvEG+wbvRviG6QbwBvaG/YbwhvHG+QbyxvOG80b0BvOG8sb0BvLG80b0BvNG9EbzxvNG84b0RvNG88b0BvPG84b0BvRG88b0hvWG9Qb0hvUG9Ub2BvWG9Ib2BvSG9Ub2BvVG9kb8BvkG8cb1xvUG9Yb1xvVG9Qb2RvVG9cb2BvXG9Yb2BvZG9cb3BvbG94b3RvbG9wb4BveG9sb4BvbG90b4BvdG+Eb3xvcG94b3xvdG9wb4RvdG98b4BvfG94b4BvhG98b8Bu8G+Qb4xvnG+Ub4xvlG+Yb6BvnG+Mb6BvjG+Yb6BvmG+kb4xvlG+cb4xvmG+Ub6RvmG+Mb6BvjG+cb6BvpG+MbxRu1G+4bvBv3G9ob+BvIG8Ab+BvAG/YbyBvTG8MbBRzMG/MbwRvDG9Mbvhv+G/EbvBvwG/Qb4hv1G8cb8BvHG/IbyRv9G+4b8Rv/G70b9hvaG/cb4hu9G/8bBRwKHMwbyRuqG/kbqhvGG/kb/xv1G+IbvhvKG/4b9RvyG8cb7BvtGwEc/RvJG/kbARztG+8bChwFHBYcABz1G/8b6xvsGw0cAxzBG9MbDBzKG+ob6xsNHMYbBBz/G/EbAhzvGwoc/xsEHAccDxzTG8gb8hv1Gwgc8BvyG/Qb/BsNHOwb+hvBGwMc+BsPHMgbDBwYHMobyhsYHP4bARz8G+wb9xu8Gwsc+RvGGw0cARzvGwIc6hv7Gwwc8Rv+GwQcHBz7G8UbxRv7G+obBRzzGy0cCBwOHPIb9BsLHLwb+RsNHBEcxRvuGxwcHhzyGw4c9xv4G/YbCRwDHNMbDxwJHNMb/Rv5GxAc7hv9GxwcABz/Gzcc8hsUHPQb+hsaHPMb+Bv3GxMcERwQHPkbHhwUHPIbFBwLHPQbBxw3HP8bBRwZHBYcPBz+GxgcIhwBHAIcChwWHAIcIhz8GwEcABwIHPUbGRwFHC0cGhz6GwMcGhwtHPMbFxwNHPwbFRwRHA0cIhwXHPwb9xsLHB8cBBwGHAccNxwSHAAcDxw2HAkcDxz4GxMcDBz7Gy8cABwSHAgcFBwuHAscHBz9GxAcDBwvHBgcBBw8HAYcLBwOHAgc9xsfHBMcExw2HA8c/hs8HAQcLhwfHAscFRwNHBccERw5HBAc+xscHC8cCBwSHCwcDhwsHB4cNBwSHDccCRwzHAMcHhwsHDIcMxwaHAMcBxwGHB0cBxwdHDccNBw3HBscFxwiHD8cFBxBHC4cGBwvHEocBhw3HB0cNxw0HBscOhwiHAIcJhwlHCMcJhwjHCAcIBwjHCQcIRwpHCocJhwgHCgcKBwgHCQcKxwpHCEcKxwhHCocLRwaHDMcJxwkHCMcKBwkHCccJhwjHCUcJhwnHCMcJhwoHCccKxwqHCkcMBwRHBUcNRwWHBkcGRwtHDUcHhxBHBQcNhwzHAkcORw7HBAcFhw6HAIcFRwXHDEcMhxBHB4cHxwuHEEcOxwvHBwcOxwcHBAcShw8HBgcMBwVHDEcEhw0HEMcORwRHDAcFxw/HD0cLRwzHDgcPxwiHDocPhwWHDUcNhwTHEYcBhxFHDccExwfHEYcFxw9HDEcPBxFHAYcEhxHHCwcLBxCHDIcOBxEHC0cQRxJHB8cHxxJHEYcSBxFHDwcQhwsHEccMhxLHEEcQBwzHDYcOBwzHEAcSBw8HEocRBw1HC0cQxxHHBIcRhxAHDYcQhxLHDIcSRxBHEscLxw7HEocPhw6HBYcMBwxHD0cRRw0HDccShw7HE0cNRxEHD4cUxwwHD0cTxw/HDocRxxDHDQcVRxCHEccURxEHDgcPhxPHDocMBxTHDkcOxw5HE0cTBw9HD8cVhw/HE8cRxw0HE4cQhxXHEscPxxWHEwcRxxOHFQcURw4HEAcShxNHGUcSxxXHFIcUhxJHEscZRxIHEocRhxJHFAcRRxOHDQcRRxbHE4cWxxFHEgcUBxAHEYcTRw5HFkcRxxUHFUcUBxJHFIcXBw+HEQcXBxEHFEcUxw9HGIcSBxlHFscWhxPHD4cXhxVHFQcWBxRHEAcWBxQHF8cUBxYHEAcPRxMHGIcURxYHFwcORxTHFkcUhxfHFAcahxaHD4cWRxTHGIcPhxcHGocZhxXHEIcYBxeHFQcVRxeHEIcZRxNHFkcVhxPHF0cYBxUHE4cThxbHGMcZhxCHF4caxxiHEwcVhxrHEwcYRxcHFgcXRxrHFYcZBxSHFccYRxYHF8cZBxfHFIcZRxoHFscVxxmHGQcThxjHGAcaBxlHFkcWhxsHE8cYRxqHFwcWxxoHGMcTxxsHF0cYBxpHF4cXhxpHGYcaBxZHGIcbBxaHGccZxxaHGocZBxhHF8cYxxyHGAcaxxoHGIcYBxyHGkccRxhHGQcaxxdHG0ccxxxHGQcbhxmHGkcbRxdHG8caBxrHG0cXRxsHG8cZhxuHGQcbhxzHGQccBxvHGwcahxhHHEccBxsHGccchxuHGkcaBxtHHUcdRxtHG8cchxjHHUcchxzHG4caBx1HGMceBxzHHIcbxxwHHUccRxnHGocdhxyHHUcdRxwHHQccBxnHHQceBx3HHMccxx3HHEcchx2HHgcdxxnHHEcZxx3HHQcdhx1HHQcdBx3HHgcdBx5HHYcdhx5HHgceBx5HHQcfRx7HHocfhx9HHocexx+HHocfhx7HHwchxx8HHschRx7HH0cjhx9HH4cgxyHHHscfByIHH4cgByBHIIcgxyAHIIchxyDHIIcfxyDHHschRx/HHscgRyQHIIchByGHIEcgRyGHJAcjByEHIEcjByBHIAcjhx+HIgcghyKHIcchxyKHHwchByPHIYchByLHI8cjByAHI0cghyQHIocgByDHI0chRyJHH8cihyQHJMcfxyVHIMciBx8HJIclRyNHIMcmByFHH0ckRyEHIwcnByLHIQcnByEHJEcihyTHJIcfByKHJIciRyXHH8cfRyOHJgcjxydHIYcixydHI8cpRyOHIgchhygHJAciRyFHJQchRyYHJQcjByNHJEclRx/HJccqBydHIschhydHKAckBygHJ8cphyRHI0ckByfHJMcphyNHJUclByXHIkckxyaHJIcmByOHKEckhyuHKccrhySHJociBySHKccmxyZHIscixycHJscmRybHJwckxyfHKkclhyUHJgcnByeHJkcmRyoHIsckxypHJocpRyhHI4cnhyiHJkcohyeHJwcmRyiHKgclByqHJccsRymHJUcqRyfHKwcnRykHKAclhyYHKEckRyjHJwcoxyRHKYcqhyxHJccohycHKMcsRyVHJccpRyIHKccpBydHKscmhypHLkcoRylHLQcuRy3HJocrhy6HKccqxydHKgcrhyaHLccthyWHKEcrBy5HKkclhyqHJQcqxyoHK8csRyjHKYctBylHKccnxygHK0ctBy2HKEcpBy4HKAcshytHKAcshyfHK0ctxy/HK4cuByyHKAcsxyiHKMcpByrHLwcsRywHKMcnxyyHKwcqhy9HLEcuhy0HKccohzFHKgcohyzHMUcxRyvHKgcvRy1HLEcwhy5HKwcvxy6HK4cvBy4HKQctRywHLEcwRy3HLkcvByrHK8cuxyzHKMcrByyHMIcvRyqHJYclhy2HL4csBy7HKMclhy+HL0cwRy5HMIcwBy2HLQctxzBHL8cwBy+HLYcvxy0HLocvRzHHLUcvxzAHLQcyxywHLUcxxy9HL4crxzFHLwcuBy8HLIcvBzCHLIcsxy7HMUcvxzBHMIcvBzFHMYcvBzKHMIcwxzEHMIcxxy+HMAcwhzEHL8cwBy/HMQcsBzLHLscwhzKHMMcvBzIHMocxhzIHLwcyxy1HMccwBzJHMccxRy7HMYcxBzDHMwcyRzAHMQczBzJHMQcyhzMHMMcyBzGHMoczhzGHLscxhzOHM0cxhzNHMocuxzLHM4cyhzNHMwcxxzOHMsczhzHHM8czRzPHMwcyRzPHMccyRzMHM8c4xzhHNAc1hzjHNAc1BziHNEc4hzUHNIc1BzhHNIc4RziHNIc4RzWHNAc4hzlHNMc1RziHNMc4hzVHNEc1RzUHNEc1xzWHOEc1RzTHOUc1xzhHNQc2hzWHNcc1BzVHNcc1xzbHNoc1RzlHNgc2RzVHNgc1RzZHNcc2RzbHNcc5RzZHNgc2hzbHNwc3RzbHNkc2xzdHNwc3RzaHNwc2RzlHN4c3RzZHN8c5RzqHN4c6hzZHN4c2RzqHOAc2RzqHN8c6hzdHN8c6hzZHOAc4xziHOEc1hzkHOMc1hzaHOQc5BzaHO4c2hzdHO4c7hzdHOoc4xzoHOIc5RziHOgc6RznHOQc5BztHOMc5hzsHOcc4xztHOgc6BzqHOUc7hzyHOQc5xztHOQc9RzuHOoc9hzmHOcc5hz2HPMc7RznHOwc5xzvHPYc5hzzHOwc6RzkHPIc8xz6HOwc9RzqHOgc7Rz7HOgc+hzzHOsc7xznHOkc7xzpHPIcAR31HOgc+Bz7HO0c8xwFHesc9hwRHfMc9RzyHO4c7Bz4HO0c6xzxHPAc6xzwHAod6xwKHfoc6Bz7HPQcBR3xHOscAR3oHPQc+xwBHfQc+RwJHfEc8RwJHfAc9xzyHPUcCh3wHAsd9xz1HBYdFh3yHPcc8hwWHe8c7Bz6HP8c+RzxHAcd8xwRHQUd7Bz/HPgcDh38HPkc/BwXHQ8d/BwPHfkcCh3+HPoc7xwUHfYcCR0LHfAcBx0OHfkcAx37HPgc+hz+HP8cFx38HAgdCB38HAAd/Bz9HAAdDh0EHfwcBR0fHfEcAR0bHfUc/Rz8HAQd/RwIHQAdFB0RHfYcAh0CHQIdAh0CHQId/RwEHQgdAh0CHQIdAh0CHQIdCB0EHQ4d+Bz/HAMdAx0tHfscLR0BHfscGB0JHfkcCB0OHQYdDh0dHQYd8RwfHQcdHR0IHQYdCR0SHQsdEB0FHREdDB0DHf8c+RwPHRgd/xz+HAwdLR0bHQEdCh0VHf4cCh0LHRUdDR0UHe8cEh0JHRgdHB0fHQUdCx0THRUdFh31HBsdEh0THQsdJh0NHe8cFx0ZHQ8dCB0dHRcdFR0MHf4cFh0mHe8cEB0RHS4dGx0hHRYdFx0dHSkdBx0dHQ4dGB0aHRIdBR0QHRwdGh0THRIdHB0QHR4dFR0oHQwdDB0oHQMdFx0pHRkdBx0lHR0dDx0aHRgdHx0cHScdJx0cHSIdHh0gHRwdFh0hHSYdLR0DHSgdER0UHS4dGh0PHTUdHx0nHQcdHh0QHSMdFR0THSgdMh0cHSAdMh0iHRwdMh0eHSMdHh0yHSAdMh0nHSIdEB0yHSMdKR0dHSUdMR0HHScdNR0PHRkdMR0lHQcdOR0hHRsdJB0mHSEdLR05HRsdKB0THT4dOR0zHSEdJB0rHQ0dDR0mHSQdMx0kHSEdEx0aHT4dKx0kHSodLB0tHSgdMx0qHSQdDR0uHRQdGR0pHTQdDR0rHS4dLh0yHRAdPh0sHSgdKx0qHS8dNB03HRkdKx0wHTIdLx0qHTsdMB0rHS8dMh0uHSsdKR0lHTQdJR0xHTYdLB05HS0dGR03HTUdGh01HT4dOB0xHTAdMB0xHScdJx0yHTAdPB0lHTYdPB00HSUdNh0xHTgdOB0wHS8dbR0sHT4dMx07HSodNh04HUEdNR08HUIdLB1tHT0dNR03HTwdPx06HT4dOx0zHTkdNx00HTwdPh01HUIdQh02HUEdQh0/HT4dXh0+HTodQh08HTYdXh1tHT4dQB07HTkdQh1BHT8dRR1BHTgdPx1OHTodbR2IHT0dbR16HYgdLB1AHTkdPx1BHVIdQx1BHUUdXh06HVUdSR1GHTgdSR04HS8doh0vHTsdPR2jHUAdPR1AHSwdqR0/HVIdTB2pHVIdTh0/HagdPx2pHagdTB2rHakdTB1SHaodQR1DHVIdUB2rHUwdrR1MHU0dTR1MHaodUh1NHaodqx1QHagdUR2tHU0dTx2sHU4dTx1OHagdTx2oHVYdVh2oHVAdrR1QHUwdUx1OHawdTx1THawdOh1OHVMdTx1WHVMdrh1QHVEdUR1QHa0dsx1RHU0dTR1SHbMdVh1QHa8drx1QHUQdrh1RHUQdUB2uHUQdtB1SHUMdVh2vHUQdUh1UHbMdRB1RHVYdVx20HUMdUR2zHbAdtB1UHVIdsR1THVYdVh1RHbAdVh2zHVQdVR1THbEdVR2xHVYdVB20HbIdVx1DHb8dOh1THVUdVh2wHbMdVh1UHbIdQx1FHbUdtB1XHbIdVx1YHbIdWB1XHb8dQx21HVsdtR1FHVsdQx1bHbYdRR1ZHVsdQx22HVodth1bHVodVR1WHbcdVh1YHbcdVh2yHVgdVR1YHbgdVR23HVgdVR24HVwduB1YHVwdWB2/HbkdQx1eHb8dQx1aHcMdXh1DHcYdQx1fHbodXB1YHbkduh1fHcEdWh1bHVkdOB1GHUUdwx1fHUMdux1aHVkdux1dHVodXR27HVkdYR1aHbwdXR28HVodWR1FHb4dXx3DHb0dvB1dHWEdXR1ZHcAdwB1ZHUcdWR2+HUcdRR1HHb4dWh1hHcsdRR1GHUcduR1VHVwdvx1eHbkdXx1gHcEdYR1dHcAdYB1fHb0dWh1uHcMdRx1GHcQdZx3GHUMdwR1gHcUdyx1uHVodYR1jHcsdRx1hHcAdQx26HcIdwx1uHc0dRx3EHUYdZx1DHcIdYh3FHWAdYh3CHcUdYx1hHcgdxx1HHUYduR1eHVUdRx3HHWEdzx1nHWIdZx3CHWIdYh1gHb0dZR3IHWEdRh3JHWEdRh1hHccdYx1mHcsdYx1lHdkdZR1jHcgdRh1IHckdZB3PHWIdZh1jHdkdyR1IHWEdvR1kHWIdaR1lHWEdSB1pHWEdZR1oHdkdZx1eHcYdZB29Hcodyx1mHdodaB1lHcwdzB1lHWkdaR3dHWgdaR1oHcwdXh1nHc0dZx3PHc0d0B1qHWodah1qHc4d0h1qHc4dZB3KHc8dax3UHdEd0R1rHdEdah3QHdAd0B1qHdIdbx3dHWkdbR1eHc0d1x3THWwdbB3THdUd1R1sHdUdax3WHdQd1h1rHWsdax3RHWsdbB3VHdcdbx1pHdgd2B1pHUgdbh1wHc0d2B1IHW8dSR1KHUYdyx1wHW4dSB3oHW8ddh3ZHWgddB3NHXAdcB3LHdodZh1zHdoddh1oHd0dSB1yHegdSh1IHUYdbR3NHXQdcx1mHdkdbx1xHd0d2x1IHUoddB1wHdod6B1xHW8dch1IHdsdSh1yHdsdcx11HdoddR1zHdwd2R12HXMd3R1xHeId3h1tHXQd3x11HXYddR3cHXYddh3cHXMdbR3eHXodeh3eHXQddB3aHXUd5h11HXcddR3fHXcddx3fHXYddh3dHeAdSh3kHXIddB11HeYddx12HeAdcR3oHeEdSh1yHeUdch1KHeUdeh10HeYdeB3gHeIdeB3iHXEdch15Hegdch3kHeMdex3vHUod5B1KHe8diB16Hecdeh3mHXcdfh3gHXgd4x15HXId5x16HYgdfh14HXEdex1KHfAdSh18HfAdeh13HeAdhR3gHX4d4R1+HXEdfB1KHekdiB16HYUdhR16HeAd6B15HeEdfB1JHfsdSR18HekdSh1JHekdhR1+HX0dfR1+HesdeR1/HeEdex3wHeodSR2DHfsdhR19HfEdfx3rHX4dfx1+HeEdfx15HeMdgB3yHe0dgR3zHewd7h2BHewdex3qHe8d8B18HfYdfx2FHfEdfR1/HfEdfx19Hesdgh2CHYIdgh30HYIdgB31HfIdgB3tHfUdgR3uHfMdgx1JHfcdgh2CHYIdhB2EHYQdgh30HYIdhB2EHYQdfB37HfYdHB6DHUsdgx33HUsd9x1JHUsdhh39Hfgd+R2GHfgdhB2EHYQdhB2EHYQdAB76HYcdhx36Hfwd/x2HHfwdgx0cHvsdhR1/HeMdhh35Hf0dAh6JHYkdiR2JHf4dBB6JHf4dhx3/HQAeBh4BHosdiR0DHgIeAx6JHQUeiR0EHgUehR3jHYodCx6MHYwdjB2MHQceDR6MHQceix0IHgYeCB6LHQkeix0BHgkeDx4KHo0djR0KHgweDB6NHQwejB0NHgseih2IHYUdih3jHQ4ejR0PHg8eDx6NHRAejR0MHhAeSx2OHRweSR0vHUsdjh1LHRMejx0THksdER6WHYgdih0RHogdlh0RHoodjh2RHRwePR2IHRIelh0SHogdlh09HRIeih2QHRQeih0OHpAdkR2OHRMeEx6PHR8ejx1LHRYelh2KHRQekB2WHRQeFR6QHQ4elR0fHpIdjx2SHR8ejx2THZIdkx2PHRceFh6UHY8dlB0WHksdmh0fHpUdGB6SHZMdFx6UHZMdFx6PHZQdlh2QHRUekR2ZHRwelR2XHRoelR2SHZcdlx2SHRgelx0YHpMdlB2XHZMdlB1LHRseGx5LHS8dkR0THhkeoh2VHRoeoh2aHZUdlx2iHRoelB2iHZcdoh2UHRseLx2iHRselh2ZHR0emR2WHRUeHB6ZHRUemR2RHRkenx09HZYdnx2WHR4eHh6WHZgdlh0dHpgdmR2YHR0enx0eHpgdmh2bHR8enB2aHSAemh2iHSAeHx6bHRkemx2aHSEemh2cHSEeoh2cHSAeGR6YHZkdnB2bHSEemx2cHSIeGR6fHZgdnB2dHSIeIx6cHaIdph0ZHpsdoB0iHp0dnR2cHSUeJR6cHZ4dnB0jHp4dnh0jHqIdJB6fHaYdnx0ZHqYdph2bHSIepR0iHqAdnR2eHSYenh2dHSUePR2fHSceox0nHp8dox2fHSQeox0kHqYdoB2dHSgenR2hHSgeoR2dHSYenh2hHSYePR0nHqMdpR2kHSIepR2gHSkeKR6gHacdoR2gHSgeoR2nHaAdph2kHSoepB2mHSIepx2lHSkenh2iHTsdox2mHSsepB2mHSoeoR2eHacdQB2jHSseph1AHSseph2kHUAdpx2kHaUdQB2kHacdnh07HacdOx1AHacdLB4vHi4eLB4uHi0eLR4vHiweLh4vHi0eLx4tHjAeLR4vHjAeLx4wHjEeLx41HjAeLx4xHjUeMh4xHjAeNB4zHjEeMh43HjEeNR46HjAeNx47HjEeNB4xHjseMR4zHjUeNx4yHjgeMB48HjIeMx42HjUeOh48HjAeNh46HjUeMx45HjYePB43HjgeMh48HjgeOh42Hj8eNh45Hj8eOx43HjweMx49HjkeNB4+HjMeMx4+Hj0ePx48HjoeOx4+HjQePx47HjwePx4+HjsePx45Hj0ePR4+HkAePx5AHj4ePx49HkAeQx5HHkkeQR5BHkEeQR5BHkEeQR5BHkEeQR5BHkEeQh5CHkIeQh5CHkIeSR5GHkMeQh5CHkIeQh5CHkIeRh5HHkMeSh5JHkceRB5EHkQeRB5EHkQeRB5EHkQeRB5EHkQeRR5KHkceRx5KHkUeRx5GHkgeTh5GHkkeRx5IHkoeTB5GHkseSB5GHkweRh5OHkseTR5KHkgeSx5IHkweSx5OHk0eSx5NHkgeSR5KHk4eVB5NHk4eSh5PHk4eTx5RHk4eSh5NHk8eTR5UHk8eVB5OHlEeUB5QHlAeUB5QHlAeVB5RHlIeUB5QHlAeUx5THlMeUB5QHlAeUx5THlMeVB5SHlEeUx5THlMeUx5THlMeTx5VHlEeTx5RHlUeUR5PHlYeTx5UHlYeVB5RHlYe';

// ---- Thick line shader (WebGL lineWidth is capped at 1 on most GPUs) ----
// Uses instanced quads: each line segment becomes a screen-aligned rectangle.
const _thickLineVert = `
  attribute vec3 instanceStart;
  attribute vec3 instanceEnd;
  attribute vec3 instanceColor;

  uniform float lineWidth;
  uniform vec2 resolution;
  uniform bool useInstanceColor;

  varying vec3 vViewPosition;
  varying vec3 vColor;

  void main() {
    vColor = useInstanceColor ? instanceColor : vec3(0.0);

    vec4 mvStart = modelViewMatrix * vec4(instanceStart, 1.0);
    vec4 mvEnd   = modelViewMatrix * vec4(instanceEnd, 1.0);
    vec4 csStart = projectionMatrix * mvStart;
    vec4 csEnd   = projectionMatrix * mvEnd;

    vec2 ndcStart = csStart.xy / csStart.w;
    vec2 ndcEnd   = csEnd.xy / csEnd.w;

    vec2 dir = ndcEnd - ndcStart;
    float len = length(dir * resolution);
    vec2 screenDir = len > 0.001 ? normalize(dir * resolution) : vec2(1.0, 0.0);
    vec2 perp = vec2(-screenDir.y, screenDir.x) / resolution * lineWidth;

    // position.x: -1 = start, +1 = end; position.y: -0.5 = left, +0.5 = right
    float t = position.x * 0.5 + 0.5;
    vec4 csPos = mix(csStart, csEnd, t);
    csPos.xy += perp * position.y * csPos.w;

    gl_Position = csPos;

    // Pass view-space position for clipping
    vec4 mvPos = mix(mvStart, mvEnd, t);
    vViewPosition = mvPos.xyz;
  }
`;

const _thickLineFrag = `
  uniform vec3 diffuse;
  uniform float opacity;
  uniform float clipZ;
  uniform bool useInstanceColor;

  varying vec3 vViewPosition;
  varying vec3 vColor;

  void main() {
    // Camera-space z-clipping: discard fragments closer than clipZ
    // vViewPosition.z is negative for objects in front of camera
    if (vViewPosition.z > clipZ) discard;

    vec3 color = useInstanceColor ? vColor : diffuse;
    gl_FragColor = vec4(color, opacity);
    #include <colorspace_fragment>
  }
`;

// Global uniforms shared by all thick-line materials
const _globalLineWidth = { value: 1.0 };
const _globalResolution = { value: new THREE.Vector2(1, 1) };
const _globalClipZ = { value: 1e10 };  // View-space z threshold for clipping (positive = no clip)

function createThickLineMaterial(color, opacity) {
    return new THREE.ShaderMaterial({
        uniforms: {
            diffuse:          { value: new THREE.Color(color) },
            opacity:          { value: opacity },
            lineWidth:        _globalLineWidth,
            resolution:       _globalResolution,
            clipZ:            _globalClipZ,
            useInstanceColor: { value: false },
        },
        vertexShader: _thickLineVert,
        fragmentShader: _thickLineFrag,
        transparent: true,
        depthWrite: true,
        side: THREE.DoubleSide,
    });
}

function createThickLineSegments(positions, material) {
    // positions: Float32Array of [x0,y0,z0, x1,y1,z1, ...] pairs
    const nVerts = positions.length / 3;
    const nSegments = nVerts / 2;

    // Quad template: 2 triangles per segment
    const quadPos = new Float32Array([ -1,-0.5,  1,-0.5,  -1,0.5,  1,0.5 ]);
    const quadIdx = new Uint16Array([ 0,1,2, 2,1,3 ]);

    const geom = new THREE.InstancedBufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(quadPos, 2));
    geom.setIndex(new THREE.BufferAttribute(quadIdx, 1));

    // Instance attributes: start and end of each segment
    const starts = new Float32Array(nSegments * 3);
    const ends = new Float32Array(nSegments * 3);
    for (let i = 0; i < nSegments; i++) {
        const base = i * 6;
        starts[i*3]   = positions[base];
        starts[i*3+1] = positions[base+1];
        starts[i*3+2] = positions[base+2];
        ends[i*3]     = positions[base+3];
        ends[i*3+1]   = positions[base+4];
        ends[i*3+2]   = positions[base+5];
    }

    geom.setAttribute('instanceStart', new THREE.InstancedBufferAttribute(starts, 3));
    geom.setAttribute('instanceEnd',   new THREE.InstancedBufferAttribute(ends, 3));
    geom.instanceCount = nSegments;

    // Set manual bounding sphere to avoid NaN warnings from the 2D quad positions
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e10);

    const mesh = new THREE.Mesh(geom, material);
    mesh.frustumCulled = false;
    return mesh;
}

// ---- Binary data decoders ----
function b64ToBuffer(b64) {
    const raw = atob(b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return bytes.buffer;
}

function decodeInt16(b64, scale) {
    const buf = b64ToBuffer(b64);
    const int16 = new Int16Array(buf);
    const out = new Float32Array(int16.length);
    const invScale = 1.0 / scale;
    for (let i = 0; i < int16.length; i++) out[i] = int16[i] * invScale;
    return out;
}

function decodeUint16(b64) {
    return new Uint16Array(b64ToBuffer(b64));
}

function decodeUint32(b64) {
    return new Uint32Array(b64ToBuffer(b64));
}

// ---- Colormaps ----
const COLORMAPS = {
    // Perceptually uniform sequential
    'Viridis':   [[0,'rgb(68,1,84)'],[0.13,'rgb(71,44,122)'],[0.25,'rgb(59,81,139)'],[0.38,'rgb(44,113,142)'],[0.5,'rgb(33,144,141)'],[0.63,'rgb(39,173,129)'],[0.75,'rgb(92,200,99)'],[0.88,'rgb(170,220,50)'],[1,'rgb(253,231,37)']],
    'Plasma':    [[0,'rgb(13,8,135)'],[0.13,'rgb(85,15,161)'],[0.25,'rgb(140,13,161)'],[0.38,'rgb(188,54,133)'],[0.5,'rgb(220,95,100)'],[0.63,'rgb(245,140,65)'],[0.75,'rgb(254,192,35)'],[1,'rgb(240,249,33)']],
    'Inferno':   [[0,'rgb(0,0,4)'],[0.13,'rgb(31,12,72)'],[0.25,'rgb(85,15,109)'],[0.38,'rgb(136,34,106)'],[0.5,'rgb(186,54,85)'],[0.63,'rgb(227,89,51)'],[0.75,'rgb(249,141,10)'],[1,'rgb(252,255,164)']],
    'Magma':     [[0,'rgb(0,0,4)'],[0.13,'rgb(28,16,68)'],[0.25,'rgb(79,18,123)'],[0.38,'rgb(129,37,129)'],[0.5,'rgb(181,54,122)'],[0.63,'rgb(229,80,100)'],[0.75,'rgb(251,135,97)'],[1,'rgb(252,253,191)']],
    'Cividis':   [[0,'rgb(0,32,77)'],[0.25,'rgb(66,78,98)'],[0.5,'rgb(124,123,120)'],[0.75,'rgb(188,176,95)'],[1,'rgb(255,234,70)']],
    'Turbo':     [[0,'rgb(48,18,59)'],[0.1,'rgb(70,96,209)'],[0.2,'rgb(41,180,233)'],[0.3,'rgb(30,227,172)'],[0.4,'rgb(89,235,75)'],[0.5,'rgb(185,232,29)'],[0.6,'rgb(249,193,0)'],[0.7,'rgb(251,123,0)'],[0.8,'rgb(222,44,7)'],[0.9,'rgb(171,0,39)'],[1,'rgb(122,4,3)']],
    // Diverging
    'Coolwarm':  [[0,'rgb(59,76,192)'],[0.25,'rgb(144,188,225)'],[0.5,'rgb(221,221,221)'],[0.75,'rgb(227,153,128)'],[1,'rgb(180,4,38)']],
    'RdBu':      [[0,'rgb(178,24,43)'],[0.25,'rgb(239,138,98)'],[0.5,'rgb(247,247,247)'],[0.75,'rgb(103,169,207)'],[1,'rgb(33,102,172)']],
    'RdYlBu':    [[0,'rgb(215,25,28)'],[0.25,'rgb(253,174,97)'],[0.5,'rgb(255,255,191)'],[0.75,'rgb(171,217,233)'],[1,'rgb(44,123,182)']],
    'RdYlGn':    [[0,'rgb(215,25,28)'],[0.25,'rgb(253,174,97)'],[0.5,'rgb(255,255,191)'],[0.75,'rgb(166,217,106)'],[1,'rgb(26,150,65)']],
    'Spectral':  [[0,'rgb(158,1,66)'],[0.25,'rgb(253,174,97)'],[0.5,'rgb(255,255,191)'],[0.75,'rgb(102,194,165)'],[1,'rgb(94,79,162)']],
    'PRGn':      [[0,'rgb(64,0,75)'],[0.25,'rgb(153,112,171)'],[0.5,'rgb(247,247,247)'],[0.75,'rgb(90,174,97)'],[1,'rgb(0,68,27)']],
    'PiYG':      [[0,'rgb(197,27,125)'],[0.25,'rgb(233,163,201)'],[0.5,'rgb(247,247,247)'],[0.75,'rgb(161,215,106)'],[1,'rgb(77,146,33)']],
    'BrBG':      [[0,'rgb(84,48,5)'],[0.25,'rgb(191,129,45)'],[0.5,'rgb(245,245,245)'],[0.75,'rgb(53,151,143)'],[1,'rgb(1,102,94)']],
    // Multi-hue sequential
    'YlOrRd':    [[0,'rgb(255,255,204)'],[0.25,'rgb(254,204,92)'],[0.5,'rgb(253,141,60)'],[0.75,'rgb(240,59,32)'],[1,'rgb(128,0,38)']],
    'YlGnBu':    [[0,'rgb(255,255,217)'],[0.25,'rgb(161,218,180)'],[0.5,'rgb(65,182,196)'],[0.75,'rgb(34,94,168)'],[1,'rgb(8,29,88)']],
    'RdPu':      [[0,'rgb(255,247,243)'],[0.25,'rgb(253,190,166)'],[0.5,'rgb(251,105,107)'],[0.75,'rgb(197,27,138)'],[1,'rgb(73,0,106)']],
    'BuPu':      [[0,'rgb(247,252,253)'],[0.25,'rgb(179,205,227)'],[0.5,'rgb(140,150,198)'],[0.75,'rgb(136,86,167)'],[1,'rgb(77,0,75)']],
    'GnBu':      [[0,'rgb(247,252,240)'],[0.25,'rgb(184,225,134)'],[0.5,'rgb(79,168,149)'],[0.75,'rgb(43,140,190)'],[1,'rgb(8,64,129)']],
    // Single-hue sequential
    'Hot':       [[0,'rgb(0,0,0)'],[0.33,'rgb(230,0,0)'],[0.67,'rgb(255,210,0)'],[1,'rgb(255,255,255)']],
    'Blues':     [[0,'rgb(247,251,255)'],[0.5,'rgb(107,174,214)'],[1,'rgb(8,48,107)']],
    'Reds':      [[0,'rgb(255,245,240)'],[0.5,'rgb(252,146,114)'],[1,'rgb(165,15,21)']],
    'Greens':    [[0,'rgb(247,252,245)'],[0.5,'rgb(116,196,118)'],[1,'rgb(0,68,27)']],
    'Purples':   [[0,'rgb(252,251,253)'],[0.5,'rgb(158,154,200)'],[1,'rgb(63,0,125)']],
    'Oranges':   [[0,'rgb(255,245,235)'],[0.5,'rgb(253,174,107)'],[1,'rgb(127,39,4)']],
    'Greys':     [[0,'rgb(255,255,255)'],[1,'rgb(0,0,0)']],
    // Cyclic / artistic
    'Rainbow':   [[0,'rgb(150,0,90)'],[0.125,'rgb(0,0,200)'],[0.25,'rgb(0,25,255)'],[0.375,'rgb(0,152,255)'],[0.5,'rgb(44,255,150)'],[0.625,'rgb(151,255,0)'],[0.75,'rgb(255,234,0)'],[0.875,'rgb(255,111,0)'],[1,'rgb(255,0,0)']],
    'Spring':    [[0,'rgb(255,0,255)'],[1,'rgb(255,255,0)']],
    'Summer':    [[0,'rgb(0,128,102)'],[1,'rgb(255,255,102)']],
    'Autumn':    [[0,'rgb(255,0,0)'],[1,'rgb(255,255,0)']],
    'Winter':    [[0,'rgb(0,0,255)'],[1,'rgb(0,255,128)']],
};

function interpolateColormap(stops, t) {
    const parseRgb = s => s.match(/\d+/g).map(Number);
    if (t <= stops[0][0]) return parseRgb(stops[0][1]);
    if (t >= stops[stops.length-1][0]) return parseRgb(stops[stops.length-1][1]);
    for (let i = 0; i < stops.length-1; i++) {
        const [t0, c0] = stops[i], [t1, c1] = stops[i+1];
        if (t >= t0 && t <= t1) {
            const f = (t-t0)/(t1-t0);
            const r0 = parseRgb(c0), r1 = parseRgb(c1);
            return [
                Math.round(r0[0] + f*(r1[0]-r0[0])),
                Math.round(r0[1] + f*(r1[1]-r0[1])),
                Math.round(r0[2] + f*(r1[2]-r0[2])),
            ];
        }
    }
    return [0,0,0];
}

// ---- Data Store ----
class DataStore {
    constructor(data) {
        this.raw = data;
        this.allTypes = data.allTypes;
        this.typeNeurons = data.typeNeurons;
        this.bidTypeMap = data.bidTypeMap;
        this.colorModes = data.colorModes;

        // Auto-inject "Instance" color mode if not already present
        // (old HTMLs generated before Instance mode was added won't have it)
        if (!this.colorModes.find(m => m.name === 'Instance')) {
            const allBids = Object.keys(data.bidTypeMap);
            const n = allBids.length;
            const instColors = {};
            const instTypeColors = {};
            for (let i = 0; i < n; i++) {
                const hue = n <= 1 ? 0 : i / (n - 1);
                const c = new THREE.Color().setHSL(hue, 0.8, 0.55);
                const rgb = `rgb(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)})`;
                instColors[allBids[i]] = rgb;
                // Use first neuron's color as type representative
                const typ = data.bidTypeMap[allBids[i]];
                if (typ && !instTypeColors[typ]) instTypeColors[typ] = rgb;
            }
            // Insert after Cell Type (index 0), before Predicted NT
            this.colorModes.splice(1, 0, {
                name: 'Instance',
                colors: instColors,
                type_colors: instTypeColors,
                is_scalar: false,
            });
        }

        // Auto-inject "Custom" color mode — starts with Cell Type colors
        const cellTypeMode = this.colorModes[0];  // Cell Type is always first
        const instanceMode = this.colorModes.find(m => m.name === 'Instance');
        const customMode = {
            name: 'Custom',
            colors: Object.assign({}, cellTypeMode.colors),
            type_colors: Object.assign({}, cellTypeMode.type_colors),
            is_scalar: false,
            is_custom: true,
        };
        // Store separate neuron-level custom colors (init from Instance)
        customMode._neuronColors = Object.assign({}, instanceMode.colors);
        this.colorModes.push(customMode);

        // ── Ensure canonical button order ──────────────────────────────
        // Cell Type (0) → Instance (1) → Predicted NT (2, if present) → user modes → Custom (last)
        {
            const cm = this.colorModes;
            const move = (pred, targetIdx) => {
                const i = cm.findIndex(pred);
                if (i >= 0 && i !== targetIdx && targetIdx < cm.length) {
                    const [m] = cm.splice(i, 1);
                    cm.splice(targetIdx, 0, m);
                }
            };
            move(m => m.name === 'Instance', 1);
            move(m => m.nt_legend, 2);
            // Custom is already last (pushed above)
        }

        this.sidebarRois = data.sidebarRois;
        this.primaryRoi = data.primaryRoi;
        this.typeRoiMap = data.typeRoiMap;
        this.roiSynapseTotals = data.roiSynapseTotals;
        this.typeRoiSynapses = data.typeRoiSynapses;
        this.neuronRoiSynapses = data.neuronRoiSynapses;
        this.instanceLookup = data.instanceLookup;
        this.typeUpstream = data.typeUpstream;
        this.typeDownstream = data.typeDownstream;
        this.neuronUpstream = data.neuronUpstream;
        this.neuronDownstream = data.neuronDownstream;
        this.roiBounds = data.roiBounds;
        this.normParams = data.normParams;
        this.camera = data.camera;
        this.initialLineWidth = data.initialLineWidth || 1;
        this.regexTerm = data.regexTerm || '';

        // Build reverse map: bodyId -> type
        this.neuronType = {};
        for (const [bid, typ] of Object.entries(this.bidTypeMap)) {
            this.neuronType[bid] = typ;
        }

        // Pre-compute sorted values for percentile mapping on scalar color modes
        const parseRgb = (s) => { const m = s.match(/\d+/g); return m ? m.map(Number) : null; };
        for (const mode of this.colorModes) {
            if (mode.type_values) {
                mode._sortedValues = Object.values(mode.type_values).sort((a, b) => a - b);
            }
            // Derive type_values from type_colors + colorscale if not present (backward compat)
            if (mode.is_scalar && !mode.type_values && mode.colorscale && mode.type_colors
                && mode.cmin !== undefined && mode.cmax !== undefined) {
                const cs = mode.colorscale;
                // Build lookup: for each type color, find its position t on the colorscale
                // then value = cmin + t * (cmax - cmin)
                // Precompute colorscale RGB stops
                const csStops = cs.map(([t, c]) => ({ t, rgb: parseRgb(c) }));
                mode.type_values = {};
                for (const [typeName, colorStr] of Object.entries(mode.type_colors)) {
                    const rgb = parseRgb(colorStr);
                    if (!rgb) continue;
                    // Find best matching t by minimizing color distance
                    let bestT = 0, bestDist = Infinity;
                    // Sample 201 points along the colorscale
                    for (let i = 0; i <= 200; i++) {
                        const t = i / 200;
                        // Interpolate colorscale at t
                        let lo = 0, hi = csStops.length - 1;
                        for (let j = 0; j < csStops.length - 1; j++) {
                            if (csStops[j].t <= t && csStops[j + 1].t >= t) { lo = j; hi = j + 1; break; }
                        }
                        const range = csStops[hi].t - csStops[lo].t;
                        const frac = range > 0 ? (t - csStops[lo].t) / range : 0;
                        const r = Math.round(csStops[lo].rgb[0] + (csStops[hi].rgb[0] - csStops[lo].rgb[0]) * frac);
                        const g = Math.round(csStops[lo].rgb[1] + (csStops[hi].rgb[1] - csStops[lo].rgb[1]) * frac);
                        const b = Math.round(csStops[lo].rgb[2] + (csStops[hi].rgb[2] - csStops[lo].rgb[2]) * frac);
                        const dist = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2;
                        if (dist < bestDist) { bestDist = dist; bestT = t; }
                    }
                    mode.type_values[typeName] = mode.cmin + bestT * (mode.cmax - mode.cmin);
                }
                mode._sortedValues = Object.values(mode.type_values).sort((a, b) => a - b);
            }
            // Derive bid_nts from colors if not already present (backward compat)
            if (mode.nt_legend && !mode.bid_nts && mode.colors) {
                // Reverse map: rgb color string -> NT name
                const colorToNt = {};
                for (const [nt, color] of Object.entries(mode.nt_legend)) {
                    colorToNt[color] = nt;
                }
                mode.bid_nts = {};
                for (const [bid, color] of Object.entries(mode.colors)) {
                    mode.bid_nts[bid] = colorToNt[color] || 'unclear';
                }
            }
        }
    }

    getTypeColor(typeName, modeIdx) {
        const mode = this.colorModes[modeIdx || 0];
        return mode.type_colors[typeName] || '#888888';
    }

    getNeuronColor(bid, modeIdx) {
        const mode = this.colorModes[modeIdx || 0];
        return mode.colors[String(bid)] || '#888888';
    }

    getROIsForType(typeName) {
        return this.typeRoiMap[typeName] || [];
    }

    getNeuronsForType(typeName) {
        return this.typeNeurons[typeName] || [];
    }
}

// ---- Scene Manager ----
class SceneManager {
    constructor(dataStore) {
        this.data = dataStore;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.canvas = null;

        // Geometry groups
        this.typeRoiGeom = new Map();   // "type|roi" -> THREE.LineSegments
        this.neuronFullGeom = new Map(); // bodyId -> THREE.LineSegments
        this.roiMeshGeom = new Map();    // roiName -> THREE.Mesh
        this.somaGeom = new Map();       // bodyId -> THREE.Mesh

        // Groups for scene organization
        this.neuronGroup = null;
        this.fullGroup = null;
        this.roiGroup = null;
        this.somaGroup = null;

        // GPU picking infrastructure
        this._pickScene = null;
        this._pickTarget = null;
        this._pickPixelBuf = new Uint8Array(4);
        this._pickIdMap = new Map();  // encoded color int -> { type, bodyId }
        this._pickGeom = new Map();   // key -> pick mesh (mirrors visibility)
        this._nextPickId = 1;

        this._init();
    }

    _init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'scene3d';
        this.canvas.style.cssText = `
            position: fixed;
            left: ${SIDEBAR_W}px;
            top: ${TOP_BAR_H}px;
            width: calc(100vw - ${SIDEBAR_W + TYPE_PANEL_W}px);
            height: calc(100vh - ${TOP_BAR_H}px);
            z-index: 0;
        `;
        document.body.appendChild(this.canvas);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
            preserveDrawingBuffer: true
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000000, 1);
        this.renderer.localClippingEnabled = true;
        this._updateSize();

        // Scene
        this.scene = new THREE.Scene();

        // Camera
        const norm = this.data.normParams;
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.001, 100);

        // Set camera from Plotly camera data
        if (this.data.camera) {
            const cam = this.data.camera;
            this.camera.position.set(cam.eye.x, cam.eye.y, cam.eye.z);
            // Use up vector from data if it's not the legacy Z-up (0,0,1) which causes
            // gimbal lock when looking along -Z. New HTMLs embed (0,-1,0) directly.
            if (cam.up && !(cam.up.x === 0 && cam.up.y === 0 && Math.abs(cam.up.z) === 1)) {
                this.camera.up.set(cam.up.x, cam.up.y, cam.up.z);
            } else {
                this.camera.up.set(0, -1, 0);  // Dorsal-up (-Y)
            }
        } else {
            this.camera.position.set(0, 0, -0.6);
            this.camera.up.set(0, -1, 0);
        }

        // TrackballControls — free rotation in all axes (no polar-angle hard stop)
        this.controls = new THREE.TrackballControls(this.camera, this.canvas);
        if (this.data.camera) {
            const cam = this.data.camera;
            this.controls.target.set(cam.center.x, cam.center.y, cam.center.z);
        }
        this.controls.staticMoving = true;  // no inertia — stops on release
        this.controls.rotateSpeed = 3.0;    // TrackballControls feels slower than Orbit; scale up
        this.controls.zoomSpeed = 2.5;
        this.controls.panSpeed = 0.8;

        // Cancel drag when the pointer exits the canvas boundary.
        // TrackballControls uses setPointerCapture, so pointermove keeps firing on the
        // canvas even when the pointer is outside. We detect the out-of-bounds case via
        // a canvas-level pointermove check and synthesise a pointerup on the canvas
        // (where TrackballControls' own pointerup listener lives) to end the drag.
        this.canvas.addEventListener('pointermove', (e) => {
            if (!this.canvas.hasPointerCapture(e.pointerId)) return;
            const r = this.canvas.getBoundingClientRect();
            if (e.clientX < r.left || e.clientX > r.right ||
                e.clientY < r.top  || e.clientY > r.bottom) {
                this.canvas.dispatchEvent(new PointerEvent('pointerup', {
                    pointerId: e.pointerId, clientX: e.clientX, clientY: e.clientY,
                    bubbles: true, cancelable: true, button: 0, buttons: 0
                }));
            }
        });

        // Store initial camera for reset
        this._initialCameraPos = this.camera.position.clone();
        this._initialCameraTarget = this.controls.target.clone();
        this._initialCameraUp = this.camera.up.clone();

        // Z-section clipping plane (camera-relative)
        // Always registered so shaders compile with clipping support;
        // disabled by setting constant to a huge value (clips nothing).
        this.clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 1e10);
        this.renderer.clippingPlanes = [this.clipPlane];
        this.clipEnabled = false;
        this.clipFraction = 0;
        this._clipDir = new THREE.Vector3();
        this._clipNormal = new THREE.Vector3();
        this._clipPoint = new THREE.Vector3();

        // Create scene groups
        this.neuronGroup = new THREE.Group();
        this.neuronGroup.name = 'clipped';
        this.scene.add(this.neuronGroup);

        this.fullGroup = new THREE.Group();
        this.fullGroup.name = 'full';
        this.scene.add(this.fullGroup);

        this.roiGroup = new THREE.Group();
        this.roiGroup.name = 'rois';
        this.scene.add(this.roiGroup);

        this.somaGroup = new THREE.Group();
        this.somaGroup.name = 'somas';
        this.scene.add(this.somaGroup);

        this.synapseGroup = new THREE.Group();
        this.synapseGroup.name = 'synapses';
        this.scene.add(this.synapseGroup);

        this.meshGroup = new THREE.Group();
        this.meshGroup.name = 'neuronMeshes';
        this.scene.add(this.meshGroup);

        // Mesh geometry map: bodyId -> THREE.Mesh
        this.neuronMeshGeom = new Map();
        this._meshesAvailable = !!this.data.raw.neuronMeshes;

        // Build geometries
        this._buildClippedGeometries();
        this._buildFullGeometries();
        this._computeBidRuns();
        this._buildROIMeshes();
        this._buildSomas();
        if (this._meshesAvailable) this._buildNeuronMeshes();
        this._buildPickScene();
        this._initScaleBar();

        // Compute scene bounding box for Z-section depth mapping.
        // We store the 8 corners so each frame we can project them onto
        // the camera axis for a tight near/far depth range.
        this._sceneBBox = new THREE.Box3();
        [this.neuronGroup, this.fullGroup, this.somaGroup].forEach(g => {
            g.traverse(obj => {
                if (obj.isMesh && obj.geometry) {
                    const attr = obj.geometry.getAttribute('instanceStart');
                    if (attr) {
                        for (let i = 0; i < attr.count; i++) {
                            this._sceneBBox.expandByPoint(
                                new THREE.Vector3(attr.getX(i), attr.getY(i), attr.getZ(i))
                            );
                        }
                    }
                    if (obj.position && !attr) {
                        this._sceneBBox.expandByPoint(obj.position);
                    }
                }
            });
        });
        // Pre-compute 8 bbox corners for per-frame depth projection
        const mn = this._sceneBBox.min, mx = this._sceneBBox.max;
        this._bboxCorners = [
            new THREE.Vector3(mn.x, mn.y, mn.z),
            new THREE.Vector3(mn.x, mn.y, mx.z),
            new THREE.Vector3(mn.x, mx.y, mn.z),
            new THREE.Vector3(mn.x, mx.y, mx.z),
            new THREE.Vector3(mx.x, mn.y, mn.z),
            new THREE.Vector3(mx.x, mn.y, mx.z),
            new THREE.Vector3(mx.x, mx.y, mn.z),
            new THREE.Vector3(mx.x, mx.y, mx.z),
        ];

        // Ambient light for mesh visibility
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        this.scene.add(new THREE.DirectionalLight(0xffffff, 0.2));

        // Resize handler
        window.addEventListener('resize', () => this._onResize());

        // Start render loop
        this._renderLoop();
    }

    resetCamera() {
        this.camera.position.copy(this._initialCameraPos);
        this.camera.up.copy(this._initialCameraUp);
        this.controls.target.copy(this._initialCameraTarget);
        this.controls.update();
    }

    _normalize(x, y, z) {
        const n = this.data.normParams;
        return [
            (x - n.cx) / n.dmax,
            (y - n.cy) / n.dmax,
            (z - n.cz) / n.dmax
        ];
    }

    _decodeCoords(data) {
        // Binary encoded: base64 Int16 string, pre-normalized
        if (typeof data === 'string') {
            const qScale = this.data.raw.quantScale || 30000;
            return decodeInt16(data, qScale);
        }
        // Legacy JSON array: normalize on the fly
        const positions = new Float32Array(data.length);
        for (let i = 0; i < data.length; i += 3) {
            const [nx, ny, nz] = this._normalize(data[i], data[i+1], data[i+2]);
            positions[i] = nx;
            positions[i+1] = ny;
            positions[i+2] = nz;
        }
        return positions;
    }

    _buildClippedGeometries() {
        const segments = this.data.raw.typeRoiSegments;
        const bidRunsAll = this.data.raw.typeRoiBidRuns || {};
        let count = 0;

        for (const [key, data] of Object.entries(segments)) {
            const [typeName, roiName] = key.split('|');
            const positions = this._decodeCoords(data);

            // Build per-segment bodyId lookup from bid runs
            const runs = bidRunsAll[key] || [];
            const bidRuns = runs;  // [[bodyId, segCount], ...]

            const color = this.data.getTypeColor(typeName, 0);
            const material = createThickLineMaterial(color, 1.0);

            const lines = createThickLineSegments(positions, material);
            lines.userData = { type: typeName, roi: roiName, kind: 'clipped', bidRuns };
            lines.visible = false;

            // Hidden LineSegments twin for raycasting (never rendered)
            const rayGeom = new THREE.BufferGeometry();
            rayGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const rayLines = new THREE.LineSegments(rayGeom, new THREE.LineBasicMaterial({visible: false}));
            rayLines.visible = false;
            rayLines.userData = lines.userData;
            rayLines.layers.set(1);  // Layer 1 = raycast only
            lines._rayTarget = rayLines;

            this.neuronGroup.add(lines);
            this.neuronGroup.add(rayLines);
            this.typeRoiGeom.set(key, lines);
            count++;
        }

        console.log(`Built ${count} clipped geometry groups`);
    }

    _buildFullGeometries() {
        const segments = this.data.raw.neuronFullSegments;
        let count = 0;

        for (const [bid, data] of Object.entries(segments)) {
            const positions = this._decodeCoords(data);

            const typeName = this.data.neuronType[bid];
            const color = this.data.getTypeColor(typeName, 0);
            const material = createThickLineMaterial(color, 1.0);

            const lines = createThickLineSegments(positions, material);
            lines.userData = { bodyId: bid, type: typeName, kind: 'full' };
            lines.visible = false;

            // Hidden LineSegments twin for raycasting
            const rayGeom = new THREE.BufferGeometry();
            rayGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const rayLines = new THREE.LineSegments(rayGeom, new THREE.LineBasicMaterial({visible: false}));
            rayLines.visible = false;
            rayLines.userData = lines.userData;
            rayLines.layers.set(1);
            lines._rayTarget = rayLines;

            this.fullGroup.add(lines);
            this.fullGroup.add(rayLines);
            this.neuronFullGeom.set(bid, lines);
            count++;
        }

        console.log(`Built ${count} full skeleton geometries`);
    }

    // Compute per-segment bodyId runs for clipped geometry by matching
    // against per-neuron full skeleton segments. Only needed if bidRuns
    // were not provided in the data bundle (backward compat with older HTMLs).
    _computeBidRuns() {
        const bidRunsAll = this.data.raw.typeRoiBidRuns || {};
        const segments = this.data.raw.typeRoiSegments;
        const fullSegs = this.data.raw.neuronFullSegments;
        const isBinary = !!this.data.raw.encoding;
        const qScale = this.data.raw.quantScale || 30000;

        // Check if any typeRoiGeom entry already has bidRuns from data bundle
        let needsCompute = false;
        for (const [key, geom] of this.typeRoiGeom) {
            if (!geom.userData.bidRuns || geom.userData.bidRuns.length === 0) {
                needsCompute = true;
                break;
            }
        }
        if (!needsCompute) return;

        // Helper: decode base64 to raw Int16Array (skip float conversion)
        function decodeToInt16(data) {
            if (typeof data === 'string') {
                return new Int16Array(b64ToBuffer(data));
            }
            // Legacy JSON array — quantize to int16 like optimize_bundle would
            const out = new Int16Array(data.length);
            for (let i = 0; i < data.length; i++) out[i] = Math.round(data[i]);
            return out;
        }

        // Build per-type segment lookup: Map<typeName, Map<segKey, bodyId>>
        const typeLookup = new Map();
        for (const [bid, data] of Object.entries(fullSegs)) {
            const typeName = this.data.neuronType[bid];
            if (!typeName) continue;
            let lookup = typeLookup.get(typeName);
            if (!lookup) { lookup = new Map(); typeLookup.set(typeName, lookup); }

            const int16 = decodeToInt16(data);
            // Each segment = 2 vertices = 6 Int16 values
            for (let i = 0; i < int16.length; i += 6) {
                const key = int16[i] + ',' + int16[i+1] + ',' + int16[i+2] + ',' +
                            int16[i+3] + ',' + int16[i+4] + ',' + int16[i+5];
                lookup.set(key, bid);
            }
        }

        // Match clipped segments to neurons
        let matched = 0, unmatched = 0;
        for (const [trKey, geom] of this.typeRoiGeom) {
            if (geom.userData.bidRuns && geom.userData.bidRuns.length > 0) continue;
            const typeName = trKey.split('|')[0];
            const lookup = typeLookup.get(typeName);
            if (!lookup) continue;

            const clippedData = segments[trKey];
            if (!clippedData) continue;
            const int16 = decodeToInt16(clippedData);

            // Build bidRuns by matching each segment
            const bidRuns = [];
            let curBid = null, curCount = 0;
            for (let i = 0; i < int16.length; i += 6) {
                const key = int16[i] + ',' + int16[i+1] + ',' + int16[i+2] + ',' +
                            int16[i+3] + ',' + int16[i+4] + ',' + int16[i+5];
                const bid = lookup.get(key) || null;
                if (bid) matched++; else unmatched++;
                if (bid === curBid) {
                    curCount++;
                } else {
                    if (curCount > 0) bidRuns.push([curBid, curCount]);
                    curBid = bid;
                    curCount = 1;
                }
            }
            if (curCount > 0) bidRuns.push([curBid, curCount]);

            // Apply to both the visible mesh and its raycast twin
            geom.userData.bidRuns = bidRuns;
            if (geom._rayTarget) geom._rayTarget.userData.bidRuns = bidRuns;
        }

        console.log(`Computed bidRuns: ${matched} matched, ${unmatched} unmatched segments`);
    }

    _buildROIMeshes() {
        const meshes = this.data.raw.roiMeshes;
        const isBinary = !!this.data.raw.encoding;
        const qScale = this.data.raw.quantScale || 30000;
        let count = 0;

        for (const [roiName, meshData] of Object.entries(meshes)) {
            let positions, faceIndices;

            if (isBinary && typeof meshData.v === 'string') {
                // Binary encoded: Int16 vertices + Uint16/32 faces
                positions = decodeInt16(meshData.v, qScale);
                if (meshData.fd === 'u2') {
                    faceIndices = new THREE.BufferAttribute(decodeUint16(meshData.f), 1);
                } else {
                    faceIndices = new THREE.BufferAttribute(decodeUint32(meshData.f), 1);
                }
            } else {
                // Legacy format
                const verts = meshData.vertices;
                positions = new Float32Array(verts.length);
                for (let i = 0; i < verts.length; i += 3) {
                    const [nx, ny, nz] = this._normalize(verts[i], verts[i+1], verts[i+2]);
                    positions[i] = nx;
                    positions[i+1] = ny;
                    positions[i+2] = nz;
                }
                faceIndices = meshData.faces;
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setIndex(faceIndices);
            geometry.computeVertexNormals();

            const material = new THREE.MeshBasicMaterial({
                color: 0xcccccc,
                transparent: true,
                opacity: 0.08,
                side: THREE.DoubleSide,
                depthWrite: false
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData = { roi: roiName, kind: 'roi_mesh' };
            mesh.visible = false;  // Start hidden (wireframes default off? actually ON)

            this.roiGroup.add(mesh);
            this.roiMeshGeom.set(roiName, mesh);
            count++;
        }

        console.log(`Built ${count} ROI meshes`);
    }

    clipMeshToRois(bid, checkedRois) {
        // Rebuild mesh index buffer to only include faces in checked ROIs.
        // If checkedRois is null, restore full mesh.
        const mesh = this.neuronMeshGeom.get(bid);
        if (!mesh || !mesh._fullIndex) return;

        if (!checkedRois || !mesh._faceRoiIndices) {
            // Restore full mesh
            mesh.geometry.setIndex(new THREE.BufferAttribute(mesh._fullIndex, 1));
            mesh.geometry.index.needsUpdate = true;
            return;
        }

        // Build set of checked ROI indices
        const roiLookup = mesh._roiLookup;
        const checkedSet = new Set();
        for (let i = 0; i < roiLookup.length; i++) {
            if (checkedRois[roiLookup[i]]) checkedSet.add(i);
        }

        // Filter faces: include only faces whose ROI index is in checkedSet
        const fri = mesh._faceRoiIndices;
        const fullIdx = mesh._fullIndex;
        const nf = fri.length;
        const filtered = [];
        for (let i = 0; i < nf; i++) {
            if (checkedSet.has(fri[i])) {
                const base = i * 3;
                filtered.push(fullIdx[base], fullIdx[base + 1], fullIdx[base + 2]);
            }
        }

        const newIndex = new Uint32Array(filtered);
        mesh.geometry.setIndex(new THREE.BufferAttribute(newIndex, 1));
        mesh.geometry.index.needsUpdate = true;
    }

    _buildSomas() {
        const somas = this.data.raw.neuronSomas;
        const sphereGeo = new THREE.SphereGeometry(0.003, 8, 6);  // Small sphere
        let count = 0;

        for (const [bid, pos] of Object.entries(somas)) {
            // Somas are pre-normalized in the data bundle
            const nx = pos.x, ny = pos.y, nz = pos.z;

            const typeName = this.data.neuronType[bid];
            const color = this.data.getTypeColor(typeName, 0);

            const material = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.02  // Dimmed by default (not highlighted)
            });

            const mesh = new THREE.Mesh(sphereGeo, material);
            mesh.position.set(nx, ny, nz);
            mesh.userData = { bodyId: bid, type: typeName, kind: 'soma' };
            mesh.visible = true;

            this.somaGroup.add(mesh);
            this.somaGeom.set(bid, mesh);
            count++;
        }

        console.log(`Built ${count} soma spheres`);
    }

    _buildNeuronMeshes() {
        const meshData = this.data.raw.neuronMeshes;
        if (!meshData) return;

        let count = 0;
        for (const [bid, md] of Object.entries(meshData)) {
            // Decode base64 vertices and faces
            const vBuf = Uint8Array.from(atob(md.v), c => c.charCodeAt(0));
            const fBuf = Uint8Array.from(atob(md.f), c => c.charCodeAt(0));
            const vertices = new Float32Array(vBuf.buffer, vBuf.byteOffset, md.nv * 3);
            const faces = new Int32Array(fBuf.buffer, fBuf.byteOffset, md.nf * 3);

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            const fullIndex = new Uint32Array(faces);
            geometry.setIndex(new THREE.BufferAttribute(fullIndex, 1));
            geometry.computeVertexNormals();

            // Decode per-face ROI assignments if available
            let faceRoiIndices = null;
            let roiLookup = null;
            if (md.fri && md.rl) {
                const friBuf = Uint8Array.from(atob(md.fri), c => c.charCodeAt(0));
                faceRoiIndices = new Uint8Array(friBuf.buffer, friBuf.byteOffset, md.nf);
                roiLookup = md.rl;
            }

            const typeName = this.data.neuronType[bid];
            const color = this.data.getTypeColor(typeName, 0);

            const material = new THREE.MeshStandardMaterial({
                color: color,
                roughness: 0,
                metalness: 0.1,
                side: THREE.DoubleSide,
                envMapIntensity: 1.0,
                clippingPlanes: [],
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData = { bodyId: bid, type: typeName, kind: 'mesh' };
            mesh._fullIndex = fullIndex;  // Store full index for restoration
            mesh._faceRoiIndices = faceRoiIndices;
            mesh._roiLookup = roiLookup;
            mesh.visible = false;

            this.meshGroup.add(mesh);
            this.neuronMeshGeom.set(bid, mesh);
            count++;
        }

        // Add lighting for mesh shading — adaptive setup
        if (count > 0) {
            this._lightIntensityScale = 4.0;
            // Hemisphere light — warm from above, cool from below
            this._meshHemi = new THREE.HemisphereLight(0xfff5ee, 0x303040, 0.6);
            this.scene.add(this._meshHemi);
            // Key light — strong, from top-right
            this._meshKey = new THREE.DirectionalLight(0xffffff, 0.9);
            this._meshKey.position.set(1.5, -1.5, -0.5);
            this.scene.add(this._meshKey);
            // Fill light — from opposite side
            this._meshFill = new THREE.DirectionalLight(0xffffff, 0.4);
            this._meshFill.position.set(-1.2, 0.3, 0.8);
            this.scene.add(this._meshFill);
            // Rim/back light for edge definition
            this._meshRim = new THREE.DirectionalLight(0xccddff, 0.3);
            this._meshRim.position.set(0, 1.5, 1.5);
            this.scene.add(this._meshRim);
            // Camera-following directional light
            this._meshCamLight = new THREE.DirectionalLight(0xffffff, 0.5);
            this._meshCamLight.position.set(0, 0, -1);
            this.camera.add(this._meshCamLight);
            this.scene.add(this.camera);

            // Generate environment map for reflections on glossy materials
            const pmrem = new THREE.PMREMGenerator(this.renderer);
            pmrem.compileEquirectangularShader();
            // Create a simple environment scene with gradient lighting
            const envScene = new THREE.Scene();
            envScene.add(new THREE.HemisphereLight(0xffffff, 0x222244, 2.0));
            const envD = new THREE.DirectionalLight(0xffffff, 1.5);
            envD.position.set(1.5, -1.5, -0.5);
            envScene.add(envD);
            const envTarget = pmrem.fromScene(envScene, 0.04);
            this.scene.environment = envTarget.texture;
            pmrem.dispose();

            // Adaptive: scale with camera distance
            this.controls.addEventListener('change', () => this._updateLightIntensity());
            this._updateLightIntensity();
        }

        console.log(`Built ${count} neuron meshes`);
    }

    _updateLightIntensity() {
        if (!this._meshCamLight) return;
        const s = this._lightIntensityScale || 1.0;
        const dist = this.camera.position.distanceTo(this.controls.target);
        const distScale = Math.max(1.0, dist / 0.4);
        this._meshCamLight.intensity = 0.5 * s * distScale;
        if (this._meshHemi) this._meshHemi.intensity = 0.6 * s * Math.max(1.0, distScale * 0.6);
        if (this._meshKey) this._meshKey.intensity = 0.9 * s;
        if (this._meshFill) this._meshFill.intensity = 0.4 * s;
        if (this._meshRim) this._meshRim.intensity = 0.3 * s;
        // Update envMapIntensity on mesh materials
        for (const [, mesh] of this.neuronMeshGeom) {
            if (mesh.material.envMapIntensity !== undefined) {
                mesh.material.envMapIntensity = s;
            }
        }
    }

    _buildPickScene() {
        // GPU picking: create a separate scene with flat-colored geometry
        // Each neuron gets a unique color encoding its identity
        this._pickScene = new THREE.Scene();
        this._pickTarget = new THREE.WebGLRenderTarget(1, 1);
        this._pickNeuronGroup = new THREE.Group();
        this._pickFullGroup = new THREE.Group();
        this._pickScene.add(this._pickNeuronGroup);
        this._pickScene.add(this._pickFullGroup);

        // Build pick meshes for clipped geometries
        for (const [key, geom] of this.typeRoiGeom) {
            this._createPickMesh(key, geom, this._pickNeuronGroup);
        }

        // Build pick meshes for full geometries
        for (const [bid, geom] of this.neuronFullGeom) {
            this._createPickMesh(bid, geom, this._pickFullGroup);
        }

        // Build pick meshes for neuron meshes (3D surfaces)
        this._pickMeshGroup = new THREE.Group();
        this._pickScene.add(this._pickMeshGroup);
        for (const [bid, mesh] of this.neuronMeshGeom) {
            this._createPickSurfaceMesh(bid, mesh, this._pickMeshGroup);
        }

        console.log(`Built pick scene: ${this._pickIdMap.size} pick IDs`);
    }

    _createPickMesh(key, sourceGeom, parentGroup) {
        const id = this._nextPickId++;
        const r = (id >> 16) & 0xFF;
        const g = (id >> 8) & 0xFF;
        const b = id & 0xFF;
        const color = new THREE.Color(r / 255, g / 255, b / 255);

        // Get the raw line geometry (from _rayTarget if thick-line, else from the mesh itself)
        const srcGeo = sourceGeom._rayTarget
            ? sourceGeom._rayTarget.geometry
            : sourceGeom.geometry;

        if (!srcGeo) return;

        const mat = new THREE.LineBasicMaterial({ color: color, linewidth: 2 });
        const pickLines = new THREE.LineSegments(srcGeo, mat);
        pickLines.visible = sourceGeom.visible;
        pickLines.userData = { sourceKey: key };

        parentGroup.add(pickLines);
        this._pickGeom.set(key, pickLines);

        // Map encoded ID back to identity
        const ud = sourceGeom.userData || {};
        this._pickIdMap.set(id, {
            type: ud.type || key.split('|')[0],
            bodyId: ud.bodyId || null,
            kind: ud.kind || 'clipped',
        });
    }

    _createPickSurfaceMesh(bid, sourceMesh, parentGroup) {
        const id = this._nextPickId++;
        const r = (id >> 16) & 0xFF;
        const g = (id >> 8) & 0xFF;
        const b = id & 0xFF;
        const color = new THREE.Color(r / 255, g / 255, b / 255);

        // Share geometry, use flat color material
        const mat = new THREE.MeshBasicMaterial({ color: color, side: THREE.DoubleSide });
        const pickMesh = new THREE.Mesh(sourceMesh.geometry, mat);
        pickMesh.visible = sourceMesh.visible;
        pickMesh.userData = { sourceKey: bid };

        parentGroup.add(pickMesh);
        this._pickGeom.set('mesh_' + bid, pickMesh);

        const ud = sourceMesh.userData || {};
        this._pickIdMap.set(id, {
            type: ud.type || null,
            bodyId: ud.bodyId || bid,
            kind: 'mesh',
        });
    }

    syncPickVisibility() {
        // Mirror main scene visibility to pick scene
        for (const [key, geom] of this.typeRoiGeom) {
            const pick = this._pickGeom.get(key);
            if (pick) pick.visible = geom.visible;
        }
        for (const [bid, geom] of this.neuronFullGeom) {
            const pick = this._pickGeom.get(bid);
            if (pick) pick.visible = geom.visible;
        }
        for (const [bid, mesh] of this.neuronMeshGeom) {
            const pick = this._pickGeom.get('mesh_' + bid);
            if (pick) pick.visible = mesh.visible;
        }
    }

    gpuPick(canvasX, canvasY) {
        if (!this._pickScene || !this._pickTarget) return null;

        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;

        // Use setViewOffset to render only the pixel at (canvasX, canvasY)
        this.camera.setViewOffset(w, h, canvasX, canvasY, 1, 1);
        this.camera.updateProjectionMatrix();

        this.syncPickVisibility();

        // Save and restore clear color so we don't clobber light mode
        const prevClearColor = new THREE.Color();
        this.renderer.getClearColor(prevClearColor);
        const prevClearAlpha = this.renderer.getClearAlpha();

        this.renderer.setRenderTarget(this._pickTarget);
        this.renderer.setClearColor(0x000000, 1);
        this.renderer.render(this._pickScene, this.camera);
        this.renderer.setRenderTarget(null);

        // Restore original clear color
        this.renderer.setClearColor(prevClearColor, prevClearAlpha);

        // Restore camera
        this.camera.clearViewOffset();
        this.camera.updateProjectionMatrix();

        // Read the single pixel
        this.renderer.readRenderTargetPixels(this._pickTarget, 0, 0, 1, 1, this._pickPixelBuf);
        const id = (this._pickPixelBuf[0] << 16) | (this._pickPixelBuf[1] << 8) | this._pickPixelBuf[2];

        if (id === 0) return null;  // Background
        return this._pickIdMap.get(id) || null;
    }

    _updateSize() {
        const w = window.innerWidth - SIDEBAR_W - TYPE_PANEL_W;
        const h = window.innerHeight - TOP_BAR_H;
        this.renderer.setSize(w, h);
        if (this.camera) {
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        }
        // Update thick-line resolution uniform
        const pr = this.renderer.getPixelRatio();
        _globalResolution.value.set(w * pr, h * pr);
    }

    _onResize() {
        this._updateSize();
    }

    _renderLoop() {
        requestAnimationFrame(() => this._renderLoop());
        if (this._recordingActive) return;  // Skip during video capture
        this.controls.update();

        // Update z-section clipping
        if (this.clipEnabled) {
            // Project the 8 bounding-box corners onto the camera axis
            // to find the tight near/far depth of actual geometry.
            this.camera.getWorldDirection(this._clipDir);
            let nearEdge = Infinity, farEdge = -Infinity;
            for (const c of this._bboxCorners) {
                const d = c.clone().sub(this.camera.position).dot(this._clipDir);
                if (d < nearEdge) nearEdge = d;
                if (d > farEdge) farEdge = d;
            }
            // Slider maps 0→nearEdge (clips nothing) to 1→farEdge (clips everything)
            const sliceDist = nearEdge + this.clipFraction * (farEdge - nearEdge);

            // View-space z threshold for custom ShaderMaterials (skeletons)
            _globalClipZ.value = -sliceDist;

            // World-space plane for built-in materials (ROIs, somas).
            // Normal = camera forward: keeps geometry beyond clip point.
            this._clipPoint.copy(this.camera.position).addScaledVector(this._clipDir, sliceDist);
            this.clipPlane.setFromNormalAndCoplanarPoint(this._clipDir, this._clipPoint);
        }

        this.renderer.render(this.scene, this.camera);
        this._renderScaleBar();
    }

    _initScaleBar() {
        // Position overlay canvas to match the 3D canvas exactly
        const overlay = document.createElement('canvas');
        const r = this.canvas.getBoundingClientRect();
        overlay.style.cssText = `position:fixed;top:${r.top}px;left:${r.left}px;width:${r.width}px;height:${r.height}px;pointer-events:none;z-index:5;`;
        document.body.appendChild(overlay);
        this._scaleBarCanvas = overlay;
        this._scaleBarCtx = overlay.getContext('2d');

        // Keep overlay aligned on resize
        window.addEventListener('resize', () => {
            const r2 = this.canvas.getBoundingClientRect();
            overlay.style.top = r2.top + 'px';
            overlay.style.left = r2.left + 'px';
            overlay.style.width = r2.width + 'px';
            overlay.style.height = r2.height + 'px';
        });
    }

    _renderScaleBar() {
        if (!this._scaleBarCanvas) return;
        const cam = this.camera;
        const ctl = this.controls;
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;

        if (this._scaleBarCanvas.width !== w || this._scaleBarCanvas.height !== h) {
            this._scaleBarCanvas.width = w;
            this._scaleBarCanvas.height = h;
            this._scaleBarCanvas.style.width = w + 'px';
            this._scaleBarCanvas.style.height = h + 'px';
        }

        const ctx = this._scaleBarCtx;
        ctx.clearRect(0, 0, w, h);

        const dist = cam.position.distanceTo(ctl.target);
        const fovRad = cam.fov * Math.PI / 180;
        const worldHeightAtTarget = 2 * dist * Math.tan(fovRad / 2);
        const dmax = this.data.normParams.dmax;
        const voxelNm = this.data.raw.voxelSizeNm || 8;
        const nmPerPixel = (worldHeightAtTarget * dmax * voxelNm) / h;

        const maxBarPx = w / 3;
        const minBarPx = w / 15;
        const niceValues = [];
        for (let exp = -1; exp < 8; exp++) {
            for (const m of [1, 2, 5]) niceValues.push(m * Math.pow(10, exp));
        }
        let barNm = 1000, barPx = barNm / nmPerPixel;
        for (const v of niceValues) {
            const px = v / nmPerPixel;
            if (px >= minBarPx && px <= maxBarPx) {
                barNm = v; barPx = px; break;
            }
        }

        let label;
        if (barNm >= 1e6) label = (barNm / 1e6).toFixed(barNm % 1e6 === 0 ? 0 : 1) + ' mm';
        else if (barNm >= 1000) label = (barNm / 1000).toFixed(barNm % 1000 === 0 ? 0 : 1) + ' \u00B5m';
        else label = barNm.toFixed(0) + ' nm';

        const x0 = 280;
        const y0 = h - 30;
        const barH = 10;
        const isCapture = this._captureScaleBar;
        const color = _currentTheme === THEMES.light ? (isCapture ? '#000' : 'rgb(212,160,23)') : (isCapture ? '#fff' : 'rgb(212,160,23)');

        ctx.shadowColor = _currentTheme === THEMES.light ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)';
        ctx.shadowBlur = 4;

        // Bar (simple rectangle)
        ctx.fillStyle = color;
        ctx.fillRect(x0, y0, barPx, barH);

        // Label
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(label, x0 + barPx / 2, y0 - 6);
        ctx.shadowBlur = 0;
    }

    _drawScaleBarOnCanvas(targetCanvas) {
        if (!this._scaleBarCanvas) return;
        const ctx = targetCanvas.getContext('2d');
        ctx.drawImage(this._scaleBarCanvas, 0, 0);
    }

    resetCamera() {
        this.camera.position.copy(this._initialCameraPos);
        this.camera.up.copy(this._initialCameraUp);
        this.controls.target.copy(this._initialCameraTarget);
        this.controls.update();
    }

    // Get all type-ROI geometries for a given type
    getTypeGeometries(typeName) {
        const result = [];
        for (const [key, geom] of this.typeRoiGeom) {
            if (key.startsWith(typeName + '|')) {
                result.push({ key, geom, roi: key.split('|')[1] });
            }
        }
        return result;
    }

    // Get all full skeleton geometries for a type
    getTypeFullGeometries(typeName) {
        const bids = this.data.getNeuronsForType(typeName);
        return bids.map(bid => ({ bid, geom: this.neuronFullGeom.get(bid) }))
                    .filter(e => e.geom);
    }
}

// ---- Synapse Manager ----
class SynapseManager {
    constructor(viewer) {
        this.viewer = viewer;
        this.data = null;       // decoded synapse arrays
        this.loaded = false;
        this.loading = false;
        this.groups = [];       // active synapse display groups
        this._nextId = 1;
        this._globalRadius = 0.002;
        this._sphereGeo = null;
    }

    _getSphereGeo() {
        if (!this._sphereGeo) {
            this._sphereGeo = new THREE.SphereGeometry(1, 8, 6);
        }
        return this._sphereGeo;
    }

    async loadData() {
        if (this.loaded || this.loading) return this.loaded;
        if (!DATA.synapseData) {
            console.warn('No synapse data embedded in DATA bundle');
            return false;
        }
        this.loading = true;
        try {
            this.data = this._decode(DATA.synapseData);
            this.loaded = true;
            console.log(`SynapseManager: loaded ${this.data.count} synapses`);
        } catch (e) {
            console.error('Failed to decode synapse data:', e);
        }
        this.loading = false;
        return this.loaded;
    }

    _decode(raw) {
        const qs = raw.quantScale || 30000;
        const decodeB64 = (b64, Type) => {
            const bin = atob(b64);
            const buf = new ArrayBuffer(bin.length);
            const u8 = new Uint8Array(buf);
            for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
            return new Type(buf);
        };
        const decode16 = (b64) => {
            const i16 = decodeB64(b64, Int16Array);
            const f32 = new Float32Array(i16.length);
            for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / qs;
            return f32;
        };
        const count = raw.count;
        const xPre = decode16(raw.xPre);
        const yPre = decode16(raw.yPre);
        const zPre = decode16(raw.zPre);
        const xPost = decode16(raw.xPost);
        const yPost = decode16(raw.yPost);
        const zPost = decode16(raw.zPost);

        // Decode base64-encoded int32 bodyId arrays
        const preBids = decodeB64(raw.preBidsB64, Int32Array);
        const postBids = decodeB64(raw.postBidsB64, Int32Array);

        // Decode ROIs from lookup + uint16 index
        const roiLookup = raw.roiLookup;
        const roiIdx = decodeB64(raw.roiIndicesB64, Uint16Array);
        const rois = new Array(count);
        for (let i = 0; i < count; i++) rois[i] = roiLookup[roiIdx[i]];

        const bidTypeMap = raw.bidTypeMap || {};

        // Build index: "preBid|postBid" -> [indices]
        const pairIndex = new Map();
        const preBidIndex = new Map();
        const postBidIndex = new Map();
        for (let i = 0; i < count; i++) {
            const key = `${preBids[i]}|${postBids[i]}`;
            if (!pairIndex.has(key)) pairIndex.set(key, []);
            pairIndex.get(key).push(i);

            const pb = preBids[i];
            if (!preBidIndex.has(pb)) preBidIndex.set(pb, []);
            preBidIndex.get(pb).push(i);

            const qb = postBids[i];
            if (!postBidIndex.has(qb)) postBidIndex.set(qb, []);
            postBidIndex.get(qb).push(i);
        }

        return {
            count, xPre, yPre, zPre, xPost, yPost, zPost,
            preBids, postBids, rois, bidTypeMap,
            pairIndex, preBidIndex, postBidIndex
        };
    }

    /**
     * Filter synapses matching criteria.
     * @param {number[]} ourBids - body IDs on "our" side
     * @param {string} partnerType - partner type name to match
     * @param {string} roi - ROI to filter (or null for all)
     * @param {string} direction - 'upstream' or 'downstream'
     * @returns {number[]} array of synapse indices
     */
    filterSynapses(ourBids, partnerType, roi, direction) {
        if (!this.data) return [];
        const d = this.data;
        const btm = d.bidTypeMap;
        const ourSet = new Set(ourBids.map(Number));
        const indices = [];

        // direction='upstream': partner is presynaptic (pre), our neurons are postsynaptic (post)
        // direction='downstream': our neurons are presynaptic (pre), partner is postsynaptic (post)
        for (let i = 0; i < d.count; i++) {
            if (roi && d.rois[i] !== roi) continue;

            if (direction === 'upstream') {
                // pre = partner, post = ours
                if (!ourSet.has(d.postBids[i])) continue;
                if ((btm[String(d.preBids[i])] || 'unknown') !== partnerType) continue;
            } else {
                // pre = ours, post = partner
                if (!ourSet.has(d.preBids[i])) continue;
                if ((btm[String(d.postBids[i])] || 'unknown') !== partnerType) continue;
            }
            indices.push(i);
        }
        return indices;
    }

    /**
     * Create a visual synapse group.
     * @param {object} opts
     * @param {string} opts.label - display label
     * @param {number[]} opts.ourBids - our body IDs
     * @param {string} opts.partnerType - partner type name
     * @param {string} opts.ourType - our type name
     * @param {string} opts.roi - ROI name (or null)
     * @param {string} opts.direction - 'upstream' or 'downstream'
     * @param {string} opts.synapseType - 'pre', 'post', or 'both'
     * @param {string|null} opts.color - CSS color or null for neuron color
     * @returns {object} group object or null
     */
    createGroup(opts) {
        const indices = this.filterSynapses(
            opts.ourBids, opts.partnerType, opts.roi, opts.direction);
        if (indices.length === 0) {
            console.warn('No synapses found for', opts.label);
            return null;
        }

        const d = this.data;
        const positions = [];

        for (const i of indices) {
            if (opts.synapseType === 'pre' || opts.synapseType === 'both') {
                positions.push({ x: d.xPre[i], y: d.yPre[i], z: d.zPre[i],
                                 idx: i, side: 'pre' });
            }
            if (opts.synapseType === 'post' || opts.synapseType === 'both') {
                positions.push({ x: d.xPost[i], y: d.yPost[i], z: d.zPost[i],
                                 idx: i, side: 'post' });
            }
        }

        if (positions.length === 0) return null;

        // Determine color
        let color = opts.color || null;
        let useNeuronColor = opts.useNeuronColor != null ? opts.useNeuronColor : !color;
        if (useNeuronColor) {
            const mode = this.viewer.data.colorModes[this.viewer.vis.activeColorMode || 0];
            color = (mode && mode.type_colors && mode.type_colors[opts.ourType]) || '#ffffff';
        }

        // Build InstancedMesh
        const geo = this._getSphereGeo();
        const mat = new THREE.MeshBasicMaterial({ color: color });
        const mesh = new THREE.InstancedMesh(geo, mat, positions.length);
        mesh.frustumCulled = false;

        const dummy = new THREE.Object3D();
        const r = this._globalRadius;
        for (let j = 0; j < positions.length; j++) {
            dummy.position.set(positions[j].x, positions[j].y, positions[j].z);
            dummy.scale.set(r, r, r);
            dummy.updateMatrix();
            mesh.setMatrixAt(j, dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;

        // Outline mesh (slightly larger, back-face only)
        const outlineMesh = this._createOutlineMesh(geo, positions, r);

        const groupId = this._nextId++;
        const group = {
            id: groupId,
            label: opts.label,
            ourType: opts.ourType,
            partnerType: opts.partnerType,
            roi: opts.roi,
            direction: opts.direction,
            synapseType: opts.synapseType,
            mesh: mesh,
            outlineMesh: outlineMesh,
            positions: positions,
            indices: indices,
            color: color,
            useNeuronColor: useNeuronColor,
            visible: true,
            count: positions.length,
        };

        mesh.userData = { isSynapse: true, groupId: groupId };

        this.viewer.scene.synapseGroup.add(mesh);
        if (outlineMesh) this.viewer.scene.synapseGroup.add(outlineMesh);
        this.groups.push(group);
        return group;
    }

    createGroupFromIndices(indices, opts) {
        // Create synapse group from pre-resolved indices (bypasses filterSynapses)
        if (!indices || indices.length === 0) return null;
        const d = this.data;
        const positions = [];
        const synapseType = opts.synapseType || 'both';
        for (const i of indices) {
            if (synapseType === 'pre' || synapseType === 'both') {
                positions.push({ x: d.xPre[i], y: d.yPre[i], z: d.zPre[i], idx: i, side: 'pre' });
            }
            if (synapseType === 'post' || synapseType === 'both') {
                positions.push({ x: d.xPost[i], y: d.yPost[i], z: d.zPost[i], idx: i, side: 'post' });
            }
        }
        if (positions.length === 0) return null;

        const color = opts.color || '#ffffff';
        const geo = this._getSphereGeo();
        const mat = new THREE.MeshBasicMaterial({ color: color });
        const mesh = new THREE.InstancedMesh(geo, mat, positions.length);
        mesh.frustumCulled = false;
        const dummy = new THREE.Object3D();
        const r = this._globalRadius;
        for (let j = 0; j < positions.length; j++) {
            dummy.position.set(positions[j].x, positions[j].y, positions[j].z);
            dummy.scale.set(r, r, r);
            dummy.updateMatrix();
            mesh.setMatrixAt(j, dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;

        // Outline mesh
        const outlineMesh = this._createOutlineMesh(geo, positions, r);

        const groupId = this._nextId++;
        const group = {
            id: groupId, label: opts.label || 'CSV group',
            ourType: null, partnerType: null, roi: null,
            direction: null, synapseType: synapseType,
            mesh: mesh, outlineMesh: outlineMesh,
            positions: positions, indices: indices,
            color: color, useNeuronColor: false, visible: true,
            count: positions.length, _fromCSV: true,
        };
        mesh.userData = { isSynapse: true, groupId: groupId };
        this.viewer.scene.synapseGroup.add(mesh);
        if (outlineMesh) this.viewer.scene.synapseGroup.add(outlineMesh);
        this.groups.push(group);
        return group;
    }

    _createOutlineMesh(geo, positions, radius) {
        const outlineScale = 1.35;  // outline is 35% larger than the sphere
        const outlineColor = _currentTheme === THEMES.light ? 0x000000 : 0xffffff;
        const outlineMat = new THREE.MeshBasicMaterial({
            color: outlineColor, side: THREE.BackSide,
            depthWrite: false,  // don't write depth so outlines never occlude each other
        });
        const outlineMesh = new THREE.InstancedMesh(geo, outlineMat, positions.length);
        outlineMesh.renderOrder = -1;  // render outlines before fill spheres
        outlineMesh.frustumCulled = false;
        const dummy = new THREE.Object3D();
        const or = radius * outlineScale;
        for (let j = 0; j < positions.length; j++) {
            dummy.position.set(positions[j].x, positions[j].y, positions[j].z);
            dummy.scale.set(or, or, or);
            dummy.updateMatrix();
            outlineMesh.setMatrixAt(j, dummy.matrix);
        }
        outlineMesh.instanceMatrix.needsUpdate = true;
        outlineMesh.userData = { isSynapseOutline: true };
        return outlineMesh;
    }

    _updateOutlineColors() {
        // Update all synapse outline mesh colors to match current theme
        const outlineColor = _currentTheme === THEMES.light ? 0x000000 : 0xffffff;
        for (const g of this.groups) {
            if (g.outlineMesh) g.outlineMesh.material.color.setHex(outlineColor);
        }
    }

    removeGroup(groupId) {
        const idx = this.groups.findIndex(g => g.id === groupId);
        if (idx < 0) return;
        const g = this.groups[idx];
        this.viewer.scene.synapseGroup.remove(g.mesh);
        if (g.outlineMesh) {
            this.viewer.scene.synapseGroup.remove(g.outlineMesh);
            g.outlineMesh.material.dispose();
            g.outlineMesh.dispose();
        }
        g.mesh.geometry !== this._sphereGeo && g.mesh.geometry.dispose();
        g.mesh.material.dispose();
        g.mesh.dispose();
        this.groups.splice(idx, 1);
    }

    toggleVisibility(groupId) {
        const g = this.groups.find(g => g.id === groupId);
        if (!g) return;
        g.visible = !g.visible;
        g.mesh.visible = g.visible;
        if (g.outlineMesh) g.outlineMesh.visible = g.visible;
    }

    setGroupColor(groupId, color, useNeuronColor) {
        const g = this.groups.find(g => g.id === groupId);
        if (!g) return;
        g.color = color;
        g.useNeuronColor = !!useNeuronColor;
        g.mesh.material.color.set(color);
    }

    setGlobalSize(radius) {
        this._globalRadius = radius;
        const outlineScale = 1.35;
        const or = radius * outlineScale;
        const dummy = new THREE.Object3D();
        for (const g of this.groups) {
            for (let j = 0; j < g.positions.length; j++) {
                dummy.position.set(g.positions[j].x, g.positions[j].y, g.positions[j].z);
                dummy.scale.set(radius, radius, radius);
                dummy.updateMatrix();
                g.mesh.setMatrixAt(j, dummy.matrix);
                if (g.outlineMesh) {
                    dummy.scale.set(or, or, or);
                    dummy.updateMatrix();
                    g.outlineMesh.setMatrixAt(j, dummy.matrix);
                }
            }
            g.mesh.instanceMatrix.needsUpdate = true;
            if (g.outlineMesh) g.outlineMesh.instanceMatrix.needsUpdate = true;
        }
    }

    getVisibleMeshes() {
        return this.groups.filter(g => g.visible).map(g => g.mesh);
    }

    resolveHit(intersection) {
        if (!intersection || !intersection.object) return null;
        const ud = intersection.object.userData;
        if (!ud || !ud.isSynapse) return null;
        const g = this.groups.find(gr => gr.id === ud.groupId);
        if (!g) return null;
        const instanceId = intersection.instanceId;
        if (instanceId == null || instanceId >= g.positions.length) return null;
        const pos = g.positions[instanceId];
        const synIdx = pos.idx;
        const d = this.data;
        return {
            isSynapse: true,
            preType: d.bidTypeMap[String(d.preBids[synIdx])] || 'unknown',
            postType: d.bidTypeMap[String(d.postBids[synIdx])] || 'unknown',
            preBid: d.preBids[synIdx],
            postBid: d.postBids[synIdx],
            roi: d.rois[synIdx],
            side: pos.side,
            groupId: g.id,
        };
    }

    /** Update neuron-colored groups when color mode changes */
    syncColors() {
        const mode = this.viewer.data.colorModes[this.viewer.vis.activeColorMode || 0];
        if (!mode) return;
        for (const g of this.groups) {
            if (!g.useNeuronColor) continue;
            let newColor;
            if (g._splitBid && mode.colors) {
                // Per-neuron color for split groups
                newColor = mode.colors[String(g._splitBid)] || mode.type_colors[g.ourType] || '#ffffff';
            } else {
                newColor = (mode.type_colors && mode.type_colors[g.ourType]) || '#ffffff';
            }
            g.color = newColor;
            g.mesh.material.color.set(newColor);
        }
    }
}

// ---- Session Manager ----
class SessionManager {
    constructor(viewer) {
        this.viewer = viewer;
        const term = (viewer.data.regexTerm || 'unknown').replace(/[^a-zA-Z0-9_]/g, '');
        this.storageKey = `neuroviz_${term}`;
        this._saveTimer = null;
        this._saveDelay = 500;  // debounce ms
    }

    debouncedSave() {
        if (this._saveTimer) clearTimeout(this._saveTimer);
        this._saveTimer = setTimeout(() => this.save(), this._saveDelay);
    }

    save() {
        const vis = this.viewer.vis;
        const ui = this.viewer.ui;
        const scene = this.viewer.scene;
        const synMgr = this.viewer.synapse;
        const customMode = this.viewer.data.colorModes.find(m => m.is_custom);

        const state = {
            version: 1,
            timestamp: Date.now(),
            // Visibility state
            highlightedSet: [...vis.highlightedSet],
            _explicitHideAll: vis._explicitHideAll,
            clipToRoi: Object.assign({}, vis.clipToRoi),
            activeColorMode: vis.activeColorMode,
            _somataVisible: vis._somataVisible,
            roiChecked: Object.assign({}, vis.roiChecked),
            showWireframes: vis.showWireframes,
            colorFilteredOutTypes: [...vis.colorFilteredOutTypes],
            colorFilteredOutNeurons: [...vis.colorFilteredOutNeurons],
            colorFilterMin: vis.colorFilterMin,
            colorFilterMax: vis.colorFilterMax,
            activeNTs: vis.activeNTs ? [...vis.activeNTs] : null,
            // UI state
            hlModeByNeuron: ui.hlModeByNeuron,
            connSelectedKey: ui.connSelectedKey,
            _magnifierEnabled: ui._magnifierEnabled || false,
            // Camera
            camera: {
                pos: { x: scene.camera.position.x, y: scene.camera.position.y, z: scene.camera.position.z },
                tgt: { x: scene.controls.target.x, y: scene.controls.target.y, z: scene.controls.target.z },
                up: { x: scene.camera.up.x, y: scene.camera.up.y, z: scene.camera.up.z },
            },
            clipEnabled: scene.clipEnabled || false,
            clipFraction: scene.clipFraction || 0,
            // Saved banks
            savedViews: (ui._savedViews || []).map(v => ({
                pos: v.pos, tgt: v.tgt, up: v.up,
                clipEnabled: v.clipEnabled, clipFraction: v.clipFraction,
            })),
            savedSets: (ui._savedSets || []).map(s => ({
                hlModeByNeuron: s.hlModeByNeuron,
                highlightedSet: [...(s.highlightedSet || [])],
                activeColorMode: s.activeColorMode,
                cbarMinPct: s.cbarMinPct,
                cbarMaxPct: s.cbarMaxPct,
                activeNTs: s.activeNTs ? [...s.activeNTs] : null,
                roiChecked: Object.assign({}, s.roiChecked),
            })),
            roiSavedSets: (ui._roiSavedSets || []).map(s => ({
                roiChecked: Object.assign({}, s.roiChecked),
            })),
            // Synapse groups (criteria only, re-created on restore)
            synapseGroups: synMgr ? synMgr.groups.map(g => ({
                ourType: g.ourType,
                partnerType: g.partnerType,
                roi: g.roi,
                direction: g.direction,
                synapseType: g.synapseType,
                label: g.label,
                color: g.color,
                useNeuronColor: g.useNeuronColor,
                visible: g.visible,
                _splitBid: g._splitBid || null,
                ourBids: g.ourBids || null,
            })) : [],
            globalRadius: synMgr ? synMgr._globalRadius : 0.002,
            // Custom color mode
            customTypeColors: customMode ? Object.assign({}, customMode.type_colors) : null,
            customNeuronColors: customMode ? Object.assign({}, customMode._neuronColors) : null,
            customColors: customMode ? Object.assign({}, customMode.colors) : null,
            // Color filter UI
            _cbarMinPct: ui._cbarMinPct || 0,
            _cbarMaxPct: ui._cbarMaxPct || 100,
            // Uploaded color modes — name-based persistence
            activeColorModeName: this.viewer.data.colorModes[vis.activeColorMode]?.name || null,
            uploadedColorModes: this.viewer.data.colorModes
                .filter(m => m.is_uploaded)
                .map(m => ({
                    name: m.name, colors: m.colors, type_colors: m.type_colors,
                    is_scalar: m.is_scalar, is_instance_level: m.is_instance_level || false,
                    is_categorical: m.is_categorical || false,
                    cmin: m.cmin, cmax: m.cmax, colorscale: m.colorscale, label: m.label,
                    type_values: m.type_values || null,
                })),
            // Uploaded synapse CSVs
            uploadedSynapseCSVs: synMgr?._uploadedCSVs || [],
        };

        try {
            localStorage.setItem(this.storageKey, JSON.stringify(state));
        } catch (e) {
            console.warn('Session save failed (quota?):', e.message);
        }
    }

    load() {
        try {
            const raw = localStorage.getItem(this.storageKey);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            console.warn('Session load failed:', e.message);
            return null;
        }
    }

    clear() {
        localStorage.removeItem(this.storageKey);
    }

    tryRestore() {
        const state = this.load();
        if (!state || state.version !== 1) return false;
        try {
            this._restore(state);
            return true;
        } catch (e) {
            console.warn('Session restore failed:', e);
            return false;
        }
    }

    _restore(s) {
        const vis = this.viewer.vis;
        const ui = this.viewer.ui;
        const scene = this.viewer.scene;
        const synMgr = this.viewer.synapse;

        // 1. Custom color mode — restore before switching color mode
        const customMode = this.viewer.data.colorModes.find(m => m.is_custom);
        if (customMode && s.customTypeColors) {
            Object.assign(customMode.type_colors, s.customTypeColors);
            if (s.customColors) Object.assign(customMode.colors, s.customColors);
            if (s.customNeuronColors && customMode._neuronColors)
                Object.assign(customMode._neuronColors, s.customNeuronColors);
        }

        // 2. Restore uploaded color modes (before switching mode or color)
        if (s.uploadedColorModes && s.uploadedColorModes.length > 0) {
            for (const um of s.uploadedColorModes) {
                // Skip if a mode with this name already exists (e.g., from re-upload)
                if (this.viewer.data.colorModes.find(m => m.name === um.name)) continue;
                const mode = Object.assign({}, um, { is_uploaded: true });
                const insertIdx = this.viewer.data.colorModes.length - 1; // before Custom
                this.viewer.data.colorModes.splice(insertIdx, 0, mode);
                if (ui._colorSection) {
                    ui._addColorModeButton(mode, insertIdx, ui._colorSection,
                        'width:32px;height:32px;border:1px solid #555;border-radius:3px;cursor:pointer;background:#222;color:#fff;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;padding:0;font-size:15px;box-sizing:border-box;');
                }
            }
            if (ui._reindexColorButtons) ui._reindexColorButtons();
            if (ui._updateInstanceBtnState) ui._updateInstanceBtnState();
        }

        // 3. Switch type/neuron mode if needed
        if (s.hlModeByNeuron !== ui.hlModeByNeuron && ui._switchMode) {
            ui._switchMode(s.hlModeByNeuron, true);
        }

        // 4. Restore highlightedSet and clipToRoi
        vis.highlightedSet = new Set(s.highlightedSet || []);
        vis._explicitHideAll = s._explicitHideAll || false;
        vis.clipToRoi = s.clipToRoi || {};

        // 5. Restore color mode (resolve by name first, fall back to index)
        {
            let targetIdx = s.activeColorMode;
            if (s.activeColorModeName) {
                const byName = this.viewer.data.colorModes.findIndex(m => m.name === s.activeColorModeName);
                if (byName >= 0) targetIdx = byName;
            }
            if (targetIdx != null && targetIdx >= 0 && targetIdx < this.viewer.data.colorModes.length) {
                vis.switchColorMode(targetIdx);
                const topBar = ui.topBar;
                if (topBar) {
                    topBar.querySelectorAll('button[data-colormode]').forEach(b => {
                        b.style.background = '#222'; b.style.color = '#fff';
                    });
                    const activeBtn = topBar.querySelector(`button[data-colormode="${targetIdx}"]`);
                    if (activeBtn) { activeBtn.style.background = 'rgb(212,160,23)'; activeBtn.style.color = '#000'; }
                }
            }
        }

        // 6. Restore ROI visibility
        if (s.roiChecked) {
            for (const [roi, checked] of Object.entries(s.roiChecked)) {
                vis.setRoiChecked(roi, checked);
            }
        }

        // 7. Restore somata visibility
        if (s._somataVisible === false) {
            vis._somataVisible = false;
            for (const [, mesh] of vis.scene.somaGeom || new Map()) {
                mesh.visible = false;
            }
        }

        // 8. Restore camera
        if (s.camera) {
            const p = s.camera.pos, t = s.camera.tgt, u = s.camera.up;
            scene.camera.position.set(p.x, p.y, p.z);
            scene.camera.up.set(u.x, u.y, u.z);
            scene.controls.target.set(t.x, t.y, t.z);
            scene.controls.update();
        }

        // 9. Restore saved view/set banks
        if (s.savedViews) {
            ui._savedViews = s.savedViews;
            ui._activeViewIdx = null;
        }
        if (s.savedSets) {
            ui._savedSets = s.savedSets.map(ss => ({
                ...ss,
                highlightedSet: new Set(ss.highlightedSet || []),
                activeNTs: ss.activeNTs ? new Set(ss.activeNTs) : null,
            }));
            ui._activeSetIdx = null;
        }
        if (s.roiSavedSets) {
            ui._roiSavedSets = s.roiSavedSets;
            ui._roiActiveSetIdx = null;
        }

        // 10. Restore color filter state
        if (s.colorFilteredOutTypes) vis.colorFilteredOutTypes = new Set(s.colorFilteredOutTypes);
        if (s.colorFilteredOutNeurons) vis.colorFilteredOutNeurons = new Set(s.colorFilteredOutNeurons);
        if (s.colorFilterMin != null) vis.colorFilterMin = s.colorFilterMin;
        if (s.colorFilterMax != null) vis.colorFilterMax = s.colorFilterMax;
        if (s.activeNTs) vis.activeNTs = new Set(s.activeNTs);

        // 11. Restore synapse groups
        if (synMgr && synMgr.loaded && s.synapseGroups && s.synapseGroups.length > 0) {
            if (s.globalRadius) synMgr._globalRadius = s.globalRadius;
            for (const sg of s.synapseGroups) {
                const newGroup = synMgr.createGroup({
                    label: sg.label,
                    ourBids: sg.ourBids || null,
                    ourType: sg.ourType,
                    partnerType: sg.partnerType,
                    roi: sg.roi,
                    direction: sg.direction,
                    synapseType: sg.synapseType,
                    color: sg.color,
                    useNeuronColor: sg.useNeuronColor,
                });
                if (newGroup) {
                    if (sg._splitBid) newGroup._splitBid = sg._splitBid;
                    if (!sg.visible) {
                        synMgr.toggleVisibility(newGroup.id);
                    }
                }
            }
        }

        // 12. Restore uploaded synapse CSVs
        if (synMgr && synMgr.loaded && s.uploadedSynapseCSVs && s.uploadedSynapseCSVs.length > 0) {
            synMgr._uploadedCSVs = s.uploadedSynapseCSVs;
            for (const csv of s.uploadedSynapseCSVs) {
                if (ui._handleSynapseCSVUpload) ui._handleSynapseCSVUpload(csv.text, csv.filename, true);
            }
        }

        // 13. Final sync
        vis._applyAllVisibility();
        if (ui._rebuildPanelContent) ui._rebuildPanelContent();
        ui.syncAllState();
        if (ui._updateColorbar) {
            ui._updateColorbar(this.viewer.data.colorModes[vis.activeColorMode]);
        }
        ui._updatePanelSwatches();
        if (synMgr) ui._updateSynapsePanel();

        // Rebuild saved view/set bank UIs
        if (ui._rebuildViewBank) ui._rebuildViewBank();
        if (ui._rebuildSetBank) ui._rebuildSetBank();
        if (ui._rebuildRoiSetBank) ui._rebuildRoiSetBank();
    }

    exportJSON() {
        this.save();  // ensure latest state
        const data = localStorage.getItem(this.storageKey);
        if (!data) return;
        const blob = new Blob([data], { type: 'application/json' });
        _saveFileAs(blob, `${this.storageKey}_session.json`, [
            { description: 'JSON Session', accept: { 'application/json': ['.json'] } }
        ]);
    }

    importJSON(jsonString) {
        try {
            const state = JSON.parse(jsonString);
            if (state.version !== 1) throw new Error('Unknown session version');
            localStorage.setItem(this.storageKey, jsonString);
            this._restore(state);
        } catch (e) {
            console.error('Session import failed:', e);
            alert('Failed to import session: ' + e.message);
        }
    }
}

// ---- Save File Helper ----
// Uses File System Access API (showSaveFilePicker) when available for directory+name control.
// Falls back to classic a.download for unsupported browsers.
// Persists the last-used directory handle across saves within a session.
// After the first save, subsequent saves default to the same folder.
let _lastDirHandle = null;

async function _saveFileAs(blob, defaultName, fileTypes) {
    if (window.showSaveFilePicker) {
        try {
            const opts = {
                suggestedName: defaultName,
                types: fileTypes || [{ description: 'File', accept: { 'application/octet-stream': [''] } }],
            };
            // Re-use last directory so the dialog opens where user last saved
            if (_lastDirHandle) opts.startIn = _lastDirHandle;
            const handle = await window.showSaveFilePicker(opts);
            const writable = await handle.createWritable();
            await writable.write(blob);
            await writable.close();
            // Remember this file's parent directory for next time.
            // getParent() is only available if the page has permission to the parent dir,
            // but we can store the file handle itself as startIn (browsers accept either).
            _lastDirHandle = handle;
            return;
        } catch (e) {
            if (e.name === 'AbortError') return;  // user cancelled
            // Fall through to legacy download
        }
    }
    // Legacy fallback
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = defaultName;
    a.click();
    URL.revokeObjectURL(url);
}

// ---- Visibility Manager ----
class VisibilityManager {
    constructor(viewer) {
        this.viewer = viewer;
        this.scene = viewer.scene;
        this.data = viewer.data;

        // State
        this.highlightedSet = new Set();
        this._explicitHideAll = false; // true = user explicitly cleared all highlights
        this.clipToRoi = {};         // key -> boolean (true = clipped)
        this._somataVisible = true;  // toggled by UI checkbox
        this._meshesVisible = this.scene._meshesAvailable;  // default ON if meshes exist
        this.roiChecked = {};        // roiName -> boolean
        this.showWireframes = true;  // ROI mesh visibility toggle
        this.activeColorMode = 0;

        // Color filter state (separate from highlightedSet)
        this.colorFilteredOutTypes = new Set();   // type names excluded by color filter
        this.colorFilteredOutNeurons = new Set(); // bodyIds excluded by color filter
        this.colorFilterMin = 0;    // percentile 0-100
        this.colorFilterMax = 100;  // percentile 0-100
        this.activeNTs = null;      // Set of active NT names (null = all active)

        // Initialize ROI checked state
        for (const roi of this.data.sidebarRois) {
            this.roiChecked[roi] = (roi === this.data.primaryRoi);
        }
        // Auto-check paired ROI
        this._checkPairedRoi();
    }

    _checkPairedRoi() {
        const pri = this.data.primaryRoi;
        const m = pri.match(/^(.+)\(([LR])\)$/);
        if (m) {
            const paired = m[1] + '(' + (m[2] === 'L' ? 'R' : 'L') + ')';
            if (this.data.sidebarRois.includes(paired)) {
                this.roiChecked[paired] = true;
            }
        }
    }

    /** Gate soma mesh visibility through the somata toggle */
    _showSoma(soma, wantVisible) {
        const vis = wantVisible && this._somataVisible;
        soma.visible = vis;
        soma.material.opacity = vis ? 1.0 : 0.02;
    }

    // Core: show/hide a type based on highlight + clip + color filter state
    applyTypeVisibility(typeName) {
        // Empty set = show all ONLY in the initial "nothing selected" state.
        // When the user explicitly clears all highlights, _explicitHideAll=true and we hide everything.
        const isHighlighted = (!this._explicitHideAll && this.highlightedSet.size === 0)
                              || this.highlightedSet.has(typeName);
        if (!isHighlighted || this.colorFilteredOutTypes.has(typeName)) {
            // Not highlighted or filtered out by color = hidden
            this._hideType(typeName);
            return;
        }

        const clipped = this.clipToRoi[typeName] === true;

        const hasNeuronFilter = this.colorFilteredOutNeurons.size > 0;

        const showMesh = this._meshesVisible;

        if (showMesh) {
            // Mesh mode: hide all skeletons, show meshes
            for (const { geom } of this.scene.getTypeGeometries(typeName)) {
                geom.visible = false;
            }
            for (const { geom } of this.scene.getTypeFullGeometries(typeName)) {
                geom.visible = false;
            }
            // Apply per-face ROI clipping to meshes
            const meshBids = this.data.getNeuronsForType(typeName);
            if (clipped) {
                for (const bid of meshBids) {
                    this.scene.clipMeshToRois(bid, this.roiChecked);
                }
            } else {
                for (const bid of meshBids) {
                    this.scene.clipMeshToRois(bid, null);  // Restore full mesh
                }
            }
        } else if (clipped) {
            // Skeleton clipped mode
            for (const { key, geom, roi } of this.scene.getTypeGeometries(typeName)) {
                geom.visible = this.roiChecked[roi] === true;
            }
            for (const { bid, geom } of this.scene.getTypeFullGeometries(typeName)) {
                geom.visible = false;
            }
        } else {
            // Skeleton full mode
            for (const { geom } of this.scene.getTypeGeometries(typeName)) {
                geom.visible = false;
            }
            for (const { bid, geom } of this.scene.getTypeFullGeometries(typeName)) {
                geom.visible = !hasNeuronFilter || !this.colorFilteredOutNeurons.has(bid);
            }
        }

        // Per-neuron: somas and meshes
        const bids = this.data.getNeuronsForType(typeName);
        for (const bid of bids) {
            const neuronFiltered = hasNeuronFilter && this.colorFilteredOutNeurons.has(bid);
            // Somas: visible in skeleton mode when not clipped
            const soma = this.scene.somaGeom.get(bid);
            if (soma) {
                this._showSoma(soma, !showMesh && !clipped && !neuronFiltered);
            }
            // Meshes: visible in mesh mode (clipping handled by face filtering)
            const meshGeom = this.scene.neuronMeshGeom.get(bid);
            if (meshGeom) {
                meshGeom.visible = showMesh && !neuronFiltered;
            }
        }
    }

    _hideType(typeName) {
        for (const { geom } of this.scene.getTypeGeometries(typeName)) {
            geom.visible = false;
        }
        for (const { geom } of this.scene.getTypeFullGeometries(typeName)) {
            geom.visible = false;
        }
        const bids = this.data.getNeuronsForType(typeName);
        for (const bid of bids) {
            const soma = this.scene.somaGeom.get(bid);
            if (soma) {
                this._showSoma(soma, false);
            }
            const meshGeom = this.scene.neuronMeshGeom.get(bid);
            if (meshGeom) meshGeom.visible = false;
        }
    }

    // Toggle highlight for a type
    toggleHighlight(key) {
        if (this.highlightedSet.has(key)) {
            this.highlightedSet.delete(key);
            delete this.clipToRoi[key];
            // If last item was removed, treat as explicit hide-all (not "show all" reset)
            if (this.highlightedSet.size === 0) this._explicitHideAll = true;
        } else {
            this._explicitHideAll = false;  // adding anything clears the explicit-hide state
            this.highlightedSet.add(key);
            this.clipToRoi[key] = false;  // Default: full skeleton (not clipped)
        }
        this._applyAllVisibility();
    }

    highlightAll() {
        this._explicitHideAll = false;
        for (const t of this.data.allTypes) {
            this.highlightedSet.add(t);
            if (!(t in this.clipToRoi)) this.clipToRoi[t] = false;
        }
        this._applyAllVisibility();
    }

    unhighlightAll() {
        this._explicitHideAll = true;  // explicit clear = hide all, not "show all"
        this.highlightedSet.clear();
        this.clipToRoi = {};
        this._applyAllVisibility();
    }

    // Dispatch visibility update: detects neuron vs type mode by key
    _applyKeyVisibility(key) {
        if (this.data.neuronType[key]) {
            this._applyNeuronVisibility(key);
        } else {
            this.applyTypeVisibility(key);
        }
    }

    toggleClip(key) {
        if (!this.highlightedSet.has(key)) return;
        this.clipToRoi[key] = this.clipToRoi[key] === true ? false : true;
        this._applyKeyVisibility(key);
    }

    // Neuron-level highlight/clip (for neuron mode)
    toggleHighlightNeuron(bid) {
        if (this.highlightedSet.has(bid)) {
            this.highlightedSet.delete(bid);
            delete this.clipToRoi[bid];
        } else {
            this.highlightedSet.add(bid);
            this.clipToRoi[bid] = false;
        }
        this._applyNeuronVisibility(bid);
    }

    toggleClipNeuron(bid) {
        if (!this.highlightedSet.has(bid)) return;
        this.clipToRoi[bid] = this.clipToRoi[bid] === true ? false : true;
        this._applyNeuronVisibility(bid);
    }

    _applyNeuronVisibility(bid) {
        const typeName = this.data.neuronType[bid];
        if (!typeName) return;
        // When nothing is highlighted, treat all neurons as highlighted
        const baseHl = this.highlightedSet.size === 0 || this.highlightedSet.has(bid);
        const isHl = baseHl && !this.colorFilteredOutNeurons.has(bid);
        const clipped = this.clipToRoi[bid] === true;

        const showMesh = this._meshesVisible;

        if (showMesh) {
            // Mesh mode: show per-neuron mesh, clip faces by ROI if clipped
            const fullGeom = this.scene.neuronFullGeom.get(bid);
            if (fullGeom) fullGeom.visible = false;

            const meshGeom = this.scene.neuronMeshGeom.get(bid);
            if (meshGeom) {
                meshGeom.visible = isHl;
                if (isHl && clipped) {
                    this.scene.clipMeshToRois(bid, this.roiChecked);
                } else if (isHl) {
                    this.scene.clipMeshToRois(bid, null);
                }
            }

            const soma = this.scene.somaGeom.get(bid);
            if (soma) this._showSoma(soma, false);
        } else {
            // Skeleton mode: full skeleton when not clipped, clipped segments when clipped
            const fullGeom = this.scene.neuronFullGeom.get(bid);
            if (fullGeom) fullGeom.visible = isHl && !clipped;

            // Clipped segments (shared per-type geometry, colored per-instance via shader)
            if (isHl && clipped) {
                for (const { geom, roi } of this.scene.getTypeGeometries(typeName)) {
                    geom.visible = this.roiChecked[roi] === true;
                }
            }

            const soma = this.scene.somaGeom.get(bid);
            if (soma) this._showSoma(soma, isHl && !clipped);

            const meshGeom = this.scene.neuronMeshGeom.get(bid);
            if (meshGeom) meshGeom.visible = false;
        }
    }

    setRoiChecked(roiName, checked) {
        this.roiChecked[roiName] = checked;
        // Update visibility for all highlighted+clipped entries
        for (const t of this.highlightedSet) {
            if (this.clipToRoi[t] === true) {
                this._applyKeyVisibility(t);
            }
        }
        // Update ROI mesh visibility
        this._applyRoiMesh(roiName);
    }

    toggleWireframes(show) {
        this.showWireframes = show;
        for (const [roi, mesh] of this.scene.roiMeshGeom) {
            mesh.visible = show && this.roiChecked[roi] === true;
        }
    }

    _applyRoiMesh(roiName) {
        const mesh = this.scene.roiMeshGeom.get(roiName);
        if (mesh) {
            mesh.visible = this.showWireframes && this.roiChecked[roiName] === true;
        }
    }

    _applyAllVisibility() {
        // Check if we're in neuron mode (highlightedSet contains bodyIds)
        const hasBodyIds = [...this.highlightedSet].some(k => this.data.neuronType[k]);
        const hasColorFilter = this.colorFilteredOutTypes.size > 0 || this.colorFilteredOutNeurons.size > 0;

        if (hasBodyIds) {
            // Neuron mode: apply per-neuron, then also hide all type-level geom first
            for (const t of this.data.allTypes) this._hideType(t);
            for (const bid of this.highlightedSet) {
                this._applyNeuronVisibility(bid);
            }
        } else if (this.highlightedSet.size === 0 && hasColorFilter) {
            // Nothing highlighted but color filter active:
            // Show all types/neurons except filtered-out ones
            // Per-neuron filtering is handled inside applyTypeVisibility
            for (const t of this.data.allTypes) {
                this.applyTypeVisibility(t);
            }
        } else {
            // Type mode (normal)
            for (const t of this.data.allTypes) {
                this.applyTypeVisibility(t);
            }
        }
        // Update ROI meshes
        for (const roi of this.data.sidebarRois) {
            this._applyRoiMesh(roi);
        }
    }

    // Set color on a material (works with both ShaderMaterial and standard materials)
    static setColor(obj, color) {
        if (obj.material.uniforms && obj.material.uniforms.diffuse) {
            obj.material.uniforms.diffuse.value.set(color);
        } else if (obj.material.color) {
            obj.material.color.set(color);
        }
    }

    // Set opacity on a material
    static setOpacity(obj, opacity) {
        if (obj.material.uniforms && obj.material.uniforms.opacity) {
            obj.material.uniforms.opacity.value = opacity;
        } else if (obj.material.opacity !== undefined) {
            obj.material.opacity = opacity;
        }
    }

    // Switch color mode
    switchColorMode(modeIdx) {
        this.activeColorMode = modeIdx;
        const mode = this.data.colorModes[modeIdx];

        // Check if this mode has distinct per-neuron colors
        const isPerNeuron = mode.name === 'Instance' || mode.is_custom;

        // Update clipped geometries
        for (const [key, geom] of this.scene.typeRoiGeom) {
            const typeName = key.split('|')[0];

            if (isPerNeuron && geom.userData && geom.userData.bidRuns && geom.userData.bidRuns.length > 0) {
                // Per-instance coloring: assign each segment its neuron's color
                const bidRuns = geom.userData.bidRuns;
                const instanceCount = geom.geometry.instanceCount || 0;
                let colorAttr = geom.geometry.getAttribute('instanceColor');
                if (!colorAttr || colorAttr.count !== instanceCount) {
                    colorAttr = new THREE.InstancedBufferAttribute(
                        new Float32Array(instanceCount * 3), 3);
                    geom.geometry.setAttribute('instanceColor', colorAttr);
                }
                const arr = colorAttr.array;
                const tmpColor = new THREE.Color();
                let segIdx = 0;
                for (const [bid, count] of bidRuns) {
                    const c = mode.colors[String(bid)] || mode.type_colors[typeName] || '#888888';
                    tmpColor.set(c);
                    for (let s = 0; s < count; s++) {
                        const off = segIdx * 3;
                        arr[off] = tmpColor.r;
                        arr[off + 1] = tmpColor.g;
                        arr[off + 2] = tmpColor.b;
                        segIdx++;
                    }
                }
                colorAttr.needsUpdate = true;
                geom.material.uniforms.useInstanceColor.value = true;
            } else {
                // Uniform color per type
                const color = mode.type_colors[typeName];
                if (color) VisibilityManager.setColor(geom, color);
                if (geom.material.uniforms.useInstanceColor) {
                    geom.material.uniforms.useInstanceColor.value = false;
                }
            }
        }

        // Update full geometries
        for (const [bid, geom] of this.scene.neuronFullGeom) {
            const color = mode.colors[bid];
            if (color) VisibilityManager.setColor(geom, color);
        }

        // Update somas
        for (const [bid, mesh] of this.scene.somaGeom) {
            const color = mode.colors[bid];
            if (color) mesh.material.color.set(color);
        }

        // Update neuron meshes
        for (const [bid, mesh] of this.scene.neuronMeshGeom) {
            const color = mode.colors[bid];
            if (color && mesh.material.color) mesh.material.color.set(color);
        }

        // Sync synapse group colors and refresh panel swatches
        if (this.viewer && this.viewer.synapse) {
            this.viewer.synapse.syncColors();
            if (this.viewer.ui) this.viewer.ui._updateSynapsePanel();
        }
    }

    // Apply color filter: compute which types/neurons are filtered out
    applyColorFilter(minPct, maxPct, activeNTs) {
        this.colorFilterMin = minPct;
        this.colorFilterMax = maxPct;

        // Restore items previously removed from highlightedSet by the filter
        if (this._filterRemovedHighlights) {
            for (const k of this._filterRemovedHighlights) {
                this.highlightedSet.add(k);
            }
        }
        this._filterRemovedHighlights = new Set();

        this.colorFilteredOutTypes.clear();
        this.colorFilteredOutNeurons.clear();

        const mode = this.data.colorModes[this.activeColorMode];
        if (!mode) return;

        if (mode.nt_legend && mode.bid_nts && activeNTs) {
            // NT mode: filter by active NTs
            this.activeNTs = activeNTs;
            for (const [bid, nt] of Object.entries(mode.bid_nts)) {
                if (!activeNTs.has(nt)) {
                    this.colorFilteredOutNeurons.add(bid);
                }
            }
            // Also mark types where ALL neurons are filtered out
            for (const typeName of this.data.allTypes) {
                const bids = this.data.getNeuronsForType(typeName);
                const allOut = bids.every(b => this.colorFilteredOutNeurons.has(b));
                if (allOut && bids.length > 0) this.colorFilteredOutTypes.add(typeName);
            }
        } else if (mode._sortedValues && mode.type_values) {
            // Scalar mode: filter by percentile range
            const sv = mode._sortedValues;
            const len = sv.length;
            if (len === 0) return;
            const minVal = sv[Math.floor(minPct / 100 * (len - 1))];
            const maxVal = sv[Math.floor(maxPct / 100 * (len - 1))];
            for (const [typeName, val] of Object.entries(mode.type_values)) {
                if (val < minVal || val > maxVal) {
                    this.colorFilteredOutTypes.add(typeName);
                }
            }
            // Mark neurons of filtered-out types
            for (const typeName of this.colorFilteredOutTypes) {
                for (const bid of this.data.getNeuronsForType(typeName)) {
                    this.colorFilteredOutNeurons.add(bid);
                }
            }
        }

        // Remove filtered-out items from highlightedSet (track for restoration)
        for (const t of this.colorFilteredOutTypes) {
            if (this.highlightedSet.has(t)) {
                this.highlightedSet.delete(t);
                this._filterRemovedHighlights.add(t);
            }
        }
        for (const bid of this.colorFilteredOutNeurons) {
            if (this.highlightedSet.has(bid)) {
                this.highlightedSet.delete(bid);
                this._filterRemovedHighlights.add(bid);
            }
        }

        this._applyAllVisibility();
    }

    // Reset color filter (clear all filtered-out sets)
    resetColorFilter() {
        this.colorFilterMin = 0;
        this.colorFilterMax = 100;
        this.activeNTs = null;
        // Restore items removed from highlightedSet by filter
        if (this._filterRemovedHighlights) {
            for (const k of this._filterRemovedHighlights) {
                this.highlightedSet.add(k);
            }
            this._filterRemovedHighlights = null;
        }
        this.colorFilteredOutTypes.clear();
        this.colorFilteredOutNeurons.clear();
        this._applyAllVisibility();
    }

    hasHighlight() {
        return this.highlightedSet.size > 0;
    }
}

// ---- Interaction Manager ----
class InteractionManager {
    constructor(viewer) {
        this.viewer = viewer;
        this.scene = viewer.scene;
        this.data = viewer.data;
        this.vis = viewer.vis;

        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Line.threshold = 0.002;
        this.mouse = new THREE.Vector2();

        // State
        this.hoveredType = null;
        this.hoveredBid = null;
        this.hoveredRoi = null;
        this._hoverClearTimer = null;
        this._hoverPreviewActive = false;
        this._hoverPreviewType = null;
        this._suppressHoverUntil = 0;
        this._lastMouseButton = 0;
        this._clickTimer = null;
        this._clickKey = null;
        this._rightClickTimer = null;
        this._rightClickRoi = null;
        this._mouseDownPos = null;

        this._setupEvents();
    }

    _setupEvents() {
        const canvas = this.scene.canvas;

        canvas.addEventListener('mousedown', (e) => {
            this._lastMouseButton = e.button;
            this._mouseDownPos = { x: e.clientX, y: e.clientY };
        }, true);

        canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
        canvas.addEventListener('click', (e) => this._onClick(e));
        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this._onRightClick(e);
        });
    }

    _getCanvasCoords(e) {
        const rect = this.scene.canvas.getBoundingClientRect();
        return {
            x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
            y: -((e.clientY - rect.top) / rect.height) * 2 + 1
        };
    }

    _gpuPick(e) {
        // Convert mouse to canvas-local coordinates
        const rect = this.viewer.scene.canvas.getBoundingClientRect();
        const canvasX = e.clientX - rect.left;
        const canvasY = e.clientY - rect.top;

        const result = this.viewer.scene.gpuPick(canvasX, canvasY);
        if (!result) return null;

        return {
            type: result.type,
            bodyId: result.bodyId,
            kind: result.kind,
            point: null,  // No 3D point from GPU pick
        };
    }

    _raycast(e) {
        const coords = this._getCanvasCoords(e);
        this.mouse.set(coords.x, coords.y);
        this.raycaster.setFromCamera(this.mouse, this.scene.camera);

        // Collect raycast targets
        const targets = [];
        const meshMode = this.viewer.vis && this.viewer.vis._meshesVisible;

        if (meshMode) {
            // Mesh mode: use GPU picking (constant-time, no per-triangle raycast)
            return this._gpuPick(e);
        } else {
            // Skeleton mode: use _rayTarget (LineSegments) for thick-line meshes
            const groups = [this.scene.neuronGroup, this.scene.fullGroup];
            for (const group of groups) {
                for (const child of group.children) {
                    if (child._rayTarget) {
                        child._rayTarget.visible = child.visible;
                        if (child.visible) targets.push(child._rayTarget);
                    } else if (child.visible && child.isLineSegments) {
                        targets.push(child);
                    }
                }
            }
        }

        // Raycast — layer 1 for skeleton ray targets, layer 0 for meshes
        this.raycaster.layers.set(meshMode ? 0 : 1);
        const intersects = this.raycaster.intersectObjects(targets, false);
        this.raycaster.layers.set(0);  // Reset

        let neuronHit = null;
        let neuronDist = Infinity;
        if (intersects.length > 0) {
            const hit = intersects[0];
            const ud = hit.object.userData;
            let bodyId = ud.bodyId || null;

            // Resolve bodyId from bidRuns for merged clipped geometry
            if (!bodyId && ud.bidRuns && ud.bidRuns.length > 0 && hit.index != null) {
                const segIdx = Math.floor(hit.index / 2);
                let cumulative = 0;
                for (const [bid, count] of ud.bidRuns) {
                    cumulative += count;
                    if (segIdx < cumulative) {
                        bodyId = bid;
                        break;
                    }
                }
            }

            neuronHit = {
                type: ud.type,
                bodyId,
                kind: ud.kind,
                point: hit.point
            };
            neuronDist = hit.distance;
        }

        // Check synapse meshes (layer 0, standard mesh raycasting)
        const synMeshes = this.viewer.synapse ? this.viewer.synapse.getVisibleMeshes() : [];
        if (synMeshes.length > 0) {
            this.raycaster.layers.set(0);
            const synHits = this.raycaster.intersectObjects(synMeshes, false);
            if (synHits.length > 0 && synHits[0].distance < neuronDist) {
                const synInfo = this.viewer.synapse.resolveHit(synHits[0]);
                if (synInfo) return synInfo;
            }
        }

        return neuronHit;
    }

    // Raycast against visible ROI meshes (fallback when no neuron lines hit)
    _raycastRoi(e) {
        const coords = this._getCanvasCoords(e);
        this.mouse.set(coords.x, coords.y);
        this.raycaster.setFromCamera(this.mouse, this.scene.camera);
        this.raycaster.layers.set(0);
        const targets = [];
        for (const child of this.scene.roiGroup.children) {
            if (child.visible) targets.push(child);
        }
        const intersects = this.raycaster.intersectObjects(targets, false);
        if (intersects.length > 0) {
            return { roi: intersects[0].object.userData.roi };
        }
        return null;
    }

    _detectROI(point) {
        if (!point) return null;
        // roiBounds are pre-normalized — compare directly with normalized point
        const bounds = this.data.roiBounds;
        const px = point.x, py = point.y, pz = point.z;

        let candidates = [];
        for (const [roi, b] of Object.entries(bounds)) {
            if (px >= b.xmin && px <= b.xmax &&
                py >= b.ymin && py <= b.ymax &&
                pz >= b.zmin && pz <= b.zmax) {
                const vol = (b.xmax-b.xmin) * (b.ymax-b.ymin) * (b.zmax-b.zmin);
                candidates.push({ roi, vol });
            }
        }

        if (candidates.length === 0) return null;
        // Return smallest bounding box (most specific ROI)
        candidates.sort((a, b) => a.vol - b.vol);
        return candidates[0].roi;
    }

    _onMouseMove(e) {
        if (Date.now() < this._suppressHoverUntil) return;

        // Use GPU picking when magnifier is active for precise 2D hover
        let hit;
        if (this.viewer.ui && this.viewer.ui._magnifierEnabled) {
            hit = this._gpuPick(e);
        } else {
            hit = this._raycast(e);
        }

        if (hit && hit.isSynapse) {
            // Synapse hit — show synapse info
            clearTimeout(this._hoverClearTimer);
            this.hoveredType = null;
            this.hoveredBid = null;
            this.hoveredRoi = hit.roi || null;
            if (this.viewer.ui) {
                this.viewer.ui.updateInfoBox(null, null, hit.roi, hit);
            }
            return;
        }

        if (hit) {
            this.hoveredType = hit.type;
            this.hoveredRoi = this._detectROI(hit.point);
            this.hoveredBid = hit.bodyId;

            // Update info box
            if (this.viewer.ui) {
                this.viewer.ui.updateInfoBox(hit.type, hit.bodyId, this.hoveredRoi, null);
            }

            // Hover preview: show full skeleton for clipped types/neurons
            if (this.viewer.hoverPreviewEnabled) {
                clearTimeout(this._hoverClearTimer);
                const previewKey = this._hoverPreviewKey(hit);
                // In neuron mode use bodyId as key for per-neuron preview
                const effectiveKey = hit.bodyId && this.vis.highlightedSet.has(hit.bodyId) ? hit.bodyId : previewKey;
                if (effectiveKey && (!this._hoverPreviewActive || this._hoverPreviewType !== effectiveKey)) {
                    this._clearHoverPreview();
                    this._showHoverPreview(hit.type, hit.bodyId);
                }
            }
        } else {
            // No neuron hit — try ROI mesh fallback
            const roiHit = this._raycastRoi(e);
            if (roiHit) {
                clearTimeout(this._hoverClearTimer);
                this._clearHoverPreview();
                this.hoveredType = null;
                this.hoveredBid = null;
                this.hoveredRoi = roiHit.roi;
                if (this.viewer.ui) {
                    this.viewer.ui.updateInfoBox(null, null, roiHit.roi);
                }
            } else if (this.hoveredType || this.hoveredRoi) {
                // Nothing hit at all — clear after delay
                this._hoverClearTimer = setTimeout(() => {
                    this._clearHoverPreview();
                    this.hoveredType = null;
                    this.hoveredBid = null;
                    this.hoveredRoi = null;
                    if (this.viewer.ui) {
                        this.viewer.ui.clearInfoBox();
                    }
                }, HOVER_CLEAR_MS);
            }
        }
    }

    // Determine the preview key for a raycast hit (type name or bodyId)
    _hoverPreviewKey(hit) {
        // Neuron mode: bodyId resolved from bidRuns or full skeleton hit
        if (hit.bodyId && this.vis.highlightedSet.has(hit.bodyId)) {
            return this.vis.clipToRoi[hit.bodyId] ? hit.bodyId : null;
        }
        // Type mode: highlightedSet contains type names
        if (this.vis.highlightedSet.has(hit.type)) {
            return this.vis.clipToRoi[hit.type] ? hit.type : null;
        }
        return null;
    }

    _showMeshFullPreview(mesh) {
        // Temporarily restore full (unclipped) mesh faces for hover preview
        if (!mesh || !mesh._fullIndex) return;
        const geom = mesh.geometry;
        const currentIndex = geom.index;
        // Save current clipped index if different from full
        if (currentIndex && currentIndex.array.length !== mesh._fullIndex.length) {
            mesh.userData._savedIndex = currentIndex;
            geom.setIndex(new THREE.BufferAttribute(new Uint32Array(mesh._fullIndex), 1));
            mesh.userData._hoverFullRestore = true;
        }
        // Also ensure visible + full opacity
        if (!mesh.visible) {
            mesh.visible = true;
            mesh.userData._hoverShown = true;
        }
        mesh.material.opacity = 1.0;
        mesh.material.transparent = false;
    }

    _clearMeshFullPreview(mesh) {
        // Restore clipped index buffer after hover preview
        if (!mesh) return;
        if (mesh.userData._hoverFullRestore && mesh.userData._savedIndex) {
            mesh.geometry.setIndex(mesh.userData._savedIndex);
            delete mesh.userData._savedIndex;
            delete mesh.userData._hoverFullRestore;
        }
    }

    _showHoverPreview(typeName, bodyId) {
        // Only preview highlighted + clipped entities
        const key = this._hoverPreviewKey({ type: typeName, bodyId });
        if (!key) return;

        this._hoverPreviewActive = true;
        this._hoverPreviewType = key;
        const showMesh = this.vis._meshesVisible;

        // Neuron mode: show full geometry + soma for the specific neuron
        if (bodyId && this.vis.highlightedSet.has(bodyId)) {
            if (showMesh) {
                const mesh = this.scene.neuronMeshGeom.get(bodyId);
                if (mesh) this._showMeshFullPreview(mesh);
            } else {
                const geom = this.scene.neuronFullGeom.get(bodyId);
                if (geom && !geom.visible) {
                    geom.visible = true;
                    VisibilityManager.setOpacity(geom, 1.0);
                    geom.userData._hoverShown = true;
                }
            }
            const soma = this.scene.somaGeom.get(bodyId);
            if (soma && this.vis._somataVisible && soma.material.opacity < 0.5) {
                soma.visible = true;
                soma.material.opacity = 1.0;
                soma.userData._hoverShown = true;
            }
        } else if (this.vis.highlightedSet.has(typeName)) {
            // Type mode: show full geometries + somas for all neurons of this type
            const bids = this.data.getNeuronsForType(typeName);
            if (showMesh) {
                for (const bid of bids) {
                    const mesh = this.scene.neuronMeshGeom.get(bid);
                    if (mesh) this._showMeshFullPreview(mesh);
                }
            } else {
                for (const { geom } of this.scene.getTypeFullGeometries(typeName)) {
                    if (!geom.visible) {
                        geom.visible = true;
                        VisibilityManager.setOpacity(geom, 1.0);
                        geom.userData._hoverShown = true;
                    }
                }
            }
            for (const bid of bids) {
                const soma = this.scene.somaGeom.get(bid);
                if (soma && this.vis._somataVisible && soma.material.opacity < 0.5) {
                    soma.visible = true;
                    soma.material.opacity = 1.0;
                    soma.userData._hoverShown = true;
                }
            }
        }
    }

    _clearHoverPreview() {
        if (!this._hoverPreviewActive) return;
        // Hide all full skeleton geometries we temporarily showed
        for (const child of this.scene.fullGroup.children) {
            if (child.userData._hoverShown) {
                child.visible = false;
                VisibilityManager.setOpacity(child, 1.0);
                delete child.userData._hoverShown;
            }
        }
        // Restore mesh geometries — revert full-index preview and hide if needed
        if (this.scene.meshGroup) {
            for (const child of this.scene.meshGroup.children) {
                this._clearMeshFullPreview(child);
                if (child.userData._hoverShown) {
                    child.visible = false;
                    delete child.userData._hoverShown;
                }
            }
        }
        // Restore somas we temporarily showed
        for (const [bid, soma] of this.scene.somaGeom) {
            if (soma.userData._hoverShown) {
                soma.material.opacity = 0.02;
                delete soma.userData._hoverShown;
            }
        }
        this._hoverPreviewActive = false;
        this._hoverPreviewType = null;
    }

    _onClick(e) {
        if (this._lastMouseButton !== 0) return;  // Left-click only
        // Suppress click if mouse was dragged (orbit/pan)
        if (this._mouseDownPos) {
            const dx = e.clientX - this._mouseDownPos.x;
            const dy = e.clientY - this._mouseDownPos.y;
            if (dx * dx + dy * dy > 25) return;
        }

        const hit = this._raycast(e);
        if (!hit) return;

        // Use bodyId when it's in highlightedSet (neuron mode), else type name
        const key = (hit.bodyId && this.vis.highlightedSet.has(hit.bodyId))
            ? hit.bodyId : hit.type;

        // Double-click detection (400ms)
        if (this._clickTimer && this._clickKey === key) {
            clearTimeout(this._clickTimer);
            this._clickTimer = null;
            this._onDoubleClick(key);
            return;
        }

        this._clickKey = key;
        this._clickTimer = setTimeout(() => {
            this._clickTimer = null;
            // Single click: select for connectivity
            if (this.vis.highlightedSet.has(key)) {
                if (this.viewer.hoverPreviewEnabled) {
                    if (this.viewer.connSelectedKey === key) {
                        // Second click on same key: re-clip it
                        this._clearHoverPreview();
                        this.vis.clipToRoi[key] = true;
                        this.vis._applyKeyVisibility(key);
                        if (this.viewer.ui) this.viewer.ui.syncAllState();
                    } else if (this.vis.clipToRoi[key] === true) {
                        // First click on clipped key: unclip it
                        this._clearHoverPreview();
                        this.vis.clipToRoi[key] = false;
                        this.vis._applyKeyVisibility(key);
                        if (this.viewer.ui) this.viewer.ui.syncAllState();
                    }
                }
                this.viewer.selectForConnectivity(key);
            }
        }, 400);
    }

    _onDoubleClick(key) {
        this._suppressHoverUntil = Date.now() + 800;
        this._clearHoverPreview();

        if (this.vis.highlightedSet.has(key)) {
            if (this.vis.clipToRoi[key] === true) {
                // Highlighted + clipped -> unclip
                this.vis.clipToRoi[key] = false;
                this.vis._applyKeyVisibility(key);
            } else {
                // Highlighted + unclipped -> unhighlight
                this.vis.toggleHighlight(key);
            }
        } else {
            // Not highlighted -> highlight
            this.vis.toggleHighlight(key);
        }

        if (this.viewer.ui) this.viewer.ui.syncAllState();
    }

    _onRightClick(e) {
        // Suppress right-click if mouse was dragged
        if (this._mouseDownPos) {
            const dx = e.clientX - this._mouseDownPos.x;
            const dy = e.clientY - this._mouseDownPos.y;
            if (dx * dx + dy * dy > 25) return;
        }
        if (!this.hoveredRoi) return;

        const roi = this.hoveredRoi;

        // Double right-click detection (500ms)
        if (this._rightClickTimer && this._rightClickRoi === roi) {
            clearTimeout(this._rightClickTimer);
            this._rightClickTimer = null;
            // Toggle ROI
            this._suppressHoverUntil = Date.now() + 800;
            this.vis.setRoiChecked(roi, !this.vis.roiChecked[roi]);
            if (this.viewer.ui) this.viewer.ui.syncAllState();
            return;
        }

        this._rightClickRoi = roi;
        this._rightClickTimer = setTimeout(() => {
            this._rightClickTimer = null;
        }, 500);
    }
}

// ---- UI Manager ----
class UIManager {
    constructor(viewer) {
        this.viewer = viewer;
        this.data = viewer.data;
        this.vis = viewer.vis;

        // DOM references
        this.sidebar = null;
        this.typePanel = null;
        this.topBar = null;
        this.infoBox = null;
        this.connPanel = null;
        this.progressBar = null;
        this.progressFill = null;

        // State
        this.hlModeByNeuron = false;
        this.connSelectedKey = null;
        this.connSelectedRoi = null;
        this.roiLabels = {};
        this.typeRows = {};
        this.typeCbs = {};
        this.clipCbs = {};

        // Saved Views bank (max 14, two rows of 7)
        this._savedViews    = [];
        this._activeViewIdx = null;
        this._viewBtns      = [];
        this._viewRow1      = null;   // buttons 1-7
        this._viewRow2      = null;   // buttons 8-14 (hidden until 8th view)
        this._addViewBtn    = null;

        // Saved Sets bank (max 10, two rows of 5)
        this._savedSets     = [];
        this._activeSetIdx  = null;
        this._setsBtns      = [];
        this._restoringSet  = false;
        this._setsRow1      = null;
        this._setsRow2      = null;
        this._addSetBtn     = null;

        // ROI Saved Sets bank (max 10, two rows of 5)
        this._roiSavedSets    = [];
        this._roiActiveSetIdx = null;
        this._roiSetsBtns     = [];
        this._restoringRoiSet = false;
        this._roiSetsRow1     = null;
        this._roiSetsRow2     = null;
        this._roiAddSetBtn    = null;

        this._build();
    }

    _build() {
        this._buildProgressBar();
        this._buildTopBar();
        this._buildSidebar();
        this._buildTypePanel();
        this._buildInfoBox();
        this._buildColorbar();
        this._buildColorFilterPanel();
        this._modeFilterState = {};  // Per-mode sticky filter memory
        this._lastColorModeIdx = 0;  // Track outgoing mode for sticky save
        this._buildConnPanel();
        this._buildSynapsePanel();
        this._buildGizmo();
        this._buildInstructionsButton();
        // Initialize colorbar for default color mode
        this._updateColorbar(this.data.colorModes[0]);
    }

    _buildProgressBar() {
        const bar = document.createElement('div');
        bar.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:3px;background:#000;z-index:9999;';
        const fill = document.createElement('div');
        fill.style.cssText = 'width:0%;height:100%;background:rgb(212,160,23);transition:width 0.4s ease;';
        bar.appendChild(fill);
        document.body.appendChild(bar);
        this.progressBar = bar;
        this.progressFill = fill;
    }

    showProgress() {
        this.progressFill.style.transition = 'width 0.4s ease';
        this.progressFill.style.width = '85%';
    }

    hideProgress() {
        this.progressFill.style.transition = 'width 0.15s ease';
        this.progressFill.style.width = '100%';
        setTimeout(() => {
            this.progressFill.style.transition = 'none';
            this.progressFill.style.width = '0%';
        }, 200);
    }

    _buildTopBar() {
        const bar = document.createElement('div');
        bar.style.cssText = `position:fixed;top:0;left:${SIDEBAR_W}px;right:${TYPE_PANEL_W}px;height:${TOP_BAR_H}px;background:rgba(20,20,20,0.95);display:flex;align-items:center;padding:0 8px;gap:10px;z-index:100;border-bottom:1px solid #333;user-select:none;overflow:hidden;`;
        this._topBarEl = bar;

        // Color-by scrollable section
        const colorSection = document.createElement('div');
        this._colorSection = colorSection;
        colorSection.className = 'color-scroll';
        colorSection.style.cssText = 'display:flex;align-items:center;gap:6px;overflow-x:auto;flex:1;min-width:0;padding-right:8px;scrollbar-width:none;';
        // Convert vertical mouse wheel to horizontal scroll on the color button bar
        colorSection.addEventListener('wheel', (e) => {
            if (colorSection.scrollWidth > colorSection.clientWidth) {
                e.preventDefault();
                colorSection.scrollLeft += e.deltaY;
            }
        }, { passive: false });
        // Hide scrollbar in webkit browsers (Chrome/Safari)
        const scrollStyle = document.createElement('style');
        scrollStyle.textContent = '.color-scroll::-webkit-scrollbar{display:none;}.right-scroll::-webkit-scrollbar{display:none;}';
        document.head.appendChild(scrollStyle);

        const colorLabel = document.createElement('span');
        colorLabel.textContent = 'Color by:';
        colorLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#fff;flex-shrink:0;';
        colorSection.appendChild(colorLabel);

        // Shared style for all square icon buttons — fixed 32x32, centered content
        const _iconBtnStyle = 'width:32px;height:32px;border:1px solid #555;border-radius:3px;cursor:pointer;background:#222;color:#fff;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;padding:0;font-size:15px;box-sizing:border-box;';

        // Color mode buttons — use name-based lookup so indices stay correct after dynamic insertion
        this._instanceLevelBtns = [];
        this.data.colorModes.forEach((mode, idx) => {
            this._addColorModeButton(mode, idx, colorSection, _iconBtnStyle);
        });

        // Shuffle button — randomizes color assignments for categorical modes
        const shuffleBtn = document.createElement('button');
        shuffleBtn.textContent = '\u{1F500}';
        shuffleBtn.dataset.tip = 'Shuffle colors';
        shuffleBtn.style.cssText = _iconBtnStyle;
        shuffleBtn.onclick = () => this._shuffleColors(shuffleBtn);
        colorSection.appendChild(shuffleBtn);
        this._shuffleBtn = shuffleBtn;

        // CSV upload button — add new color mode from file
        const uploadBtn = document.createElement('button');
        uploadBtn.textContent = '+';
        uploadBtn.dataset.tip = 'Upload color CSV';
        uploadBtn.style.cssText = _iconBtnStyle + 'font-size:18px;font-weight:bold;';
        uploadBtn.onclick = () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.csv';
            input.onchange = (ev) => {
                const file = ev.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (re) => this._handleColorCSVUpload(re.target.result, file.name);
                reader.readAsText(file);
            };
            input.click();
        };
        colorSection.appendChild(uploadBtn);

        bar.appendChild(colorSection);
        // Initial state for Instance button (grayed in type mode)
        setTimeout(() => this._updateInstanceBtnState(), 0);

        // Right-side group: pushed to far right
        const rightGroup = document.createElement('div');
        rightGroup.className = 'right-scroll';
        rightGroup.style.cssText = 'margin-left:auto;display:flex;align-items:center;gap:10px;flex-shrink:1;min-width:0;max-width:60%;overflow-x:auto;scrollbar-width:none;';
        // Convert vertical mouse wheel to horizontal scroll on the right button bar
        rightGroup.addEventListener('wheel', (e) => {
            if (rightGroup.scrollWidth > rightGroup.clientWidth) {
                e.preventDefault();
                rightGroup.scrollLeft += e.deltaY;
            }
        }, { passive: false });

        // Z-section divider and slider
        const zDivider = document.createElement('div');
        zDivider.style.cssText = 'width:1px;height:24px;background:#555;flex-shrink:0;';
        rightGroup.appendChild(zDivider);

        const zLabel = document.createElement('span');
        zLabel.textContent = 'Z-section:';
        zLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#fff;white-space:nowrap;';
        rightGroup.appendChild(zLabel);

        const zSlider = document.createElement('input');
        zSlider.type = 'range';
        zSlider.min = '0';
        zSlider.max = '100';
        zSlider.step = '1';
        zSlider.value = '0';
        zSlider.style.cssText = 'width:100px;';
        this._zSlider = zSlider;
        rightGroup.appendChild(zSlider);

        const zVal = document.createElement('span');
        zVal.textContent = 'Off';
        zVal.style.cssText = 'font-size:12px;color:#ccc;width:30px;';
        this._zVal = zVal;
        rightGroup.appendChild(zVal);

        const sceneRef = this.viewer.scene;
        zSlider.oninput = () => {
            const v = parseInt(zSlider.value);
            if (v === 0) {
                zVal.textContent = 'Off';
                sceneRef.clipEnabled = false;
                sceneRef.clipPlane.constant = 1e10;
                _globalClipZ.value = 1e10;  // disable for ShaderMaterials
            } else {
                zVal.textContent = v + '%';
                sceneRef.clipEnabled = true;
                sceneRef.clipFraction = v / 100;
            }
            // Sync camera panel z-section input
            if (this.cameraInputs && this.cameraInputs.zsec) {
                this.cameraInputs.zsec.value = v;
            }
            // Update gizmo clip-plane indicator
            this._updateGizmo();
        };

        // Pan mode toggle
        const panBtn = document.createElement('button');
        panBtn.textContent = '\u2725';
        panBtn.dataset.tip = 'Pan mode';
        panBtn.style.cssText = _iconBtnStyle;
        let panMode = false;
        panBtn.onclick = () => {
            panMode = !panMode;
            const c = this.viewer.scene.controls;
            if (panMode) {
                c.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
                panBtn.style.background = 'rgb(212,160,23)';
                panBtn.style.color = '#000';
            } else {
                c.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
                panBtn.style.background = '#222';
                panBtn.style.color = '#fff';
            }
        };
        rightGroup.appendChild(panBtn);

        // Magnifier toggle button
        const magBtn = document.createElement('button');
        magBtn.textContent = '\uD83D\uDD0D';
        magBtn.dataset.tip = 'Magnifier';
        magBtn.style.cssText = _iconBtnStyle;
        this._magnifierEnabled = false;
        magBtn.onclick = () => {
            this._magnifierEnabled = !this._magnifierEnabled;
            magBtn.style.borderColor = this._magnifierEnabled ? 'rgb(212,160,23)' : '#555';
            magBtn.style.background = this._magnifierEnabled ? 'rgba(212,160,23,0.2)' : '#222';
            if (this._magOverlay) {
                this._magOverlay.style.display = this._magnifierEnabled ? 'block' : 'none';
            }
        };
        rightGroup.appendChild(magBtn);

        // Camera reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = '\u21BA';
        resetBtn.dataset.tip = 'Reset camera';
        resetBtn.style.cssText = _iconBtnStyle;
        resetBtn.onclick = () => { this.viewer.scene.resetCamera(); };
        rightGroup.appendChild(resetBtn);

        // Session export button
        const exportBtn = document.createElement('button');
        exportBtn.textContent = '\uD83D\uDCBE';
        exportBtn.dataset.tip = 'Save session';
        exportBtn.style.cssText = _iconBtnStyle;
        exportBtn.onclick = () => { if (this.viewer.session) this.viewer.session.exportJSON(); };
        rightGroup.appendChild(exportBtn);

        // Session import button
        const importBtn = document.createElement('button');
        importBtn.textContent = '\uD83D\uDCC2';
        importBtn.dataset.tip = 'Load session';
        importBtn.style.cssText = _iconBtnStyle;
        importBtn.onclick = () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (ev) => {
                const file = ev.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (re) => {
                    if (this.viewer.session) this.viewer.session.importJSON(re.target.result);
                };
                reader.readAsText(file);
            };
            input.click();
        };
        rightGroup.appendChild(importBtn);

        // Screenshot button
        const snapBtn = document.createElement('button');
        snapBtn.textContent = '\uD83D\uDCF7';
        snapBtn.dataset.tip = 'Screenshot';
        snapBtn.style.cssText = _iconBtnStyle;
        snapBtn.onclick = () => {
            const s = this.viewer.scene;
            s.renderer.render(s.scene, s.camera);
            s._captureScaleBar = true;
            s._renderScaleBar();
            s._captureScaleBar = false;
            // Composite 3D canvas + scale bar overlay
            const compCanvas = document.createElement('canvas');
            compCanvas.width = s.canvas.width;
            compCanvas.height = s.canvas.height;
            const compCtx = compCanvas.getContext('2d');
            compCtx.drawImage(s.canvas, 0, 0);
            if (s._scaleBarCanvas) compCtx.drawImage(s._scaleBarCanvas, 0, 0, compCanvas.width, compCanvas.height);
            const p = s.camera.position;
            const t = s.controls.target;
            const d = p.distanceTo(t);
            const term = this.viewer.data.regexTerm || 'neuron_viewer';
            const f = v => v.toFixed(4);
            const zPct = parseInt(this._zSlider.value);
            const cmIdx = this.vis.activeColorMode;
            const curMode = this.viewer.data.colorModes[cmIdx];
            let cfStr = `_cm${cmIdx}`;
            if (curMode && curMode.nt_legend && this._activeNTs) {
                const nts = [...this._activeNTs].sort().join(',');
                cfStr += `_nt${nts}`;
            } else {
                cfStr += `_pmin${this._cbarMinPct || 0}_pmax${this._cbarMaxPct !== undefined ? this._cbarMaxPct : 100}`;
            }
            const fname = `${term}_pos${f(p.x)}_${f(p.y)}_${f(p.z)}_tgt${f(t.x)}_${f(t.y)}_${f(t.z)}_d${f(d)}_z${zPct}${cfStr}.png`;
            compCanvas.toBlob((blob) => {
                _saveFileAs(blob, fname, [
                    { description: 'PNG Image', accept: { 'image/png': ['.png'] } }
                ]);
            }, 'image/png');
        };
        rightGroup.appendChild(snapBtn);

        // Video export button
        const videoBtn = document.createElement('button');
        videoBtn.textContent = '\u{1F3AC}';
        videoBtn.dataset.tip = 'Record rotation video';
        videoBtn.style.cssText = _iconBtnStyle;
        videoBtn.onclick = () => this._showVideoExportDialog();
        rightGroup.appendChild(videoBtn);

        bar.appendChild(rightGroup);
        document.body.appendChild(bar);
        this.topBar = bar;

        // Build magnifier overlay
        this._buildMagnifier();

        // ---- Camera info panel (floating, top-right) ----
        this._buildCameraPanel();
    }

    // ── Color mode button creation (shared by init + dynamic upload) ────
    _addColorModeButton(mode, idx, colorSection, iconStyle) {
        const btn = document.createElement('button');
        btn.textContent = mode.name;
        btn.dataset.tip = `Color by ${mode.name}`;
        btn.dataset.colormodename = mode.name;
        btn.dataset.colormode = idx;
        btn.style.cssText = `height:32px;padding:0 10px;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:12px;flex-shrink:0;white-space:nowrap;box-sizing:border-box;${idx === 0 ? 'background:rgb(212,160,23);color:#000;' : 'background:#222;color:#fff;'}`;
        btn.onclick = () => {
            const mIdx = this.data.colorModes.findIndex(m => m.name === mode.name);
            if (mIdx < 0) return;
            // If Predicted NT auto-switched us to neuron mode, silently restore
            const wasNtAutoSwitched = this._ntAutoSwitchedNeuron && this.hlModeByNeuron;
            if (wasNtAutoSwitched && this._switchMode) {
                this._ntAutoSwitchedNeuron = false;
                this._switchMode(false, true);
            }
            const clipAllWasChecked = this.clipAllCb && this.clipAllCb.checked;
            if (!wasNtAutoSwitched && this.vis._filterRemovedHighlights) {
                for (const k of this.vis._filterRemovedHighlights) {
                    this.vis.highlightedSet.add(k);
                    if (clipAllWasChecked) this.vis.clipToRoi[k] = true;
                }
            }
            this.vis._filterRemovedHighlights = null;
            this.vis.colorFilteredOutTypes.clear();
            this.vis.colorFilteredOutNeurons.clear();
            this.vis.activeNTs = null;
            this._filterUncheckedRois = null;
            this.vis.switchColorMode(mIdx);
            this.vis._applyAllVisibility();
            this._colorSection.querySelectorAll('button[data-colormode]').forEach(b => {
                b.style.background = '#222'; b.style.color = '#fff';
            });
            btn.style.background = 'rgb(212,160,23)'; btn.style.color = '#000';
            this._updateColorbar(this.data.colorModes[mIdx]);
            this._updatePanelSwatches();
            this.syncAllState();
        };
        btn.oncontextmenu = (e) => {
            e.preventDefault();
            const mIdx = this.data.colorModes.findIndex(m => m.name === mode.name);
            if (mIdx >= 0 && !mode.is_custom) {
                // Allow right-click for active mode (colormap swap) OR any uploaded mode (remove option)
                if (this.vis.activeColorMode === mIdx || mode.is_uploaded)
                    this._showColormapMenu(e, mIdx);
            }
        };
        if (mode.name === 'Instance' || mode.is_instance_level || mode.nt_legend) {
            this._instanceLevelBtns.push(btn);
        }
        // Insert before shuffle button (if present) or append
        if (this._shuffleBtn && colorSection.contains(this._shuffleBtn)) {
            colorSection.insertBefore(btn, this._shuffleBtn);
        } else {
            colorSection.appendChild(btn);
        }
        return btn;
    }

    _reindexColorButtons() {
        // Update data-colormode indices on all buttons to match current array positions
        this._colorSection.querySelectorAll('button[data-colormodename]').forEach(btn => {
            const name = btn.dataset.colormodename;
            const idx = this.data.colorModes.findIndex(m => m.name === name);
            if (idx >= 0) btn.dataset.colormode = idx;
        });
    }

    // ── CSV parsing and color mode upload ────────────────────────────
    _parseCSV(text) {
        const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
        if (lines.length < 2) return null;
        const parseLine = (line) => {
            const fields = [];
            let cur = '', inQ = false;
            for (let i = 0; i < line.length; i++) {
                const ch = line[i];
                if (ch === '"') { inQ = !inQ; continue; }
                if (ch === ',' && !inQ) { fields.push(cur.trim()); cur = ''; continue; }
                cur += ch;
            }
            fields.push(cur.trim());
            return fields;
        };
        const headers = parseLine(lines[0]);
        const rows = [];
        for (let i = 1; i < lines.length; i++) {
            const r = parseLine(lines[i]);
            if (r.length >= headers.length) rows.push(r);
        }
        return { headers, rows };
    }

    _interpolateColorscale(stops, t) {
        // stops: [[0, "rgb(r,g,b)"], [0.016, "rgb(...)"], ...] — 64 entries
        t = Math.max(0, Math.min(1, t));
        let lo = 0, hi = stops.length - 1;
        for (let i = 0; i < stops.length - 1; i++) {
            if (t >= stops[i][0] && t <= stops[i+1][0]) { lo = i; hi = i + 1; break; }
        }
        const frac = stops[hi][0] === stops[lo][0] ? 0 : (t - stops[lo][0]) / (stops[hi][0] - stops[lo][0]);
        const parse = (s) => s.match(/\d+/g).map(Number);
        const cLo = parse(stops[lo][1]), cHi = parse(stops[hi][1]);
        const r = Math.round(cLo[0] + (cHi[0] - cLo[0]) * frac);
        const g = Math.round(cLo[1] + (cHi[1] - cLo[1]) * frac);
        const b = Math.round(cLo[2] + (cHi[2] - cLo[2]) * frac);
        return `rgb(${r},${g},${b})`;
    }

    _buildColorscale(divergent) {
        // Generate 64-stop colorscale. Divergent: blue→white→red. Sequential: white→orange.
        const stops = [];
        for (let i = 0; i < 64; i++) {
            const t = i / 63;
            let r, g, b;
            if (divergent) {
                // RdBu-like: blue(0) → white(0.5) → red(1)
                if (t < 0.5) {
                    const f = t / 0.5;
                    r = Math.round(33 + (255 - 33) * f);
                    g = Math.round(102 + (255 - 102) * f);
                    b = Math.round(172 + (255 - 172) * f);
                } else {
                    const f = (t - 0.5) / 0.5;
                    r = Math.round(255);
                    g = Math.round(255 - (255 - 44) * f);
                    b = Math.round(255 - (255 - 37) * f);
                }
            } else {
                // Oranges-like: white(0) → orange(1)
                r = Math.round(255);
                g = Math.round(245 - (245 - 69) * t);
                b = Math.round(235 - (235 - 0) * t);
            }
            stops.push([Math.round(t * 1000) / 1000, `rgb(${r},${g},${b})`]);
        }
        return stops;
    }

    _createUploadedColorMode(name, keyCol, valueMap, isInstanceLevel, customColorMap) {
        // keyCol: 'type' or 'bodyid' — determines how to map colors to neurons
        // valueMap: {key: rawValue} from CSV
        // customColorMap: optional {key: cssColor} from a paired color column
        const allBids = Object.keys(this.data.bidTypeMap);
        const values = Object.values(valueMap);
        const isNumeric = values.every(v => v !== '' && isFinite(Number(v)));

        const mode = {
            name: name,
            colors: {},
            type_colors: {},
            is_scalar: false,
            is_instance_level: isInstanceLevel,
            is_uploaded: true,
            label: name,
        };

        if (isNumeric) {
            // ── Continuous / scalar mode ──
            const numMap = {};
            for (const [k, v] of Object.entries(valueMap)) numMap[k] = Number(v);
            const vals = Object.values(numMap);
            const cmin = Math.min(...vals);
            const cmax = Math.max(...vals);
            const divergent = cmin < 0 && cmax > 0;
            const range = cmax - cmin || 1;

            // Build colorscale: use custom colors if provided, else auto-generate
            let colorscale;
            if (customColorMap) {
                // Build colorscale from value→color pairs sorted by numeric value
                const colorPairs = [];
                for (const [key, val] of Object.entries(numMap)) {
                    if (customColorMap[key]) colorPairs.push([val, customColorMap[key]]);
                }
                colorPairs.sort((a, b) => a[0] - b[0]);
                if (colorPairs.length >= 2) {
                    colorscale = colorPairs.map(([v, c]) => [(v - cmin) / range, c]);
                    // Ensure we have stops at 0 and 1
                    if (colorscale[0][0] > 0) colorscale.unshift([0, colorscale[0][1]]);
                    if (colorscale[colorscale.length - 1][0] < 1) colorscale.push([1, colorscale[colorscale.length - 1][1]]);
                } else {
                    colorscale = this._buildColorscale(divergent);
                }
            } else {
                colorscale = this._buildColorscale(divergent);
            }

            mode.is_scalar = true;
            mode.cmin = cmin;
            mode.cmax = cmax;
            mode.colorscale = colorscale;

            if (isInstanceLevel) {
                // keyed by bodyId
                const typeFirstColor = {};
                for (const bid of allBids) {
                    const val = numMap[bid];
                    if (val != null) {
                        // Use direct custom color if available, else interpolate colorscale
                        if (customColorMap && customColorMap[bid]) {
                            mode.colors[bid] = customColorMap[bid];
                        } else {
                            const t = (val - cmin) / range;
                            mode.colors[bid] = this._interpolateColorscale(colorscale, t);
                        }
                    } else {
                        mode.colors[bid] = 'rgb(128,128,128)';
                    }
                    const typ = this.data.bidTypeMap[bid];
                    if (typ && !typeFirstColor[typ]) typeFirstColor[typ] = mode.colors[bid];
                }
                mode.type_colors = typeFirstColor;
            } else {
                // keyed by type
                const typeColors = {};
                for (const [typ, val] of Object.entries(numMap)) {
                    if (customColorMap && customColorMap[typ]) {
                        typeColors[typ] = customColorMap[typ];
                    } else {
                        const t = (val - cmin) / range;
                        typeColors[typ] = this._interpolateColorscale(colorscale, t);
                    }
                }
                for (const bid of allBids) {
                    const typ = this.data.bidTypeMap[bid];
                    mode.colors[bid] = typeColors[typ] || 'rgb(128,128,128)';
                }
                mode.type_colors = typeColors;
            }
            // Pre-compute type_values and _sortedValues for scalar percentile filtering + colorbar
            if (!isInstanceLevel) {
                mode.type_values = {};
                for (const [typ, val] of Object.entries(numMap)) mode.type_values[typ] = val;
            }
            mode._sortedValues = Object.values(numMap).sort((a, b) => a - b);
        } else {
            // ── Categorical mode ──
            mode.is_categorical = true;
            const uniqueVals = [...new Set(values)];
            const catColors = {};
            if (customColorMap) {
                // Build category→color from value→color: map each value to its row's custom color
                const valToColor = {};
                for (const [key, val] of Object.entries(valueMap)) {
                    if (customColorMap[key] && !valToColor[val]) valToColor[val] = customColorMap[key];
                }
                for (const v of uniqueVals) {
                    catColors[v] = valToColor[v] || 'rgb(128,128,128)';
                }
            } else {
                for (let i = 0; i < uniqueVals.length; i++) {
                    const hue = uniqueVals.length <= 1 ? 0 : i / uniqueVals.length;
                    const c = new THREE.Color().setHSL(hue, 0.7, 0.55);
                    catColors[uniqueVals[i]] = `rgb(${Math.round(c.r*255)},${Math.round(c.g*255)},${Math.round(c.b*255)})`;
                }
            }
            if (isInstanceLevel) {
                const typeFirstColor = {};
                for (const bid of allBids) {
                    const cat = valueMap[bid];
                    mode.colors[bid] = cat != null ? (catColors[cat] || 'rgb(128,128,128)') : 'rgb(128,128,128)';
                    const typ = this.data.bidTypeMap[bid];
                    if (typ && !typeFirstColor[typ]) typeFirstColor[typ] = mode.colors[bid];
                }
                mode.type_colors = typeFirstColor;
            } else {
                const typeColors = {};
                for (const [typ, cat] of Object.entries(valueMap)) {
                    typeColors[typ] = catColors[cat] || 'rgb(128,128,128)';
                }
                for (const bid of allBids) {
                    const typ = this.data.bidTypeMap[bid];
                    mode.colors[bid] = typeColors[typ] || 'rgb(128,128,128)';
                }
                mode.type_colors = typeColors;
            }
            // Store category→color mapping for legend display
            mode._catLegend = catColors;
        }
        return mode;
    }

    _handleColorCSVUpload(text, filename) {
        const parsed = this._parseCSV(text);
        if (!parsed || parsed.rows.length === 0) {
            alert('Could not parse CSV or no data rows found.');
            return;
        }
        const firstCol = parsed.headers[0].toLowerCase().trim();
        let isInstanceLevel = false;
        if (firstCol === 'bodyid' || firstCol === 'body_id') {
            isInstanceLevel = true;
        } else if (firstCol !== 'type') {
            alert('First column must be "type" or "bodyid" / "body_id". Found: "' + parsed.headers[0] + '"');
            return;
        }

        const newModes = [];
        for (let col = 1; col < parsed.headers.length; col++) {
            const colName = parsed.headers[col].trim();
            const valueMap = {};
            for (const row of parsed.rows) {
                const key = row[0].trim();
                const val = row[col] != null ? row[col].trim() : '';
                if (key && val !== '') valueMap[key] = val;
            }
            if (Object.keys(valueMap).length === 0) continue;

            // Check if next column is a paired color column (all valid CSS colors)
            let customColorMap = null;
            if (col + 1 < parsed.headers.length) {
                const nextColVals = {};
                let allColors = true;
                for (const row of parsed.rows) {
                    const key = row[0].trim();
                    const cv = row[col + 1] != null ? row[col + 1].trim() : '';
                    if (key && cv !== '') {
                        if (!this._isValidCSSColor(cv)) { allColors = false; break; }
                        nextColVals[key] = cv;
                    }
                }
                if (allColors && Object.keys(nextColVals).length > 0) {
                    customColorMap = nextColVals;
                    col++;  // skip the color column
                }
            }

            // Check for duplicate name — append suffix if needed
            let modeName = colName;
            let suffix = 2;
            while (this.data.colorModes.find(m => m.name === modeName)) {
                modeName = `${colName} (${suffix++})`;
            }

            const mode = this._createUploadedColorMode(modeName, firstCol, valueMap, isInstanceLevel, customColorMap);
            // Insert before Custom (always last)
            const insertIdx = this.data.colorModes.length - 1;
            this.data.colorModes.splice(insertIdx, 0, mode);
            this._addColorModeButton(mode, insertIdx, this._colorSection,
                'width:32px;height:32px;border:1px solid #555;border-radius:3px;cursor:pointer;background:#222;color:#fff;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0;padding:0;font-size:15px;box-sizing:border-box;');
            newModes.push(mode);
        }

        if (newModes.length === 0) {
            alert('No valid data columns found in CSV.');
            return;
        }

        this._reindexColorButtons();

        // Auto-switch to the first new mode
        const firstIdx = this.data.colorModes.findIndex(m => m.name === newModes[0].name);
        if (firstIdx >= 0) {
            this.vis.switchColorMode(firstIdx);
            this.vis._applyAllVisibility();
            this._colorSection.querySelectorAll('button[data-colormode]').forEach(b => {
                b.style.background = '#222'; b.style.color = '#fff';
            });
            const activeBtn = this._colorSection.querySelector(`button[data-colormodename="${newModes[0].name}"]`);
            if (activeBtn) { activeBtn.style.background = 'rgb(212,160,23)'; activeBtn.style.color = '#000'; }
            this._updateColorbar(newModes[0]);
            this._updatePanelSwatches();
            this.syncAllState();
        }
        this._updateInstanceBtnState();

        // Save uploaded CSV text for session persistence
        if (!this._uploadedColorCSVs) this._uploadedColorCSVs = [];
        this._uploadedColorCSVs.push({ filename, text });

        // Trigger session save
        if (this.viewer.session) this.viewer.session.debouncedSave();
    }

    _isValidCSSColor(str) {
        if (!str || typeof str !== 'string') return false;
        if (/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(str)) return true;
        if (/^rgb\(/.test(str)) return true;
        // Check named colors via canvas
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.fillStyle = str;
        return ctx.fillStyle !== '#000000' || str.toLowerCase() === 'black';
    }

    async _handleSynapseCSVUpload(text, filename, isRestore) {
        const synMgr = this.viewer.synapse;
        if (!synMgr) {
            if (!isRestore) alert('Synapse manager not available.');
            return;
        }
        // Auto-load synapse data if not yet loaded
        if (!synMgr.loaded) {
            if (!DATA.synapseData) {
                if (!isRestore) alert('No synapse data is embedded in this visualization. Generate the HTML with synapse data enabled.');
                return;
            }
            const ok = await synMgr.loadData();
            if (!ok || !synMgr.data) {
                if (!isRestore) alert('Failed to load synapse data.');
                return;
            }
        }

        const parsed = this._parseCSV(text);
        if (!parsed || parsed.rows.length === 0) {
            if (!isRestore) alert('Could not parse CSV or no data rows found.');
            return;
        }

        // Validate headers: need bodyid_pre, bodyid_post, and a 3rd column
        const h = parsed.headers.map(s => s.toLowerCase().trim());
        const preCol = h.indexOf('bodyid_pre');
        const postCol = h.indexOf('bodyid_post');
        if (preCol < 0 || postCol < 0 || parsed.headers.length < 3) {
            if (!isRestore) alert('Synapse CSV must have columns: bodyid_pre, bodyid_post, and a 3rd column (color or category).');
            return;
        }
        // Find the 3rd column (first that isn't pre or post) and optional 4th color column
        let valCol = -1, colorCol = -1;
        for (let i = 0; i < parsed.headers.length; i++) {
            if (i !== preCol && i !== postCol) {
                if (valCol < 0) valCol = i;
                else if (colorCol < 0) colorCol = i;
            }
        }
        if (valCol < 0) return;

        // Check if colorCol contains all valid CSS colors (paired category+color format)
        let pairedColorMap = null;
        if (colorCol >= 0) {
            let allColors = true;
            const catToColor = {};
            for (const row of parsed.rows) {
                const cv = row[colorCol] ? row[colorCol].trim() : '';
                const cat = row[valCol] ? row[valCol].trim() : '';
                if (cv && cat) {
                    if (!this._isValidCSSColor(cv)) { allColors = false; break; }
                    if (!catToColor[cat]) catToColor[cat] = cv;
                }
            }
            if (allColors && Object.keys(catToColor).length > 0) {
                pairedColorMap = catToColor;  // {categoryValue: cssColor}
            }
        }

        // Gather data: Map<"preBid|postBid", value>
        const pairValues = new Map();
        for (const row of parsed.rows) {
            const pre = row[preCol].trim();
            const post = row[postCol].trim();
            const val = row[valCol] ? row[valCol].trim() : '';
            if (pre && post && val) pairValues.set(`${pre}|${post}`, val);
        }

        // Auto-detect: are all values valid CSS colors? (3-column format: pre, post, color)
        const allVals = [...pairValues.values()];
        const isDirectColor = !pairedColorMap && allVals.every(v => this._isValidCSSColor(v));

        // Group by color/category
        const groups = new Map(); // value -> [pairKeys]
        for (const [key, val] of pairValues) {
            if (!groups.has(val)) groups.set(val, []);
            groups.get(val).push(key);
        }

        // Assign colors for category mode
        let catColors = null;
        if (pairedColorMap) {
            catColors = pairedColorMap;  // use user-provided colors
        } else if (!isDirectColor) {
            catColors = {};
            const cats = [...groups.keys()];
            for (let i = 0; i < cats.length; i++) {
                const hue = cats.length <= 1 ? 0 : i / cats.length;
                const c = new THREE.Color().setHSL(hue, 0.7, 0.55);
                catColors[cats[i]] = `#${c.getHexString()}`;
            }
        }

        // Create synapse groups
        let created = 0;
        for (const [val, pairKeys] of groups) {
            const allIndices = [];
            for (const pk of pairKeys) {
                const idxArr = synMgr.data.pairIndex.get(pk);
                if (idxArr) allIndices.push(...idxArr);
            }
            if (allIndices.length === 0) continue;

            const color = isDirectColor ? val : catColors[val];
            const label = isDirectColor ? `CSV: ${val}` : `CSV: ${val}`;
            synMgr.createGroupFromIndices(allIndices, {
                label: label,
                color: color,
                synapseType: 'both',
            });
            created++;
        }

        if (created > 0) {
            this._updateSynapsePanel();
            // Make synapse panel visible
            if (this.synPanel) this.synPanel.style.display = 'flex';

            // Store for session persistence
            if (!isRestore) {
                if (!synMgr._uploadedCSVs) synMgr._uploadedCSVs = [];
                synMgr._uploadedCSVs.push({ filename, text });
                if (this.viewer.session) this.viewer.session.debouncedSave();
            }
        } else if (!isRestore) {
            alert('No matching synapse pairs found in the loaded data. Check that the body IDs in your CSV match neurons in this visualization.');
        }
    }

    _buildMagnifier() {
        const MAG_SIZE = 150;
        const SRC_SIZE = 75;  // 2x zoom

        const overlay = document.createElement('canvas');
        overlay.width = MAG_SIZE;
        overlay.height = MAG_SIZE;
        overlay.style.cssText = `position:fixed;z-index:9999;width:${MAG_SIZE}px;height:${MAG_SIZE}px;border-radius:50%;border:2px solid rgb(212,160,23);pointer-events:none;display:none;box-shadow:0 0 8px rgba(0,0,0,0.6);`;
        document.body.appendChild(overlay);
        this._magOverlay = overlay;
        this._magCtx = overlay.getContext('2d');

        let lastMagUpdate = 0;
        const MAG_INTERVAL = 50; // ~20fps

        const canvas = this.viewer.scene.canvas;
        canvas.addEventListener('mousemove', (e) => {
            if (!this._magnifierEnabled) return;
            const now = Date.now();
            if (now - lastMagUpdate < MAG_INTERVAL) return;
            lastMagUpdate = now;

            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const sx = (e.clientX - rect.left) * dpr;
            const sy = (e.clientY - rect.top) * dpr;
            const srcW = SRC_SIZE * dpr;
            const srcH = SRC_SIZE * dpr;

            this._magCtx.clearRect(0, 0, MAG_SIZE, MAG_SIZE);
            try {
                this._magCtx.drawImage(
                    canvas,
                    sx - srcW / 2, sy - srcH / 2, srcW, srcH,
                    0, 0, MAG_SIZE, MAG_SIZE
                );
            } catch(err) { /* cross-origin or empty canvas */ }

            // Draw crosshair at center — white outline for contrast, then bold orange
            const cx = MAG_SIZE / 2, cy = MAG_SIZE / 2, gap = 4, arm = 16;
            const drawCross = () => {
                this._magCtx.beginPath();
                this._magCtx.moveTo(cx - gap - arm, cy); this._magCtx.lineTo(cx - gap, cy);
                this._magCtx.moveTo(cx + gap, cy); this._magCtx.lineTo(cx + gap + arm, cy);
                this._magCtx.moveTo(cx, cy - gap - arm); this._magCtx.lineTo(cx, cy - gap);
                this._magCtx.moveTo(cx, cy + gap); this._magCtx.lineTo(cx, cy + gap + arm);
                this._magCtx.stroke();
            };
            // White outline pass
            this._magCtx.strokeStyle = 'rgba(255,255,255,0.95)';
            this._magCtx.lineWidth = 5;
            drawCross();
            // Black inner pass
            this._magCtx.strokeStyle = 'rgba(0,0,0,0.9)';
            this._magCtx.lineWidth = 2;
            drawCross();

            // Center magnifier on cursor
            overlay.style.left = (e.clientX - MAG_SIZE / 2) + 'px';
            overlay.style.top = (e.clientY - MAG_SIZE / 2) + 'px';
        });

        // Hide during drag
        canvas.addEventListener('mousedown', () => {
            if (this._magnifierEnabled && this._magOverlay) {
                this._magOverlay.style.opacity = '0';
            }
        });
        canvas.addEventListener('mouseup', () => {
            if (this._magnifierEnabled && this._magOverlay) {
                this._magOverlay.style.opacity = '1';
            }
        });
        // Hide during scroll/zoom
        canvas.addEventListener('wheel', () => {
            if (this._magnifierEnabled && this._magOverlay) {
                this._magOverlay.style.opacity = '0';
                clearTimeout(this._magScrollTimer);
                this._magScrollTimer = setTimeout(() => {
                    // Wait two frames for the renderer to catch up at new zoom
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        if (this._magnifierEnabled && this._magOverlay) {
                            this._magOverlay.style.opacity = '1';
                        }
                    }));
                }, 150);
            }
        });
    }

    _buildCameraPanel() {
        const panel = document.createElement('div');
        panel.style.cssText = `position:fixed;top:${TOP_BAR_H + PANEL_PAD}px;right:${TYPE_PANEL_W + PANEL_PAD}px;z-index:101;background:rgba(20,20,20,0.92);border:1px solid #444;border-radius:6px;padding:0;font-size:11px;font-family:monospace;color:#ccc;min-width:260px;`;

        // Header with collapse toggle
        const hdr = document.createElement('div');
        hdr.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 8px;cursor:pointer;user-select:none;border-bottom:1px solid #333;';
        const hdrLabel = document.createElement('span');
        hdrLabel.textContent = 'Camera';
        hdrLabel.style.cssText = 'font-weight:bold;font-size:11px;color:#fff;';
        const collapseBtn = document.createElement('span');
        collapseBtn.textContent = '\u25BC';
        collapseBtn.style.cssText = 'font-size:9px;color:#888;';
        hdr.appendChild(hdrLabel);
        hdr.appendChild(collapseBtn);
        panel.appendChild(hdr);

        const body = document.createElement('div');
        body.style.cssText = 'padding:6px 8px;user-select:none;';

        const inputStyle = 'width:62px;background:#111;border:1px solid #555;color:#fff;text-align:right;font-family:monospace;font-size:11px;padding:2px 4px;border-radius:2px;';
        const labelStyle = 'color:#888;font-size:10px;width:14px;display:inline-block;text-align:right;margin-right:2px;';
        const rowStyle = 'display:flex;align-items:center;gap:4px;margin-bottom:3px;';

        const makeInput = () => {
            const inp = document.createElement('input');
            inp.type = 'text';
            inp.style.cssText = inputStyle;
            inp.value = '0.0000';
            return inp;
        };

        const makeRow = (labels) => {
            const row = document.createElement('div');
            row.style.cssText = rowStyle;
            const inputs = [];
            for (const lbl of labels) {
                const span = document.createElement('span');
                span.textContent = lbl;
                span.style.cssText = labelStyle;
                row.appendChild(span);
                const inp = makeInput();
                row.appendChild(inp);
                inputs.push(inp);
            }
            return { row, inputs };
        };

        // Section: Position
        const posLabel = document.createElement('div');
        posLabel.textContent = 'Position';
        posLabel.style.cssText = 'font-size:10px;color:#aaa;font-weight:bold;margin-bottom:2px;';
        body.appendChild(posLabel);

        const posRow = makeRow(['X', 'Y', 'Z']);
        body.appendChild(posRow.row);

        // Section: Target
        const tgtLabel = document.createElement('div');
        tgtLabel.textContent = 'Target';
        tgtLabel.style.cssText = 'font-size:10px;color:#aaa;font-weight:bold;margin-top:4px;margin-bottom:2px;';
        body.appendChild(tgtLabel);

        const tgtRow = makeRow(['X', 'Y', 'Z']);
        body.appendChild(tgtRow.row);

        // Section: Distance
        const distLabel = document.createElement('div');
        distLabel.textContent = 'Distance';
        distLabel.style.cssText = 'font-size:10px;color:#aaa;font-weight:bold;margin-top:4px;margin-bottom:2px;';
        body.appendChild(distLabel);

        const distRow = makeRow(['D']);
        body.appendChild(distRow.row);

        // Section: Z-section
        const zSecLabel = document.createElement('div');
        zSecLabel.textContent = 'Z-section';
        zSecLabel.style.cssText = 'font-size:10px;color:#aaa;font-weight:bold;margin-top:4px;margin-bottom:2px;';
        body.appendChild(zSecLabel);

        const zSecRow = makeRow(['%']);
        body.appendChild(zSecRow.row);

        // ── Saved Views ──────────────────────────────────────────────
        const viewsLabel = document.createElement('div');
        viewsLabel.textContent = 'Saved Views';
        viewsLabel.style.cssText = 'font-size:10px;color:#aaa;font-weight:bold;margin-top:10px;margin-bottom:4px;letter-spacing:0.04em;';
        body.appendChild(viewsLabel);

        const viewRow1 = document.createElement('div');
        viewRow1.style.cssText = 'display:flex;gap:5px;margin-bottom:5px;';
        body.appendChild(viewRow1);
        this._viewRow1 = viewRow1;

        const viewRow2 = document.createElement('div');
        viewRow2.style.cssText = 'display:none;flex;gap:5px;margin-bottom:5px;';
        body.appendChild(viewRow2);
        this._viewRow2 = viewRow2;

        this._viewBtns = [];

        const addViewBtn = document.createElement('button');
        addViewBtn.textContent = '+ Add view';
        addViewBtn.style.cssText = 'padding:4px 10px;border:1px solid #555;border-radius:3px;'
            + 'cursor:pointer;font-size:12px;background:#222;color:#fff;width:100%;margin-bottom:6px;user-select:none;';
        addViewBtn.onclick = () => {
            const v = this._captureView();
            this._addSavedView(v);
        };
        body.appendChild(addViewBtn);
        this._addViewBtn = addViewBtn;

        // Parse from PNG input
        const parseLabel = document.createElement('div');
        parseLabel.textContent = 'Parse from PNG';
        parseLabel.style.cssText = 'font-size:10px;color:#aaa;font-weight:bold;margin-top:6px;margin-bottom:2px;';
        body.appendChild(parseLabel);

        const pngInput = document.createElement('input');
        pngInput.type = 'text';
        pngInput.placeholder = 'Paste filename...';
        pngInput.style.cssText = 'width:100%;background:#111;border:1px solid #555;color:#fff;font-family:monospace;font-size:10px;padding:3px 4px;border-radius:2px;box-sizing:border-box;';
        body.appendChild(pngInput);

        // Apply button
        const applyRow = document.createElement('div');
        applyRow.style.cssText = 'margin-top:5px;text-align:center;';
        const applyBtn = document.createElement('button');
        applyBtn.textContent = 'Apply';
        applyBtn.style.cssText = 'padding:3px 16px;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:11px;background:#333;color:#fff;font-family:monospace;';
        applyRow.appendChild(applyBtn);
        body.appendChild(applyRow);

        panel.appendChild(body);
        document.body.appendChild(panel);

        // Store input references
        this.cameraInputs = {
            px: posRow.inputs[0], py: posRow.inputs[1], pz: posRow.inputs[2],
            tx: tgtRow.inputs[0], ty: tgtRow.inputs[1], tz: tgtRow.inputs[2],
            dist: distRow.inputs[0],
            zsec: zSecRow.inputs[0]
        };
        this._cameraPanel = panel;
        this._cameraPanelBody = body;

        // Collapse toggle
        let collapsed = false;
        hdr.onclick = () => {
            collapsed = !collapsed;
            body.style.display = collapsed ? 'none' : 'block';
            collapseBtn.textContent = collapsed ? '\u25B6' : '\u25BC';
        };

        // Try to parse a filename string; check PNG input first, then other fields
        this._parsedColorFilter = null;  // Store parsed color filter for Apply
        const tryParseFilename = () => {
            const re = /pos(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_tgt(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_d(-?\d+\.\d+)(?:_z(\d+))?(?:_cm(\d+))?(?:_pmin(\d+)_pmax(\d+))?(?:_nt([a-z,]+))?/;
            const sources = [pngInput, ...Object.values(this.cameraInputs)];
            for (const inp of sources) {
                const m = inp.value.match(re);
                if (m) {
                    this.cameraInputs.px.value = m[1];
                    this.cameraInputs.py.value = m[2];
                    this.cameraInputs.pz.value = m[3];
                    this.cameraInputs.tx.value = m[4];
                    this.cameraInputs.ty.value = m[5];
                    this.cameraInputs.tz.value = m[6];
                    this.cameraInputs.dist.value = m[7];
                    this.cameraInputs.zsec.value = m[8] || '0';
                    // Parse color filter info
                    this._parsedColorFilter = null;
                    if (m[9] !== undefined) {
                        const cmIdx = parseInt(m[9]);
                        if (m[12]) {
                            // NT filter
                            this._parsedColorFilter = { cmIdx, nts: m[12].split(',') };
                        } else if (m[10] !== undefined && m[11] !== undefined) {
                            // Percentile filter
                            this._parsedColorFilter = { cmIdx, pmin: parseInt(m[10]), pmax: parseInt(m[11]) };
                        } else {
                            this._parsedColorFilter = { cmIdx };
                        }
                    }
                    pngInput.value = '';
                    return true;
                }
            }
            return false;
        };

        // Apply button: parse filename if present, then set camera + z-section + color filter
        applyBtn.onclick = () => {
            tryParseFilename();
            const s = this.viewer.scene;
            const px = parseFloat(this.cameraInputs.px.value);
            const py = parseFloat(this.cameraInputs.py.value);
            const pz = parseFloat(this.cameraInputs.pz.value);
            const tx = parseFloat(this.cameraInputs.tx.value);
            const ty = parseFloat(this.cameraInputs.ty.value);
            const tz = parseFloat(this.cameraInputs.tz.value);
            // Read z-section BEFORE controls.update() which triggers live-update
            const zv = parseInt(this.cameraInputs.zsec.value) || 0;
            if ([px,py,pz,tx,ty,tz].some(isNaN)) return;
            s.camera.position.set(px, py, pz);
            // PNG filenames don't encode camera.up — reset to canonical initial up
            // so lookAt produces the same orientation as when the screenshot was taken.
            s.camera.up.copy(s._initialCameraUp);
            s.controls.target.set(tx, ty, tz);
            s.controls.update();
            // Apply z-section
            this._zSlider.value = zv;
            this._zSlider.oninput();
            // Apply color filter from parsed filename
            if (this._parsedColorFilter) {
                const cf = this._parsedColorFilter;
                // Switch color mode
                if (cf.cmIdx !== undefined && cf.cmIdx < this.viewer.data.colorModes.length) {
                    // Click the corresponding color mode button
                    const btn = document.querySelector(`button[data-colormode="${cf.cmIdx}"]`);
                    if (btn) btn.click();
                }
                // Apply filter after mode switch
                setTimeout(() => {
                    if (cf.nts) {
                        // NT filter: set active NTs
                        const ntSet = new Set(cf.nts);
                        this._activeNTs = ntSet;
                        // Sync legend and checkboxes
                        for (const [nt, info] of Object.entries(this._ntLegendRows || {})) {
                            const active = ntSet.has(nt);
                            info.row.style.opacity = active ? '1' : '0.35';
                            info.row.style.textDecoration = active ? 'none' : 'line-through';
                            if (this._ntCheckboxes && this._ntCheckboxes[nt]) {
                                this._ntCheckboxes[nt].checked = active;
                            }
                        }
                        this._onNtFilterChange();
                    } else if (cf.pmin !== undefined && cf.pmax !== undefined) {
                        // Percentile filter
                        this._cbarMinPct = cf.pmin;
                        this._cbarMaxPct = cf.pmax;
                        if (this._filterMinInput) this._filterMinInput.value = cf.pmin;
                        if (this._filterMaxInput) this._filterMaxInput.value = cf.pmax;
                        if (this._cbarPercentToPos) {
                            this._cbarMaxHandle.style.top = this._cbarPercentToPos(cf.pmax) + 'px';
                            this._cbarMinHandle.style.top = this._cbarPercentToPos(cf.pmin) + 'px';
                            this._cbarUpdateRangeHighlight();
                        }
                        this._onColorFilterChange();
                    }
                }, 50);
                this._parsedColorFilter = null;
            }
            // Auto-save the parsed state as a new view
            this._addSavedView(this._captureView());
        };

        // Paste-to-parse: detect pasted filename and auto-populate all fields
        const parsePaste = (e) => {
            const text = (e.clipboardData || window.clipboardData).getData('text');
            const m = text.match(/pos(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_tgt(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)_d(-?\d+\.\d+)(?:_z(\d+))?(?:_cm(\d+))?(?:_pmin(\d+)_pmax(\d+))?(?:_nt([a-z,]+))?/);
            if (m) {
                e.preventDefault();
                this.cameraInputs.px.value = m[1];
                this.cameraInputs.py.value = m[2];
                this.cameraInputs.pz.value = m[3];
                this.cameraInputs.tx.value = m[4];
                this.cameraInputs.ty.value = m[5];
                this.cameraInputs.tz.value = m[6];
                this.cameraInputs.dist.value = m[7];
                this.cameraInputs.zsec.value = m[8] || '0';
                // Store color filter for Apply
                this._parsedColorFilter = null;
                if (m[9] !== undefined) {
                    const cmIdx = parseInt(m[9]);
                    if (m[12]) {
                        this._parsedColorFilter = { cmIdx, nts: m[12].split(',') };
                    } else if (m[10] !== undefined && m[11] !== undefined) {
                        this._parsedColorFilter = { cmIdx, pmin: parseInt(m[10]), pmax: parseInt(m[11]) };
                    } else {
                        this._parsedColorFilter = { cmIdx };
                    }
                }
            }
        };
        for (const inp of Object.values(this.cameraInputs)) {
            inp.addEventListener('paste', parsePaste);
        }

        // Live-update camera inputs on controls change
        const updateCameraInputs = () => {
            const s = this.viewer.scene;
            const p = s.camera.position;
            const t = s.controls.target;
            const f = v => v.toFixed(4);
            this.cameraInputs.px.value = f(p.x);
            this.cameraInputs.py.value = f(p.y);
            this.cameraInputs.pz.value = f(p.z);
            this.cameraInputs.tx.value = f(t.x);
            this.cameraInputs.ty.value = f(t.y);
            this.cameraInputs.tz.value = f(t.z);
            this.cameraInputs.dist.value = f(p.distanceTo(t));
            this.cameraInputs.zsec.value = this._zSlider.value;
        };
        this.viewer.scene.controls.addEventListener('change', updateCameraInputs);
        // Deselect active saved view when the user manually moves the camera
        this.viewer.scene.controls.addEventListener('change', () => {
            if (this._restoringView) return;
            if (this._activeViewIdx !== null) {
                this._activeViewIdx = null;
                this._viewBtns.forEach(b => { b.style.background = '#222'; b.style.color = '#fff'; });
            }
        });
        // Set initial values
        updateCameraInputs();
    }

    // ── Saved Views ───────────────────────────────────────────────────────────

    _captureView() {
        const sc = this.viewer.scene;
        return {
            pos: { x: sc.camera.position.x, y: sc.camera.position.y, z: sc.camera.position.z },
            tgt: { x: sc.controls.target.x,  y: sc.controls.target.y,  z: sc.controls.target.z  },
            up:  { x: sc.camera.up.x,        y: sc.camera.up.y,        z: sc.camera.up.z        },
            clipEnabled:  sc.clipEnabled,
            clipFraction: sc.clipFraction,
        };
    }

    _addSavedView(viewObj) {
        if (!this._viewRow1) return;
        if (this._savedViews.length >= 14) return;
        const idx = this._savedViews.length;
        this._savedViews.push(viewObj);

        const btn = document.createElement('button');
        btn.textContent = String(idx + 1);
        btn.title = `Jump to view ${idx + 1}`;
        btn.style.cssText = 'width:32px;height:32px;border:1px solid #555;border-radius:4px;'
            + 'cursor:pointer;font-size:13px;font-weight:bold;background:#222;color:#fff;user-select:none;';
        btn.onclick = () => this._restoreView(idx);
        btn.oncontextmenu = (e) => { e.preventDefault(); this._showViewContextMenu(e, idx); };

        if (idx < 7) {
            this._viewRow1.appendChild(btn);
        } else {
            // Reveal row 2 on the 8th view
            if (idx === 7) this._viewRow2.style.display = 'flex';
            this._viewRow2.appendChild(btn);
        }
        this._viewBtns.push(btn);

        if (this._savedViews.length >= 14 && this._addViewBtn) {
            this._addViewBtn.disabled = true;
            this._addViewBtn.style.opacity = '0.4';
            this._addViewBtn.style.cursor = 'default';
        }
    }

    _restoreView(idx) {
        const v = this._savedViews[idx];
        if (!v) return;
        const sc = this.viewer.scene;

        this._restoringView = true;
        sc.camera.position.set(v.pos.x, v.pos.y, v.pos.z);
        sc.camera.up.set(v.up.x, v.up.y, v.up.z);
        sc.controls.target.set(v.tgt.x, v.tgt.y, v.tgt.z);
        sc.controls.update();
        this._restoringView = false;

        const pct = v.clipEnabled ? Math.round(v.clipFraction * 100) : 0;
        if (this._zSlider) { this._zSlider.value = pct; this._zSlider.oninput(); }

        this._viewBtns.forEach((b, i) => {
            b.style.background = i === idx ? 'rgb(212,160,23)' : '#222';
            b.style.color      = i === idx ? '#000' : '#fff';
        });
        this._activeViewIdx = idx;
    }

    _showViewContextMenu(e, idx) {
        // Remove any existing context menu
        const existing = document.getElementById('_viewCtxMenu');
        if (existing) existing.remove();

        const menu = document.createElement('div');
        menu.id = '_viewCtxMenu';
        menu.style.cssText = 'position:fixed;z-index:9999;background:rgba(30,30,30,0.97);'
            + 'border:1px solid #555;border-radius:4px;overflow:hidden;font-family:monospace;'
            + 'font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.6);user-select:none;';
        menu.style.left = e.clientX + 'px';
        menu.style.top  = e.clientY + 'px';

        const item = document.createElement('div');
        item.textContent = 'Delete view';
        item.style.cssText = 'padding:7px 14px;color:#ff6b6b;cursor:pointer;white-space:nowrap;';
        item.onmouseenter = () => { item.style.background = 'rgba(255,107,107,0.15)'; };
        item.onmouseleave = () => { item.style.background = ''; };
        item.onclick = () => { menu.remove(); this._removeView(idx); };
        menu.appendChild(item);
        document.body.appendChild(menu);

        // Dismiss on any outside click
        const dismiss = (ev) => {
            if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener('mousedown', dismiss, true); }
        };
        document.addEventListener('mousedown', dismiss, true);
    }

    _removeView(idx) {
        if (idx < 0 || idx >= this._savedViews.length) return;

        // Splice out the view data
        this._savedViews.splice(idx, 1);

        // Update active index
        if (this._activeViewIdx === idx) {
            this._activeViewIdx = null;
        } else if (this._activeViewIdx > idx) {
            this._activeViewIdx--;
        }

        // Clear both rows
        while (this._viewRow1.firstChild) this._viewRow1.removeChild(this._viewRow1.firstChild);
        while (this._viewRow2.firstChild) this._viewRow2.removeChild(this._viewRow2.firstChild);
        this._viewBtns = [];
        this._viewRow2.style.display = 'none';

        // Re-enable Add button
        if (this._addViewBtn) {
            this._addViewBtn.disabled = false;
            this._addViewBtn.style.opacity = '1';
            this._addViewBtn.style.cursor = 'pointer';
        }

        // Re-render all remaining buttons
        const views = this._savedViews.slice();
        this._savedViews = [];
        for (const v of views) this._addSavedView(v);

        // Re-apply gold highlight if active view still exists
        if (this._activeViewIdx !== null && this._viewBtns[this._activeViewIdx]) {
            this._viewBtns[this._activeViewIdx].style.background = 'rgb(212,160,23)';
            this._viewBtns[this._activeViewIdx].style.color = '#000';
        }
    }

    // ── Saved Sets ────────────────────────────────────────────────────────────

    _captureSet() {
        return {
            hlModeByNeuron: this.hlModeByNeuron,
            highlightedSet: new Set(this.vis.highlightedSet),
            activeColorMode: this.vis.activeColorMode,
            cbarMinPct: this._cbarMinPct  ?? 0,
            cbarMaxPct: this._cbarMaxPct  ?? 100,
            activeNTs:  this._activeNTs   ? new Set(this._activeNTs) : null,
            roiChecked: Object.assign({}, this.vis.roiChecked),
        };
    }

    _addSavedSet(setObj) {
        if (!this._setsRow1) return;
        if (this._savedSets.length >= 10) return;
        const idx = this._savedSets.length;
        this._savedSets.push(setObj);

        const btn = document.createElement('button');
        btn.textContent = String(idx + 1);
        btn.title = `Restore set ${idx + 1}`;
        btn.style.cssText = 'width:32px;height:32px;border:1px solid #555;border-radius:4px;'
            + 'cursor:pointer;font-size:13px;font-weight:bold;background:#222;color:#fff;user-select:none;';
        btn.onclick = () => this._restoreSet(idx);
        btn.oncontextmenu = (e) => { e.preventDefault(); this._showSetContextMenu(e, idx); };

        if (idx < 5) {
            this._setsRow1.appendChild(btn);
        } else {
            if (idx === 5) this._setsRow2.style.display = 'flex';
            this._setsRow2.appendChild(btn);
        }
        this._setsBtns.push(btn);

        if (this._savedSets.length >= 10 && this._addSetBtn) {
            this._addSetBtn.disabled = true;
            this._addSetBtn.style.opacity = '0.4';
            this._addSetBtn.style.cursor = 'default';
        }
    }

    _restoreSet(idx) {
        const s = this._savedSets[idx];
        if (!s) return;
        this._restoringSet = true;

        // 1. Clear stale filter state (mirrors what the color-mode button click does)
        if (this.vis._filterRemovedHighlights) {
            for (const k of this.vis._filterRemovedHighlights) this.vis.highlightedSet.add(k);
        }
        this.vis._filterRemovedHighlights = null;
        this.vis.colorFilteredOutTypes.clear();
        this.vis.colorFilteredOutNeurons.clear();
        this.vis.activeNTs = null;
        this._filterUncheckedRois = null;

        // 2. Switch color mode if needed
        if (s.activeColorMode !== this.vis.activeColorMode) {
            this.vis.switchColorMode(s.activeColorMode);
            this._updateColorbar(this.viewer.data.colorModes[s.activeColorMode]);
            this._updatePanelSwatches();
            if (this._colorSection) {
                this._colorSection.querySelectorAll('button[data-colormode]').forEach(b => {
                    const i = parseInt(b.dataset.colormode);
                    b.style.background = i === s.activeColorMode ? 'rgb(212,160,23)' : '#222';
                    b.style.color      = i === s.activeColorMode ? '#000' : '#fff';
                });
            }
        }

        // 3. Switch type/neuron mode if needed (updates UI via stored _switchMode)
        if (s.hlModeByNeuron !== this.hlModeByNeuron && this._switchMode) {
            this._switchMode(s.hlModeByNeuron, true);
        }

        // 4. Overwrite highlighted set with the saved copy
        this.vis.highlightedSet = new Set(s.highlightedSet);
        this.vis._filterRemovedHighlights = null;
        this.vis._explicitHideAll = (s.highlightedSet.size === 0);

        // 5. Apply saved color filter
        const mode = this.viewer.data.colorModes[s.activeColorMode];
        if (mode && mode.is_scalar) {
            this._cbarMinPct = s.cbarMinPct;
            this._cbarMaxPct = s.cbarMaxPct;
            if (this._cbarPercentToPos) {
                this._cbarMaxHandle.style.top = this._cbarPercentToPos(s.cbarMaxPct) + 'px';
                this._cbarMinHandle.style.top = this._cbarPercentToPos(s.cbarMinPct) + 'px';
                this._cbarUpdateRangeHighlight();
            }
            if (this._filterMinInput) this._filterMinInput.value = s.cbarMinPct;
            if (this._filterMaxInput) this._filterMaxInput.value = s.cbarMaxPct;
            this.vis.applyColorFilter(s.cbarMinPct, s.cbarMaxPct, null);
        } else if (mode && mode.nt_legend && s.activeNTs) {
            this._activeNTs = new Set(s.activeNTs);
            if (this._ntCheckboxes) {
                for (const [nt, cb] of Object.entries(this._ntCheckboxes)) {
                    cb.checked = this._activeNTs.has(nt);
                }
            }
            this.vis.applyColorFilter(0, 100, this._activeNTs);
        } else {
            this.vis._applyAllVisibility();
        }

        // 6. Restore ROI visibility state
        if (s.roiChecked) {
            for (const [roi, checked] of Object.entries(s.roiChecked)) {
                if (this.vis.roiChecked[roi] !== checked) {
                    this.vis.setRoiChecked(roi, checked);
                }
            }
        }

        // 7. Sync sidebar visuals and rebuild
        this._applySidebarColorFilter();
        this._drawColorbarOverlay();
        this._rebuildPanelContent();
        this.syncAllState();
        this._restoringSet = false;

        // 8. Highlight active set button in gold
        this._setsBtns.forEach((b, i) => {
            b.style.background = i === idx ? 'rgb(212,160,23)' : '#222';
            b.style.color      = i === idx ? '#000' : '#fff';
        });
        this._activeSetIdx = idx;
    }

    _setMatchesCurrent(s) {
        if (s.activeColorMode !== this.vis.activeColorMode) return false;
        if (s.hlModeByNeuron !== this.hlModeByNeuron) return false;
        const mode = this.viewer.data.colorModes[s.activeColorMode];
        if (mode && mode.is_scalar) {
            if (s.cbarMinPct !== (this._cbarMinPct ?? 0)) return false;
            if (s.cbarMaxPct !== (this._cbarMaxPct ?? 100)) return false;
        } else if (mode && mode.nt_legend) {
            const cur = this._activeNTs;
            if (!!s.activeNTs !== !!cur) return false;
            if (s.activeNTs && cur) {
                if (s.activeNTs.size !== cur.size) return false;
                for (const nt of s.activeNTs) if (!cur.has(nt)) return false;
            }
        }
        // Compare ROI visibility state
        if (s.roiChecked) {
            for (const [roi, checked] of Object.entries(s.roiChecked)) {
                if (this.vis.roiChecked[roi] !== checked) return false;
            }
        }
        // Compare full highlighted set (union of visible + filter-hidden items)
        const curFull = new Set(this.vis.highlightedSet);
        if (this.vis._filterRemovedHighlights) {
            for (const k of this.vis._filterRemovedHighlights) curFull.add(k);
        }
        if (s.highlightedSet.size !== curFull.size) return false;
        for (const k of s.highlightedSet) if (!curFull.has(k)) return false;
        return true;
    }

    _showSetContextMenu(e, idx) {
        const existing = document.getElementById('_setCtxMenu');
        if (existing) existing.remove();

        const menu = document.createElement('div');
        menu.id = '_setCtxMenu';
        menu.style.cssText = 'position:fixed;z-index:9999;background:rgba(30,30,30,0.97);'
            + 'border:1px solid #555;border-radius:4px;overflow:hidden;font-family:monospace;'
            + 'font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.6);user-select:none;';
        menu.style.left = e.clientX + 'px';
        menu.style.top  = e.clientY + 'px';

        const item = document.createElement('div');
        item.textContent = `Delete set ${idx + 1}`;
        item.style.cssText = 'padding:7px 14px;color:#ff6b6b;cursor:pointer;white-space:nowrap;';
        item.onmouseenter = () => { item.style.background = 'rgba(255,107,107,0.15)'; };
        item.onmouseleave = () => { item.style.background = ''; };
        item.onclick = () => { menu.remove(); this._removeSet(idx); };
        menu.appendChild(item);
        document.body.appendChild(menu);

        const dismiss = (ev) => {
            if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener('mousedown', dismiss, true); }
        };
        document.addEventListener('mousedown', dismiss, true);
    }

    _removeSet(idx) {
        if (idx < 0 || idx >= this._savedSets.length) return;
        this._savedSets.splice(idx, 1);

        if (this._activeSetIdx === idx) {
            this._activeSetIdx = null;
        } else if (this._activeSetIdx > idx) {
            this._activeSetIdx--;
        }

        // Clear both rows
        while (this._setsRow1.firstChild) this._setsRow1.removeChild(this._setsRow1.firstChild);
        while (this._setsRow2.firstChild) this._setsRow2.removeChild(this._setsRow2.firstChild);
        this._setsBtns = [];
        this._setsRow2.style.display = 'none';

        // Re-enable Add button
        if (this._addSetBtn) {
            this._addSetBtn.disabled = false;
            this._addSetBtn.style.opacity = '1';
            this._addSetBtn.style.cursor = 'pointer';
        }

        // Re-render all remaining buttons
        const sets = this._savedSets.slice();
        this._savedSets = [];
        for (const s of sets) this._addSavedSet(s);

        // Re-apply gold highlight if active set still exists
        if (this._activeSetIdx !== null && this._setsBtns[this._activeSetIdx]) {
            this._setsBtns[this._activeSetIdx].style.background = 'rgb(212,160,23)';
            this._setsBtns[this._activeSetIdx].style.color = '#000';
        }
    }

    // ── ROI Saved Sets ─────────────────────────────────────────────────────

    _captureRoiSet() {
        return { roiChecked: Object.assign({}, this.vis.roiChecked) };
    }

    _addRoiSavedSet(setObj) {
        if (!this._roiSetsRow1) return;
        if (this._roiSavedSets.length >= 10) return;
        const idx = this._roiSavedSets.length;
        this._roiSavedSets.push(setObj);

        const btn = document.createElement('button');
        btn.textContent = String(idx + 1);
        btn.title = `Restore ROI set ${idx + 1}`;
        btn.style.cssText = 'width:28px;height:28px;border:1px solid #555;border-radius:4px;'
            + 'cursor:pointer;font-size:12px;font-weight:bold;background:#222;color:#fff;user-select:none;';
        btn.onclick = () => this._restoreRoiSet(idx);
        btn.oncontextmenu = (e) => { e.preventDefault(); this._showRoiSetContextMenu(e, idx); };

        if (idx < 5) {
            this._roiSetsRow1.appendChild(btn);
        } else {
            if (idx === 5) this._roiSetsRow2.style.display = 'flex';
            this._roiSetsRow2.appendChild(btn);
        }
        this._roiSetsBtns.push(btn);

        if (this._roiSavedSets.length >= 10 && this._roiAddSetBtn) {
            this._roiAddSetBtn.disabled = true;
            this._roiAddSetBtn.style.opacity = '0.4';
            this._roiAddSetBtn.style.cursor = 'default';
        }
    }

    _restoreRoiSet(idx) {
        const s = this._roiSavedSets[idx];
        if (!s) return;
        this._restoringRoiSet = true;
        for (const [roi, checked] of Object.entries(s.roiChecked)) {
            if (this.vis.roiChecked[roi] !== checked) {
                this.vis.setRoiChecked(roi, checked);
            }
        }
        this.syncAllState();
        this._restoringRoiSet = false;

        this._roiSetsBtns.forEach((b, i) => {
            b.style.background = i === idx ? 'rgb(212,160,23)' : '#222';
            b.style.color      = i === idx ? '#000' : '#fff';
        });
        this._roiActiveSetIdx = idx;
    }

    _roiSetMatchesCurrent(s) {
        for (const [roi, checked] of Object.entries(s.roiChecked)) {
            if (this.vis.roiChecked[roi] !== checked) return false;
        }
        return true;
    }

    _showRoiSetContextMenu(e, idx) {
        const existing = document.getElementById('_roiSetCtxMenu');
        if (existing) existing.remove();

        const menu = document.createElement('div');
        menu.id = '_roiSetCtxMenu';
        menu.style.cssText = 'position:fixed;z-index:9999;background:rgba(30,30,30,0.97);'
            + 'border:1px solid #555;border-radius:4px;overflow:hidden;font-family:monospace;'
            + 'font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.6);user-select:none;';
        menu.style.left = e.clientX + 'px';
        menu.style.top  = e.clientY + 'px';

        const item = document.createElement('div');
        item.textContent = `Delete set ${idx + 1}`;
        item.style.cssText = 'padding:7px 14px;color:#ff6b6b;cursor:pointer;white-space:nowrap;';
        item.onmouseenter = () => { item.style.background = 'rgba(255,107,107,0.15)'; };
        item.onmouseleave = () => { item.style.background = ''; };
        item.onclick = () => { menu.remove(); this._removeRoiSet(idx); };
        menu.appendChild(item);
        document.body.appendChild(menu);

        const dismiss = (ev) => {
            if (!menu.contains(ev.target)) { menu.remove(); document.removeEventListener('mousedown', dismiss, true); }
        };
        document.addEventListener('mousedown', dismiss, true);
    }

    _removeRoiSet(idx) {
        if (idx < 0 || idx >= this._roiSavedSets.length) return;
        this._roiSavedSets.splice(idx, 1);

        if (this._roiActiveSetIdx === idx) {
            this._roiActiveSetIdx = null;
        } else if (this._roiActiveSetIdx > idx) {
            this._roiActiveSetIdx--;
        }

        while (this._roiSetsRow1.firstChild) this._roiSetsRow1.removeChild(this._roiSetsRow1.firstChild);
        while (this._roiSetsRow2.firstChild) this._roiSetsRow2.removeChild(this._roiSetsRow2.firstChild);
        this._roiSetsBtns = [];
        this._roiSetsRow2.style.display = 'none';

        if (this._roiAddSetBtn) {
            this._roiAddSetBtn.disabled = false;
            this._roiAddSetBtn.style.opacity = '1';
            this._roiAddSetBtn.style.cursor = 'pointer';
        }

        const sets = this._roiSavedSets.slice();
        this._roiSavedSets = [];
        for (const s of sets) this._addRoiSavedSet(s);

        if (this._roiActiveSetIdx !== null && this._roiSetsBtns[this._roiActiveSetIdx]) {
            this._roiSetsBtns[this._roiActiveSetIdx].style.background = 'rgb(212,160,23)';
            this._roiSetsBtns[this._roiActiveSetIdx].style.color = '#000';
        }
    }

    // ── Colormap picker ───────────────────────────────────────────────────────

    _shuffleColors() {
        const modeIdx = this.vis.activeColorMode;
        const mode = this.data.colorModes[modeIdx];

        // Don't shuffle scalar/continuous modes
        if (mode.is_scalar || mode.colorscale) return;

        if (!this.hlModeByNeuron) {
            // Type mode: shuffle type_colors
            const types = Object.keys(mode.type_colors);
            const colors = types.map(t => mode.type_colors[t]);
            // Fisher-Yates shuffle
            for (let i = colors.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [colors[i], colors[j]] = [colors[j], colors[i]];
            }
            types.forEach((t, i) => { mode.type_colors[t] = colors[i]; });
            // Propagate to per-neuron colors
            for (const bid of Object.keys(mode.colors)) {
                const typ = this.data.neuronType[bid];
                if (typ && mode.type_colors[typ]) mode.colors[bid] = mode.type_colors[typ];
            }
        } else {
            // Neuron mode: shuffle per-neuron colors
            const bids = Object.keys(mode.colors);
            const colors = bids.map(b => mode.colors[b]);
            for (let i = colors.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [colors[i], colors[j]] = [colors[j], colors[i]];
            }
            bids.forEach((b, i) => { mode.colors[b] = colors[i]; });
        }

        // Re-apply to 3D + UI
        this.vis.switchColorMode(modeIdx);
        this._updateColorbar(mode);
        this._updatePanelSwatches();
    }

    _showColormapMenu(e, modeIdx) {
        const mode = this.data.colorModes[modeIdx];

        const existing = document.getElementById('_cmapCtxMenu');
        if (existing) existing.remove();

        const menu = document.createElement('div');
        menu.id = '_cmapCtxMenu';
        menu.style.cssText = 'position:fixed;z-index:9999;background:rgba(30,30,30,0.97);'
            + 'border:1px solid #555;border-radius:4px;overflow:hidden;font-family:monospace;'
            + 'font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.6);user-select:none;min-width:160px;';
        menu.style.left = e.clientX + 'px';
        menu.style.top  = e.clientY + 'px';

        // Pinned header
        const hdr = document.createElement('div');
        hdr.textContent = 'Color map';
        hdr.style.cssText = 'padding:5px 14px;color:#888;font-size:10px;border-bottom:1px solid #444;letter-spacing:0.04em;flex-shrink:0;';
        menu.appendChild(hdr);

        // Scrollable item list — fixed height matches ~13 rows so the list feels the same
        // size as before but now scrolls to reveal the extra entries
        const itemList = document.createElement('div');
        itemList.style.cssText = 'overflow-y:auto;max-height:390px;scrollbar-width:thin;scrollbar-color:#555 transparent;';
        menu.appendChild(itemList);

        const dismiss = (ev) => {
            if (!menu.contains(ev.target)) {
                menu.remove();
                document.removeEventListener('mousedown', dismiss, true);
            }
        };

        const addItem = (label, cmapName) => {
            const item = document.createElement('div');
            item.style.cssText = 'padding:5px 14px;color:#ccc;cursor:pointer;display:flex;align-items:center;gap:8px;';

            // Mini gradient swatch
            const swatch = document.createElement('canvas');
            swatch.width = 48; swatch.height = 8;
            swatch.style.cssText = 'border-radius:2px;flex-shrink:0;';
            if (cmapName) {
                const ctx2d = swatch.getContext('2d');
                const grad  = ctx2d.createLinearGradient(0, 0, 48, 0);
                for (const [t, c] of COLORMAPS[cmapName]) grad.addColorStop(t, c);
                ctx2d.fillStyle = grad;
                ctx2d.fillRect(0, 0, 48, 8);
            } else {
                const ctx2d = swatch.getContext('2d');
                ctx2d.fillStyle = '#444';
                ctx2d.fillRect(0, 0, 48, 8);
            }
            item.appendChild(swatch);

            const lbl = document.createElement('span');
            lbl.textContent = label;
            item.appendChild(lbl);

            item.onmouseenter = () => { item.style.background = 'rgba(255,255,255,0.08)'; };
            item.onmouseleave = () => { item.style.background = ''; };
            item.onclick = () => {
                menu.remove();
                document.removeEventListener('mousedown', dismiss, true);
                this._applyColormap(modeIdx, cmapName);
            };
            itemList.appendChild(item);
        };

        // "Default" always first
        addItem('Default', null);
        for (const name of Object.keys(COLORMAPS)) addItem(name, name);

        // "Remove" option for uploaded color modes
        if (mode.is_uploaded) {
            const sep = document.createElement('div');
            sep.style.cssText = 'border-top:1px solid #444;margin:4px 0;';
            itemList.appendChild(sep);
            const removeItem = document.createElement('div');
            removeItem.style.cssText = 'padding:5px 14px;color:#e55;cursor:pointer;display:flex;align-items:center;gap:8px;';
            removeItem.innerHTML = '<span style="font-size:13px;">\u2715</span><span>Remove color mode</span>';
            removeItem.onmouseenter = () => { removeItem.style.background = 'rgba(255,80,80,0.15)'; };
            removeItem.onmouseleave = () => { removeItem.style.background = ''; };
            removeItem.onclick = () => {
                menu.remove();
                document.removeEventListener('mousedown', dismiss, true);
                this._removeUploadedColorMode(modeIdx);
            };
            itemList.appendChild(removeItem);
        }

        document.body.appendChild(menu);
        document.addEventListener('mousedown', dismiss, true);
    }

    _removeUploadedColorMode(modeIdx) {
        const mode = this.data.colorModes[modeIdx];
        if (!mode) return;
        const modeName = mode.name;

        // If this is the active mode, switch to Cell Type first
        if (this.vis.activeColorMode === modeIdx) {
            this.vis.switchColorMode(0);
            this.vis._applyAllVisibility();
            this._colorSection.querySelectorAll('button[data-colormode]').forEach(b => {
                b.style.background = '#222'; b.style.color = '#fff';
            });
            const cellTypeBtn = this._colorSection.querySelector('button[data-colormode="0"]');
            if (cellTypeBtn) { cellTypeBtn.style.background = 'rgb(212,160,23)'; cellTypeBtn.style.color = '#000'; }
            this._updateColorbar(this.data.colorModes[0]);
            this._updatePanelSwatches();
        }

        // Remove from colorModes array
        this.data.colorModes.splice(modeIdx, 1);

        // Remove button from DOM
        const btn = this._colorSection.querySelector(`button[data-colormodename="${modeName}"]`);
        if (btn) btn.remove();

        // Remove from _instanceLevelBtns
        if (this._instanceLevelBtns) {
            this._instanceLevelBtns = this._instanceLevelBtns.filter(b => b.dataset.colormodename !== modeName);
        }

        // Re-index remaining buttons
        this._reindexColorButtons();

        // If active mode index shifted, update
        if (this.vis.activeColorMode >= this.data.colorModes.length) {
            this.vis.activeColorMode = 0;
        }

        // Remove from uploadedColorCSVs by matching mode name
        if (this._uploadedColorCSVs) {
            // The CSV may have produced multiple modes; remove entries whose columns match
            this._uploadedColorCSVs = this._uploadedColorCSVs.filter(entry => {
                const p = this._parseCSV(entry.text);
                if (!p) return true;
                return !p.headers.slice(1).some(h => h.trim() === modeName || h.trim().replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) === modeName);
            });
        }

        this._updateInstanceBtnState();
        this.syncAllState();
        if (this.viewer.session) this.viewer.session.debouncedSave();
    }

    _applyColormap(modeIdx, cmapName) {
        const mode = this.data.colorModes[modeIdx];

        // Snapshot originals on first use so "Default" can restore them
        if (!mode._defaultTypeColors) {
            mode._defaultTypeColors  = Object.assign({}, mode.type_colors);
            mode._defaultColors      = Object.assign({}, mode.colors);
            if (mode.colorscale) mode._defaultColorscale = mode.colorscale.map(s => [s[0], s[1]]);
            if (mode.nt_legend)  mode._defaultNtLegend   = Object.assign({}, mode.nt_legend);
        }

        if (!cmapName) {
            // ── Restore defaults ─────────────────────────────────────────────
            mode.type_colors = Object.assign({}, mode._defaultTypeColors);
            mode.colors      = Object.assign({}, mode._defaultColors);
            if (mode._defaultColorscale) mode.colorscale = mode._defaultColorscale.map(s => [s[0], s[1]]);
            if (mode._defaultNtLegend)   mode.nt_legend  = Object.assign({}, mode._defaultNtLegend);
            mode._activeColorscale = null;

        } else {
            const stops = COLORMAPS[cmapName];

            if (mode.type_values) {
                // ── Scalar mode: map each type by its numeric value ───────────
                const vals  = Object.values(mode.type_values);
                const cmin  = mode.cmin !== undefined ? mode.cmin : Math.min(...vals);
                const cmax  = mode.cmax !== undefined ? mode.cmax : Math.max(...vals);
                const range = cmax - cmin || 1;
                for (const [typeName, val] of Object.entries(mode.type_values)) {
                    const t = Math.max(0, Math.min(1, (val - cmin) / range));
                    const [r, g, b] = interpolateColormap(stops, t);
                    mode.type_colors[typeName] = `rgb(${r},${g},${b})`;
                }
                for (const bid of Object.keys(mode.colors)) {
                    const tn = this.data.neuronType[bid];
                    if (tn && mode.type_colors[tn]) mode.colors[bid] = mode.type_colors[tn];
                }
                // Keep mode.colorscale in sync so the colorbar gradient matches
                mode.colorscale = stops;

            } else if (mode.nt_legend) {
                // ── NT mode: evenly space each NT along the colormap ──────────
                // Build dominant-NT map for type_colors on first use
                if (!mode._ntDominantMap) {
                    mode._ntDominantMap = {};
                    for (const typeName of Object.keys(mode.type_colors)) {
                        const bids = this.data.getNeuronsForType(typeName);
                        const counts = {};
                        for (const bid of bids) {
                            const nt = mode.bid_nts && mode.bid_nts[String(bid)];
                            if (nt) counts[nt] = (counts[nt] || 0) + 1;
                        }
                        let dom = null, max = 0;
                        for (const [nt, c] of Object.entries(counts)) {
                            if (c > max) { max = c; dom = nt; }
                        }
                        mode._ntDominantMap[typeName] = dom;
                    }
                }
                const nts = Object.keys(mode.nt_legend);
                nts.forEach((nt, i) => {
                    const t = nts.length > 1 ? i / (nts.length - 1) : 0.5;
                    const [r, g, b] = interpolateColormap(stops, t);
                    mode.nt_legend[nt] = `rgb(${r},${g},${b})`;
                });
                // Update bid colors and type_colors from updated nt_legend
                if (mode.bid_nts) {
                    for (const [bid, nt] of Object.entries(mode.bid_nts)) {
                        if (mode.nt_legend[nt]) mode.colors[bid] = mode.nt_legend[nt];
                    }
                }
                for (const [typeName, dom] of Object.entries(mode._ntDominantMap)) {
                    if (dom && mode.nt_legend[dom]) mode.type_colors[typeName] = mode.nt_legend[dom];
                }

            } else if (mode.name === 'Instance') {
                // ── Instance mode: evenly space each neuron along the colormap ──
                const bids = Object.keys(mode.colors).sort();
                bids.forEach((bid, i) => {
                    const t = bids.length > 1 ? i / (bids.length - 1) : 0.5;
                    const [r, g, b] = interpolateColormap(stops, t);
                    mode.colors[bid] = `rgb(${r},${g},${b})`;
                });
                // Update type_colors to first neuron's color per type
                for (const typeName of Object.keys(mode.type_colors)) {
                    const typeBids = this.data.getNeuronsForType(typeName);
                    if (typeBids.length > 0 && mode.colors[String(typeBids[0])]) {
                        mode.type_colors[typeName] = mode.colors[String(typeBids[0])];
                    }
                }

            } else {
                // ── Categorical mode: evenly space each type along the colormap ─
                const types = Object.keys(mode.type_colors).sort();
                types.forEach((typeName, i) => {
                    const t = types.length > 1 ? i / (types.length - 1) : 0.5;
                    const [r, g, b] = interpolateColormap(stops, t);
                    mode.type_colors[typeName] = `rgb(${r},${g},${b})`;
                });
                for (const bid of Object.keys(mode.colors)) {
                    const tn = this.data.neuronType[bid];
                    if (tn && mode.type_colors[tn]) mode.colors[bid] = mode.type_colors[tn];
                }
            }

            mode._activeColorscale = stops;
        }

        // Re-render and update UI if this mode is currently active
        if (this.vis.activeColorMode === modeIdx) {
            this.vis.switchColorMode(modeIdx);
            this._updateColorbar(mode);
            this._updatePanelSwatches();
        }
    }

    // ── Connection CSV export ─────────────────────────────────────────────────

    _exportConnCsv() {
        if (!this.connSelectedKey) return;
        const key   = this.connSelectedKey;
        const label = (this.data.instanceLookup[key] || key).replace(/[^a-zA-Z0-9_\-]/g, '_');
        const term  = (this.data.regexTerm || 'viewer').replace(/[^a-zA-Z0-9_\-]/g, '_');

        // Reconstruct up/downstream the same way _updateConnPanel does
        const isType = (this.data.typeUpstream[key] !== undefined || this.data.typeDownstream[key] !== undefined);
        let upstream, downstream;
        if (isType) {
            upstream   = this.data.typeUpstream[key]   || {};
            downstream = this.data.typeDownstream[key] || {};
        } else {
            const nUp = this.data.neuronUpstream[key]   || {};
            const nDn = this.data.neuronDownstream[key] || {};
            upstream = {}; downstream = {};
            for (const [roi, d] of Object.entries(nUp)) upstream[roi]   = d.__types__ || d;
            for (const [roi, d] of Object.entries(nDn)) downstream[roi] = d.__types__ || d;
        }

        const rows = ['Direction,ROI,Partner,Synapses'];
        const addRows = (data, dir) => {
            for (const [roi, partners] of Object.entries(data)) {
                const sorted = Object.entries(partners).sort((a, b) => b[1] - a[1]);
                for (const [partner, weight] of sorted) {
                    const safe = partner.includes(',') ? `"${partner}"` : partner;
                    rows.push(`${dir},${roi},${safe},${weight}`);
                }
            }
        };
        addRows(upstream,   'Upstream');
        addRows(downstream, 'Downstream');

        const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
        _saveFileAs(blob, `${term}_${label}_connections.csv`, [
            { description: 'CSV File', accept: { 'text/csv': ['.csv'] } }
        ]);
    }

    _buildSidebar() {
        const sb = document.createElement('div');
        sb.style.cssText = `position:fixed;left:0;top:0;width:${SIDEBAR_W}px;height:100vh;background:rgba(20,20,20,0.95);overflow-y:auto;padding:0;z-index:100;border-right:1px solid #444;user-select:none;`;

        // Sticky header container
        const stickyHdr = document.createElement('div');
        stickyHdr.style.cssText = 'position:sticky;top:0;z-index:2;background:rgba(20,20,20,0.98);padding:14px 10px 4px 10px;';

        // ── Light / Dark mode toggle ──────────────────────────────────────────
        const themeWrap = document.createElement('div');
        themeWrap.style.cssText = 'margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;';
        const themeLabel = document.createElement('span');
        themeLabel.textContent = '\u2600 Light Mode';
        themeLabel.style.cssText = 'font-size:15px;font-weight:bold;color:#fff;';
        themeWrap.appendChild(themeLabel);
        const themeSwitch = document.createElement('div');
        themeSwitch.style.cssText = `position:relative;width:40px;height:20px;background:${GOLD};border-radius:10px;cursor:pointer;`;
        const themeThumb = document.createElement('div');
        themeThumb.style.cssText = 'position:absolute;top:2px;left:20px;width:16px;height:16px;background:#fff;border-radius:50%;transition:left 0.2s;';
        themeSwitch.appendChild(themeThumb);
        themeSwitch.onclick = () => {
            const isLight = themeThumb.style.left === '20px';
            const t = isLight ? THEMES.dark : THEMES.light;
            _currentTheme = t;
            themeThumb.style.left = isLight ? '2px' : '20px';
            themeSwitch.style.background = isLight ? '#555' : GOLD;
            this._applyTheme(t);
        };
        themeWrap.appendChild(themeSwitch);
        stickyHdr.appendChild(themeWrap);
        this._themeLabel      = themeLabel;
        this._sidebarEl       = sb;
        this._stickyHdrEl     = stickyHdr;

        // Light intensity slider (below Light Mode, only if meshes available)
        if (this.viewer.scene._meshesAvailable) {
            const lightWrap = document.createElement('div');
            lightWrap.style.cssText = 'margin-bottom:12px;display:flex;align-items:center;gap:6px;overflow:hidden;';
            const lightLabel = document.createElement('span');
            lightLabel.textContent = 'Light';
            lightLabel.style.cssText = 'font-size:15px;font-weight:bold;color:#fff;flex-shrink:0;';
            lightWrap.appendChild(lightLabel);
            const lightSlider = document.createElement('input');
            lightSlider.type = 'range';
            lightSlider.min = '0.5';
            lightSlider.max = '7.5';
            lightSlider.step = '0.1';
            lightSlider.value = '4.0';
            lightSlider.style.cssText = 'flex:1;min-width:0;max-width:100%;accent-color:rgb(212,160,23);';
            lightSlider.oninput = () => {
                const v = parseFloat(lightSlider.value);
                this.viewer.scene._lightIntensityScale = v;
                this.viewer.scene._updateLightIntensity();
            };
            lightWrap.appendChild(lightSlider);
            stickyHdr.appendChild(lightWrap);
        }

        // Show ROI Bounds toggle
        const toggleWrap = document.createElement('div');
        toggleWrap.style.cssText = 'margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;';

        const toggleLabel = document.createElement('span');
        toggleLabel.textContent = 'Show ROI Bounds';
        toggleLabel.style.cssText = 'font-size:15px;font-weight:bold;color:#fff;';
        toggleWrap.appendChild(toggleLabel);

        const toggleSwitch = document.createElement('div');
        toggleSwitch.style.cssText = `position:relative;width:40px;height:20px;background:rgb(212,160,23);border-radius:10px;cursor:pointer;`;
        const toggleThumb = document.createElement('div');
        toggleThumb.style.cssText = 'position:absolute;top:2px;left:20px;width:16px;height:16px;background:#fff;border-radius:50%;transition:left 0.2s;';
        toggleSwitch.appendChild(toggleThumb);
        toggleSwitch.onclick = () => {
            const on = toggleThumb.style.left === '20px';
            toggleThumb.style.left = on ? '2px' : '20px';
            toggleSwitch.style.background = on ? '#555' : 'rgb(212,160,23)';
            this.vis.toggleWireframes(!on);
        };
        toggleWrap.appendChild(toggleSwitch);
        stickyHdr.appendChild(toggleWrap);

        // ROIs header
        const roiHeader = document.createElement('div');
        roiHeader.textContent = 'ROIs';
        roiHeader.style.cssText = 'font-size:15px;font-weight:bold;color:#fff;margin-bottom:6px;';
        stickyHdr.appendChild(roiHeader);
        const roiHr = document.createElement('hr');
        roiHr.style.cssText = 'border:none;border-top:1px solid #444;margin:4px 0 8px 0;';
        stickyHdr.appendChild(roiHr);

        // ── ROI Saved Sets ────────────────────────────────────────────────
        const roiSetsWrap = document.createElement('div');
        roiSetsWrap.style.cssText = 'padding:2px 6px 6px 6px;';

        const roiSetsTopRow = document.createElement('div');
        roiSetsTopRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;';
        const roiSetsLabel = document.createElement('span');
        roiSetsLabel.textContent = 'Saved Sets';
        roiSetsLabel.style.cssText = 'font-size:14px;color:#aaa;font-weight:bold;';
        roiSetsTopRow.appendChild(roiSetsLabel);
        const roiAddSetBtn = document.createElement('button');
        roiAddSetBtn.textContent = '+ Add';
        roiAddSetBtn.style.cssText = 'padding:2px 7px;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:11px;background:#222;color:#fff;user-select:none;';
        roiAddSetBtn.onclick = () => this._addRoiSavedSet(this._captureRoiSet());
        roiSetsTopRow.appendChild(roiAddSetBtn);
        this._roiAddSetBtn = roiAddSetBtn;
        roiSetsWrap.appendChild(roiSetsTopRow);

        const roiSetsRow1 = document.createElement('div');
        roiSetsRow1.style.cssText = 'display:flex;gap:4px;';
        roiSetsWrap.appendChild(roiSetsRow1);
        this._roiSetsRow1 = roiSetsRow1;

        const roiSetsRow2 = document.createElement('div');
        roiSetsRow2.style.cssText = 'display:none;gap:4px;margin-top:4px;';
        roiSetsWrap.appendChild(roiSetsRow2);
        this._roiSetsRow2 = roiSetsRow2;

        this._roiSetsBtns = [];
        stickyHdr.appendChild(roiSetsWrap);

        // Select all ROIs checkbox
        const selAllWrap = document.createElement('div');
        selAllWrap.style.cssText = 'padding:4px 6px;margin:2px 0 4px 0;display:flex;align-items:center;gap:6px;';
        const selAllCb = document.createElement('input');
        selAllCb.type = 'checkbox';
        selAllCb.id = 'roiSelectAllCb';
        selAllCb.indeterminate = true;
        selAllCb.style.cssText = 'accent-color:#d4a017;';
        selAllCb.onchange = () => {
            const checked = selAllCb.checked;
            for (const roi of this.data.sidebarRois) {
                this.vis.setRoiChecked(roi, checked);
            }
            this.syncAllState();
        };
        selAllWrap.appendChild(selAllCb);
        const selAllLabel = document.createElement('span');
        selAllLabel.textContent = 'Select all ROIs';
        selAllLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;';
        selAllWrap.appendChild(selAllLabel);
        stickyHdr.appendChild(selAllWrap);
        sb.appendChild(stickyHdr);

        // ROI list
        const roiList = document.createElement('div');
        roiList.style.cssText = 'padding:0 10px 14px 10px;';
        this.roiContainer = roiList;
        for (const roi of this.data.sidebarRois) {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:6px;padding:3px 4px;cursor:pointer;border-radius:3px;';
            row.dataset.disabled = 'false';
            row.dataset.themedRow = '1';

            // Gold border for checked ROIs (no checkbox)
            if (this.vis.roiChecked[roi] === true) {
                row.style.border = '1px solid #d4a017';
            } else {
                row.style.border = '1px solid transparent';
            }

            const label = document.createElement('span');
            const synCount = this.data.roiSynapseTotals[roi] || 0;
            label.textContent = `${roi} (${synCount.toLocaleString()})`;
            label.style.cssText = 'font-size:12px;color:#ccc;flex:1;';
            row.appendChild(label);

            // Click handlers (single = select for conn, double = toggle)
            let clickTimer = null;
            row.onclick = () => {
                if (row.dataset.disabled === 'true') return;
                if (clickTimer) {
                    clearTimeout(clickTimer);
                    clickTimer = null;
                    // Double-click: toggle ROI
                    this.vis.setRoiChecked(roi, !this.vis.roiChecked[roi]);
                    row.style.border = this.vis.roiChecked[roi] === true ? '1px solid #d4a017' : '1px solid transparent';
                    this.syncAllState();
                } else {
                    clickTimer = setTimeout(() => {
                        clickTimer = null;
                        // Single-click: select for connections
                        this._selectRoiForConn(roi);
                    }, 400);
                }
            };

            roiList.appendChild(row);
            this.roiLabels[roi] = { row, label };
        }
        sb.appendChild(roiList);

        document.body.appendChild(sb);
        this.sidebar = sb;
    }

    _selectRoiForConn(roiName) {
        if (this.connSelectedRoi === roiName) {
            this.connSelectedRoi = null;
        } else {
            this.connSelectedRoi = roiName;
        }
        this._updateRoiHighlight();
        if (this.connSelectedKey) {
            this._updateConnPanel();
        }
    }

    _updateRoiHighlight() {
        for (const [roi, els] of Object.entries(this.roiLabels)) {
            els.row.style.background = (roi === this.connSelectedRoi) ? 'rgba(100,149,237,0.25)' : 'none';
        }
    }

    _buildTypePanel() {
        const panel = document.createElement('div');
        panel.style.cssText = `position:fixed;right:0;top:0;width:${TYPE_PANEL_W}px;height:100vh;background:rgba(20,20,20,0.95);display:flex;flex-direction:column;overflow:hidden;padding:0;z-index:100;border-left:1px solid #444;user-select:none;`;
        this._typePanelEl = panel;

        // Sticky header container
        const typeStickyHdr = document.createElement('div');
        typeStickyHdr.style.cssText = 'flex-shrink:0;background:rgba(20,20,20,0.98);padding:14px 10px 4px 10px;';

        // Type / Neuron toggle (visual toggle switch)
        const toggleWrap = document.createElement('div');
        toggleWrap.style.cssText = 'margin-bottom:18px;display:flex;align-items:center;gap:8px;';
        const typeLabel = document.createElement('span');
        typeLabel.textContent = 'Type';
        typeLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#fff;cursor:pointer;';
        toggleWrap.appendChild(typeLabel);

        const toggleSwitch = document.createElement('div');
        toggleSwitch.style.cssText = 'position:relative;width:40px;height:20px;background:#555;border-radius:10px;cursor:pointer;';
        const toggleThumb = document.createElement('div');
        toggleThumb.style.cssText = 'position:absolute;top:2px;left:2px;width:16px;height:16px;background:#fff;border-radius:50%;transition:left 0.2s;';
        toggleSwitch.appendChild(toggleThumb);
        toggleWrap.appendChild(toggleSwitch);

        const neuronLabel = document.createElement('span');
        neuronLabel.textContent = 'Neuron';
        neuronLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#888;cursor:pointer;';
        toggleWrap.appendChild(neuronLabel);
        typeStickyHdr.appendChild(toggleWrap);

        // Mesh / Skeleton toggle (only if mesh data is available) — above somata
        if (this.viewer.scene._meshesAvailable) {
            const meshToggleWrap = document.createElement('div');
            meshToggleWrap.style.cssText = 'margin-bottom:18px;display:flex;align-items:center;gap:8px;';
            const meshLabel = document.createElement('span');
            meshLabel.textContent = 'Mesh';
            meshLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#fff;cursor:pointer;';
            meshToggleWrap.appendChild(meshLabel);

            const meshToggleSwitch = document.createElement('div');
            meshToggleSwitch.style.cssText = 'position:relative;width:40px;height:20px;background:#555;border-radius:10px;cursor:pointer;';
            const meshToggleThumb = document.createElement('div');
            // Left = Mesh (default ON), Right = Skeleton
            meshToggleThumb.style.cssText = 'position:absolute;top:2px;left:2px;width:16px;height:16px;background:#fff;border-radius:50%;transition:left 0.2s;';
            meshToggleSwitch.appendChild(meshToggleThumb);
            meshToggleWrap.appendChild(meshToggleSwitch);

            const skelLabel = document.createElement('span');
            skelLabel.textContent = 'Skeleton';
            skelLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#888;cursor:pointer;';
            meshToggleWrap.appendChild(skelLabel);

            const switchMeshMode = (toMesh) => {
                this.vis._meshesVisible = toMesh;
                // Left = Mesh, Right = Skeleton
                meshToggleThumb.style.left = toMesh ? '2px' : '20px';
                meshLabel.style.color = toMesh ? '#fff' : '#888';
                skelLabel.style.color = toMesh ? '#888' : '#fff';
                this.vis._applyAllVisibility();
            };
            meshToggleSwitch.onclick = () => switchMeshMode(!this.vis._meshesVisible);
            skelLabel.onclick = () => switchMeshMode(false);
            meshLabel.onclick = () => switchMeshMode(true);
            this._switchMeshMode = switchMeshMode;

            typeStickyHdr.appendChild(meshToggleWrap);

            // Neuron size slider (skeleton line width) — only visible in skeleton mode
            const sizeWrap = document.createElement('div');
            sizeWrap.style.cssText = 'margin-bottom:10px;display:flex;align-items:center;gap:6px;';
            const sizeLabel = document.createElement('span');
            sizeLabel.textContent = 'Neuron size:';
            sizeLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;white-space:nowrap;';
            sizeWrap.appendChild(sizeLabel);
            const sizeSlider = document.createElement('input');
            sizeSlider.type = 'range';
            sizeSlider.min = '0.1';
            sizeSlider.max = '10';
            sizeSlider.step = '0.1';
            sizeSlider.value = String(this.data.initialLineWidth);
            sizeSlider.style.cssText = 'flex:1;min-width:0;max-width:100px;accent-color:rgb(212,160,23);';
            sizeWrap.appendChild(sizeSlider);
            const sizeVal = document.createElement('span');
            sizeVal.textContent = String(this.data.initialLineWidth);
            sizeVal.style.cssText = 'font-size:12px;color:#ccc;width:24px;';
            sizeSlider.oninput = () => {
                sizeVal.textContent = sizeSlider.value;
                _globalLineWidth.value = parseFloat(sizeSlider.value);
            };
            sizeWrap.appendChild(sizeVal);
            typeStickyHdr.appendChild(sizeWrap);
            this._neuronSizeWrap = sizeWrap;

            // Somata visibility toggle (skeleton-only when meshes available)
            const somataWrap = document.createElement('div');
            somataWrap.style.cssText = 'margin-bottom:10px;display:flex;align-items:center;gap:6px;';
            const somataCheck = document.createElement('input');
            somataCheck.type = 'checkbox';
            somataCheck.checked = true;
            somataCheck.style.cssText = 'margin:0;cursor:pointer;accent-color:rgb(212,160,23);';
            somataWrap.appendChild(somataCheck);
            const somataLabel = document.createElement('span');
            somataLabel.textContent = 'Show somata';
            somataLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;cursor:pointer;';
            somataLabel.onclick = () => { somataCheck.click(); };
            somataWrap.appendChild(somataLabel);
            somataCheck.onchange = () => {
                this.vis._somataVisible = somataCheck.checked;
                this.vis._applyAllVisibility();
            };
            typeStickyHdr.appendChild(somataWrap);

            // Hook into mesh/skeleton toggle to show/hide skeleton-only controls
            const origSwitch = switchMeshMode;
            const wrappedSwitch = (toMesh) => {
                origSwitch(toMesh);
                sizeWrap.style.display = toMesh ? 'none' : 'flex';
                somataWrap.style.display = toMesh ? 'none' : 'flex';
            };
            this._switchMeshMode = wrappedSwitch;
            meshToggleSwitch.onclick = () => wrappedSwitch(!this.vis._meshesVisible);
            skelLabel.onclick = () => wrappedSwitch(false);
            meshLabel.onclick = () => wrappedSwitch(true);
            // Initial visibility: if mesh mode is default, hide skeleton-only controls
            if (this.vis._meshesVisible !== false) {
                sizeWrap.style.display = 'none';
                somataWrap.style.display = 'none';
            }

        }

        // Neuron size + Somata — shown always when no mesh data (skeleton-only visualization)
        if (!this.viewer.scene._meshesAvailable) {
            const sizeWrap = document.createElement('div');
            sizeWrap.style.cssText = 'margin-bottom:10px;display:flex;align-items:center;gap:6px;';
            const sizeLabel = document.createElement('span');
            sizeLabel.textContent = 'Neuron size:';
            sizeLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;white-space:nowrap;';
            sizeWrap.appendChild(sizeLabel);
            const sizeSlider = document.createElement('input');
            sizeSlider.type = 'range';
            sizeSlider.min = '0.1';
            sizeSlider.max = '10';
            sizeSlider.step = '0.1';
            sizeSlider.value = String(this.data.initialLineWidth);
            sizeSlider.style.cssText = 'flex:1;min-width:0;max-width:100px;accent-color:rgb(212,160,23);';
            sizeWrap.appendChild(sizeSlider);
            const sizeVal = document.createElement('span');
            sizeVal.textContent = String(this.data.initialLineWidth);
            sizeVal.style.cssText = 'font-size:12px;color:#ccc;width:24px;';
            sizeSlider.oninput = () => {
                sizeVal.textContent = sizeSlider.value;
                _globalLineWidth.value = parseFloat(sizeSlider.value);
            };
            sizeWrap.appendChild(sizeVal);
            typeStickyHdr.appendChild(sizeWrap);

            const somataWrap = document.createElement('div');
            somataWrap.style.cssText = 'margin-bottom:10px;display:flex;align-items:center;gap:6px;';
            const somataCheck = document.createElement('input');
            somataCheck.type = 'checkbox';
            somataCheck.checked = true;
            somataCheck.style.cssText = 'margin:0;cursor:pointer;accent-color:rgb(212,160,23);';
            somataWrap.appendChild(somataCheck);
            const somataLabel = document.createElement('span');
            somataLabel.textContent = 'Show somata';
            somataLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;cursor:pointer;';
            somataLabel.onclick = () => { somataCheck.click(); };
            somataWrap.appendChild(somataLabel);
            somataCheck.onchange = () => {
                this.vis._somataVisible = somataCheck.checked;
                this.vis._applyAllVisibility();
            };
            typeStickyHdr.appendChild(somataWrap);
        }

        // Preview on hover toggle (always visible)
        const hoverWrap = document.createElement('div');
        hoverWrap.style.cssText = 'margin-bottom:10px;display:flex;align-items:center;gap:6px;';
        const hoverCheck = document.createElement('input');
        hoverCheck.type = 'checkbox';
        hoverCheck.checked = false;
        hoverCheck.style.cssText = 'margin:0;cursor:pointer;accent-color:rgb(212,160,23);';
        hoverWrap.appendChild(hoverCheck);
        const hoverLabel = document.createElement('span');
        hoverLabel.textContent = 'Preview on hover';
        hoverLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;cursor:pointer;';
        hoverLabel.onclick = () => { hoverCheck.click(); };
        hoverWrap.appendChild(hoverLabel);
        hoverCheck.onchange = () => {
            this.viewer.hoverPreviewEnabled = hoverCheck.checked;
        };
        typeStickyHdr.appendChild(hoverWrap);

        // Types (N) header
        this.panelHeader = document.createElement('div');
        this.panelHeader.style.cssText = 'font-size:14px;font-weight:bold;color:#fff;margin-bottom:6px;';
        this.panelHeader.textContent = `Types (${this.data.allTypes.length})`;
        typeStickyHdr.appendChild(this.panelHeader);
        const typePanelHr = document.createElement('hr');
        typePanelHr.style.cssText = 'border:none;border-top:1px solid #444;margin:4px 0 8px 0;';
        typeStickyHdr.appendChild(typePanelHr);

        const switchMode = (toNeuron, isAuto = false) => {
            if (toNeuron === this.hlModeByNeuron) return;
            // A manual toggle always clears the NT auto-switch flag
            if (!isAuto) this._ntAutoSwitchedNeuron = false;
            // Convert highlighted set between type and neuron keys
            if (toNeuron) {
                // Type -> Neuron: expand type names to bodyIds
                const newHl = new Set();
                const newClip = {};
                for (const key of this.vis.highlightedSet) {
                    const bids = this.data.getNeuronsForType(key);
                    if (bids.length > 0) {
                        for (const bid of bids) {
                            newHl.add(bid);
                            newClip[bid] = this.vis.clipToRoi[key] || false;
                        }
                    } else {
                        // Already a bodyId
                        newHl.add(key);
                        newClip[key] = this.vis.clipToRoi[key] || false;
                    }
                }
                this.vis.highlightedSet = newHl;
                this.vis.clipToRoi = newClip;
                // Convert connSelectedKey
                if (this.connSelectedKey && this.data.getNeuronsForType(this.connSelectedKey).length > 0) {
                    this.connSelectedKey = null;
                    this.viewer.connSelectedKey = null;
                }
            } else {
                // Neuron -> Type: collapse bodyIds to type names
                const newHl = new Set();
                const newClip = {};
                for (const key of this.vis.highlightedSet) {
                    const typeName = this.data.neuronType[key];
                    if (typeName) {
                        newHl.add(typeName);
                        // Preserve clip state (use any neuron's state)
                        if (this.vis.clipToRoi[key] === true) newClip[typeName] = true;
                        else if (!(typeName in newClip)) newClip[typeName] = false;
                    } else {
                        newHl.add(key);
                    }
                }
                this.vis.highlightedSet = newHl;
                this.vis.clipToRoi = newClip;
                if (this.connSelectedKey && this.data.neuronType[this.connSelectedKey]) {
                    this.connSelectedKey = null;
                    this.viewer.connSelectedKey = null;
                }
            }

            this.hlModeByNeuron = toNeuron;

            // Auto-switch away from Instance / instance-level modes when going back to type mode
            if (!toNeuron) {
                const curMode = this.data.colorModes[this.vis.activeColorMode];
                if (curMode && (curMode.name === 'Instance' || curMode.is_instance_level || curMode.nt_legend)) {
                    // Switch to Cell Type (index 0)
                    this.vis.switchColorMode(0);
                    const topBar = this.topBar;
                    if (topBar) {
                        topBar.querySelectorAll('button[data-colormode]').forEach(b => {
                            b.style.background = '#222'; b.style.color = '#fff';
                        });
                        const cellTypeBtn = topBar.querySelector('button[data-colormode="0"]');
                        if (cellTypeBtn) { cellTypeBtn.style.background = 'rgb(212,160,23)'; cellTypeBtn.style.color = '#000'; }
                    }
                    this._updateColorbar(this.data.colorModes[0]);
                }
            }

            if (toNeuron) {
                toggleThumb.style.left = '20px';
                typeLabel.style.color = '#888';
                neuronLabel.style.color = '#fff';
            } else {
                toggleThumb.style.left = '2px';
                typeLabel.style.color = '#fff';
                neuronLabel.style.color = '#888';
            }
            this.vis._applyAllVisibility();
            this._rebuildPanelContent();
            this.syncAllState();
            this._updateInstanceBtnState();
            this._updatePanelSwatches();
        };
        this._switchMode = switchMode;
        toggleSwitch.onclick = () => switchMode(!this.hlModeByNeuron);
        typeLabel.onclick = () => switchMode(false);
        neuronLabel.onclick = () => switchMode(true);

        // ── Saved Sets ────────────────────────────────────────────────────
        const setsWrap = document.createElement('div');
        setsWrap.style.cssText = 'padding:2px 6px 6px 6px;';

        const setsTopRow = document.createElement('div');
        setsTopRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;';
        const setsLabel = document.createElement('span');
        setsLabel.textContent = 'Saved Sets';
        setsLabel.style.cssText = 'font-size:14px;color:#aaa;font-weight:bold;';
        setsTopRow.appendChild(setsLabel);
        const addSetBtn = document.createElement('button');
        addSetBtn.textContent = '+ Add';
        addSetBtn.style.cssText = 'padding:2px 7px;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:11px;background:#222;color:#fff;user-select:none;';
        addSetBtn.onclick = () => this._addSavedSet(this._captureSet());
        setsTopRow.appendChild(addSetBtn);
        this._addSetBtn = addSetBtn;
        setsWrap.appendChild(setsTopRow);

        const setsRow1 = document.createElement('div');
        setsRow1.style.cssText = 'display:flex;gap:4px;';
        setsWrap.appendChild(setsRow1);
        this._setsRow1 = setsRow1;

        const setsRow2 = document.createElement('div');
        setsRow2.style.cssText = 'display:none;gap:4px;margin-top:4px;';
        setsWrap.appendChild(setsRow2);
        this._setsRow2 = setsRow2;

        this._setsBtns = [];
        typeStickyHdr.appendChild(setsWrap);

        // Highlight all checkbox
        const hlAllWrap = document.createElement('div');
        hlAllWrap.style.cssText = 'padding:4px 6px;margin:0 0 4px 0;display:flex;align-items:center;gap:6px;';
        this.hlAllCb = document.createElement('input');
        this.hlAllCb.type = 'checkbox';
        this.hlAllCb.style.cssText = 'accent-color:#d4a017;';
        this.hlAllCb.onchange = () => {
            if (this.hlModeByNeuron) {
                // Neuron mode: highlight/unhighlight all bodyIds
                if (this.hlAllCb.checked) {
                    this.vis._explicitHideAll = false;
                    for (const t of this.data.allTypes) {
                        for (const bid of this.data.getNeuronsForType(t)) {
                            this.vis.highlightedSet.add(bid);
                            if (!(bid in this.vis.clipToRoi)) this.vis.clipToRoi[bid] = false;
                        }
                    }
                } else {
                    this.vis._explicitHideAll = true;
                    this.vis.highlightedSet.clear();
                    this.vis.clipToRoi = {};
                }
                this.vis._applyAllVisibility();
            } else {
                if (this.hlAllCb.checked) this.vis.highlightAll();
                else this.vis.unhighlightAll();
            }
            this.syncAllState();
        };
        hlAllWrap.appendChild(this.hlAllCb);
        const hlAllLabel = document.createElement('span');
        hlAllLabel.textContent = 'Highlight all types';
        hlAllLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;';
        this.hlAllLabel = hlAllLabel;
        hlAllWrap.appendChild(hlAllLabel);
        typeStickyHdr.appendChild(hlAllWrap);

        // Clip all checkbox
        const clipAllWrap = document.createElement('div');
        clipAllWrap.style.cssText = 'padding:4px 6px;margin:0 0 8px 0;display:flex;align-items:center;gap:6px;';
        this.clipAllCb = document.createElement('input');
        this.clipAllCb.type = 'checkbox';
        this.clipAllCb.style.cssText = 'accent-color:#d4a017;';
        this.clipAllCb.onchange = () => {
            const clip = this.clipAllCb.checked;
            for (const t of this.vis.highlightedSet) {
                this.vis.clipToRoi[t] = clip;
                this.vis._applyKeyVisibility(t);
            }
            // Also apply to items currently removed by color filter so clip persists across mode switches
            if (this.vis._filterRemovedHighlights) {
                for (const k of this.vis._filterRemovedHighlights) {
                    this.vis.clipToRoi[k] = clip;
                }
            }
            this.syncAllState();
        };
        clipAllWrap.appendChild(this.clipAllCb);
        const clipAllLabel = document.createElement('span');
        clipAllLabel.textContent = 'Clip all to ROI';
        clipAllLabel.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;';
        clipAllWrap.appendChild(clipAllLabel);
        typeStickyHdr.appendChild(clipAllWrap);
        panel.appendChild(typeStickyHdr);

        // Scrollable wrapper for type/neuron list
        const scrollWrap = document.createElement('div');
        scrollWrap.style.cssText = 'flex:1;overflow-y:auto;';

        // Content container for type/neuron list
        this.panelContent = document.createElement('div');
        this.panelContent.style.cssText = 'padding:0 10px;';
        scrollWrap.appendChild(this.panelContent);
        panel.appendChild(scrollWrap);

        document.body.appendChild(panel);
        this.typePanel = panel;

        // Build initial type list
        this._rebuildPanelContent();
    }

    _rebuildPanelContent() {
        this.panelContent.innerHTML = '';
        this.typeRows = {};
        this.typeCbs = {};
        this.clipCbs = {};
        this.swatches = {};

        if (!this.hlModeByNeuron) {
            this.hlAllLabel.textContent = 'Highlight all types';
            if (this.panelHeader) this.panelHeader.textContent = `Types (${this.data.allTypes.length})`;
            this._buildTypeModePanel();
        } else {
            this.hlAllLabel.textContent = 'Highlight all neurons';
            const totalNeurons = this.data.allTypes.reduce((sum, t) => sum + (this.data.typeNeurons[t] || []).length, 0);
            if (this.panelHeader) this.panelHeader.textContent = `Neurons (${totalNeurons})`;
            this._buildNeuronModePanel();
        }

        // Reapply color filter styling if active
        if (this.vis.colorFilteredOutTypes.size > 0 || this.vis.colorFilteredOutNeurons.size > 0) {
            this._applySidebarColorFilter();
        }
    }

    _updateInstanceBtnState() {
        // Gray out Instance and instance-level color buttons when not in neuron mode
        const isNeuronMode = this.hlModeByNeuron;
        for (const btn of (this._instanceLevelBtns || [])) {
            const modeName = btn.dataset.colormodename;
            const mIdx = this.data.colorModes.findIndex(m => m.name === modeName);
            const isActive = this.vis.activeColorMode === mIdx;
            if (!isNeuronMode && !isActive) {
                btn.style.opacity = '0.35';
                btn.style.pointerEvents = 'none';
            } else {
                btn.style.opacity = '1';
                btn.style.pointerEvents = 'auto';
            }
        }
    }

    _updatePanelSwatches() {
        if (!this.swatches) return;
        const modeIdx = this.vis.activeColorMode;
        const isCustom = this.data.colorModes[modeIdx] && this.data.colorModes[modeIdx].is_custom;
        for (const [key, info] of Object.entries(this.swatches)) {
            const color = info.kind === 'type'
                ? this.data.getTypeColor(key, modeIdx)
                : this.data.getNeuronColor(key, modeIdx);
            info.el.style.background = color;
            info.el.style.cursor = isCustom ? 'pointer' : 'default';
            info.el.style.border = isCustom ? '1px solid #888' : 'none';
        }
    }

    _buildTypeModePanel() {
        for (const typeName of this.data.allTypes) {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:4px;padding:2px 0;cursor:pointer;';
            row.dataset.themedRow = '1';

            // Gold border for highlighted types (no highlight checkbox — border IS the indicator)
            if (this.vis.highlightedSet.has(typeName)) {
                row.style.border = '1px solid #d4a017';
                row.style.borderRadius = '3px';
                row.style.padding = '2px 4px';
            } else {
                row.style.border = '1px solid transparent';
                row.style.borderRadius = '3px';
                row.style.padding = '2px 4px';
            }

            // Clip checkbox (only checkbox per row)
            const clipCb = document.createElement('input');
            clipCb.type = 'checkbox';
            clipCb.checked = this.vis.clipToRoi[typeName] === true;
            clipCb.title = 'Clip to ROI';
            clipCb.onclick = (e) => {
                e.stopPropagation();
                this.vis.toggleClip(typeName);
                this.syncAllState();
            };
            row.appendChild(clipCb);

            // Color swatch
            const swatch = document.createElement('div');
            const color = this.data.getTypeColor(typeName, this.vis.activeColorMode);
            swatch.style.cssText = `width:12px;height:12px;border-radius:2px;background:${color};flex-shrink:0;`;
            ((tn) => {
                swatch.addEventListener('click', (e) => {
                    const mode = this.data.colorModes[this.vis.activeColorMode];
                    if (!mode.is_custom) return;
                    e.stopPropagation();
                    this._showCustomColorPicker(e, tn, 'type');
                });
            })(typeName);
            row.appendChild(swatch);
            this.swatches[typeName] = { el: swatch, kind: 'type' };

            // Type name + neuron count
            const nameSpan = document.createElement('span');
            const nCount = (this.data.typeNeurons[typeName] || []).length;
            nameSpan.textContent = typeName;
            nameSpan.style.cssText = 'font-size:12px;color:#ccc;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
            row.appendChild(nameSpan);

            // Neuron count label
            const countSpan = document.createElement('span');
            countSpan.textContent = `(${nCount})`;
            countSpan.style.cssText = 'font-size:11px;color:#888;flex-shrink:0;';
            row.appendChild(countSpan);

            // Click handler: single = select for conn, double = toggle highlight
            let clickTimer = null;
            row.onclick = () => {
                if (clickTimer) {
                    clearTimeout(clickTimer);
                    clickTimer = null;
                    // Double-click: toggle highlight
                    this.vis.toggleHighlight(typeName);
                    this.syncAllState();
                } else {
                    clickTimer = setTimeout(() => {
                        clickTimer = null;
                        // Single-click: select for connectivity
                        if (this.vis.highlightedSet.has(typeName)) {
                            this.viewer.selectForConnectivity(typeName);
                        }
                    }, 400);
                }
            };

            this.panelContent.appendChild(row);
            this.typeRows[typeName] = row;
            this.clipCbs[typeName] = clipCb;
        }
    }

    _buildNeuronModePanel() {
        this._neuronTypeHeaders = {};
        // Group neurons by type, display each with instance name
        for (const typeName of this.data.allTypes) {
            const bids = this.data.getNeuronsForType(typeName);
            if (!bids.length) continue;

            // Type group header
            const typeHeader = document.createElement('div');
            typeHeader.style.cssText = 'font-size:11px;color:#888;margin-top:6px;margin-bottom:2px;font-weight:bold;';
            typeHeader.textContent = typeName;
            this.panelContent.appendChild(typeHeader);
            this._neuronTypeHeaders[typeName] = typeHeader;

            for (const bid of bids) {
                const row = document.createElement('div');
                row.style.cssText = 'display:flex;align-items:center;gap:4px;padding:2px 0;padding-left:8px;cursor:pointer;';

                const isHl = this.vis.highlightedSet.has(bid);
                if (isHl) {
                    row.style.border = '1px solid #d4a017';
                    row.style.borderRadius = '3px';
                    row.style.padding = '2px 4px 2px 8px';
                } else {
                    row.style.border = '1px solid transparent';
                    row.style.borderRadius = '3px';
                    row.style.padding = '2px 4px 2px 8px';
                }

                // Clip checkbox (only checkbox per row)
                const clipCb = document.createElement('input');
                clipCb.type = 'checkbox';
                clipCb.checked = this.vis.clipToRoi[bid] === true;
                clipCb.onclick = (e) => {
                    e.stopPropagation();
                    this.vis.toggleClipNeuron(bid);
                    this.syncAllState();
                };
                row.appendChild(clipCb);

                // Color swatch
                const swatch = document.createElement('div');
                const color = this.data.getNeuronColor(bid, this.vis.activeColorMode);
                swatch.style.cssText = `width:12px;height:12px;border-radius:2px;background:${color};flex-shrink:0;`;
                ((b) => {
                    swatch.addEventListener('click', (e) => {
                        const mode = this.data.colorModes[this.vis.activeColorMode];
                        if (!mode.is_custom) return;
                        e.stopPropagation();
                        this._showCustomColorPicker(e, b, 'neuron');
                    });
                })(bid);
                row.appendChild(swatch);
                this.swatches[bid] = { el: swatch, kind: 'neuron' };

                // Instance name
                const nameSpan = document.createElement('span');
                const instance = (this.data.instanceLookup || {})[bid] || bid;
                nameSpan.textContent = instance + ' (' + bid + ')';
                nameSpan.style.cssText = 'font-size:11px;color:#ccc;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
                row.appendChild(nameSpan);

                // Click handler
                let clickTimer = null;
                row.onclick = () => {
                    if (clickTimer) {
                        clearTimeout(clickTimer);
                        clickTimer = null;
                        this.vis.toggleHighlightNeuron(bid);
                        this.syncAllState();
                    } else {
                        clickTimer = setTimeout(() => {
                            clickTimer = null;
                            if (this.vis.highlightedSet.has(bid)) {
                                this.viewer.selectForConnectivity(bid);
                            }
                        }, 400);
                    }
                };

                this.panelContent.appendChild(row);
                this.typeRows[bid] = row;
                this.clipCbs[bid] = clipCb;
            }
        }
    }

    _buildInfoBox() {
        // Sits PANEL_PAD above the orientation panel header — same left offset (SIDEBAR_W + PANEL_PAD).
        // bottom = PANEL_PAD + hdrH + sz + PANEL_PAD  (updated by _buildGizmo collapse toggle)
        const box = document.createElement('div');
        box.style.cssText = `position:fixed;bottom:${PANEL_PAD + 28 + 240 + PANEL_PAD}px;left:${SIDEBAR_W + PANEL_PAD}px;`
            + `z-index:100;font-family:monospace;font-size:11px;letter-spacing:0.04em;padding:6px 10px;`
            + `border:1px solid #555;border-radius:6px;background:rgba(0,0,0,0.6);user-select:none;`;
        box.innerHTML = `
            <div><span style="color:#aaa">Type: </span><span id="infoType" style="font-weight:bold;color:#fff"></span></div>
            <div><span style="color:#aaa">ROI: </span><span id="infoRoi" style="font-weight:bold;color:#fff"></span></div>
            <div id="infoSynRow"><span style="color:#aaa">Synapse: </span><span id="infoSynapse" style="font-weight:bold;color:#fff"></span></div>
        `;
        document.body.appendChild(box);
        this.infoBox = box;
        this.infoType = document.getElementById('infoType');
        this.infoRoi = document.getElementById('infoRoi');
    }

    _buildGizmo() {
        const sz = 240, left = SIDEBAR_W + PANEL_PAD;
        const panelBot = PANEL_PAD, hdrH = 28;

        // ── Panel container ────────────────────────────────────────────────
        const panel = document.createElement('div');
        panel.style.cssText = `position:fixed;bottom:${panelBot}px;left:${left}px;width:${sz}px;z-index:100;`;
        document.body.appendChild(panel);
        this._gizmoPanel = panel;

        // Header with collapse toggle
        const hdr = document.createElement('div');
        hdr.style.cssText = `height:${hdrH}px;background:rgba(20,20,20,0.92);border:1px solid #444;`
            + `border-radius:6px 6px 0 0;display:flex;align-items:center;`
            + `justify-content:space-between;padding:0 10px;cursor:pointer;user-select:none;`;
        const hdrLabel = document.createElement('span');
        hdrLabel.textContent = 'Orientation';
        hdrLabel.style.cssText = 'font-size:11px;color:#aaa;font-family:monospace;letter-spacing:0.04em;';
        const collapseBtn = document.createElement('span');
        collapseBtn.textContent = '\u25bc';
        collapseBtn.style.cssText = 'font-size:9px;color:#888;';
        hdr.appendChild(hdrLabel);
        hdr.appendChild(collapseBtn);
        panel.appendChild(hdr);

        // Gizmo body — holds the two canvases; background fills the corners outside the circle
        const gizmoBody = document.createElement('div');
        gizmoBody.style.cssText = `width:${sz}px;height:${sz}px;position:relative;`
            + `border:1px solid #444;border-top:none;border-radius:0 0 6px 6px;overflow:hidden;`
            + `background:${_currentTheme.gizmoBg};`;
        panel.appendChild(gizmoBody);
        this._gizmoBody = gizmoBody;

        // Toggle: collapse/expand the gizmo body and shift the info box accordingly
        let expanded = true;
        hdr.addEventListener('click', () => {
            expanded = !expanded;
            gizmoBody.style.display = expanded ? '' : 'none';
            hdr.style.borderRadius = expanded ? '6px 6px 0 0' : '6px';
            collapseBtn.textContent = expanded ? '\u25bc' : '\u25b6';
            if (this.infoBox) {
                this.infoBox.style.bottom = (panelBot + hdrH + (expanded ? sz : 0) + PANEL_PAD) + 'px';
            }
        });

        // ── Layer 1: WebGL canvas — renders the mini brain ─────────────────
        const wgl = document.createElement('canvas');
        wgl.width = sz; wgl.height = sz;
        wgl.style.cssText = `position:absolute;top:0;left:0;`
            + `width:${sz}px;height:${sz}px;border-radius:50%;`
            + `background:rgba(15,15,15,0.82);pointer-events:none;z-index:0;overflow:hidden;`;
        gizmoBody.appendChild(wgl);
        this._gizmoWebGLCanvas = wgl;

        this._miniRenderer = new THREE.WebGLRenderer({ canvas: wgl, alpha: true, antialias: true });
        this._miniRenderer.setSize(sz, sz);
        this._miniRenderer.setPixelRatio(1);
        this._miniRenderer.setClearColor(0x000000, 0);  // transparent clear

        this._miniScene = new THREE.Scene();
        this._miniCamera = new THREE.PerspectiveCamera(38, 1, 0.001, 10);
        this._miniCamera.up.set(0, -1, 0);
        this._miniCamera.lookAt(0, 0, 0);

        // Ambient + key light (key light position updated each frame to track camera)
        this._miniScene.add(new THREE.AmbientLight(0xffffff, 0.55));
        this._miniKeyLight = new THREE.DirectionalLight(0xffffff, 0.9);
        this._miniScene.add(this._miniKeyLight);

        this._buildMiniBrain();

        // ── Layer 2: 2D canvas — renders axis lines and labels ─────────────
        // pointer-events:none — document-level listeners handle cursor and dblclick
        const el = document.createElement('canvas');
        el.width = sz; el.height = sz;
        el.style.cssText = `position:absolute;top:0;left:0;`
            + `width:${sz}px;height:${sz}px;pointer-events:none;z-index:1;`;
        gizmoBody.appendChild(el);
        this._gizmoCanvas = el;
        this._gizmoCtx = el.getContext('2d');

        // Hit-test helper: returns projected axis entry if client coords are near a label
        const gizmoHit = (clientX, clientY) => {
            if (!this._gizmoProjectedAxes) return null;
            const rect = el.getBoundingClientRect();
            const x = clientX - rect.left, y = clientY - rect.top;
            if (x < 0 || y < 0 || x > sz || y > sz) return null;
            const { cx, cy, projected } = this._gizmoProjectedAxes;
            for (const p of projected) {
                const lx = cx + p.sx * 1.65, ly = cy + p.sy * 1.65;
                if ((x - lx) ** 2 + (y - ly) ** 2 < 18 ** 2) return p;
            }
            return null;
        };

        // ── Drag-to-orbit state ──────────────────────────────────────────────
        let _gizmoDrag = null;    // { x, y } of last drag position
        let _gizmoDragged = false; // true once we've moved >1px (suppresses click)

        // Mousedown on gizmoBody starts orbit drag
        gizmoBody.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            _gizmoDrag = { x: e.clientX, y: e.clientY };
            _gizmoDragged = false;
            e.preventDefault();
        });

        // Mousemove: orbit while dragging, else update cursor
        document.addEventListener('mousemove', (e) => {
            if (_gizmoDrag) {
                // Release drag if button was lifted outside the window (no mouseup fired)
                if (e.buttons === 0) { _gizmoDrag = null; return; }
                const dx = e.clientX - _gizmoDrag.x;
                const dy = e.clientY - _gizmoDrag.y;
                if (Math.abs(dx) > 1 || Math.abs(dy) > 1) _gizmoDragged = true;
                _gizmoDrag = { x: e.clientX, y: e.clientY };
                const cam = this.viewer.scene.camera;
                const ctl = this.viewer.scene.controls;
                const speed = Math.PI / sz;   // half-turn per full panel-width drag
                const offset = new THREE.Vector3().subVectors(cam.position, ctl.target);
                const right = new THREE.Vector3().crossVectors(cam.up, offset).normalize();
                const qH = new THREE.Quaternion().setFromAxisAngle(cam.up, -dx * speed);
                const qV = new THREE.Quaternion().setFromAxisAngle(right,  -dy * speed);
                offset.applyQuaternion(qH).applyQuaternion(qV);
                cam.up.applyQuaternion(qV);
                cam.position.copy(ctl.target).add(offset);
                ctl.update();
                this._updateGizmo();
            } else {
                document.body.style.cursor = gizmoHit(e.clientX, e.clientY) ? 'pointer' : '';
            }
        });

        document.addEventListener('mouseup', () => { _gizmoDrag = null; });

        // Double-click detection via two rapid clicks on the document
        let _gLastClickTime = 0;
        document.addEventListener('click', (e) => {
            const hit = gizmoHit(e.clientX, e.clientY);
            if (!hit || _gizmoDragged) { _gLastClickTime = 0; return; }
            const now = performance.now();
            if (now - _gLastClickTime < 400) {
                _gLastClickTime = 0;
                this._snapAxisToTop(hit.d);
            } else {
                _gLastClickTime = now;
            }
        });

        // Right-click anywhere over the gizmo → flip context menu
        const inGizmo = (clientX, clientY) => {
            const rect = el.getBoundingClientRect();
            const x = clientX - rect.left, y = clientY - rect.top;
            return x >= 0 && y >= 0 && x <= sz && y <= sz;
        };
        document.addEventListener('contextmenu', (e) => {
            if (!inGizmo(e.clientX, e.clientY)) return;
            e.preventDefault();
            this._showGizmoMenu(e.clientX, e.clientY);
        });

        this.viewer.scene.controls.addEventListener('change', () => this._updateGizmo());
        this._updateGizmo();
    }

    _showGizmoMenu(mx, my) {
        const existing = document.getElementById('_gizmoCtxMenu');
        if (existing) existing.remove();

        const BASE = 'background:rgba(30,30,30,0.97);border:1px solid #555;border-radius:4px;'
            + 'overflow:hidden;font-family:monospace;font-size:12px;'
            + 'box-shadow:0 4px 12px rgba(0,0,0,0.6);user-select:none;';
        const ROW  = 'padding:7px 14px;cursor:pointer;color:#fff;white-space:nowrap;';
        const HOV  = 'rgba(255,255,255,0.08)';

        const menu = document.createElement('div');
        menu.id = '_gizmoCtxMenu';
        menu.style.cssText = 'position:fixed;z-index:9999;' + BASE;

        // Active submenu management
        let activeSub = null, subTimer = null;
        const closeSub  = () => { if (activeSub) { activeSub.remove(); activeSub = null; } if (subTimer) { clearTimeout(subTimer); subTimer = null; } };
        const delaySub  = () => { subTimer = setTimeout(closeSub, 150); };
        const cancelSub = () => { if (subTimer) { clearTimeout(subTimer); subTimer = null; } };

        const addDivider = () => {
            const d = document.createElement('div');
            d.style.cssText = 'height:1px;background:#3a3a3a;margin:3px 0;';
            menu.appendChild(d);
        };

        const addItem = (label, action) => {
            const row = document.createElement('div');
            row.textContent = label;
            row.style.cssText = ROW;
            row.onmouseenter = () => { row.style.background = HOV; closeSub(); };
            row.onmouseleave = () => row.style.background = '';
            row.onclick = () => { closeSub(); menu.remove(); action(); };
            menu.appendChild(row);
        };

        const addSubmenu = (label, subitems) => {
            const row = document.createElement('div');
            row.style.cssText = ROW + 'display:flex;justify-content:space-between;align-items:center;gap:20px;';
            const lbl = document.createElement('span'); lbl.textContent = label;
            const arr = document.createElement('span'); arr.textContent = '\u25b8'; arr.style.cssText = 'font-size:10px;opacity:0.5;';
            row.appendChild(lbl); row.appendChild(arr);

            row.onmouseenter = () => {
                row.style.background = HOV;
                cancelSub();
                closeSub();
                const sub = document.createElement('div');
                sub.style.cssText = 'position:fixed;z-index:10000;' + BASE;
                subitems.forEach(({ label: sl, action: sa }) => {
                    const sr = document.createElement('div');
                    sr.textContent = sl;
                    sr.style.cssText = ROW;
                    sr.onmouseenter = () => { sr.style.background = HOV; cancelSub(); };
                    sr.onmouseleave = () => { sr.style.background = ''; delaySub(); };
                    sr.onclick = () => { closeSub(); menu.remove(); sa(); };
                    sub.appendChild(sr);
                });
                document.body.appendChild(sub);
                activeSub = sub;
                const rect = row.getBoundingClientRect();
                sub.style.top = rect.top + 'px';
                const sw = sub.offsetWidth;
                sub.style.left = (rect.right + 2 + sw > window.innerWidth)
                    ? (rect.left - sw - 2) + 'px'
                    : (rect.right + 2) + 'px';
                sub.addEventListener('mouseleave', delaySub);
            };
            row.onmouseleave = () => { row.style.background = ''; delaySub(); };
            menu.appendChild(row);
        };

        // ── Standard Views ──────────────────────────────────────────
        const V = (x, y, z) => new THREE.Vector3(x, y, z);
        addSubmenu('Standard Views', [
            { label: (DATA.axisLabels || {}).zNeg || 'Anterior',  action: () => this._snapToView(V( 0, 0,-1), V(0,-1, 0)) },
            { label: (DATA.axisLabels || {}).zPos || 'Posterior', action: () => this._snapToView(V( 0, 0, 1), V(0,-1, 0)) },
            { label: (DATA.axisLabels || {}).yNeg || 'Dorsal',    action: () => this._snapToView(V( 0,-1, 0), V(0, 0,-1)) },
            { label: (DATA.axisLabels || {}).yPos || 'Ventral',   action: () => this._snapToView(V( 0, 1, 0), V(0, 0,-1)) },
            { label: (DATA.axisLabels || {}).xPos || 'Left',      action: () => this._snapToView(V( 1, 0, 0), V(0,-1, 0)) },
            { label: (DATA.axisLabels || {}).xNeg || 'Right',     action: () => this._snapToView(V(-1, 0, 0), V(0,-1, 0)) },
        ]);

        // ── Invert ──────────────────────────────────────────────────
        addSubmenu('Invert', [
            { label: 'Invert L/R', action: () => this._flipAxis('y') },
            { label: 'Invert D/V', action: () => this._flipAxis('x') },
            { label: 'Invert A/P', action: () => this._flipAxis('z') },
        ]);

        addDivider();

        // ── Utility ─────────────────────────────────────────────────
        addItem('Reset View',  () => { this.viewer.scene.resetCamera(); });
        addItem('Zoom/Pan to Fit', () => this._zoomToFit());

        document.body.appendChild(menu);
        const mw = menu.offsetWidth, mh = menu.offsetHeight;
        menu.style.left = Math.min(mx, window.innerWidth  - mw - 4) + 'px';
        menu.style.top  = Math.min(my, window.innerHeight - mh - 4) + 'px';

        const dismiss = (e) => {
            if (!menu.contains(e.target) && !(activeSub && activeSub.contains(e.target))) {
                closeSub(); menu.remove(); document.removeEventListener('mousedown', dismiss);
            }
        };
        setTimeout(() => document.addEventListener('mousedown', dismiss), 0);
    }

    // Snap camera to a canonical anatomical view (animated)
    _snapToView(dir, up) {
        const cam = this.viewer.scene.camera;
        const ctl = this.viewer.scene.controls;
        const dist = cam.position.distanceTo(ctl.target);
        const startPos = cam.position.clone(), startUp = cam.up.clone();
        const endPos   = ctl.target.clone().add(dir.clone().normalize().multiplyScalar(dist));
        const endUp    = up.clone().normalize();
        const startTime = performance.now(), duration = 400;
        const animate = (now) => {
            const t = Math.min((now - startTime) / duration, 1.0);
            const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
            cam.position.lerpVectors(startPos, endPos, ease);
            cam.up.lerpVectors(startUp, endUp, ease).normalize();
            this._updateGizmo();
            if (t < 1) { requestAnimationFrame(animate); } else { ctl.update(); }
        };
        requestAnimationFrame(animate);
    }

    // Pan + zoom to fit all currently visible rendered objects (no rotation)
    _zoomToFit() {
        const sc  = this.viewer.scene;
        const cam = sc.camera, ctl = sc.controls;
        if (sc.scene) sc.scene.updateMatrixWorld(true);
        const box = new THREE.Box3();
        const tmp = new THREE.Vector3();

        // Some neuron objects are LineSegments2 (fat lines rendered as ShaderMaterial Mesh quads).
        // Their geometry stores segment endpoints in 'instanceStart' / 'instanceEnd' BufferAttributes
        // instead of 'position', so THREE.Box3.setFromObject() returns NaN for them.
        // We handle both cases explicitly.
        const expandWithObj = (obj) => {
            if (!obj || !obj.visible) return;
            let p = obj.parent;
            while (p) { if (!p.visible) return; p = p.parent; }
            const geo = obj.geometry;
            if (!geo) return;
            const mw = obj.matrixWorld;
            const instStart = geo.attributes.instanceStart;
            const instEnd   = geo.attributes.instanceEnd;
            if (instStart && instEnd) {
                // LineSegments2 fat-line: step through instance endpoints (sample every 4th for speed)
                const step = Math.max(1, Math.floor(instStart.count / 512));
                for (let i = 0; i < instStart.count; i += step) {
                    tmp.set(instStart.getX(i), instStart.getY(i), instStart.getZ(i));
                    if (!isNaN(tmp.x)) box.expandByPoint(tmp.applyMatrix4(mw));
                    tmp.set(instEnd.getX(i), instEnd.getY(i), instEnd.getZ(i));
                    if (!isNaN(tmp.x)) box.expandByPoint(tmp.applyMatrix4(mw));
                }
            } else {
                // Standard LineSegments / Mesh with a position attribute
                try {
                    const b = new THREE.Box3().setFromObject(obj);
                    if (!b.isEmpty() && !isNaN(b.min.x)) box.union(b);
                } catch(e) {}
            }
        };
        if (sc.typeRoiGeom)    sc.typeRoiGeom.forEach(expandWithObj);
        if (sc.neuronFullGeom) sc.neuronFullGeom.forEach(expandWithObj);
        if (sc.somaGeom)       sc.somaGeom.forEach(expandWithObj);
        if (sc.neuronMeshGeom) sc.neuronMeshGeom.forEach(expandWithObj);
        if (box.isEmpty() || isNaN(box.min.x)) return;
        const center = new THREE.Vector3(); box.getCenter(center);

        // Project box corners onto camera plane to find tight screen bounds
        const dir = cam.position.clone().sub(ctl.target).normalize();
        const right = new THREE.Vector3().crossVectors(dir, cam.up).normalize();
        const up = cam.up.clone().normalize();
        const corners = [
            new THREE.Vector3(box.min.x, box.min.y, box.min.z),
            new THREE.Vector3(box.max.x, box.min.y, box.min.z),
            new THREE.Vector3(box.min.x, box.max.y, box.min.z),
            new THREE.Vector3(box.max.x, box.max.y, box.min.z),
            new THREE.Vector3(box.min.x, box.min.y, box.max.z),
            new THREE.Vector3(box.max.x, box.min.y, box.max.z),
            new THREE.Vector3(box.min.x, box.max.y, box.max.z),
            new THREE.Vector3(box.max.x, box.max.y, box.max.z),
        ];
        let maxH = 0, maxW = 0;
        for (const c of corners) {
            const rel = c.clone().sub(center);
            maxH = Math.max(maxH, Math.abs(rel.dot(up)));
            maxW = Math.max(maxW, Math.abs(rel.dot(right)));
        }

        const fovRad = cam.fov * Math.PI / 180;
        const aspect = cam.aspect;
        // Distance needed to fit vertically and horizontally
        const distV = maxH / Math.tan(fovRad / 2);
        const distH = maxW / Math.tan(fovRad / 2 * aspect);
        const newDist = Math.max(distV, distH) * 1.05;  // 5% padding
        const startPos = cam.position.clone(), startTgt = ctl.target.clone();
        const endPos   = center.clone().add(dir.clone().multiplyScalar(newDist));
        const startTime = performance.now(), duration = 450;
        const animate = (now) => {
            const t = Math.min((now - startTime) / duration, 1.0);
            const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
            cam.position.lerpVectors(startPos, endPos, ease);
            ctl.target.lerpVectors(startTgt, center, ease);
            this._updateGizmo();
            if (t < 1) { requestAnimationFrame(animate); } else { ctl.update(); }
        };
        requestAnimationFrame(animate);
    }

    // Rotate camera 180° around the given world axis (x/y/z), keeping distance to target
    _flipAxis(axis) {
        const cam = this.viewer.scene.camera;
        const ctl = this.viewer.scene.controls;
        const tgt = ctl.target;
        const axisVec = new THREE.Vector3(
            axis === 'x' ? 1 : 0,
            axis === 'y' ? 1 : 0,
            axis === 'z' ? 1 : 0
        );
        const startRel = cam.position.clone().sub(tgt);  // position relative to target
        const startUp  = cam.up.clone();
        const q = new THREE.Quaternion();
        const startTime = performance.now();
        const duration = 300;
        const animate = (now) => {
            const t = Math.min((now - startTime) / duration, 1.0);
            const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
            // Rotate the starting rel-position and up by the eased angle around the axis
            q.setFromAxisAngle(axisVec, ease * Math.PI);
            cam.position.copy(tgt).add(startRel.clone().applyQuaternion(q));
            cam.up.copy(startUp.clone().applyQuaternion(q)).normalize();
            this._updateGizmo();
            if (t < 1) {
                requestAnimationFrame(animate);
            } else {
                ctl.update();
            }
        };
        requestAnimationFrame(animate);
    }

    // Smoothly reorient camera so the given world axis points to the top of the viewport
    _snapAxisToTop(d) {
        const cam = this.viewer.scene.camera;
        const ctl = this.viewer.scene.controls;
        const targetUp = new THREE.Vector3(d[0], d[1], d[2]).normalize();
        // Skip degenerate case: axis is nearly parallel to the current view direction
        const viewDir = new THREE.Vector3().subVectors(cam.position, ctl.target).normalize();
        if (Math.abs(targetUp.dot(viewDir)) > 0.98) return;
        const startUp = cam.up.clone();
        const startTime = performance.now();
        const duration = 300;
        const animate = (now) => {
            const t = Math.min((now - startTime) / duration, 1.0);
            const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;  // ease in-out quad
            cam.up.lerpVectors(startUp, targetUp, ease).normalize();
            this._updateGizmo();  // keep gizmo in sync visually during animation
            if (t < 1) {
                requestAnimationFrame(animate);
            } else {
                cam.up.copy(targetUp);  // ensure exact final value, no lerp drift
            }
        };
        requestAnimationFrame(animate);
    }

    _buildMiniBrain() {
        if (!BRAIN_HULL_V || !BRAIN_HULL_F) return;
        const qScale = 30000;

        // Decode Int16 vertices
        const vRaw = Uint8Array.from(atob(BRAIN_HULL_V), c => c.charCodeAt(0));
        const vInts = new Int16Array(vRaw.buffer);
        const positions = new Float32Array(vInts.length);
        for (let i = 0; i < vInts.length; i++) positions[i] = vInts[i] / qScale;

        // Decode Uint16 face indices
        const fRaw = Uint8Array.from(atob(BRAIN_HULL_F), c => c.charCodeAt(0));
        const faceIndices = new Uint16Array(fRaw.buffer);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(faceIndices, 1));
        geometry.computeVertexNormals();

        // Semi-transparent brain shell — front-face only so back faces naturally dissolve
        const material = new THREE.MeshLambertMaterial({
            color: 0xd4a017,
            transparent: true,
            opacity: 0.35,
            side: THREE.FrontSide,
            depthWrite: false,
        });

        this._miniBrainMesh = new THREE.Mesh(geometry, material);
        this._miniScene.add(this._miniBrainMesh);

        // Store bounding box — used for clip-plane indicator and mini-renderer clipping
        geometry.computeBoundingBox();
        this._miniBrainBBox = geometry.boundingBox;
    }

    _updateGizmo() {
        const ctx = this._gizmoCtx;
        if (!ctx) return;

        // ── Sync and render mini brain ─────────────────────────────────────
        const scene = this.viewer.scene;
        const cam   = scene.camera;
        const vd    = new THREE.Vector3();
        cam.getWorldDirection(vd);  // direction camera is looking (toward scene)

        if (this._miniCamera && this._miniRenderer) {
            const dist = 2.0;
            this._miniCamera.position.copy(vd).negate().multiplyScalar(dist);
            this._miniCamera.up.copy(cam.up);
            this._miniCamera.lookAt(0, 0, 0);
            this._miniKeyLight.position.copy(this._miniCamera.position);
        }

        // ── Clip mini brain to match main viewer z-section ────────────────
        // Project all 8 bounding-box corners onto vd to find the depth range,
        // then set a matching clipping plane on the mini renderer.
        let nearProj = Infinity, farProj = -Infinity, sliceProj = null;
        if (this._miniBrainBBox) {
            const bb = this._miniBrainBBox;
            for (let ix = 0; ix < 2; ix++) { const px = ix ? bb.max.x : bb.min.x;
            for (let iy = 0; iy < 2; iy++) { const py = iy ? bb.max.y : bb.min.y;
            for (let iz = 0; iz < 2; iz++) { const pz = iz ? bb.max.z : bb.min.z;
                const p = px * vd.x + py * vd.y + pz * vd.z;
                if (p < nearProj) nearProj = p;
                if (p > farProj)  farProj  = p;
            }}}
            if (scene.clipEnabled) {
                sliceProj = nearProj + scene.clipFraction * (farProj - nearProj);
                // THREE.Plane clips where (n·p + c) < 0; constant = –sliceProj keeps far side
                this._miniRenderer.clippingPlanes = [new THREE.Plane(vd.clone(), -sliceProj)];
            } else {
                this._miniRenderer.clippingPlanes = [];
            }
        }

        if (this._miniRenderer) {
            this._miniRenderer.render(this._miniScene, this._miniCamera);
        }

        // Sphere radius (108) is sized so the widest label ("Posterior") never bleeds outside:
        //   arm tip at r*1.65 = 56px, "Posterior" half-width ≈ 27px → max reach 83px < 108
        const W = 240, H = 240, cx = 120, cy = 120, r = 34, sphereR = 108;
        ctx.clearRect(0, 0, W, H);

        // Border only — background is the WebGL canvas below
        ctx.beginPath();
        ctx.arc(cx, cy, sphereR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(80,80,80,0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // ── View frustum box ─────────────────────────────────────────────
        // Main camera and mini brain share the same world-space coords, so no
        // scale conversion is needed. Projects the main frustum rectangle at
        // the look-at target depth onto the gizmo canvas.
        let tCx = cx, tCy = cy, boxHalfW = sphereR * 2, boxHalfH = sphereR * 2;
        if (this._miniCamera && scene.controls) {
            const target      = scene.controls.target;
            const tNDC        = target.clone().project(this._miniCamera);
            tCx = cx + tNDC.x * (W / 2);
            tCy = cy - tNDC.y * (H / 2);

            const distToTarget = cam.position.distanceTo(target);
            const tanHalfMain  = Math.tan(cam.fov * Math.PI / 360);
            const halfH        = distToTarget * tanHalfMain;
            const halfW        = halfH * cam.aspect;

            const miniCamDist  = this._miniCamera.position.distanceTo(target);
            const tanHalfMini  = Math.tan(19 * Math.PI / 180);
            const pxPerUnit    = (W / 2) / (tanHalfMini * Math.max(miniCamDist, 0.01));

            boxHalfW = halfW * pxPerUnit;
            boxHalfH = halfH * pxPerUnit;

            ctx.save();
            ctx.beginPath();
            ctx.arc(cx, cy, sphereR, 0, Math.PI * 2);
            ctx.clip();
            ctx.strokeStyle = 'rgba(180,180,180,0.55)';
            ctx.lineWidth = 1;
            ctx.strokeRect(tCx - boxHalfW, tCy - boxHalfH, boxHalfW * 2, boxHalfH * 2);
            ctx.restore();
        }

        // ── Z-section clip-plane indicator ────────────────────────────────
        // Draw an ellipse whose semi-axes match the brain bounding box's
        // extent along the camera right/up directions at the cut depth.
        // Shrinks to zero at either edge of the brain; changes colour at the
        // midpoint so the user knows they've passed the centre.
        if (scene.clipEnabled && this._miniBrainBBox && sliceProj !== null) {
            const bb  = this._miniBrainBBox;
            const hx  = (bb.max.x - bb.min.x) / 2;
            const hy  = (bb.max.y - bb.min.y) / 2;
            const hz  = (bb.max.z - bb.min.z) / 2;

            // Camera right / up in world space (matrixWorld cols 0 and 1)
            const me = cam.matrixWorld.elements;
            const rx = me[0], ry = me[1], rz = me[2];   // right
            const ux = me[4], uy = me[5], uz = me[6];   // up

            // Maximum half-extents of the bbox cross-section in screen X and Y
            const maxSemiW = Math.abs(hx*rx) + Math.abs(hy*ry) + Math.abs(hz*rz);
            const maxSemiH = Math.abs(hx*ux) + Math.abs(hy*uy) + Math.abs(hz*uz);

            // Cross-section scale: ellipsoid approximation — shrinks toward edges
            const R_along   = (farProj - nearProj) / 2;
            const d_ctr     = sliceProj - (nearProj + farProj) / 2;
            const crossScale = Math.sqrt(Math.max(0, 1 - (d_ctr / Math.max(R_along, 1e-6)) ** 2));

            // Perspective: pixels per world unit at the clip plane's depth from mini cam
            const camToPlane  = 2.0 + sliceProj;
            const tanHalfFov  = Math.tan(19 * Math.PI / 180);
            const pxPerUnit   = (W / 2) / (tanHalfFov * Math.max(camToPlane, 0.1));

            const semiW = Math.min(maxSemiW * pxPerUnit * crossScale, sphereR - 1);
            const semiH = Math.min(maxSemiH * pxPerUnit * crossScale, sphereR - 1);

            if (crossScale > 0.02 && semiW > 1 && semiH > 1) {
                // Orange = front half of brain still visible; blue = past midpoint
                const pastMid = scene.clipFraction > 0.5;
                const fillC = pastMid ? 'rgba(80,160,255,0.07)' : 'rgba(255,165,50,0.07)';

                ctx.save();
                ctx.beginPath();
                ctx.arc(cx, cy, sphereR, 0, Math.PI * 2);
                ctx.clip();
                // Further clip to the frustum box so the ellipse never bleeds outside it
                ctx.beginPath();
                ctx.rect(tCx - boxHalfW, tCy - boxHalfH, boxHalfW * 2, boxHalfH * 2);
                ctx.clip();

                ctx.beginPath();
                ctx.ellipse(cx, cy, semiW, semiH, 0, 0, Math.PI * 2);
                ctx.fillStyle = fillC;
                ctx.fill();
                ctx.restore();
            }
        }

        // Camera basis vectors from matrixWorld columns (world-space)
        // col0=right, col1=up, col2=camera-local+Z (backward — camera looks in -Z)
        const e = this.viewer.scene.camera.matrixWorld.elements;
        const right = [e[0], e[1], e[2]];
        const up    = [e[4], e[5], e[6]];
        const back  = [e[8], e[9], e[10]];
        function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

        // Anatomical axis definitions — read from DATA if available, else use Drosophila defaults
        const al = (DATA.axisLabels || {});
        const axes = [
            { d: [ 1, 0, 0], label: al.xPos || 'Left',      color: '#b05848' },
            { d: [-1, 0, 0], label: al.xNeg || 'Right',     color: '#b05848' },
            { d: [ 0, 1, 0], label: al.yPos || 'Ventral',   color: '#5a9660' },
            { d: [ 0,-1, 0], label: al.yNeg || 'Dorsal',    color: '#5a9660' },
            { d: [ 0, 0,-1], label: al.zNeg || 'Anterior',  color: '#4d6eaa' },
            { d: [ 0, 0, 1], label: al.zPos || 'Posterior', color: '#4d6eaa' },
        ];

        // Project each world axis into screen space via camera basis
        const projected = axes.map(ax => ({
            sx:    dot(ax.d, right) * r,
            sy:   -dot(ax.d, up)   * r,
            depth: dot(ax.d, back),        // >0 = toward viewer, <0 = away
            label: ax.label,
            color: ax.color,
            d:     ax.d,                   // world-space axis vector (for snap-to-top)
        }));
        // Store for double-click hit-testing in _snapAxisToTop
        this._gizmoProjectedAxes = { cx, cy, projected };

        // Render back-to-front (farthest first)
        projected.sort((a, b) => a.depth - b.depth);

        // Center dot
        ctx.beginPath();
        ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#666';
        ctx.globalAlpha = 1.0;
        ctx.fill();

        for (const p of projected) {
            const front = p.depth > 0;
            ctx.globalAlpha = front ? 1.0 : 0.28;

            // Axis line
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + p.sx, cy + p.sy);
            ctx.strokeStyle = p.color;
            ctx.lineWidth = front ? 2.5 : 1.5;
            ctx.stroke();

            // Dot at tip
            ctx.beginPath();
            ctx.arc(cx + p.sx, cy + p.sy, front ? 3.5 : 2, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();

            // Label beyond tip
            ctx.font = 'bold 9px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = front ? (_currentTheme === THEMES.light ? '#222222' : '#dddddd') : '#888888';
            ctx.fillText(p.label, cx + p.sx * 1.65, cy + p.sy * 1.65);
        }
        ctx.globalAlpha = 1.0;
    }

    updateInfoBox(typeName, bodyId, roiName, synapseHit) {
        const synRow = document.getElementById('infoSynRow');
        const synSpan = document.getElementById('infoSynapse');

        if (synapseHit && synapseHit.isSynapse) {
            // Synapse hit — show synapse info, dim type/roi
            if (this.infoType) {
                this.infoType.textContent = '';
            }
            if (this.infoRoi) this.infoRoi.textContent = synapseHit.roi || '';
            if (synSpan) {
                const il = this.viewer.data.instanceLookup || {};
                const preLabel = il[synapseHit.preBid] || synapseHit.preType;
                const postLabel = il[synapseHit.postBid] || synapseHit.postType;
                synSpan.textContent = preLabel + ' \u2192 ' + postLabel;
            }
            return;
        }

        // Normal neuron/ROI hit — clear synapse text
        if (synSpan) synSpan.textContent = '';

        if (this.infoType) {
            if (bodyId) {
                const instance = (this.viewer.data.instanceLookup || {})[bodyId] || bodyId;
                this.infoType.textContent = instance + ' (' + bodyId + ')';
                const color = this.viewer.data.getNeuronColor(bodyId, this.viewer.vis.activeColorMode);
                this.infoType.style.color = color || '#fff';
            } else {
                this.infoType.textContent = typeName || '';
                const color = typeName ? this.viewer.data.getTypeColor(typeName, this.viewer.vis.activeColorMode) : '';
                this.infoType.style.color = color || '#fff';
            }
        }
        if (this.infoRoi) this.infoRoi.textContent = roiName || '';
    }

    clearInfoBox() {
        if (this.infoType) {
            this.infoType.textContent = '';
            this.infoType.style.color = '#fff';
        }
        if (this.infoRoi) this.infoRoi.textContent = '';
        const synSpan = document.getElementById('infoSynapse');
        if (synSpan) synSpan.textContent = '';
    }

    _buildColorbar() {
        const container = document.createElement('div');
        container.style.cssText = `position:fixed;left:${SIDEBAR_W + PANEL_PAD}px;top:${TOP_BAR_H + PANEL_PAD}px;z-index:100;display:none;font-family:sans-serif;user-select:none;`;

        // Outer flex row: histogram on left, right column holds labels + cbar + slider
        const outerRow = document.createElement('div');
        outerRow.style.cssText = 'display:flex;align-items:flex-end;gap:4px;';

        // Histogram canvas (marginal, left of gradient)
        const histCanvas = document.createElement('canvas');
        histCanvas.width = 40;
        histCanvas.height = 200;
        histCanvas.style.cssText = 'display:block;align-self:flex-end;';
        outerRow.appendChild(histCanvas);
        this._histogramCanvas = histCanvas;

        // Right column: label, maxLabel, [canvas + slider], minLabel
        const rightCol = document.createElement('div');
        rightCol.style.cssText = 'display:flex;flex-direction:column;align-items:center;';

        // Label at top (centered over colorbar only)
        const label = document.createElement('div');
        label.style.cssText = 'font-size:10px;color:#ccc;text-align:center;margin-bottom:4px;white-space:pre-line;width:52px;line-height:1.2;';
        rightCol.appendChild(label);

        // Max value label
        const maxLabel = document.createElement('div');
        maxLabel.style.cssText = 'font-size:10px;color:#ccc;text-align:center;margin-bottom:2px;width:52px;';
        rightCol.appendChild(maxLabel);

        // Inner row: canvas + slider track side by side
        const cbarRow = document.createElement('div');
        cbarRow.style.cssText = 'display:flex;align-items:stretch;gap:4px;';

        // Canvas for gradient
        const canvas = document.createElement('canvas');
        canvas.width = 20;
        canvas.height = 200;
        canvas.style.cssText = 'border:1px solid #555;border-radius:2px;display:block;';
        cbarRow.appendChild(canvas);

        // Vertical range slider track (to the right of the colorbar)
        const sliderTrack = document.createElement('div');
        sliderTrack.style.cssText = 'position:relative;width:16px;height:200px;';

        // Track line
        const trackLine = document.createElement('div');
        trackLine.style.cssText = 'position:absolute;left:7px;top:0;width:2px;height:100%;background:#555;border-radius:1px;';
        sliderTrack.appendChild(trackLine);

        // Selected range highlight
        const rangeHighlight = document.createElement('div');
        rangeHighlight.style.cssText = 'position:absolute;left:5px;width:6px;background:rgba(212,160,23,0.6);border-radius:2px;pointer-events:none;';
        sliderTrack.appendChild(rangeHighlight);
        this._cbarRangeHighlight = rangeHighlight;

        // Top handle (max percentile — top of bar = 100%)
        const makeHandle = (isTop) => {
            const h = document.createElement('div');
            h.style.cssText = `position:absolute;left:0;width:16px;height:8px;background:#d4a017;border-radius:2px;cursor:ns-resize;${isTop ? 'top:0px;' : 'bottom:0px;'}`;
            h.title = isTop ? 'Max percentile' : 'Min percentile';
            return h;
        };
        const maxHandle = makeHandle(true);
        const minHandle = makeHandle(false);
        sliderTrack.appendChild(maxHandle);
        sliderTrack.appendChild(minHandle);
        this._cbarMaxHandle = maxHandle;
        this._cbarMinHandle = minHandle;
        this._cbarSliderTrack = sliderTrack;

        // Drag logic for handles
        const trackH = 200;
        const handleH = 8;
        let dragTarget = null, dragStartY = 0, dragStartTop = 0;

        const posToPercent = (topPx) => {
            // top=0 means 100%, top=trackH-handleH means 0%
            return Math.round(100 * (1 - topPx / (trackH - handleH)));
        };
        const percentToPos = (pct) => {
            return (1 - pct / 100) * (trackH - handleH);
        };

        // Initialize positions
        this._cbarMinPct = 0;
        this._cbarMaxPct = 100;
        maxHandle.style.top = percentToPos(100) + 'px';
        minHandle.style.top = percentToPos(0) + 'px';

        const updateRangeHighlight = () => {
            const minTop = parseFloat(minHandle.style.top);
            const maxTop = parseFloat(maxHandle.style.top);
            const top = Math.min(minTop, maxTop) + handleH / 2;
            const bot = Math.max(minTop, maxTop) + handleH / 2;
            rangeHighlight.style.top = top + 'px';
            rangeHighlight.style.height = (bot - top) + 'px';
        };
        updateRangeHighlight();

        const onDragMove = (e) => {
            if (!dragTarget) return;
            const dy = e.clientY - dragStartY;
            let newTop = Math.max(0, Math.min(trackH - handleH, dragStartTop + dy));
            dragTarget.style.top = newTop + 'px';
            // Enforce min <= max
            const maxPos = parseFloat(maxHandle.style.top);
            const minPos = parseFloat(minHandle.style.top);
            if (dragTarget === maxHandle && maxPos > minPos) {
                maxHandle.style.top = minPos + 'px';
            } else if (dragTarget === minHandle && minPos < maxPos) {
                minHandle.style.top = maxPos + 'px';
            }
            updateRangeHighlight();
            // Update percentiles and sync filter panel
            this._cbarMaxPct = posToPercent(parseFloat(maxHandle.style.top));
            this._cbarMinPct = posToPercent(parseFloat(minHandle.style.top));
            this._onColorFilterChange();
        };

        const onDragEnd = () => {
            dragTarget = null;
            document.removeEventListener('mousemove', onDragMove);
            document.removeEventListener('mouseup', onDragEnd);
        };

        for (const handle of [maxHandle, minHandle]) {
            handle.addEventListener('mousedown', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragTarget = handle;
                dragStartY = e.clientY;
                dragStartTop = parseFloat(handle.style.top);
                document.addEventListener('mousemove', onDragMove);
                document.addEventListener('mouseup', onDragEnd);
            });
        }

        // Store percentToPos for external updates
        this._cbarPercentToPos = percentToPos;
        this._cbarUpdateRangeHighlight = updateRangeHighlight;

        cbarRow.appendChild(sliderTrack);
        rightCol.appendChild(cbarRow);
        outerRow.appendChild(rightCol);
        container.appendChild(outerRow);

        // Min value label — sits below outerRow, offset right past histogram to center under colorbar
        const minLabel = document.createElement('div');
        minLabel.style.cssText = 'font-size:10px;color:#ccc;text-align:center;margin-top:2px;margin-left:44px;width:52px;';
        container.appendChild(minLabel);

        document.body.appendChild(container);
        this._colorbarContainer = container;
        this._colorbarCanvas = canvas;
        this._colorbarLabel = label;
        this._colorbarMin = minLabel;
        this._colorbarMax = maxLabel;

        // NT legend container (same position, shown instead of colorbar for NT mode)
        const legend = document.createElement('div');
        legend.style.cssText = `position:fixed;left:${SIDEBAR_W + PANEL_PAD}px;top:${TOP_BAR_H + PANEL_PAD}px;z-index:100;display:none;font-family:sans-serif;user-select:none;`;
        const legendTitle = document.createElement('div');
        legendTitle.style.cssText = 'font-size:11px;color:#ccc;font-weight:bold;margin-bottom:6px;';
        legendTitle.textContent = 'Predicted NT';
        legend.appendChild(legendTitle);
        document.body.appendChild(legend);
        this._ntLegend = legend;
        this._ntLegendTitle = legendTitle;
        this._ntLegendRows = {};  // nt -> {row, swatch, label, active}
    }

    _drawColorbarGradient(stops) {
        const canvas = this._colorbarCanvas;
        const ctx = canvas.getContext('2d');
        const h = canvas.height, w = canvas.width;
        for (let y = 0; y < h; y++) {
            const t = 1.0 - y / (h - 1);
            const [r, g, b] = interpolateColormap(stops, t);
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(0, y, w, 1);
        }
    }

    _updateColorbar(mode) {
        // Hide both first
        this._colorbarContainer.style.display = 'none';
        this._ntLegend.style.display = 'none';
        if (this._histogramCanvas) this._histogramCanvas.style.display = 'none';

        // Save outgoing mode's filter state before reset
        const prevModeIdx = this._lastColorModeIdx !== undefined ? this._lastColorModeIdx : 0;
        const prevMode = this.viewer.data.colorModes[prevModeIdx];
        this._lastColorModeIdx = this.vis.activeColorMode;
        if (prevMode && this._modeFilterState) {
            if (prevMode.nt_legend && this._activeNTs) {
                const allNTs = Object.keys(prevMode.nt_legend);
                const allChecked = this._activeNTs.size >= allNTs.length;
                if (!allChecked) {
                    this._modeFilterState[prevModeIdx] = { type: 'nt', activeNTs: new Set(this._activeNTs) };
                } else {
                    delete this._modeFilterState[prevModeIdx];
                }
            } else if (prevMode.is_scalar) {
                if (this._cbarMinPct !== 0 || this._cbarMaxPct !== 100) {
                    this._modeFilterState[prevModeIdx] = { type: 'scalar', min: this._cbarMinPct, max: this._cbarMaxPct };
                } else {
                    delete this._modeFilterState[prevModeIdx];
                }
            }
        }

        // Reset color filter when switching modes
        this._cbarMinPct = 0;
        this._cbarMaxPct = 100;
        if (this._cbarMinHandle && this._cbarPercentToPos) {
            this._cbarMaxHandle.style.top = this._cbarPercentToPos(100) + 'px';
            this._cbarMinHandle.style.top = this._cbarPercentToPos(0) + 'px';
            this._cbarUpdateRangeHighlight();
        }
        // Reset filter panel inputs
        if (this._filterMinInput) this._filterMinInput.value = '0';
        if (this._filterMaxInput) this._filterMaxInput.value = '100';

        // Reset the actual filter
        this.vis.resetColorFilter();

        if (!mode) return;

        // NT legend mode
        if (mode.nt_legend) {
            // Clear old entries (keep title)
            while (this._ntLegend.children.length > 1) {
                this._ntLegend.removeChild(this._ntLegend.lastChild);
            }
            this._ntLegendRows = {};
            // Initialize active NTs set
            this._activeNTs = new Set(Object.keys(mode.nt_legend));
            // Ensure 'unclear' is always in the legend
            const entries = Object.entries(mode.nt_legend);
            if (!entries.some(([k]) => k === 'unclear')) {
                entries.push(['unclear', 'rgb(140,140,140)']);
                this._activeNTs.add('unclear');
            }
            for (const [nt, color] of entries) {
                const row = document.createElement('div');
                row.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:3px;cursor:pointer;';
                const swatch = document.createElement('div');
                swatch.style.cssText = `width:12px;height:12px;border-radius:2px;background:${color};flex-shrink:0;`;
                if (color === 'rgb(255,255,255)') swatch.style.border = '1px solid #666';
                row.appendChild(swatch);
                const label = document.createElement('div');
                label.style.cssText = 'font-size:11px;color:#ccc;text-transform:capitalize;';
                label.textContent = nt;
                row.appendChild(label);

                // Click to toggle this NT
                row.onclick = () => {
                    if (this._activeNTs.has(nt)) {
                        this._activeNTs.delete(nt);
                        row.style.opacity = '0.35';
                        row.style.textDecoration = 'line-through';
                    } else {
                        this._activeNTs.add(nt);
                        row.style.opacity = '1';
                        row.style.textDecoration = 'none';
                    }
                    // Sync filter panel checkboxes
                    if (this._ntCheckboxes && this._ntCheckboxes[nt]) {
                        this._ntCheckboxes[nt].checked = this._activeNTs.has(nt);
                    }
                    this._onNtFilterChange();
                };

                this._ntLegend.appendChild(row);
                this._ntLegendRows[nt] = { row, swatch, label, color };
            }
            this._ntLegend.style.display = 'block';

            // Update filter panel to show NT checkboxes
            this._updateFilterPanelForMode(mode);

            // Restore saved NT filter state if available
            const ntModeIdx = this.vis.activeColorMode;
            const ntSaved = this._modeFilterState && this._modeFilterState[ntModeIdx];
            if (ntSaved && ntSaved.type === 'nt') {
                this._activeNTs = new Set(ntSaved.activeNTs);
                // Update legend row and checkbox styling
                for (const [nt, info] of Object.entries(this._ntLegendRows)) {
                    const active = this._activeNTs.has(nt);
                    info.row.style.opacity = active ? '1' : '0.35';
                    info.row.style.textDecoration = active ? 'none' : 'line-through';
                    if (this._ntCheckboxes && this._ntCheckboxes[nt]) {
                        this._ntCheckboxes[nt].checked = active;
                    }
                }
                this._onNtFilterChange();
            }
            return;
        }

        // Non-scalar, non-NT mode — show categorical legend if available
        if (!mode.is_scalar) {
            if (mode._catLegend && Object.keys(mode._catLegend).length > 0) {
                // Show categorical legend using NT legend container
                while (this._ntLegend.children.length > 1) this._ntLegend.removeChild(this._ntLegend.lastChild);
                this._ntLegendTitle.textContent = mode.name || mode.label || 'Legend';
                this._ntLegendRows = {};
                for (const [cat, color] of Object.entries(mode._catLegend)) {
                    const row = document.createElement('div');
                    row.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:3px;cursor:default;';
                    const swatch = document.createElement('div');
                    swatch.style.cssText = `width:12px;height:12px;border-radius:2px;background:${color};flex-shrink:0;`;
                    row.appendChild(swatch);
                    const label = document.createElement('div');
                    label.style.cssText = 'font-size:11px;color:#ccc;';
                    label.textContent = cat;
                    row.appendChild(label);
                    this._ntLegend.appendChild(row);
                    this._ntLegendRows[cat] = { row, swatch, label, color };
                }
                this._ntLegend.style.display = 'block';
            }
            this._updateFilterPanelForMode(mode);
            return;
        }

        // Scalar mode with no filterable numeric data — show colorbar if has colorscale
        if (!mode.type_values && !mode._sortedValues) {
            this._updateFilterPanelForMode(mode);
            return;
        }

        this._colorbarLabel.textContent = mode.label || '';
        const fmtCbar = (v) => {
            if (v === 0) return '0';
            if (Math.abs(v) < 0.001 || Math.abs(v) >= 1e6) return v.toExponential(1);
            return v.toFixed(3);
        };
        this._colorbarMin.textContent = fmtCbar(mode.cmin);
        this._colorbarMax.textContent = fmtCbar(mode.cmax);

        // Draw gradient — use custom colormap if one was applied, else original colorscale
        this._drawColorbarGradient(mode._activeColorscale || mode.colorscale);
        this._colorbarContainer.style.display = 'block';

        // Show/hide slider track based on whether mode has type_values
        if (this._cbarSliderTrack) {
            this._cbarSliderTrack.style.display = mode._sortedValues ? 'block' : 'none';
        }

        // Update filter panel for scalar mode
        this._updateFilterPanelForMode(mode);

        // Restore saved scalar filter state if available
        const scalarModeIdx = this.vis.activeColorMode;
        const scalarSaved = this._modeFilterState && this._modeFilterState[scalarModeIdx];
        if (scalarSaved && scalarSaved.type === 'scalar') {
            this._cbarMinPct = scalarSaved.min;
            this._cbarMaxPct = scalarSaved.max;
            if (this._filterMinInput) this._filterMinInput.value = scalarSaved.min;
            if (this._filterMaxInput) this._filterMaxInput.value = scalarSaved.max;
            if (this._cbarMinHandle && this._cbarPercentToPos) {
                this._cbarMinHandle.style.top = this._cbarPercentToPos(scalarSaved.min) + 'px';
                this._cbarMaxHandle.style.top = this._cbarPercentToPos(scalarSaved.max) + 'px';
                this._cbarUpdateRangeHighlight();
            }
            this._onColorFilterChange();
        } else {
            // No saved filter — still draw the histogram for the fresh mode
            this._drawHistogram();
        }
    }

    // Called when colorbar slider handles are dragged (scalar modes)
    _onColorFilterChange() {
        // Sync filter panel inputs
        if (this._filterMinInput) this._filterMinInput.value = this._cbarMinPct;
        if (this._filterMaxInput) this._filterMaxInput.value = this._cbarMaxPct;
        // Apply filter (also updates highlightedSet)
        this.vis.applyColorFilter(this._cbarMinPct, this._cbarMaxPct, null);
        this._drawColorbarOverlay();
        this.syncAllState();  // Update borders, ROI counts, and sidebar filter styling
    }

    // Called when NT legend items are toggled
    _onNtFilterChange() {
        const mode = this.viewer.data.colorModes[this.vis.activeColorMode];
        const allNTs = mode && mode.nt_legend ? Object.keys(mode.nt_legend) : [];
        const allChecked = this._activeNTs && this._activeNTs.size >= allNTs.length;

        // Auto-switch to Neuron mode when any NT is unchecked (mixed types need per-neuron view)
        if (!allChecked && !this.hlModeByNeuron && this._switchMode) {
            this._ntAutoSwitchedNeuron = true;
            this._switchMode(true, true);  // isAuto = true — won't clear the flag
        }
        // Auto-restore Type mode when all NTs are re-checked (conflicts resolved)
        if (allChecked && this._ntAutoSwitchedNeuron && this.hlModeByNeuron && this._switchMode) {
            this._ntAutoSwitchedNeuron = false;
            this._switchMode(false, true);
        }

        this.vis.applyColorFilter(0, 100, this._activeNTs);
        this.syncAllState();  // Update borders, ROI counts, and sidebar filter styling
    }

    // Draw semi-transparent overlay on colorbar for filtered-out regions
    _drawColorbarOverlay() {
        const canvas = this._colorbarCanvas;
        if (!canvas) return;
        const mode = this.viewer.data.colorModes[this.vis.activeColorMode];
        if (!mode || !mode.is_scalar) return;

        // Redraw the gradient first
        const ctx = canvas.getContext('2d');
        const h = canvas.height, w = canvas.width;
        const cs = mode.colorscale;
        if (!cs) return;
        const parseRgb = (s) => s.match(/\d+/g).map(Number);
        for (let y = 0; y < h; y++) {
            const t = 1.0 - y / (h - 1);
            let lo = 0, hi = cs.length - 1;
            for (let i = 0; i < cs.length - 1; i++) {
                if (cs[i][0] <= t && cs[i + 1][0] >= t) { lo = i; hi = i + 1; break; }
            }
            const range = cs[hi][0] - cs[lo][0];
            const frac = range > 0 ? (t - cs[lo][0]) / range : 0;
            const [r0, g0, b0] = parseRgb(cs[lo][1]);
            const [r1, g1, b1] = parseRgb(cs[hi][1]);
            ctx.fillStyle = `rgb(${Math.round(r0 + (r1 - r0) * frac)},${Math.round(g0 + (g1 - g0) * frac)},${Math.round(b0 + (b1 - b0) * frac)})`;
            ctx.fillRect(0, y, w, 1);
        }

        // Draw dark overlay on filtered-out regions
        if (this._cbarMinPct > 0 || this._cbarMaxPct < 100) {
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            // Bottom region (below min percentile): y from minPctY to h
            const minY = Math.round((1 - this._cbarMinPct / 100) * (h - 1));
            const maxY = Math.round((1 - this._cbarMaxPct / 100) * (h - 1));
            if (minY < h) ctx.fillRect(0, minY, w, h - minY);
            // Top region (above max percentile): y from 0 to maxPctY
            if (maxY > 0) ctx.fillRect(0, 0, w, maxY);
        }

        this._drawHistogram();
    }

    _drawHistogram() {
        const hc = this._histogramCanvas;
        if (!hc) return;
        const mode = this.viewer.data.colorModes[this.vis.activeColorMode];
        if (!mode || !mode.is_scalar || !mode.type_values) {
            hc.style.display = 'none';
            return;
        }
        hc.style.display = 'block';

        const ctx = hc.getContext('2d');
        const H = hc.height, W = hc.width;
        ctx.clearRect(0, 0, W, H);

        // Build histogram — 20 bins evenly spread across the value range
        const N_BINS = 20;
        const binH = H / N_BINS;
        const values = Object.values(mode.type_values);
        const cmin = mode.cmin, cmax = mode.cmax;
        const counts = new Array(N_BINS).fill(0);
        for (const v of values) {
            const t = cmax > cmin ? (v - cmin) / (cmax - cmin) : 0.5;
            counts[Math.min(N_BINS - 1, Math.floor(t * N_BINS))]++;
        }
        const maxCount = Math.max(...counts, 1);

        // Draw bars — colors match gradient at bin midpoint; bars extend left from right edge
        const cs = mode._activeColorscale || mode.colorscale;
        if (!cs) return;
        const parseRgb = s => s.match(/\d+/g).map(Number);
        for (let b = 0; b < N_BINS; b++) {
            if (counts[b] === 0) continue;
            const barW = Math.round((counts[b] / maxCount) * (W - 2));
            const t = (b + 0.5) / N_BINS;   // b=0 → bottom of canvas = low values = t≈0
            let lo = 0, hi = cs.length - 1;
            for (let i = 0; i < cs.length - 1; i++) {
                if (cs[i][0] <= t && cs[i + 1][0] >= t) { lo = i; hi = i + 1; break; }
            }
            const range = cs[hi][0] - cs[lo][0];
            const frac  = range > 0 ? (t - cs[lo][0]) / range : 0;
            const [r0, g0, b0] = parseRgb(cs[lo][1]);
            const [r1, g1, b1] = parseRgb(cs[hi][1]);
            const r  = Math.round(r0 + (r1 - r0) * frac);
            const g  = Math.round(g0 + (g1 - g0) * frac);
            const bv = Math.round(b0 + (b1 - b0) * frac);
            const y = H - (b + 1) * binH;
            ctx.fillStyle = `rgba(${r},${g},${bv},0.82)`;
            ctx.fillRect(W - barW, y, barW, binH - 1);
        }

        // Apply same filtered-out overlay as the gradient canvas
        if (this._cbarMinPct > 0 || this._cbarMaxPct < 100) {
            ctx.fillStyle = 'rgba(0,0,0,0.7)';
            const minY = Math.round((1 - this._cbarMinPct / 100) * (H - 1));
            const maxY = Math.round((1 - this._cbarMaxPct / 100) * (H - 1));
            if (minY < H) ctx.fillRect(0, minY, W, H - minY);
            if (maxY > 0) ctx.fillRect(0, 0, W, maxY);
        }
    }

    // Apply sidebar color filter styling (gray out, strikethrough, sort to bottom)
    _applySidebarColorFilter() {
        const filteredTypes = this.vis.colorFilteredOutTypes;
        const filteredNeurons = this.vis.colorFilteredOutNeurons;

        // Determine if we're in neuron mode
        const panelKeys = Object.keys(this.typeRows);
        if (panelKeys.length === 0) return;

        const isNeuronMode = panelKeys.some(k => this.viewer.data.neuronType[k]);

        if (isNeuronMode) {
            // Re-append type groups: types with passing neurons first, all-filtered types last
            const headers = this._neuronTypeHeaders || {};
            const passingTypes = [];
            const filteredOutTypes = [];
            for (const typeName of this.data.allTypes) {
                const bids = this.data.getNeuronsForType(typeName);
                if (!bids.length || !headers[typeName]) continue;
                const allOut = bids.every(b => filteredNeurons.has(b));
                if (allOut) filteredOutTypes.push(typeName);
                else passingTypes.push(typeName);
            }
            // Append passing types first, then filtered-out types
            for (const group of [passingTypes, filteredOutTypes]) {
                const isFilteredGroup = group === filteredOutTypes;
                for (const typeName of group) {
                    const hdr = headers[typeName];
                    hdr.style.opacity = isFilteredGroup ? '0.35' : '1';
                    hdr.style.textDecoration = isFilteredGroup ? 'line-through' : 'none';
                    if (isFilteredGroup) hdr.style.border = 'none';
                    this.panelContent.appendChild(hdr);
                    const bids = this.data.getNeuronsForType(typeName);
                    // Within each type: passing neurons first, filtered last
                    const passing = bids.filter(b => !filteredNeurons.has(b));
                    const filtered = bids.filter(b => filteredNeurons.has(b));
                    for (const bid of [...passing, ...filtered]) {
                        const row = this.typeRows[bid];
                        if (!row) continue;
                        const isOut = filteredNeurons.has(bid);
                        row.style.opacity = isOut ? '0.35' : '1';
                        row.style.textDecoration = isOut ? 'line-through' : 'none';
                        if (isOut) row.style.border = 'none';
                        this.panelContent.appendChild(row);
                    }
                }
            }
        } else {
            // Type mode: sort types
            const allTypes = this.data.allTypes;
            const sorted = allTypes.slice().sort((a, b) => {
                const aOut = filteredTypes.has(a) ? 1 : 0;
                const bOut = filteredTypes.has(b) ? 1 : 0;
                return aOut - bOut;
            });
            for (const typeName of sorted) {
                const row = this.typeRows[typeName];
                if (!row) continue;
                this.panelContent.appendChild(row);
                if (filteredTypes.has(typeName)) {
                    row.style.opacity = '0.35';
                    row.style.textDecoration = 'line-through';
                    row.style.border = 'none';
                } else {
                    row.style.opacity = '1';
                    row.style.textDecoration = 'none';
                }
            }
        }
    }

    _buildColorFilterPanel() {
        const panel = document.createElement('div');
        // Position to the left of the colorbar
        panel.style.cssText = `position:fixed;left:${SIDEBAR_W + PANEL_PAD}px;top:${TOP_BAR_H + PANEL_PAD}px;z-index:101;background:rgba(20,20,20,0.92);border:1px solid #444;border-radius:6px;padding:0;font-size:11px;font-family:monospace;color:#ccc;min-width:160px;display:none;`;

        // Header with collapse toggle
        const hdr = document.createElement('div');
        hdr.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 8px;cursor:pointer;user-select:none;border-bottom:1px solid #333;';
        const hdrLabel = document.createElement('span');
        hdrLabel.textContent = 'Color Filter';
        hdrLabel.style.cssText = 'font-weight:bold;font-size:11px;color:#fff;';
        const collapseBtn = document.createElement('span');
        collapseBtn.textContent = '\u25BC';
        collapseBtn.style.cssText = 'font-size:9px;color:#888;';
        hdr.appendChild(hdrLabel);
        hdr.appendChild(collapseBtn);
        panel.appendChild(hdr);

        const body = document.createElement('div');
        body.style.cssText = 'padding:6px 8px;';

        // Mode label (read-only)
        const modeLabel = document.createElement('div');
        modeLabel.style.cssText = 'font-size:10px;color:#aaa;margin-bottom:4px;';
        modeLabel.textContent = 'Mode: —';
        body.appendChild(modeLabel);
        this._filterModeLabel = modeLabel;

        // Scalar filter inputs container
        const scalarSection = document.createElement('div');
        scalarSection.style.cssText = 'display:none;';

        const inputStyle = 'width:50px;background:#111;border:1px solid #555;color:#fff;text-align:right;font-family:monospace;font-size:11px;padding:2px 4px;border-radius:2px;';

        // Max percentile (top, matching colorbar orientation)
        const maxRow = document.createElement('div');
        maxRow.style.cssText = 'display:flex;align-items:center;gap:4px;margin-bottom:3px;';
        const maxLbl = document.createElement('span');
        maxLbl.textContent = 'Max %:';
        maxLbl.style.cssText = 'color:#888;font-size:10px;';
        const maxInp = document.createElement('input');
        maxInp.type = 'text';
        maxInp.value = '100';
        maxInp.style.cssText = inputStyle;
        maxRow.appendChild(maxLbl);
        maxRow.appendChild(maxInp);
        scalarSection.appendChild(maxRow);
        this._filterMaxInput = maxInp;

        // Min percentile (bottom, matching colorbar orientation)
        const minRow = document.createElement('div');
        minRow.style.cssText = 'display:flex;align-items:center;gap:4px;margin-bottom:3px;';
        const minLbl = document.createElement('span');
        minLbl.textContent = 'Min %:';
        minLbl.style.cssText = 'color:#888;font-size:10px;';
        const minInp = document.createElement('input');
        minInp.type = 'text';
        minInp.value = '0';
        minInp.style.cssText = inputStyle;
        minRow.appendChild(minLbl);
        minRow.appendChild(minInp);
        scalarSection.appendChild(minRow);
        this._filterMinInput = minInp;

        // Apply filter on input change
        const applyScalarFilter = () => {
            let minV = parseInt(minInp.value) || 0;
            let maxV = parseInt(maxInp.value) || 100;
            minV = Math.max(0, Math.min(100, minV));
            maxV = Math.max(0, Math.min(100, maxV));
            if (minV > maxV) { const tmp = minV; minV = maxV; maxV = tmp; }
            minInp.value = minV;
            maxInp.value = maxV;
            this._cbarMinPct = minV;
            this._cbarMaxPct = maxV;
            // Sync colorbar handles
            if (this._cbarPercentToPos) {
                this._cbarMaxHandle.style.top = this._cbarPercentToPos(maxV) + 'px';
                this._cbarMinHandle.style.top = this._cbarPercentToPos(minV) + 'px';
                this._cbarUpdateRangeHighlight();
            }
            this.vis.applyColorFilter(minV, maxV, null);
            this._applySidebarColorFilter();
            this._drawColorbarOverlay();
        };
        minInp.addEventListener('change', applyScalarFilter);
        maxInp.addEventListener('change', applyScalarFilter);

        body.appendChild(scalarSection);
        this._filterScalarSection = scalarSection;

        // NT filter section (checkboxes)
        const ntSection = document.createElement('div');
        ntSection.style.cssText = 'display:none;';
        body.appendChild(ntSection);
        this._filterNtSection = ntSection;
        this._ntCheckboxes = {};

        panel.appendChild(body);
        document.body.appendChild(panel);
        this._colorFilterPanel = panel;
        this._colorFilterPanelBody = body;

        // Collapse toggle
        let collapsed = false;
        hdr.onclick = () => {
            collapsed = !collapsed;
            body.style.display = collapsed ? 'none' : 'block';
            collapseBtn.textContent = collapsed ? '\u25B6' : '\u25BC';
        };
    }

    // Update the filter panel contents when color mode changes
    _updateFilterPanelForMode(mode) {
        if (!this._colorFilterPanel) return;

        this._filterModeLabel.textContent = 'Mode: ' + (mode ? mode.name : '—');

        if (mode && mode.nt_legend) {
            // NT mode: show checkboxes
            this._filterScalarSection.style.display = 'none';
            this._filterNtSection.style.display = 'block';
            this._filterNtSection.innerHTML = '';
            this._ntCheckboxes = {};

            const allNTs = Object.keys(mode.nt_legend);
            if (!allNTs.includes('unclear')) allNTs.push('unclear');

            for (const nt of allNTs) {
                const row = document.createElement('div');
                row.style.cssText = 'display:flex;align-items:center;gap:4px;margin-bottom:2px;';
                const cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.checked = true;
                cb.style.cssText = 'cursor:pointer;';
                const lbl = document.createElement('span');
                lbl.textContent = nt;
                lbl.style.cssText = 'font-size:10px;color:#ccc;text-transform:capitalize;cursor:pointer;';
                row.appendChild(cb);
                row.appendChild(lbl);

                cb.onchange = () => {
                    if (cb.checked) {
                        this._activeNTs.add(nt);
                    } else {
                        this._activeNTs.delete(nt);
                    }
                    // Sync legend row
                    const legendRow = this._ntLegendRows[nt];
                    if (legendRow) {
                        legendRow.row.style.opacity = cb.checked ? '1' : '0.35';
                        legendRow.row.style.textDecoration = cb.checked ? 'none' : 'line-through';
                    }
                    this._onNtFilterChange();
                };

                lbl.onclick = () => { cb.click(); };
                this._filterNtSection.appendChild(row);
                this._ntCheckboxes[nt] = cb;
            }

            this._showColorFilterPanel();
        } else if (mode && mode._sortedValues) {
            // Scalar mode: show min/max inputs
            this._filterScalarSection.style.display = 'block';
            this._filterNtSection.style.display = 'none';
            this._showColorFilterPanel();
        } else {
            // No filtering available
            this._colorFilterPanel.style.display = 'none';
        }
    }

    // Show the color-filter panel, positioned just to the right of the visible colorbar element
    _showColorFilterPanel() {
        const cEl = (this._colorbarContainer && this._colorbarContainer.style.display !== 'none')
            ? this._colorbarContainer
            : (this._ntLegend && this._ntLegend.style.display !== 'none')
                ? this._ntLegend : null;
        if (cEl) {
            const r = cEl.getBoundingClientRect();
            this._colorFilterPanel.style.left = (r.right + PANEL_PAD * 3) + 'px';
        }
        this._colorFilterPanel.style.display = 'block';
    }

    _buildConnPanel() {
        const CONN_PANEL_W = 500;
        const panel = document.createElement('div');
        panel.style.cssText = `position:fixed;bottom:${PANEL_PAD}px;right:${TYPE_PANEL_W + PANEL_PAD}px;width:${CONN_PANEL_W}px;height:${CONN_PANEL_H};background:rgba(20,20,20,0.92);color:#ccc;z-index:998;display:flex;flex-direction:column;font-family:monospace;font-size:11px;border:1px solid #444;border-radius:6px;box-sizing:border-box;user-select:none;`;

        // Title bar with collapse toggle
        const titleBar = document.createElement('div');
        titleBar.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 8px;flex-shrink:0;cursor:pointer;user-select:none;border-bottom:1px solid #333;';
        this.connTitle = document.createElement('div');
        this.connTitle.style.cssText = 'font-weight:bold;font-size:11px;color:#fff;flex:1;';
        titleBar.appendChild(this.connTitle);

        const exportCsvBtn = document.createElement('span');
        exportCsvBtn.textContent = '\u2193 CSV';
        exportCsvBtn.title = 'Export connections as CSV';
        exportCsvBtn.style.cssText = 'font-size:10px;color:#8cb4d8;cursor:pointer;padding:1px 6px;border:1px solid #445;border-radius:3px;margin-right:6px;flex-shrink:0;';
        exportCsvBtn.onmouseenter = () => { exportCsvBtn.style.color = '#fff'; exportCsvBtn.style.borderColor = '#8cb4d8'; };
        exportCsvBtn.onmouseleave = () => { exportCsvBtn.style.color = '#8cb4d8'; exportCsvBtn.style.borderColor = '#445'; };
        exportCsvBtn.onclick = (e) => { e.stopPropagation(); this._exportConnCsv(); };
        titleBar.appendChild(exportCsvBtn);

        const collapseBtn = document.createElement('span');
        collapseBtn.textContent = '\u25BC';
        collapseBtn.style.cssText = 'font-size:9px;color:#888;cursor:pointer;padding:0 4px;';
        collapseBtn.title = 'Collapse/expand';
        titleBar.appendChild(collapseBtn);
        panel.appendChild(titleBar);

        // Collapsible body
        const connBody = document.createElement('div');
        connBody.style.cssText = 'display:flex;flex-direction:column;flex:1;overflow:hidden;';

        // Dropdown to select which highlighted item
        const ddWrap = document.createElement('div');
        ddWrap.style.cssText = 'padding:0 12px 4px 12px;flex-shrink:0;';
        this.connDropdown = document.createElement('select');
        this.connDropdown.style.cssText = 'width:100%;padding:3px 6px;background:#222;color:#ccc;border:1px solid #555;border-radius:3px;font-size:11px;cursor:pointer;';
        this.connDropdown.addEventListener('change', () => {
            const key = this.connDropdown.value;
            if (key && this.vis.highlightedSet.has(key)) {
                this.connSelectedKey = key;
                this.viewer.connSelectedKey = key;
                this._updateConnPanel();
                this.syncAllState();
            }
        });
        ddWrap.appendChild(this.connDropdown);
        connBody.appendChild(ddWrap);

        // Columns container
        const connColumns = document.createElement('div');
        connColumns.style.cssText = 'display:flex;flex:1;overflow:hidden;';

        // Upstream column
        this.connUpCol = document.createElement('div');
        this.connUpCol.style.cssText = 'flex:1;overflow-y:auto;padding:0 10px 8px 10px;border-right:1px solid #333;';
        connColumns.appendChild(this.connUpCol);

        // Downstream column
        this.connDnCol = document.createElement('div');
        this.connDnCol.style.cssText = 'flex:1;overflow-y:auto;padding:0 10px 8px 10px;';
        connColumns.appendChild(this.connDnCol);

        connBody.appendChild(connColumns);
        panel.appendChild(connBody);

        // Collapse toggle
        let collapsed = false;
        titleBar.onclick = () => {
            collapsed = !collapsed;
            connBody.style.display = collapsed ? 'none' : 'flex';
            panel.style.height = collapsed ? 'auto' : CONN_PANEL_H;
            collapseBtn.textContent = collapsed ? '\u25B6' : '\u25BC';
        };

        document.body.appendChild(panel);
        this.connPanel = panel;
        this.CONN_PANEL_W = CONN_PANEL_W;
        this._clearConnPanel();
    }

    _buildSynapsePanel() {
        const SYN_PANEL_W = 280;
        const connW = 500;
        const panel = document.createElement('div');
        panel.style.cssText = `position:fixed;bottom:${PANEL_PAD}px;right:${TYPE_PANEL_W + PANEL_PAD + connW + 4}px;width:${SYN_PANEL_W}px;height:${CONN_PANEL_H};background:rgba(20,20,20,0.92);color:#ccc;z-index:998;display:flex;flex-direction:column;font-family:monospace;font-size:11px;border:1px solid #444;border-radius:6px;box-sizing:border-box;user-select:none;`;

        // Title bar
        const titleBar = document.createElement('div');
        titleBar.style.cssText = 'display:flex;align-items:center;justify-content:space-between;padding:5px 8px;flex-shrink:0;cursor:pointer;user-select:none;border-bottom:1px solid #333;';

        const titleText = document.createElement('div');
        titleText.textContent = 'Synapse Groups';
        titleText.style.cssText = 'font-weight:bold;font-size:11px;color:#fff;flex:1;';
        titleBar.appendChild(titleText);

        // Synapse CSV upload button
        const synUploadBtn = document.createElement('span');
        synUploadBtn.textContent = '+';
        synUploadBtn.dataset.tip = 'Upload synapse CSV';
        synUploadBtn.style.cssText = 'font-size:14px;font-weight:bold;color:#aaa;cursor:pointer;padding:0 6px;';
        synUploadBtn.onmouseenter = () => { synUploadBtn.style.color = '#fff'; };
        synUploadBtn.onmouseleave = () => { synUploadBtn.style.color = '#aaa'; };
        synUploadBtn.onclick = (e) => {
            e.stopPropagation();
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.csv';
            input.onchange = (ev) => {
                const file = ev.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (re) => this._handleSynapseCSVUpload(re.target.result, file.name);
                reader.readAsText(file);
            };
            input.click();
        };
        titleBar.appendChild(synUploadBtn);

        const collapseBtn = document.createElement('span');
        collapseBtn.textContent = '\u25BC';
        collapseBtn.style.cssText = 'font-size:9px;color:#888;cursor:pointer;padding:0 4px;';
        titleBar.appendChild(collapseBtn);
        panel.appendChild(titleBar);

        // Body
        const body = document.createElement('div');
        body.style.cssText = 'display:flex;flex-direction:column;flex:1;overflow:hidden;';

        // Size slider
        const sliderWrap = document.createElement('div');
        sliderWrap.style.cssText = 'padding:6px 10px;display:flex;align-items:center;gap:6px;flex-shrink:0;border-bottom:1px solid #333;';
        const sliderLabel = document.createElement('span');
        sliderLabel.textContent = 'Size:';
        sliderLabel.style.cssText = 'font-size:10px;color:#aaa;';
        sliderWrap.appendChild(sliderLabel);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0.1';
        slider.max = '5';
        slider.step = '0.05';
        slider.value = '2';
        slider.style.cssText = 'flex:1;height:12px;cursor:pointer;accent-color:#d4a017;';
        slider.addEventListener('input', () => {
            const radius = parseFloat(slider.value) * 0.001;
            this.viewer.synapse.setGlobalSize(radius);
        });
        sliderWrap.appendChild(slider);
        body.appendChild(sliderWrap);

        // Scrollable group list
        const groupList = document.createElement('div');
        groupList.style.cssText = 'flex:1;overflow-y:auto;padding:4px 0;';
        body.appendChild(groupList);

        panel.appendChild(body);

        // Collapse toggle
        let collapsed = false;
        titleBar.onclick = () => {
            collapsed = !collapsed;
            body.style.display = collapsed ? 'none' : 'flex';
            panel.style.height = collapsed ? 'auto' : CONN_PANEL_H;
            collapseBtn.textContent = collapsed ? '\u25B6' : '\u25BC';
        };

        document.body.appendChild(panel);
        this.synPanel = panel;
        this.synGroupList = groupList;
    }

    _updateSynapsePanel() {
        const groups = this.viewer.synapse.groups;

        this.synGroupList.innerHTML = '';
        if (groups.length === 0) {
            const empty = document.createElement('div');
            empty.textContent = 'No synapse groups. Click + to upload a CSV.';
            empty.style.cssText = 'padding:12px 10px;color:#666;font-size:10px;text-align:center;';
            this.synGroupList.appendChild(empty);
            return;
        }
        for (const g of groups) {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:4px;padding:3px 8px;font-size:10px;';

            // Eye toggle
            const eye = document.createElement('span');
            eye.textContent = g.visible ? '\u25C9' : '\u25CB';
            eye.style.cssText = 'cursor:pointer;font-size:12px;width:16px;text-align:center;flex-shrink:0;';
            eye.title = 'Toggle visibility';
            eye.addEventListener('click', () => {
                this.viewer.synapse.toggleVisibility(g.id);
                this._updateSynapsePanel();
            });
            row.appendChild(eye);

            // Color swatch
            const swatch = document.createElement('span');
            swatch.style.cssText = `width:12px;height:12px;border-radius:2px;cursor:pointer;flex-shrink:0;background:${g.color};border:1px solid #666;`;
            swatch.title = 'Click to change color';
            swatch.addEventListener('click', (e) => {
                e.stopPropagation();
                this._showSynapseColorPicker(e, g);
            });
            row.appendChild(swatch);

            // Label
            const label = document.createElement('span');
            label.textContent = g.label;
            label.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#ccc;';
            label.title = g.label;
            row.appendChild(label);

            // Count
            const cnt = document.createElement('span');
            cnt.textContent = `[${g.count}]`;
            cnt.style.cssText = 'color:#888;flex-shrink:0;font-size:9px;';
            row.appendChild(cnt);

            // Remove button
            const removeBtn = document.createElement('span');
            removeBtn.textContent = '\u00D7';
            removeBtn.style.cssText = 'cursor:pointer;color:#888;font-size:14px;flex-shrink:0;padding:0 2px;';
            removeBtn.title = 'Remove synapse group';
            removeBtn.addEventListener('click', () => {
                this.viewer.synapse.removeGroup(g.id);
                this._updateSynapsePanel();
            });
            row.appendChild(removeBtn);

            // Right-click for split by instance
            row.addEventListener('contextmenu', (ev) => {
                ev.preventDefault();
                ev.stopPropagation();
                this._showSynapseGroupContextMenu(ev, g);
            });

            this.synGroupList.appendChild(row);
        }
    }

    _showSynapseGroupContextMenu(e, group) {
        // Only offer split if the group represents a type with multiple neurons
        const bids = this.data.typeNeurons[group.ourType] || [];
        if (bids.length <= 1 || group._splitBid) return;  // Already split or single neuron

        const menu = document.createElement('div');
        menu.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;z-index:10001;background:#1a1a1a;border:1px solid #555;border-radius:4px;padding:4px 0;box-shadow:0 4px 12px rgba(0,0,0,0.5);`;

        const item = document.createElement('div');
        item.textContent = 'Split by instance';
        item.style.cssText = 'padding:6px 16px;font-size:12px;color:#ccc;cursor:pointer;white-space:nowrap;';
        item.onmouseenter = () => { item.style.background = '#333'; };
        item.onmouseleave = () => { item.style.background = 'transparent'; };
        item.addEventListener('click', () => {
            menu.remove();
            this._splitSynapseGroup(group);
        });
        menu.appendChild(item);

        document.body.appendChild(menu);
        setTimeout(() => {
            const dismiss = (ev) => {
                if (!menu.contains(ev.target)) {
                    menu.remove();
                    document.removeEventListener('click', dismiss);
                }
            };
            document.addEventListener('click', dismiss);
        }, 0);
    }

    _splitSynapseGroup(group) {
        const synMgr = this.viewer.synapse;
        const bids = this.data.typeNeurons[group.ourType] || [];
        const il = this.data.instanceLookup || {};
        const mode = this.data.colorModes[this.viewer.vis.activeColorMode || 0];

        // Remove the original group
        synMgr.removeGroup(group.id);

        // Create per-neuron groups
        for (const bid of bids) {
            const neuronLabel = il[bid] || `${group.ourType} (${bid})`;
            const arrow = group.direction === 'upstream'
                ? `${group.partnerType} \u2192 ${neuronLabel}`
                : `${neuronLabel} \u2192 ${group.partnerType}`;
            const label = group.roi ? `${arrow} (${group.roi})` : arrow;

            // Determine per-neuron color from current mode
            const neuronColor = (mode && mode.colors && mode.colors[String(bid)]) ||
                                (mode && mode.type_colors && mode.type_colors[group.ourType]) || '#ffffff';

            const newGroup = synMgr.createGroup({
                label: label,
                ourBids: [parseInt(bid)],
                ourType: group.ourType,
                partnerType: group.partnerType,
                roi: group.roi,
                direction: group.direction,
                synapseType: group.synapseType,
                color: neuronColor,
                useNeuronColor: true,
            });

            if (newGroup) {
                newGroup._splitBid = String(bid);
                newGroup.useNeuronColor = true;
            }
        }

        this._updateSynapsePanel();
    }

    _buildColorPicker(e, onSelect, opts = {}) {
        const picker = document.createElement('div');
        picker.style.cssText = `position:fixed;left:${e.clientX}px;top:${Math.max(10, e.clientY - 260)}px;z-index:10001;background:#222;border:1px solid #555;border-radius:4px;padding:6px;box-shadow:0 4px 12px rgba(0,0,0,0.5);display:flex;flex-wrap:wrap;gap:3px;width:149px;overflow:hidden;box-sizing:border-box;`;

        // Optional "Auto" button (for synapse pickers)
        if (opts.autoLabel) {
            const autoOpt = document.createElement('div');
            autoOpt.textContent = opts.autoLabel;
            autoOpt.style.cssText = 'width:100%;padding:3px 6px;font-size:10px;color:#ccc;cursor:pointer;border:1px solid #555;border-radius:3px;text-align:center;margin-bottom:2px;';
            autoOpt.title = opts.autoTitle || '';
            autoOpt.addEventListener('click', () => { opts.onAuto(); picker.remove(); });
            picker.appendChild(autoOpt);
        }

        // Preset swatches
        const presets = [
            '#ff4444','#ff8800','#ffcc00','#88ff00','#44cc44','#00cccc',
            '#4488ff','#8844ff','#cc44cc','#ff88aa','#d4a017','#ffffff',
            '#cc0000','#cc6600','#cc9900','#448800','#008844','#006688',
            '#2244aa','#5500aa','#880066','#aa4466','#666666','#000000',
        ];
        for (const color of presets) {
            const sw = document.createElement('div');
            sw.style.cssText = `width:20px;height:20px;border-radius:3px;cursor:pointer;background:${color};border:1px solid #666;`;
            sw.addEventListener('click', () => { onSelect(color); picker.remove(); });
            picker.appendChild(sw);
        }

        // RGB sliders
        const rgbWrap = document.createElement('div');
        rgbWrap.style.cssText = 'width:100%;margin-top:4px;border-top:1px solid #444;padding-top:4px;overflow:hidden;';

        const preview = document.createElement('div');
        preview.style.cssText = 'width:100%;height:16px;border-radius:3px;background:#808080;border:1px solid #666;margin-bottom:3px;box-sizing:border-box;';

        const sliders = {};
        let r = 128, g = 128, b = 128;
        const updatePreview = () => {
            preview.style.background = `rgb(${r},${g},${b})`;
        };

        for (const [ch, label] of [['r','R'],['g','G'],['b','B']]) {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:2px;overflow:hidden;';
            const lbl = document.createElement('span');
            lbl.textContent = label;
            lbl.style.cssText = 'font-size:9px;color:#888;width:10px;flex-shrink:0;';
            const slider = document.createElement('input');
            slider.type = 'range'; slider.min = '0'; slider.max = '255'; slider.value = '128';
            slider.style.cssText = 'flex:1;min-width:0;max-width:100px;height:12px;accent-color:rgb(212,160,23);';
            const val = document.createElement('span');
            val.textContent = '128';
            val.style.cssText = 'font-size:9px;color:#888;width:20px;text-align:right;flex-shrink:0;';
            slider.addEventListener('input', () => {
                const v = parseInt(slider.value);
                val.textContent = v;
                if (ch === 'r') r = v; else if (ch === 'g') g = v; else b = v;
                updatePreview();
            });
            sliders[ch] = slider;
            row.appendChild(lbl); row.appendChild(slider); row.appendChild(val);
            rgbWrap.appendChild(row);
        }

        const applyBtn = document.createElement('button');
        applyBtn.textContent = 'Apply';
        applyBtn.style.cssText = 'width:100%;margin-top:3px;padding:3px;font-size:10px;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;cursor:pointer;';
        applyBtn.addEventListener('click', () => {
            onSelect(`rgb(${r},${g},${b})`);
            picker.remove();
        });

        rgbWrap.appendChild(preview);
        // Move preview after sliders visually — insert before Apply
        rgbWrap.insertBefore(preview, rgbWrap.firstChild);
        rgbWrap.appendChild(applyBtn);
        picker.appendChild(rgbWrap);

        return picker;
    }

    _showSynapseColorPicker(e, group) {
        if (this._synColorPicker) this._synColorPicker.remove();

        const picker = this._buildColorPicker(e,
            (color) => {
                this.viewer.synapse.setGroupColor(group.id, color, false);
                this._updateSynapsePanel();
            },
            {
                autoLabel: 'Auto',
                autoTitle: 'Use neuron type color (auto-updates with color mode)',
                onAuto: () => {
                    const mode = this.viewer.data.colorModes[this.viewer.vis.activeColorMode || 0];
                    const c = (mode && mode.type_colors && mode.type_colors[group.ourType]) || '#ffffff';
                    this.viewer.synapse.setGroupColor(group.id, c, true);
                    this._updateSynapsePanel();
                },
            }
        );

        document.body.appendChild(picker);
        this._synColorPicker = picker;
        setTimeout(() => {
            const dismiss = (ev) => {
                if (!picker.contains(ev.target)) {
                    picker.remove(); this._synColorPicker = null;
                    document.removeEventListener('click', dismiss);
                }
            };
            document.addEventListener('click', dismiss);
        }, 0);
    }

    _showCustomColorPicker(e, key, kind) {
        if (this._customColorPicker) this._customColorPicker.remove();

        const picker = this._buildColorPicker(e,
            (color) => {
                this._applyCustomColor(key, kind, color);
                this._customColorPicker = null;
            }
        );

        document.body.appendChild(picker);
        this._customColorPicker = picker;
        setTimeout(() => {
            const dismiss = (ev) => {
                if (!picker.contains(ev.target)) {
                    picker.remove(); this._customColorPicker = null;
                    document.removeEventListener('click', dismiss);
                }
            };
            document.addEventListener('click', dismiss);
        }, 0);
    }

    _applyCustomColor(key, kind, color) {
        const mode = this.data.colorModes[this.vis.activeColorMode];
        if (!mode.is_custom) return;

        if (kind === 'type') {
            // Update type color and all neurons of this type
            mode.type_colors[key] = color;
            const bids = this.data.typeNeurons[key] || [];
            for (const bid of bids) {
                mode.colors[String(bid)] = color;
            }
        } else {
            // Update single neuron color
            mode.colors[String(key)] = color;
            if (mode._neuronColors) mode._neuronColors[String(key)] = color;
        }

        // Apply to 3D
        this.vis.switchColorMode(this.vis.activeColorMode);
        this._updatePanelSwatches();
    }

    _updateConnPanel() {
        if (!this.connSelectedKey) {
            this._clearConnPanel();
            return;
        }

        const key = this.connSelectedKey;

        // Update title
        const label = this.data.instanceLookup[key] || key;
        const bidSuffix = this.data.neuronType[key] ? ' (' + key + ')' : '';
        this.connTitle.textContent = 'Connections for: ' + label + bidSuffix;

        // Update dropdown options
        this.connDropdown.innerHTML = '';
        for (const k of this.vis.highlightedSet) {
            const opt = document.createElement('option');
            opt.value = k;
            const dlabel = this.data.instanceLookup[k] || k;
            const dsuffix = this.data.neuronType[k] ? ' (' + k + ')' : '';
            opt.textContent = dlabel + dsuffix;
            this.connDropdown.appendChild(opt);
        }
        this.connDropdown.value = key;

        // Check if key is a type or a neuron bodyId
        let upstream, downstream;
        const isType = (this.data.typeUpstream[key] !== undefined || this.data.typeDownstream[key] !== undefined);
        if (isType) {
            upstream = this.data.typeUpstream[key] || {};
            downstream = this.data.typeDownstream[key] || {};
        } else {
            // Neuron mode: data is {roi: {__types__: {partner: w}, __instances__: ...}}
            // Extract __types__ per ROI to get {roi: {partner: w}}
            const nUp = this.data.neuronUpstream[key] || {};
            const nDn = this.data.neuronDownstream[key] || {};
            upstream = {};
            for (const [roi, roiData] of Object.entries(nUp)) {
                upstream[roi] = roiData.__types__ || roiData;
            }
            downstream = {};
            for (const [roi, roiData] of Object.entries(nDn)) {
                downstream[roi] = roiData.__types__ || roiData;
            }
        }

        this._buildConnList(this.connUpCol, upstream, 'Upstream', 'upstream');
        this._buildConnList(this.connDnCol, downstream, 'Downstream', 'downstream');
        this.connUpCol.scrollTop = 0;
        this.connDnCol.scrollTop = 0;
    }

    _buildConnList(container, roiData, title, direction) {
        container.innerHTML = `<div style="font-weight:bold;font-size:12px;color:#aaa;margin-bottom:2px;padding:4px 0;position:sticky;top:0;background:rgba(15,15,15,0.95);z-index:2;">${title}</div>`;

        // Build ROI entries sorted by total
        const entries = [];
        for (const [roi, partners] of Object.entries(roiData)) {
            let total = 0;
            for (const w of Object.values(partners)) total += w;
            entries.push({ roi, partners, total });
        }
        entries.sort((a, b) => b.total - a.total);

        // Grand total for fraction bars
        let grandTotal = 0;
        for (const e of entries) grandTotal += e.total;

        // Priority ROI at top
        if (this.connSelectedRoi) {
            const idx = entries.findIndex(e => e.roi === this.connSelectedRoi);
            if (idx > 0) {
                const [item] = entries.splice(idx, 1);
                entries.unshift(item);
            } else if (idx < 0) {
                entries.unshift({ roi: this.connSelectedRoi, partners: {}, total: 0 });
            }
        }

        for (const { roi, partners, total } of entries) {
            const roiPct = grandTotal > 0 ? (total / grandTotal * 100) : 0;
            const roiHdr = document.createElement('div');
            roiHdr.style.cssText = 'font-weight:bold;color:#8cb4d8;font-size:11px;margin-top:4px;padding:3px 4px;position:sticky;top:22px;background:rgb(15,15,15);z-index:1;display:flex;align-items:center;gap:4px;cursor:pointer;';
            roiHdr.title = 'Double-click to toggle ROI';

            const roiLabel = document.createElement('span');
            roiLabel.textContent = `${roi} (${total.toLocaleString()})`;
            roiHdr.appendChild(roiLabel);

            const roiSpacer = document.createElement('span');
            roiSpacer.style.cssText = 'flex:1;';
            roiHdr.appendChild(roiSpacer);

            const roiBarWrap = document.createElement('span');
            roiBarWrap.style.cssText = 'flex-shrink:0;width:60px;height:8px;background:rgba(140,180,216,0.15);border-radius:2px;overflow:hidden;';
            const roiBar = document.createElement('span');
            roiBar.style.cssText = `display:block;height:100%;background:rgba(140,180,216,0.6);width:${roiPct.toFixed(1)}%;border-radius:2px;`;
            roiBarWrap.appendChild(roiBar);
            roiHdr.appendChild(roiBarWrap);

            // Double-click to toggle ROI visibility
            ((rName, hdr) => {
                let _lastClick = 0;
                hdr.addEventListener('click', () => {
                    const now = Date.now();
                    if (now - _lastClick < 500) {
                        const isOn = !this.vis.roiChecked[rName];
                        this.vis.setRoiChecked(rName, isOn);
                        this.syncAllState();
                    }
                    _lastClick = now;
                });
            })(roi, roiHdr);

            container.appendChild(roiHdr);

            // Partner rows sorted by weight
            const sorted = Object.entries(partners).sort((a, b) => b[1] - a[1]);
            for (const [partner, weight] of sorted.slice(0, 20)) {
                const pct = total > 0 ? (weight / total * 100) : 0;
                const row = document.createElement('div');
                row.style.cssText = 'display:flex;align-items:center;gap:4px;padding:1px 0;font-size:11px;color:#ccc;cursor:context-menu;';
                row.innerHTML = `
                    <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${partner}</span>
                    <div style="width:60px;height:6px;background:#333;border-radius:3px;overflow:hidden;">
                        <div style="width:${pct}%;height:100%;background:rgb(212,160,23);"></div>
                    </div>
                    <span style="width:30px;text-align:right;font-size:10px;color:#888;">${weight}</span>
                `;

                // Right-click to show synapse context menu
                ((partnerType, roiName, dir) => {
                    row.addEventListener('contextmenu', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        this._showSynapseContextMenu(e, partnerType, roiName, dir);
                    });
                })(partner, roi, direction);

                container.appendChild(row);
            }
        }
    }

    _showSynapseContextMenu(e, partnerType, roi, direction) {
        this._dismissSynapseContextMenu();
        if (!DATA.synapseData) {
            console.warn('No synapse data available for this visualization');
            return;
        }

        const menu = document.createElement('div');
        menu.id = 'synapseContextMenu';
        menu.style.cssText = `position:fixed;left:${e.clientX}px;top:${e.clientY}px;z-index:10000;background:#222;border:1px solid #555;border-radius:4px;padding:4px 0;font-family:sans-serif;font-size:12px;min-width:180px;box-shadow:0 4px 12px rgba(0,0,0,0.5);`;

        const ourKey = this.connSelectedKey;
        const ourType = this.viewer.data.bidTypeMap[String(ourKey)] || ourKey;

        const items = direction === 'upstream'
            ? [{ label: 'Show presynapses', type: 'pre' }]
            : [{ label: 'Show postsynapses', type: 'post' }];

        for (const item of items) {
            const row = document.createElement('div');
            row.textContent = item.label;
            row.style.cssText = 'padding:6px 14px;cursor:pointer;color:#ddd;';
            row.onmouseenter = () => { row.style.background = '#444'; };
            row.onmouseleave = () => { row.style.background = 'transparent'; };
            row.addEventListener('click', async () => {
                this._dismissSynapseContextMenu();
                await this._createSynapseGroup(ourKey, ourType, partnerType, roi, direction, item.type);
            });
            menu.appendChild(row);
        }

        document.body.appendChild(menu);
        this._synapseContextMenu = menu;

        // Dismiss on click outside or Escape
        const dismiss = (ev) => {
            if (ev.type === 'keydown' && ev.key !== 'Escape') return;
            this._dismissSynapseContextMenu();
            document.removeEventListener('click', dismiss);
            document.removeEventListener('keydown', dismiss);
        };
        setTimeout(() => {
            document.addEventListener('click', dismiss);
            document.addEventListener('keydown', dismiss);
        }, 0);
    }

    _dismissSynapseContextMenu() {
        if (this._synapseContextMenu) {
            this._synapseContextMenu.remove();
            this._synapseContextMenu = null;
        }
    }

    async _createSynapseGroup(ourKey, ourType, partnerType, roi, direction, synapseType) {
        const synMgr = this.viewer.synapse;

        // Ensure data is loaded
        if (!synMgr.loaded) {
            console.log('Loading synapse data...');
            const ok = await synMgr.loadData();
            if (!ok) {
                console.warn('Failed to load synapse data');
                return;
            }
        }

        // Resolve "our" body IDs
        let ourBids;
        const isType = this.viewer.data.typeNeurons.hasOwnProperty(ourKey);
        if (isType) {
            ourBids = this.viewer.data.typeNeurons[ourKey];
        } else {
            ourBids = [Number(ourKey)];
        }

        const il = this.viewer.data.instanceLookup || {};
        const ourLabel = isType ? ourType : (il[ourKey] || ourType);
        const arrow = direction === 'upstream' ? partnerType + ' \u2192 ' + ourLabel : ourLabel + ' \u2192 ' + partnerType;
        const label = `${arrow} (${roi})`;

        const group = synMgr.createGroup({
            label: label,
            ourBids: ourBids,
            ourType: ourType,
            partnerType: partnerType,
            roi: roi,
            direction: direction,
            synapseType: synapseType,
            color: null,  // neuron color by default
        });

        if (group) {
            this._updateSynapsePanel();
        } else {
            console.warn(`No synapses found for ${label}`);
        }
    }

    _clearConnPanel() {
        this.connTitle.textContent = 'Connections';
        this.connDropdown.innerHTML = '';
        this.connUpCol.innerHTML = '<div style="color:#555;font-size:12px;padding:20px 4px;text-align:center;">Click a highlighted type or neuron to view connections</div>';
        this.connDnCol.innerHTML = '';
    }

    _showVideoExportDialog() {
        // Dismiss existing dialog
        if (this._videoDialog) this._videoDialog.remove();

        const dialog = document.createElement('div');
        dialog.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:10002;background:#1a1a1a;border:1px solid #555;border-radius:8px;padding:0;box-shadow:0 8px 24px rgba(0,0,0,0.6);color:#ccc;font-size:13px;min-width:300px;';

        // Draggable title bar
        const titleBar = document.createElement('div');
        titleBar.style.cssText = 'padding:12px 24px 8px 24px;cursor:move;user-select:none;border-bottom:1px solid #333;';
        const title = document.createElement('div');
        title.textContent = 'Record a video';
        title.style.cssText = 'font-size:16px;font-weight:bold;color:#d4a017;';
        titleBar.appendChild(title);
        dialog.appendChild(titleBar);

        // Make draggable
        let dragX = 0, dragY = 0, isDragging = false;
        titleBar.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragX = e.clientX - dialog.offsetLeft;
            dragY = e.clientY - dialog.offsetTop;
            dialog.style.transform = 'none'; // Remove centering transform once dragging
        });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            dialog.style.left = (e.clientX - dragX) + 'px';
            dialog.style.top = (e.clientY - dragY) + 'px';
        });
        document.addEventListener('mouseup', () => { isDragging = false; });

        // Body content
        const body = document.createElement('div');
        body.style.cssText = 'padding:12px 24px 20px 24px;';

        const _selectStyle = 'width:100%;padding:4px 6px;background:#222;color:#fff;border:1px solid #555;border-radius:3px;font-size:13px;margin-bottom:10px;';
        const _labelStyle = 'margin-bottom:4px;font-weight:bold;font-size:12px;';

        // Helpers
        const getSelectionCenter = () => {
            const sc = this.viewer.scene;
            const box = new THREE.Box3();
            const meshMode = this.vis._meshesVisible;
            if (meshMode && sc.neuronMeshGeom) {
                for (const [, m] of sc.neuronMeshGeom) {
                    if (m.visible) { try { const b = new THREE.Box3().setFromObject(m); if (!b.isEmpty()) box.union(b); } catch(e){} }
                }
            }
            if (box.isEmpty()) {
                for (const map of [sc.typeRoiGeom, sc.neuronFullGeom]) {
                    if (!map) continue;
                    for (const [, g] of map) {
                        if (!g.visible) continue;
                        try { const b = new THREE.Box3().setFromObject(g); if (!b.isEmpty()) box.union(b); } catch(e){}
                    }
                }
            }
            if (box.isEmpty()) return null;
            const c = new THREE.Vector3(); box.getCenter(c); return c;
        };
        const resolveAxis = (axisVal, coordVal) => {
            if (coordVal === 'viewport') {
                const cam = this.viewer.scene.camera;
                const ctl = this.viewer.scene.controls;
                const fwd = new THREE.Vector3().subVectors(ctl.target, cam.position).normalize();
                const right = new THREE.Vector3().crossVectors(fwd, cam.up).normalize();
                const up = cam.up.clone().normalize();
                if (axisVal === 'Y') return [up.x, up.y, up.z];
                if (axisVal === 'X') return [right.x, right.y, right.z];
                return [fwd.x, fwd.y, fwd.z];
            }
            // World axes
            if (axisVal === 'Y') return [0, 1, 0];
            if (axisVal === 'X') return [1, 0, 0];
            return [0, 0, 1];
        };

        // 1. Motion type
        const motionLabel = document.createElement('div');
        motionLabel.textContent = 'Motion:';
        motionLabel.style.cssText = _labelStyle;
        body.appendChild(motionLabel);
        const motionSelect = document.createElement('select');
        motionSelect.style.cssText = _selectStyle;
        [['360', 'Full 360\u00B0 rotation'], ['pivot', 'Pivot (\u00B1 N\u00B0)']].forEach(([v, l]) => {
            const o = document.createElement('option'); o.value = v; o.textContent = l; motionSelect.appendChild(o);
        });
        body.appendChild(motionSelect);

        // Pivot angle (hidden until pivot selected)
        const pivotRow = document.createElement('div');
        pivotRow.style.cssText = 'display:none;align-items:center;gap:6px;margin-bottom:10px;';
        const pivotLabel = document.createElement('span');
        pivotLabel.textContent = 'Pivot angle (\u00B0):';
        pivotLabel.style.cssText = 'font-size:12px;color:#ccc;';
        pivotRow.appendChild(pivotLabel);
        const pivotInput = document.createElement('input');
        pivotInput.type = 'number'; pivotInput.value = '45'; pivotInput.min = '1'; pivotInput.max = '180';
        pivotInput.style.cssText = 'width:50px;padding:3px 5px;background:#222;color:#fff;border:1px solid #555;border-radius:3px;font-size:13px;';
        pivotRow.appendChild(pivotInput);
        body.appendChild(pivotRow);
        motionSelect.onchange = () => {
            pivotRow.style.display = motionSelect.value === 'pivot' ? 'flex' : 'none';
            updateDuration();
        };

        // 2. Axis
        const axisLabel = document.createElement('div');
        axisLabel.textContent = 'Axis:';
        axisLabel.style.cssText = _labelStyle;
        body.appendChild(axisLabel);
        const axisSelect = document.createElement('select');
        axisSelect.style.cssText = _selectStyle;
        [['Y', 'Y (Dorsal / Ventral)'], ['X', 'X (Left / Right)'], ['Z', 'Z (Anterior / Posterior)']].forEach(([v, l]) => {
            const o = document.createElement('option'); o.value = v; o.textContent = l; axisSelect.appendChild(o);
        });
        body.appendChild(axisSelect);

        // 3. Orbit center + coordinate space (merged)
        const centerLabel = document.createElement('div');
        centerLabel.textContent = 'Orbit around:';
        centerLabel.style.cssText = _labelStyle;
        body.appendChild(centerLabel);
        const centerSelect = document.createElement('select');
        centerSelect.style.cssText = _selectStyle;
        const orbitOptions = [
            ['scene_world', 'Scene center (world axes)', 'Rotate around camera target using anatomical D/V, L/R, A/P axes'],
            ['scene_viewport', 'Scene center (screen axes)', 'Rotate around camera target using your current screen orientation'],
            ['selection_world', 'Selection center (world axes)', 'Rotate around the centroid of visible neurons using anatomical axes'],
            ['selection_viewport', 'Selection center (screen axes)', 'Rotate around the centroid of visible neurons using screen orientation'],
        ];
        for (const [v, l, desc] of orbitOptions) {
            const o = document.createElement('option'); o.value = v; o.textContent = l; o.title = desc; centerSelect.appendChild(o);
        }
        body.appendChild(centerSelect);

        // 5. Speed
        const speedRow = document.createElement('div');
        speedRow.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:10px;';
        const speedLabel = document.createElement('span');
        speedLabel.textContent = 'Speed (\u00B0/sec):';
        speedLabel.style.cssText = 'font-size:12px;font-weight:bold;color:#ccc;';
        speedRow.appendChild(speedLabel);
        const speedInput = document.createElement('input');
        speedInput.type = 'number'; speedInput.value = '30'; speedInput.min = '1'; speedInput.max = '360';
        speedInput.style.cssText = 'width:50px;padding:3px 5px;background:#222;color:#fff;border:1px solid #555;border-radius:3px;font-size:13px;';
        speedRow.appendChild(speedInput);
        const durationSpan = document.createElement('span');
        durationSpan.style.cssText = 'color:#888;font-size:12px;';
        speedRow.appendChild(durationSpan);
        body.appendChild(speedRow);

        const updateDuration = () => {
            const speed = parseFloat(speedInput.value) || 30;
            const isPivot = motionSelect.value === 'pivot';
            const angle = isPivot ? (parseFloat(pivotInput.value) || 45) * 4 : 360;
            const dur = angle / speed;
            durationSpan.textContent = `\u2192 ${dur.toFixed(1)}s`;
        };
        speedInput.oninput = updateDuration;
        pivotInput.oninput = updateDuration;
        updateDuration();


        // Format selection
        const fmtLabel = document.createElement('div');
        fmtLabel.textContent = 'Format:';
        fmtLabel.style.cssText = 'margin-bottom:6px;font-weight:bold;';
        body.appendChild(fmtLabel);

        const fmtSelect = document.createElement('select');
        fmtSelect.style.cssText = 'width:100%;padding:4px 6px;background:#222;color:#fff;border:1px solid #555;border-radius:3px;font-size:13px;margin-bottom:14px;';
        const formats = [
            { label: 'AVI', type: 'avi' },
            { label: 'WebM', type: 'webm' },
            { label: 'GIF', type: 'gif' },
        ];
        for (const fmt of formats) {
            const opt = document.createElement('option');
            opt.value = fmt.type;
            opt.textContent = fmt.label;
            fmtSelect.appendChild(opt);
        }
        body.appendChild(fmtSelect);

        // Buttons
        const btnRow = document.createElement('div');
        btnRow.style.cssText = 'display:flex;gap:10px;justify-content:flex-end;margin-top:8px;';

        const cancelBtn = document.createElement('button');
        cancelBtn.textContent = 'Cancel';
        cancelBtn.style.cssText = 'padding:6px 16px;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:13px;';
        cancelBtn.onclick = () => {
            if (this._previewAnimId) { cancelAnimationFrame(this._previewAnimId); this._previewAnimId = null; }
            dialog.remove();
            this._videoDialog = null;
        };
        btnRow.appendChild(cancelBtn);

        // Gather current settings from dropdowns
        const getSettings = () => {
            const speed = parseFloat(speedInput.value) || 30;
            const isPivot = motionSelect.value === 'pivot';
            const pivotAngle = isPivot ? (parseFloat(pivotInput.value) || 45) : 360;
            const [centerType, coordType] = centerSelect.value.split('_');
            const axis = resolveAxis(axisSelect.value, coordType);
            const center = centerType === 'selection' ? getSelectionCenter() : null;
            return { speed, isPivot, pivotAngle, axis, center };
        };

        // Preview button
        const previewBtn = document.createElement('button');
        previewBtn.textContent = 'Preview';
        previewBtn.style.cssText = 'padding:6px 16px;background:#444;color:#fff;border:1px solid #666;border-radius:3px;cursor:pointer;font-size:13px;';
        previewBtn.onclick = () => {
            if (this._previewAnimId) {
                cancelAnimationFrame(this._previewAnimId);
                this._previewAnimId = null;
                this._restorePreviewCamera();
                previewBtn.textContent = 'Preview';
                previewBtn.style.background = '#444';
            } else {
                previewBtn.textContent = 'Stop';
                previewBtn.style.background = '#844';
                const s = getSettings();
                this._runPreview(s.axis, s.speed, previewBtn, s.center, s.isPivot, s.pivotAngle);
            }
        };
        btnRow.appendChild(previewBtn);

        const recordBtn = document.createElement('button');
        recordBtn.textContent = 'Record';
        recordBtn.style.cssText = 'padding:6px 16px;background:rgb(212,160,23);color:#000;border:none;border-radius:3px;cursor:pointer;font-size:13px;font-weight:bold;';
        recordBtn.onclick = () => {
            if (this._previewAnimId) {
                cancelAnimationFrame(this._previewAnimId);
                this._previewAnimId = null;
                this._restorePreviewCamera();
            }
            const s = getSettings();
            const fmtType = fmtSelect.value;
            dialog.remove();
            this._videoDialog = null;
            // For pivot: full cycle = +angle, center, -angle, center = 4x angle
            const totalDeg = s.isPivot ? s.pivotAngle * 4 : 360;
            if (fmtType === 'gif') {
                this._recordRotationGif(s.axis, s.speed, 'viewport', s.center, totalDeg, s.isPivot ? s.pivotAngle : 0);
            } else if (fmtType === 'avi') {
                this._recordRotationAvi(s.axis, s.speed, 'viewport', s.center, totalDeg, s.isPivot ? s.pivotAngle : 0);
            } else {
                const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
                    ? 'video/webm;codecs=vp9' : 'video/webm';
                this._recordRotationVideo(s.axis, s.speed,
                    { mime, ext: 'webm' }, 'viewport', s.center, totalDeg, s.isPivot ? s.pivotAngle : 0);
            }
        };
        btnRow.appendChild(recordBtn);
        body.appendChild(btnRow);
        dialog.appendChild(body);

        document.body.appendChild(dialog);
        this._videoDialog = dialog;
    }

    _runPreview(axis, degPerSec, previewBtn, orbitCenter, isPivot, pivotAngle) {
        const scene = this.viewer.scene;
        const camera = scene.camera;
        const controls = scene.controls;
        const target = orbitCenter || controls.target.clone();

        scene._recordingActive = true;
        this._previewSavedPos = camera.position.clone();
        this._previewSavedUp = camera.up.clone();
        this._previewSavedTarget = controls.target.clone();

        // If orbiting a different center, preserve camera distance along view direction
        const viewDir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
        const dist = camera.position.distanceTo(controls.target);
        const initialOffset = viewDir.multiplyScalar(dist);
        const initialUp = camera.up.clone();
        const rotAxis = new THREE.Vector3(axis[0], axis[1], axis[2]).normalize();
        const radPerSec = degPerSec * Math.PI / 180;
        const pivotRad = isPivot ? (pivotAngle || 45) * Math.PI / 180 : 0;
        const pivotPeriod = isPivot ? (pivotAngle * 4 / degPerSec) : 0;

        // Use frozen copies to prevent drift
        const frozenOffset = new Float64Array([initialOffset.x, initialOffset.y, initialOffset.z]);
        const frozenUp = new Float64Array([initialUp.x, initialUp.y, initialUp.z]);
        const frozenTarget = new Float64Array([target.x, target.y, target.z]);

        let startTime = null;
        const animate = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const elapsed = (timestamp - startTime) / 1000;

            let angle;
            if (isPivot && pivotPeriod > 0) {
                const t = (elapsed % pivotPeriod) / pivotPeriod;
                angle = pivotRad * Math.sin(t * 2 * Math.PI);
            } else {
                angle = (elapsed * radPerSec) % (Math.PI * 2);
            }

            // Always compute from frozen originals to prevent drift
            const off = new THREE.Vector3(frozenOffset[0], frozenOffset[1], frozenOffset[2]);
            off.applyAxisAngle(rotAxis, angle);
            camera.position.set(frozenTarget[0] + off.x, frozenTarget[1] + off.y, frozenTarget[2] + off.z);
            const up = new THREE.Vector3(frozenUp[0], frozenUp[1], frozenUp[2]);
            up.applyAxisAngle(rotAxis, angle);
            camera.up.copy(up);
            camera.lookAt(frozenTarget[0], frozenTarget[1], frozenTarget[2]);
            camera.updateMatrixWorld();
            scene.renderer.render(scene.scene, camera);
            scene._captureScaleBar = true; scene._renderScaleBar(); scene._captureScaleBar = false;

            this._previewAnimId = requestAnimationFrame(animate);
        };
        this._previewAnimId = requestAnimationFrame(animate);
    }

    _restorePreviewCamera() {
        if (!this._previewSavedPos) return;
        const scene = this.viewer.scene;
        scene._recordingActive = false;
        // Restore exact camera state
        scene.camera.position.copy(this._previewSavedPos);
        scene.camera.up.copy(this._previewSavedUp);
        scene.controls.target.copy(this._previewSavedTarget);
        scene.camera.updateMatrixWorld();
        scene.controls.update();
        this._previewSavedPos = null;
    }

    _recordRotationVideo(axis, degPerSec, fmt, resolution, orbitCenter, totalDeg, pivotAngle) {
        const scene = this.viewer.scene;
        const renderer = scene.renderer;
        const camera = scene.camera;
        const controls = scene.controls;
        const canvas = scene.canvas;

        // Handle resolution: resize renderer temporarily if not 'viewport'
        let origW, origH, restoreSize = false;
        if (resolution && resolution !== 'viewport') {
            const [w, h] = resolution.split('x').map(Number);
            origW = canvas.clientWidth;
            origH = canvas.clientHeight;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            restoreSize = true;
        }

        const fps = 30;
        const _totalDeg = totalDeg || 360;
        const _pivotRad = pivotAngle ? pivotAngle * Math.PI / 180 : 0;
        const totalFrames = Math.ceil(_totalDeg / degPerSec * fps);
        const radPerFrame = (_totalDeg * Math.PI / 180) / totalFrames;
        // Angle function: for pivot, go 0→+pivot→0; for 360°, go 0→2π
        const getAngle = (frame) => {
            if (_pivotRad > 0) {
                // Smooth out-and-back: 0 → +pivot → 0
                const t = frame / totalFrames;  // 0 to 1
                return _pivotRad * Math.sin(t * 2 * Math.PI);
            }
            return radPerFrame * frame;
        };
        const rotAxis = new THREE.Vector3(axis[0], axis[1], axis[2]).normalize();

        const target = orbitCenter || controls.target.clone();
        const initialOffset = camera.position.clone().sub(target);
        const initialUp = camera.up.clone();

        // Recording overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:9998;pointer-events:none;';
        const badge = document.createElement('div');
        badge.style.cssText = 'position:fixed;top:50px;left:50%;transform:translateX(-50%);z-index:9999;background:rgba(200,0,0,0.8);color:#fff;padding:8px 20px;border-radius:20px;font-size:14px;font-weight:bold;';
        badge.textContent = '\u{1F534} Recording...  0%';
        overlay.appendChild(badge);
        document.body.appendChild(overlay);

        const mimeType = fmt.mime;
        const fileExt = fmt.ext;
        const bitrate = resolution === '3840x2160' ? 20000000 : resolution === '1920x1080' ? 12000000 : 8000000;

        const stream = canvas.captureStream(fps);
        const chunks = [];
        const recorder = new MediaRecorder(stream, {
            mimeType: mimeType,
            videoBitsPerSecond: bitrate,
        });
        recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
        recorder.onstop = () => {
            overlay.remove();
            camera.position.copy(target).add(initialOffset);
            camera.up.copy(initialUp);
            controls.update();

            if (restoreSize) {
                renderer.setSize(origW, origH);
                camera.aspect = origW / origH;
                camera.updateProjectionMatrix();
            }

            const blob = new Blob(chunks, { type: mimeType });
            const term = this.viewer.data.regexTerm || 'neuron';
            _saveFileAs(blob, `${term}_rotation.${fileExt}`, [
                { description: 'Video', accept: { [mimeType]: [`.${fileExt}`] } }
            ]);
        };

        scene._recordingActive = true;
        recorder.start();

        let frame = 0;
        const animate = () => {
            if (frame >= totalFrames) {
                scene._recordingActive = false;
                recorder.stop();
                return;
            }

            // Rotate camera around target
            const angle = getAngle(frame);
            const offset = initialOffset.clone().applyAxisAngle(rotAxis, angle);
            camera.position.copy(target).add(offset);
            camera.up.copy(initialUp).applyAxisAngle(rotAxis, angle);
            camera.lookAt(target);
            camera.updateMatrixWorld();

            // Force render
            renderer.render(scene.scene, camera);
            scene._captureScaleBar = true; scene._renderScaleBar(); scene._captureScaleBar = false;

            badge.textContent = `\u{1F534} Recording...  ${Math.round(frame / totalFrames * 100)}%`;
            frame++;
            setTimeout(animate, 0);
        };
        animate();
    }

    _recordRotationAvi(axis, degPerSec, resolution, orbitCenter, totalDeg, pivotAngle) {
        const scene = this.viewer.scene;
        const renderer = scene.renderer;
        const camera = scene.camera;
        const controls = scene.controls;
        const canvas = scene.canvas;

        // Render at 1x pixel ratio for reasonable file sizes
        const pr = renderer.getPixelRatio();
        const origW = canvas.clientWidth;
        const origH = canvas.clientHeight;
        // Set to 1x pixel ratio for capture (Retina would be 2-3x too large)
        renderer.setPixelRatio(1);
        renderer.setSize(origW, origH);
        camera.aspect = origW / origH;
        camera.updateProjectionMatrix();
        let w = origW & ~1;  // Even dimensions for codecs
        let h = origH & ~1;

        const fps = 30;
        const _totalDeg = totalDeg || 360;
        const _pivotRad = pivotAngle ? pivotAngle * Math.PI / 180 : 0;
        const totalFrames = Math.ceil(_totalDeg / degPerSec * fps);
        const radPerFrame = (_totalDeg * Math.PI / 180) / totalFrames;
        // Angle function: for pivot, go 0→+pivot→0; for 360°, go 0→2π
        const getAngle = (frame) => {
            if (_pivotRad > 0) {
                // Smooth out-and-back: 0 → +pivot → 0
                const t = frame / totalFrames;  // 0 to 1
                return _pivotRad * Math.sin(t * 2 * Math.PI);
            }
            return radPerFrame * frame;
        };
        const rotAxis = new THREE.Vector3(axis[0], axis[1], axis[2]).normalize();

        const target = controls.target.clone();
        const initialOffset = camera.position.clone().sub(target);
        const initialUp = camera.up.clone();

        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:9998;pointer-events:none;';
        const badge = document.createElement('div');
        badge.style.cssText = 'position:fixed;top:50px;left:50%;transform:translateX(-50%);z-index:9999;background:rgba(200,0,0,0.8);color:#fff;padding:8px 20px;border-radius:20px;font-size:14px;font-weight:bold;';
        badge.textContent = '\u{1F534} Capturing frames...  0%';
        overlay.appendChild(badge);
        document.body.appendChild(overlay);

        // Capture JPEG frames
        const jpegFrames = [];
        let frame = 0;

        // Pause the main render loop so it doesn't overwrite our frames
        scene._recordingActive = true;

        const captureFrame = () => {
            if (frame >= totalFrames) {
                scene._recordingActive = false;
                camera.position.copy(target).add(initialOffset);
                camera.up.copy(initialUp);
                // Restore pixel ratio and size
                renderer.setPixelRatio(pr);
                renderer.setSize(origW, origH);
                camera.aspect = origW / origH;
                camera.updateProjectionMatrix();
                controls.update();
                badge.textContent = '\u{1F534} Building AVI...';
                setTimeout(() => this._buildAvi(jpegFrames, w, h, fps, overlay), 100);
                return;
            }

            const angle = getAngle(frame);
            const offset = initialOffset.clone().applyAxisAngle(rotAxis, angle);
            camera.position.copy(target).add(offset);
            camera.up.copy(initialUp).applyAxisAngle(rotAxis, angle);
            camera.lookAt(target);
            camera.updateMatrixWorld();
            renderer.render(scene.scene, camera);
            scene._captureScaleBar = true; scene._renderScaleBar(); scene._captureScaleBar = false;

            // Synchronous JPEG capture via dataURL → binary
            // Composite scale bar onto frame
            const compC = document.createElement('canvas');
            compC.width = canvas.width; compC.height = canvas.height;
            const compX = compC.getContext('2d');
            compX.drawImage(canvas, 0, 0);
            if (scene._scaleBarCanvas) compX.drawImage(scene._scaleBarCanvas, 0, 0, compC.width, compC.height);
            const dataUrl = compC.toDataURL('image/jpeg', 0.92);
            const b64 = dataUrl.split(',')[1];
            const raw = atob(b64);
            const arr = new Uint8Array(raw.length);
            for (let j = 0; j < raw.length; j++) arr[j] = raw.charCodeAt(j);
            jpegFrames.push(arr);

            badge.textContent = `\u{1F534} Capturing frames...  ${Math.round((frame + 1) / totalFrames * 100)}%`;
            frame++;
            setTimeout(captureFrame, 0);
        };
        captureFrame();
    }

    _buildAvi(jpegFrames, width, height, fps, overlay) {
        const badge = overlay.querySelector('div');
        badge.textContent = '\u{1F534} Encoding AVI...';
        const frames = jpegFrames;
        const nFrames = frames.length;
        const usPerFrame = Math.round(1000000 / fps);

        // Build into a single ArrayBuffer for byte-accurate sizing
        // First pass: compute total size
        let moviPayload = 0;  // bytes inside movi LIST after 'movi' tag
        const frameSizes = [];
        const frameOffsets = [];
        for (const f of frames) {
            frameOffsets.push(4 + moviPayload);  // offset from 'movi' start (after LIST+size)
            const padded = f.length + (f.length & 1);
            moviPayload += 8 + padded;
            frameSizes.push(f.length);
        }
        const moviListSize = 4 + moviPayload;  // 'movi' + frame chunks

        // avih: 8 (chunk hdr) + 56 (data) = 64
        // strh: 8 + 56 = 64
        // strf: 8 + 40 = 48
        // strl LIST: 8 (LIST hdr) + 4 ('strl') + 64 + 48 = 124
        // hdrl LIST: 8 (LIST hdr) + 4 ('hdrl') + 64 + 124 = 200
        const strlPayload = 4 + 64 + 48;  // 116
        const hdrlPayload = 4 + 64 + (8 + strlPayload);  // 4 + 64 + 124 = 192
        const idx1Payload = nFrames * 16;

        const totalFileSize = 12  // RIFF + size + 'AVI '
            + (8 + hdrlPayload)   // hdrl LIST
            + (8 + moviListSize)  // movi LIST
            + (8 + idx1Payload);  // idx1

        const buf = new ArrayBuffer(totalFileSize);
        const d = new DataView(buf);
        const u = new Uint8Array(buf);
        let p = 0;

        const ws = (s) => { for (let i = 0; i < s.length; i++) u[p++] = s.charCodeAt(i); };
        const w4 = (v) => { d.setUint32(p, v, true); p += 4; };
        const w2 = (v) => { d.setUint16(p, v, true); p += 2; };

        // RIFF
        ws('RIFF'); w4(totalFileSize - 8); ws('AVI ');

        // hdrl LIST
        ws('LIST'); w4(hdrlPayload); ws('hdrl');

        // avih chunk
        ws('avih'); w4(56);
        w4(usPerFrame); w4(0); w4(0); w4(0x10);
        w4(nFrames); w4(0); w4(1); w4(0);
        w4(width); w4(height);
        w4(0); w4(0); w4(0); w4(0);

        // strl LIST
        ws('LIST'); w4(strlPayload); ws('strl');

        // strh chunk
        ws('strh'); w4(56);
        ws('vids'); ws('MJPG');
        w4(0); w2(0); w2(0); w4(0);
        w4(1); w4(fps); w4(0); w4(nFrames);
        w4(0); w4(0xFFFFFFFF); w4(0);
        w2(0); w2(0); w2(width); w2(height);

        // strf chunk (BITMAPINFOHEADER)
        ws('strf'); w4(40);
        w4(40); w4(width); w4(height);
        w2(1); w2(24);
        ws('MJPG'); w4(width * height * 3);
        w4(0); w4(0); w4(0); w4(0);

        // movi LIST
        ws('LIST'); w4(moviListSize); ws('movi');

        // Frame chunks
        for (let i = 0; i < nFrames; i++) {
            ws('00dc'); w4(frames[i].length);
            u.set(frames[i], p);
            p += frames[i].length;
            if (frames[i].length & 1) u[p++] = 0;
        }

        // idx1 chunk
        ws('idx1'); w4(idx1Payload);
        for (let i = 0; i < nFrames; i++) {
            ws('00dc'); w4(0x10);
            w4(frameOffsets[i]);
            w4(frameSizes[i]);
        }

        // Verify we wrote exactly the right amount
        console.log(`AVI: ${nFrames} frames, ${totalFileSize} bytes, wrote ${p} bytes`);

        overlay.remove();

        const blob = new Blob([buf], { type: 'video/avi' });
        const term = this.viewer.data.regexTerm || 'neuron';
        _saveFileAs(blob, `${term}_rotation.avi`, [
            { description: 'AVI Video', accept: { 'video/avi': ['.avi'] } }
        ]);
    }

    _recordRotationGif(axis, degPerSec, resolution, orbitCenter, totalDeg, pivotAngle) {
        const scene = this.viewer.scene;
        const renderer = scene.renderer;
        const camera = scene.camera;
        const controls = scene.controls;
        const canvas = scene.canvas;

        // GIF settings: lower fps for reasonable file size
        const fps = 15;
        let w, h, origW, origH, restoreSize = false;
        if (resolution && resolution !== 'viewport') {
            [w, h] = resolution.split('x').map(Number);
            origW = canvas.clientWidth;
            origH = canvas.clientHeight;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            restoreSize = true;
        } else {
            w = canvas.width;
            h = canvas.height;
        }

        // Scale down for GIF (max 800px wide to keep size reasonable)
        const gifScale = Math.min(1, 800 / w);
        const gw = Math.round(w * gifScale);
        const gh = Math.round(h * gifScale);

        const _totalDeg = totalDeg || 360;
        const _pivotRad = pivotAngle ? pivotAngle * Math.PI / 180 : 0;
        const totalFrames = Math.ceil(_totalDeg / degPerSec * fps);
        const radPerFrame = (_totalDeg * Math.PI / 180) / totalFrames;
        // Angle function: for pivot, go 0→+pivot→0; for 360°, go 0→2π
        const getAngle = (frame) => {
            if (_pivotRad > 0) {
                // Smooth out-and-back: 0 → +pivot → 0
                const t = frame / totalFrames;  // 0 to 1
                return _pivotRad * Math.sin(t * 2 * Math.PI);
            }
            return radPerFrame * frame;
        };
        const rotAxis = new THREE.Vector3(axis[0], axis[1], axis[2]).normalize();
        const delay = Math.round(1000 / fps);

        const target = controls.target.clone();
        const initialOffset = camera.position.clone().sub(target);
        const initialUp = camera.up.clone();

        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;z-index:9998;pointer-events:none;';
        const badge = document.createElement('div');
        badge.style.cssText = 'position:fixed;top:50px;left:50%;transform:translateX(-50%);z-index:9999;background:rgba(200,0,0,0.8);color:#fff;padding:8px 20px;border-radius:20px;font-size:14px;font-weight:bold;';
        badge.textContent = '\u{1F534} Capturing frames...  0%';
        overlay.appendChild(badge);
        document.body.appendChild(overlay);

        // Capture frames as data URLs
        const offscreen = document.createElement('canvas');
        offscreen.width = gw;
        offscreen.height = gh;
        const offCtx = offscreen.getContext('2d');
        const frames = [];
        let frame = 0;
        scene._recordingActive = true;

        const captureFrame = () => {
            if (frame >= totalFrames) {
                scene._recordingActive = false;
                // Restore
                camera.position.copy(target).add(initialOffset);
                camera.up.copy(initialUp);
                controls.update();
                if (restoreSize) {
                    renderer.setSize(origW, origH);
                    camera.aspect = origW / origH;
                    camera.updateProjectionMatrix();
                }
                badge.textContent = '\u{1F534} Encoding GIF...';
                // Encode GIF
                setTimeout(() => this._encodeGif(frames, gw, gh, delay, overlay), 50);
                return;
            }

            const angle = getAngle(frame);
            const offset = initialOffset.clone().applyAxisAngle(rotAxis, angle);
            camera.position.copy(target).add(offset);
            camera.up.copy(initialUp).applyAxisAngle(rotAxis, angle);
            camera.lookAt(target);
            camera.updateMatrixWorld();
            renderer.render(scene.scene, camera);
            scene._captureScaleBar = true; scene._renderScaleBar(); scene._captureScaleBar = false;

            // Capture frame
            offCtx.drawImage(canvas, 0, 0, gw, gh);
            frames.push(offCtx.getImageData(0, 0, gw, gh));

            badge.textContent = `\u{1F534} Capturing frames...  ${Math.round(frame / totalFrames * 100)}%`;
            frame++;
            setTimeout(captureFrame, 0);
        };
        captureFrame();
    }

    _encodeGif(frames, w, h, delay, overlay) {
        // Minimal GIF89a encoder with LZW compression
        // Quantize each frame to 256 colors, write GIF with animation extension
        const badge = overlay.querySelector('div');

        const out = [];
        const writeByte = (b) => out.push(b & 0xFF);
        const writeShort = (v) => { writeByte(v); writeByte(v >> 8); };
        const writeStr = (s) => { for (let i = 0; i < s.length; i++) writeByte(s.charCodeAt(i)); };
        const writeBytes = (arr) => { for (let i = 0; i < arr.length; i++) out.push(arr[i]); };

        // Build global palette from first frame (median cut simplified to uniform 6x6x6 = 216 colors)
        const palSize = 256;
        const palette = new Uint8Array(palSize * 3);
        // 6x6x6 color cube + 40 grays
        let idx = 0;
        for (let r = 0; r < 6; r++)
            for (let g = 0; g < 6; g++)
                for (let b = 0; b < 6; b++) {
                    palette[idx*3]   = Math.round(r * 255 / 5);
                    palette[idx*3+1] = Math.round(g * 255 / 5);
                    palette[idx*3+2] = Math.round(b * 255 / 5);
                    idx++;
                }
        // Fill remaining with grays
        for (let i = idx; i < palSize; i++) {
            const v = Math.round((i - idx) * 255 / (palSize - idx - 1));
            palette[i*3] = palette[i*3+1] = palette[i*3+2] = v;
        }

        // Map RGBA pixel to nearest palette index
        const colorMap = new Map();
        const nearest = (r, g, b) => {
            const key = ((r >> 2) << 16) | ((g >> 2) << 8) | (b >> 2);
            if (colorMap.has(key)) return colorMap.get(key);
            // Check 6x6x6 cube first
            const ri = Math.round(r * 5 / 255);
            const gi = Math.round(g * 5 / 255);
            const bi = Math.round(b * 5 / 255);
            const ci = ri * 36 + gi * 6 + bi;
            colorMap.set(key, ci);
            return ci;
        };

        // GIF Header
        writeStr('GIF89a');
        writeShort(w);
        writeShort(h);
        writeByte(0xF7); // GCT flag, 8 bits, 256 colors
        writeByte(0);     // Background color index
        writeByte(0);     // Pixel aspect ratio

        // Global Color Table
        writeBytes(palette);

        // Netscape extension for looping
        writeByte(0x21); writeByte(0xFF); writeByte(11);
        writeStr('NETSCAPE2.0');
        writeByte(3); writeByte(1);
        writeShort(0); // loop forever
        writeByte(0);

        // LZW encode helper
        const lzwEncode = (indexedPixels, minCodeSize) => {
            const clearCode = 1 << minCodeSize;
            const eoiCode = clearCode + 1;
            let codeSize = minCodeSize + 1;
            let nextCode = eoiCode + 1;
            const table = new Map();

            const output = [];
            let buf = 0, bits = 0;
            const emit = (code) => {
                buf |= (code << bits);
                bits += codeSize;
                while (bits >= 8) {
                    output.push(buf & 0xFF);
                    buf >>= 8;
                    bits -= 8;
                }
            };

            // Init table
            const resetTable = () => {
                table.clear();
                for (let i = 0; i < clearCode; i++) table.set(String(i), i);
                nextCode = eoiCode + 1;
                codeSize = minCodeSize + 1;
            };

            emit(clearCode);
            resetTable();

            let cur = String(indexedPixels[0]);
            for (let i = 1; i < indexedPixels.length; i++) {
                const px = String(indexedPixels[i]);
                const key = cur + ',' + px;
                if (table.has(key)) {
                    cur = key;
                } else {
                    emit(table.get(cur));
                    if (nextCode < 4096) {
                        table.set(key, nextCode++);
                        if (nextCode > (1 << codeSize) && codeSize < 12) codeSize++;
                    } else {
                        emit(clearCode);
                        resetTable();
                    }
                    cur = px;
                }
            }
            emit(table.get(cur));
            emit(eoiCode);
            if (bits > 0) output.push(buf & 0xFF);
            return output;
        };

        // Write each frame
        for (let f = 0; f < frames.length; f++) {
            if (f % 10 === 0 && badge) {
                badge.textContent = `\u{1F534} Encoding GIF...  ${Math.round(f / frames.length * 100)}%`;
            }

            // Graphic Control Extension
            writeByte(0x21); writeByte(0xF9); writeByte(4);
            writeByte(0x00); // No transparency, no disposal
            writeShort(Math.round(delay / 10)); // Delay in centiseconds
            writeByte(0); // Transparent color index
            writeByte(0); // Block terminator

            // Image Descriptor (no local color table)
            writeByte(0x2C);
            writeShort(0); writeShort(0); // position
            writeShort(w); writeShort(h);
            writeByte(0); // No local color table

            // Quantize frame
            const data = frames[f].data;
            const pixels = new Uint8Array(w * h);
            for (let i = 0; i < w * h; i++) {
                pixels[i] = nearest(data[i*4], data[i*4+1], data[i*4+2]);
            }

            // LZW encode
            const minCodeSize = 8;
            writeByte(minCodeSize);
            const lzwData = lzwEncode(pixels, minCodeSize);
            // Write in sub-blocks (max 255 bytes each)
            let pos = 0;
            while (pos < lzwData.length) {
                const blockSize = Math.min(255, lzwData.length - pos);
                writeByte(blockSize);
                for (let j = 0; j < blockSize; j++) out.push(lzwData[pos + j]);
                pos += blockSize;
            }
            writeByte(0); // Block terminator
        }

        // GIF Trailer
        writeByte(0x3B);

        overlay.remove();

        // Download
        const blob = new Blob([new Uint8Array(out)], { type: 'image/gif' });
        const term = this.viewer.data.regexTerm || 'neuron';
        _saveFileAs(blob, `${term}_rotation.gif`, [
            { description: 'GIF Image', accept: { 'image/gif': ['.gif'] } }
        ]);
    }

    _buildInstructionsButton() {
        // Fixed footer inside type panel — always at bottom via flex layout
        const footer = document.createElement('div');
        footer.style.cssText = 'flex-shrink:0;background:rgba(20,20,20,0.98);padding:8px 10px 10px 10px;border-top:1px solid #444;';

        const btn = document.createElement('button');
        btn.textContent = 'Instructions';
        btn.style.cssText = `width:100%;padding:6px;font-size:12px;font-family:sans-serif;font-weight:bold;color:#fff;background:rgba(30,30,30,0.97);border:1px solid #555;border-radius:4px;cursor:pointer;`;
        btn.onmouseenter = () => { btn.style.background = 'rgba(60,60,60,0.97)'; };
        btn.onmouseleave = () => { btn.style.background = 'rgba(30,30,30,0.97)'; };

        let overlay = null;
        btn.onclick = () => {
            if (overlay) {
                overlay.remove();
                overlay = null;
                return;
            }
            overlay = this._buildInstructionsOverlay();
        };

        footer.appendChild(btn);
        this.typePanel.appendChild(footer);
    }

    _buildInstructionsOverlay() {
        const ov = document.createElement('div');
        ov.style.cssText = `position:fixed;bottom:60px;right:${TYPE_PANEL_W + 20}px;width:520px;max-height:70vh;background:rgba(15,15,15,0.97);color:#ccc;border:1px solid #555;border-radius:8px;font-family:sans-serif;font-size:13px;line-height:1.6;overflow:hidden;display:flex;flex-direction:column;z-index:2000;`;

        const header = document.createElement('div');
        header.style.cssText = 'flex-shrink:0;background:rgba(15,15,15,0.97);padding:14px 20px 8px 20px;text-align:right;border-bottom:1px solid #333;';
        const close = document.createElement('span');
        close.textContent = '\u2715';
        close.style.cssText = 'font-size:20px;cursor:pointer;color:#888;';
        close.onclick = () => ov.remove();
        header.appendChild(close);
        ov.appendChild(header);

        const body = document.createElement('div');
        body.style.cssText = 'overflow-y:auto;padding:16px 20px 20px 20px;flex:1;';
        ov.appendChild(body);

        const content = document.createElement('div');
        body.appendChild(content);
        content.innerHTML = `
            <h3 style="color:#d4a017;font-size:14px;">⬡ Orientation Gizmo</h3>
            <p style="border:1px solid #444;border-radius:5px;padding:8px;background:rgba(255,255,255,0.04);">The lower-left widget shows a whole-brain reference with anatomical axes (D/V, L/R, A/P). The <b>gray box</b> tracks your zoom/pan position. When Z-section is active, an <b>orange/blue ellipse</b> shows the clipping plane. Right-click the gizmo for snap-to-view and axis inversion options.</p>

            <h3 style="color:#d4a017;font-size:14px;">3D View — Navigation</h3>
            <p><b>Orbit:</b> Left-drag anywhere in the 3D canvas. <b>Pan:</b> Right-drag or Ctrl+drag. <b>Zoom:</b> Scroll wheel. Rotation stops immediately on mouse release (no drift). Full rotation is available in all directions.</p>

            <h3 style="color:#d4a017;font-size:14px;">3D View — Interaction</h3>
            <p><b>Hover</b> over a neuron to see its type and ROI in the info box. <b>Double-click</b> a neuron to highlight/unhighlight its type. <b>Right-double-click</b> to toggle the hovered ROI on/off.</p>

            <h3 style="color:#d4a017;font-size:14px;">Top Bar — Color Modes</h3>
            <p><b>Color by:</b> Switch between built-in modes (<em>Cell Type, Instance, Predicted NT, Custom</em>) and any uploaded CSV modes. <b>+</b> button uploads a new color CSV. <b>\u{1F500}</b> shuffles color assignments. <b>Right-click</b> a color mode button to change its colormap or remove an uploaded mode.</p>
            <p><b>CSV format:</b> First column must be <code>type</code> (type-level) or <code>bodyid</code> (instance-level). Remaining columns become color modes. If a column immediately after a value column contains CSS colors (e.g. <code>#ff0000</code>), those colors are used as the default mapping. Instance-level modes are only active in Neuron mode.</p>

            <h3 style="color:#d4a017;font-size:14px;">Top Bar — Controls</h3>
            <p><b>Z-section:</b> Slice the volume at a given depth. <b>\u2725 Pan mode:</b> Swap left/right mouse button for pan vs orbit. <b>\u{1F50D} Magnifier:</b> GPU-based pixel picking for precise neuron identification. <b>\u21BA Reset camera.</b> <b>\u{1F4BE} Save session:</b> Export current state (highlights, camera, colors, synapse groups) as JSON. <b>\u{1F4C2} Load session:</b> Restore from a saved JSON file. <b>\u{1F4F7} Screenshot:</b> Save a PNG with camera position encoded in the filename. <b>\u{1F3AC} Video:</b> Record a rotation video (AVI, WebM, or GIF). All exports open a Save As dialog so you can choose the directory and filename.</p>

            <h3 style="color:#d4a017;font-size:14px;">Right Sidebar — Type / Neuron Panel</h3>
            <p>Toggle between <b>Type</b> and <b>Neuron</b> mode with the switch at top. <b>Mesh / Skeleton</b> toggle switches between 3D surface meshes and wire-frame skeletons (when meshes are available). In skeleton mode: <b>Neuron size</b> adjusts line width and <b>Show somata</b> toggles cell bodies. <b>Preview on hover</b> shows the full mesh or skeleton of a highlighted+clipped neuron when you hover over it.</p>
            <p><b>Double-click</b> a row to highlight/unhighlight. <b>Single-click</b> a highlighted row to load it in the Connections panel. The <b>clip</b> checkbox clips each type/neuron to its ROI boundaries. <b>Saved Sets</b> let you store and recall highlight+clip configurations.</p>

            <h3 style="color:#d4a017;font-size:14px;">Left Sidebar — ROIs</h3>
            <p><b>Double-click</b> an ROI to toggle its visibility. <b>Single-click</b> to select it for the Connections panel. Synapse counts reflect only currently highlighted neurons. <b>Select all ROIs</b> checkbox at top. <b>Saved Sets</b> let you store ROI visibility configurations.</p>

            <h3 style="color:#d4a017;font-size:14px;">Connections Panel</h3>
            <p>Shows upstream and downstream synaptic partners for the selected type or neuron, broken down by ROI. <b>Double-click</b> an ROI header to toggle visibility. Click a partner name to highlight it. <b>\u2193 CSV</b> exports the connection table. <b>Right-click</b> a partner row to show presynapses or postsynapses in the 3D view as colored spheres.</p>

            <h3 style="color:#d4a017;font-size:14px;">Synapse Groups Panel</h3>
            <p>Manage synapse visualization groups. Click <b>+</b> to upload a synapse CSV (<code>bodyid_pre, bodyid_post, category</code> columns; optional 4th color column). Groups appear as colored spheres with outlines. Use the <b>eye</b> toggle for visibility, <b>color swatch</b> to change colors, and <b>\u00D7</b> to remove. <b>Right-click</b> a group to split it by individual neuron. The <b>Size</b> slider adjusts sphere radius.</p>

            <h3 style="color:#d4a017;font-size:14px;">Session Persistence</h3>
            <p>Your viewing state (highlights, camera, color mode, synapse groups, uploaded CSVs) is automatically saved to browser localStorage and restored when you reopen the same HTML. Use <b>\u{1F4BE} Save</b> / <b>\u{1F4C2} Load</b> to export and share session files across browsers or machines.</p>

            <h3 style="color:#d4a017;font-size:14px;">Video Export</h3>
            <p>Click <b>\u{1F3AC}</b> to record a rotation video. Choose motion type (360\u00B0 or pivot), axis of rotation, center point, and coordinate space. Available formats: <b>AVI</b> (MJPEG, universal), <b>WebM</b>, and <b>GIF</b>. Use Preview to test the animation before recording.</p>

            <h3 style="color:#d4a017;font-size:14px;">CSV Upload Formats</h3>
            <div style="border:1px solid #444;border-radius:5px;padding:10px;background:rgba(255,255,255,0.03);margin-bottom:10px;">
                <p style="margin:0 0 6px 0;"><b>Color Mode CSV</b> (+ button in Color by bar):</p>
                <p style="margin:0 0 4px 0;">First column: <code>type</code> (all neurons of a type share one color) or <code>bodyid</code> (each neuron gets its own color).</p>
                <p style="margin:0 0 4px 0;">Remaining columns: numeric values (continuous colormap) or text categories (discrete colors).</p>
                <p style="margin:0 0 4px 0;">Optional: a color column (CSS colors like <code>#ff0000</code>, <code>rgb(255,0,0)</code>, or <code>red</code>) immediately after a value column overrides the auto-generated colors.</p>
                <pre style="background:#1a1a1a;padding:6px 8px;border-radius:3px;font-size:11px;color:#aaa;margin:6px 0 0 0;overflow-x:auto;">type,score,color        bodyid,cluster,color
FB1A,0.85,#e41a1c       10539,alpha,#e41a1c
FB1B,-0.32,#377eb8      11307,beta,#377eb8</pre>
            </div>
            <div style="border:1px solid #444;border-radius:5px;padding:10px;background:rgba(255,255,255,0.03);">
                <p style="margin:0 0 6px 0;"><b>Synapse Group CSV</b> (+ button in Synapse Groups panel):</p>
                <p style="margin:0 0 4px 0;">Required columns: <code>bodyid_pre</code>, <code>bodyid_post</code>, plus a category or color column.</p>
                <p style="margin:0 0 4px 0;">3-column format: third column is either category names (auto-colored) or direct CSS colors.</p>
                <p style="margin:0 0 4px 0;">4-column format: third column is category, fourth column is CSS color per category.</p>
                <pre style="background:#1a1a1a;padding:6px 8px;border-radius:3px;font-size:11px;color:#aaa;margin:6px 0 0 0;overflow-x:auto;">bodyid_pre,bodyid_post,pathway,color
10539,11307,excitatory,#e41a1c
11307,12449,inhibitory,#377eb8</pre>
            </div>

            <h3 style="color:#d4a017;font-size:14px;">Data Source</h3>
            <p>Connectome data from <a href="https://${this.viewer.data.raw.dataSource ? this.viewer.data.raw.dataSource.server : 'neuprint-cns.janelia.org'}" style="color:#6a9fc0;">${this.viewer.data.raw.dataSource ? this.viewer.data.raw.dataSource.server : 'neuprint-cns.janelia.org'}</a>, dataset <em>${this.viewer.data.raw.dataSource ? this.viewer.data.raw.dataSource.dataset : 'cns'}</em>. Skeletons via navis; connectivity via neuprint-python.</p>
        `;
        document.body.appendChild(ov);
        return ov;
    }

    syncAllState() {
        // Deselect active saved set if current state has diverged from it
        if (this._activeSetIdx !== null && !this._restoringSet) {
            const s = this._savedSets[this._activeSetIdx];
            if (!s || !this._setMatchesCurrent(s)) {
                this._setsBtns.forEach(b => { b.style.background = '#222'; b.style.color = '#fff'; });
                this._activeSetIdx = null;
            }
        }
        // Deselect active ROI saved set if ROI state has diverged
        if (this._roiActiveSetIdx !== null && !this._restoringRoiSet) {
            const rs = this._roiSavedSets[this._roiActiveSetIdx];
            if (!rs || !this._roiSetMatchesCurrent(rs)) {
                this._roiSetsBtns.forEach(b => { b.style.background = '#222'; b.style.color = '#fff'; });
                this._roiActiveSetIdx = null;
            }
        }
        // Sync panel row borders and clip checkboxes (works for both type and neuron mode)
        for (const [key, row] of Object.entries(this.typeRows)) {
            const clipCb = this.clipCbs[key];
            if (clipCb) clipCb.checked = this.vis.clipToRoi[key] === true;
            if (row) {
                row.style.border = this.vis.highlightedSet.has(key) ? '1px solid #d4a017' : '1px solid transparent';
                if (key === this.connSelectedKey) {
                    row.style.background = 'rgba(100,149,237,0.25)';
                } else {
                    row.style.background = 'none';
                }
            }
        }

        // Sync highlight all checkbox
        const panelKeys = Object.keys(this.typeRows);
        const total = panelKeys.length;
        const highlighted = panelKeys.filter(k => this.vis.highlightedSet.has(k)).length;
        this.hlAllCb.checked = highlighted === total && total > 0;
        this.hlAllCb.indeterminate = highlighted > 0 && highlighted < total;

        // Sync clip all checkbox
        let clipCount = 0;
        for (const k of panelKeys) {
            if (this.vis.highlightedSet.has(k) && this.vis.clipToRoi[k] === true) clipCount++;
        }
        this.clipAllCb.checked = highlighted > 0 && clipCount === highlighted;
        this.clipAllCb.indeterminate = clipCount > 0 && clipCount < highlighted;

        // Sync ROI checkboxes
        let roiAllChecked = true, roiNoneChecked = true;
        for (const roi of this.data.sidebarRois) {
            const els = this.roiLabels[roi];
            if (els) {
                els.row.style.border = this.vis.roiChecked[roi] === true ? '1px solid #d4a017' : '1px solid transparent';
            }
            if (this.vis.roiChecked[roi] === true) roiNoneChecked = false;
            else roiAllChecked = false;
        }
        const roiSelectAll = document.getElementById('roiSelectAllCb');
        if (roiSelectAll) {
            roiSelectAll.checked = roiAllChecked;
            roiSelectAll.indeterminate = !roiAllChecked && !roiNoneChecked;
        }

        // Update sidebar synapse counts based on highlighted types
        this._updateSidebarCounts();

        // Update ROI highlight
        this._updateRoiHighlight();

        // Re-apply or clear color filter sidebar styling
        if (this.vis.colorFilteredOutTypes.size > 0 || this.vis.colorFilteredOutNeurons.size > 0) {
            this._applySidebarColorFilter();
        } else {
            // Clear any leftover filter styling: reset opacity, textDecoration, and restore sort order
            this._clearSidebarColorFilter();
        }

        // Auto-save session
        if (this.viewer && this.viewer.session) {
            this.viewer.session.debouncedSave();
        }
    }

    _clearSidebarColorFilter() {
        // Reset all type/neuron row styling and restore original sort order
        for (const [key, row] of Object.entries(this.typeRows)) {
            if (!row) continue;
            row.style.opacity = '1';
            row.style.textDecoration = 'none';
            row.style.border = this.vis.highlightedSet.has(key) ? '1px solid #d4a017' : '1px solid transparent';
        }
        // Also reset neuron type headers
        if (this._neuronTypeHeaders) {
            for (const [t, hdr] of Object.entries(this._neuronTypeHeaders)) {
                if (hdr) { hdr.style.opacity = '1'; hdr.style.textDecoration = 'none'; hdr.style.border = '1px solid transparent'; }
            }
        }
        // Restore original sort order: re-append rows in allTypes order
        if (this.hlModeByNeuron) {
            for (const typeName of this.data.allTypes) {
                const hdr = this._neuronTypeHeaders && this._neuronTypeHeaders[typeName];
                if (hdr) this.panelContent.appendChild(hdr);
                const bids = this.data.getNeuronsForType(typeName);
                for (const bid of bids) {
                    const row = this.typeRows[bid];
                    if (row) this.panelContent.appendChild(row);
                }
            }
        } else {
            for (const typeName of this.data.allTypes) {
                const row = this.typeRows[typeName];
                if (row) this.panelContent.appendChild(row);
            }
        }
    }

    _updateSidebarCounts() {
        // Recompute synapse counts based on highlighted types, respecting color filter
        const hasColorFilter = this.vis.colorFilteredOutTypes.size > 0 || this.vis.colorFilteredOutNeurons.size > 0;
        if (this.vis.highlightedSet.size === 0 && !hasColorFilter) {
            // Show totals, reset all styles, restore original order
            for (const roi of this.data.sidebarRois) {
                const els = this.roiLabels[roi];
                if (!els) continue;
                const count = this.data.roiSynapseTotals[roi] || 0;
                els.label.textContent = `${roi} (${count.toLocaleString()})`;
                els.row.style.opacity = '1';
                els.row.style.textDecoration = 'none';
                els.row.style.cursor = 'pointer';
                els.row.dataset.disabled = 'false';
                this.roiContainer.appendChild(els.row);
            }
        } else {
            // Build the set of items to sum: highlighted items minus color-filtered items
            let itemsToSum;
            if (this.vis.highlightedSet.size === 0 && hasColorFilter) {
                // All highlighted but color filter active — sum all non-filtered types
                itemsToSum = this.data.allTypes.filter(t => !this.vis.colorFilteredOutTypes.has(t));
            } else {
                // Specific items highlighted — filter out color-filtered ones
                itemsToSum = [...this.vis.highlightedSet].filter(
                    t => !this.vis.colorFilteredOutTypes.has(t) && !this.vis.colorFilteredOutNeurons.has(t)
                );
            }
            // Sum connectivity for visible items
            const roiTotals = {};
            const unionRois = new Set();
            for (const t of itemsToSum) {
                const isType = (this.data.typeUpstream[t] !== undefined || this.data.typeDownstream[t] !== undefined);
                if (isType) {
                    const up = this.data.typeUpstream[t] || {};
                    const dn = this.data.typeDownstream[t] || {};
                    for (const [roi, partners] of Object.entries(up)) {
                        unionRois.add(roi);
                        let sum = 0;
                        for (const w of Object.values(partners)) sum += w;
                        roiTotals[roi] = (roiTotals[roi] || 0) + sum;
                    }
                    for (const [roi, partners] of Object.entries(dn)) {
                        unionRois.add(roi);
                        let sum = 0;
                        for (const w of Object.values(partners)) sum += w;
                        roiTotals[roi] = (roiTotals[roi] || 0) + sum;
                    }
                } else {
                    const nUp = this.data.neuronUpstream[t];
                    const nDn = this.data.neuronDownstream[t];
                    if (nUp) {
                        for (const [roi, roiData] of Object.entries(nUp)) {
                            unionRois.add(roi);
                            const types = roiData.__types__ || roiData;
                            let sum = 0;
                            for (const w of Object.values(types)) sum += w;
                            roiTotals[roi] = (roiTotals[roi] || 0) + sum;
                        }
                    }
                    if (nDn) {
                        for (const [roi, roiData] of Object.entries(nDn)) {
                            unionRois.add(roi);
                            const types = roiData.__types__ || roiData;
                            let sum = 0;
                            for (const w of Object.values(types)) sum += w;
                            roiTotals[roi] = (roiTotals[roi] || 0) + sum;
                        }
                    }
                }
            }

            // Sort: ROIs with synapses first (by count desc), then zero ROIs
            const sorted = this.data.sidebarRois.slice().sort((a, b) => {
                const aIn = unionRois.has(a) ? 1 : 0;
                const bIn = unionRois.has(b) ? 1 : 0;
                if (aIn !== bIn) return bIn - aIn;
                return (roiTotals[b] || 0) - (roiTotals[a] || 0);
            });

            // Reorder DOM and update styles
            for (const roi of sorted) {
                const els = this.roiLabels[roi];
                if (!els) continue;
                this.roiContainer.appendChild(els.row);
                const count = roiTotals[roi] || 0;
                if (unionRois.has(roi)) {
                    els.label.textContent = `${roi} (${count.toLocaleString()})`;
                    els.row.style.opacity = '1';
                    els.row.style.textDecoration = 'none';
                    els.row.style.cursor = 'pointer';
                    els.row.dataset.disabled = 'false';
                    // Auto-recheck ROIs that were unchecked by color filter
                    if (this._filterUncheckedRois && this._filterUncheckedRois.has(roi)) {
                        this._filterUncheckedRois.delete(roi);
                        this.vis.roiChecked[roi] = true;
                        els.row.style.border = '1px solid #d4a017';
                        this.vis._applyRoiMesh(roi);
                    }
                } else {
                    els.label.textContent = `${roi} (0)`;
                    els.row.style.opacity = '0.35';
                    els.row.style.textDecoration = 'line-through';
                    els.row.style.cursor = 'default';
                    els.row.dataset.disabled = 'true';
                    // Auto-uncheck zero ROIs (track for re-check when filter widens)
                    if (this.vis.roiChecked[roi]) {
                        if (!this._filterUncheckedRois) this._filterUncheckedRois = new Set();
                        this._filterUncheckedRois.add(roi);
                        this.vis.roiChecked[roi] = false;
                        els.row.style.border = '1px solid transparent';
                        this.vis._applyRoiMesh(roi);
                    }
                    // Deselect for conn if zero
                    if (roi === this.connSelectedRoi) {
                        this.connSelectedRoi = null;
                        els.row.style.background = 'none';
                    }
                }
            }
        }
    }

    _applyTheme(t) {
        // Only the viewer background changes — UI panels stay dark in both modes
        document.body.style.background = t.bodyBg;
        if (this.viewer && this.viewer.scene)
            this.viewer.scene.renderer.setClearColor(t.clearColor, 1);
        // Gizmo: update WebGL canvas and body div background so corners match the canvas
        if (this._gizmoWebGLCanvas) this._gizmoWebGLCanvas.style.background = t.gizmoBg;
        if (this._gizmoBody)       this._gizmoBody.style.background       = t.gizmoBg;
        // Update synapse outline colors to match theme (black outlines in light mode, white in dark)
        if (this.viewer && this.viewer.synapse) this.viewer.synapse._updateOutlineColors();
    }
}

// ---- Main Viewer ----
class NeuronViewer {
    constructor(rawData) {
        this.data = new DataStore(rawData);
        this.scene = new SceneManager(this.data);
        this.vis = new VisibilityManager(this);
        this.synapse = new SynapseManager(this);
        this.interaction = new InteractionManager(this);
        this.ui = new UIManager(this);

        this.session = new SessionManager(this);
        this.hoverPreviewEnabled = false;  // Default OFF, matching Plotly
        this.connSelectedKey = null;

        // Auto-save on camera changes (debounced through session manager)
        this.scene.controls.addEventListener('change', () => {
            if (this.session) this.session.debouncedSave();
        });

        // Default: all types highlighted (matching Plotly behavior)
        this.vis.highlightAll();
        this.ui.syncAllState();

        // Apply default light theme
        this.ui._applyTheme(THEMES.light);

        // Try restoring saved session
        const restored = this.session.tryRestore();
        if (restored) {
            console.log('Session restored from localStorage');
        }

        // Auto-create embedded synapse groups from generation-time CSVs
        if (DATA.synapseData && DATA.embeddedSynapseGroups && DATA.embeddedSynapseGroups.length > 0) {
            this.synapse.loadData().then(() => {
                if (!this.synapse.loaded) return;
                const existingLabels = new Set(this.synapse.groups.map(g => g.label));
                let created = 0;
                for (const sg of DATA.embeddedSynapseGroups) {
                    if (existingLabels.has(sg.label)) continue;
                    const allIdx = [];
                    for (const pk of sg.pairs) {
                        const arr = this.synapse.pairIndex.get(pk);
                        if (arr) allIdx.push(...arr);
                    }
                    if (allIdx.length > 0) {
                        this.synapse.createGroupFromIndices(allIdx, {
                            label: sg.label, color: sg.color, synapseType: 'both'
                        });
                        created++;
                    }
                }
                if (created > 0 && this.ui) this.ui._updateSynapsePanel();
            });
        }

        console.log('NeuronViewer initialized');
        console.log(`  Types: ${this.data.allTypes.length}`);
        console.log(`  ROIs: ${this.data.sidebarRois.length}`);
        console.log(`  Clipped geometries: ${this.scene.typeRoiGeom.size}`);
        console.log(`  Full geometries: ${this.scene.neuronFullGeom.size}`);
    }

    selectForConnectivity(key) {
        if (this.connSelectedKey === key) {
            this.connSelectedKey = null;
        } else {
            this.connSelectedKey = key;
        }
        this.ui.connSelectedKey = this.connSelectedKey;
        this.ui._updateConnPanel();
        this.ui.syncAllState();
    }
}

// ---- Initialize ----
window.addEventListener('DOMContentLoaded', function() {
    window.viewer = new NeuronViewer(DATA);
    // Hide loading screen
    const ls = document.getElementById('_loadScreen');
    if (ls) ls.style.display = 'none';
});

})();
"""


if __name__ == '__main__':
    main()
