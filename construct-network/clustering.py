#!/usr/bin/env python3
# Force stdout/stderr to flush immediately — prevents log buffering on SLURM
import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("── Script started ────────────────────────────────────────────────────────", flush=True)
print(f"   Python {sys.version}", flush=True)
print(f"   PID {os.getpid()}", flush=True)

"""
Manx Shearwater Foraging Patch Clustering
Celtic Seas – GPS Tracking Data
====================================================

GPU-accelerated (CUDA via cupy/cuml if available) with memory-efficient
CPU fallback using chunked distance computation.

METHOD: HDBSCAN
    Unlike DBSCAN, HDBSCAN builds a hierarchy of clusters at all density
    levels and extracts the most stable ones. This handles the Irish Sea
    correctly: instead of merging all continuously-dense fixes into one giant
    cluster (as DBSCAN does at any single eps), HDBSCAN finds the natural
    density breaks within the dense region and splits it into separate patches.

    Key parameters:
        min_cluster_size : minimum number of fixes to constitute a foraging
                           patch. Set to ~150 (≈ ~0.07% of 202k fixes) —
                           small enough to recover genuinely distinct patches
                           in sparse offshore areas, large enough to suppress
                           artefacts from single-bird repeated transits.
        min_samples      : controls how conservative cluster boundaries are.
                           Higher = tighter cores, more noise. Set lower than
                           min_cluster_size to allow sparser patches at the
                           shelf edge to still be captured.
        cluster_selection_epsilon : soft minimum separation between cluster
                           cores (km). Prevents HDBSCAN from over-splitting
                           genuinely contiguous patches into dozens of tiny
                           sub-clusters. Set to 3 km — below the ~5 km
                           within-patch revisitation scale.

Outputs (consumed by plot.py):
    ./checkpoints/foraging_clustered.csv — foraging fixes with cl_hdbscan column
    ./manx_shearwater_foraging_nodes.csv — node centroid table
"""

import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ── GPU detection ──────────────────────────────────────────────────────────────
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    _ = cp.array([1.0])
    GPU_AVAILABLE = True
    print("✔  GPU (CUDA) detected — using cuML HDBSCAN", flush=True)
except Exception:
    import hdbscan as hdbscan_cpu
    print("ℹ  No GPU / cuML — using CPU hdbscan", flush=True)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "./data/processed_tracks_hmm.csv"
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

CELTIC_BBOX = dict(lon_min=-14, lon_max=-4, lat_min=47.5, lat_max=61)

# HDBSCAN parameters
MIN_CLUSTER_SIZE          = 500   # minimum fixes per foraging patch
MIN_SAMPLES               = 50    # core point threshold — lower than
                                  # min_cluster_size to capture shelf-edge patches
CLUSTER_SELECTION_EPSILON = 0   # km — minimum inter-cluster separation

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── 1. Load & filter ───────────────────────────────────────────────────────────
print("\n── Loading data ──────────────────────────────────────────────────────────", flush=True)
tracks = pd.read_csv(DATA_PATH, low_memory=False)
print(f"   Loaded {len(tracks):,} rows, {tracks.shape[1]} columns")

foraging = (
    tracks
    .query("behaviour == 'Foraging'")
    .query("@CELTIC_BBOX['lon_min'] <= lon <= @CELTIC_BBOX['lon_max']")
    .query("@CELTIC_BBOX['lat_min'] <= lat <= @CELTIC_BBOX['lat_max']")
    .dropna(subset=["lon", "lat"])
    .drop_duplicates(subset=["lon", "lat"])
    .reset_index(drop=True)
)
print(f"   {len(foraging):,} foraging fixes after Celtic Sea filter")
lons = foraging["lon"].values.astype(np.float64)
lats = foraging["lat"].values.astype(np.float64)
n    = len(foraging)

# ── 2. Projected coordinates (LAEA km) ────────────────────────────────────────
# HDBSCAN runs on euclidean distance in projected space.
# LAEA centred on the study area gives < 0.5% planar error across the Celtic Seas.
LON_C = (CELTIC_BBOX["lon_min"] + CELTIC_BBOX["lon_max"]) / 2
LAT_C = (CELTIC_BBOX["lat_min"] + CELTIC_BBOX["lat_max"]) / 2
proj_str    = f"+proj=laea +lon_0={LON_C} +lat_0={LAT_C} +units=km +datum=WGS84"
transformer = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
x_km, y_km  = transformer.transform(lons, lats)
coords_km   = np.column_stack([x_km, y_km])

# ── 3. HDBSCAN ────────────────────────────────────────────────────────────────
print("\n── HDBSCAN ───────────────────────────────────────────────────────────────", flush=True)
print(f"   min_cluster_size          = {MIN_CLUSTER_SIZE}", flush=True)
print(f"   min_samples               = {MIN_SAMPLES}", flush=True)
print(f"   cluster_selection_epsilon = {CLUSTER_SELECTION_EPSILON} km", flush=True)

t0 = time.time()

if GPU_AVAILABLE:
    clusterer = cuHDBSCAN(
        min_cluster_size          = MIN_CLUSTER_SIZE,
        min_samples               = MIN_SAMPLES,
        cluster_selection_epsilon = CLUSTER_SELECTION_EPSILON,
        metric                    = "euclidean",
        output_type               = "numpy",
        cluster_selection_method  = "leaf",
    )
    cl_hdbscan = clusterer.fit_predict(coords_km.astype(np.float32)).astype(int)
else:
    clusterer = hdbscan_cpu.HDBSCAN(
        min_cluster_size          = MIN_CLUSTER_SIZE,
        min_samples               = MIN_SAMPLES,
        cluster_selection_epsilon = CLUSTER_SELECTION_EPSILON,
        metric                    = "euclidean",
        core_dist_n_jobs          = -1,
        cluster_selection_method  = "leaf",  
    )
    cl_hdbscan = clusterer.fit_predict(coords_km)

print(f"   Done in {time.time()-t0:.1f}s", flush=True)

foraging["cl_hdbscan"] = cl_hdbscan
n_clusters = len(set(cl_hdbscan[cl_hdbscan >= 0]))
n_noise    = int(np.sum(cl_hdbscan == -1))
print(f"   → {n_clusters} clusters, {n_noise:,} noise points "
      f"({100*n_noise/n:.1f}% of fixes)")

# Cluster size distribution for quick sanity check
sizes = pd.Series(cl_hdbscan).value_counts().sort_index()
sizes = sizes[sizes.index >= 0]
print(f"\n   Cluster size summary:")
print(f"     min    : {sizes.min():,}")
print(f"     median : {int(sizes.median()):,}")
print(f"     max    : {sizes.max():,}")
print(f"     top 5  : {sizes.nlargest(5).to_dict()}")

# ── 4. Centroids ──────────────────────────────────────────────────────────────
assigned  = foraging[foraging["cl_hdbscan"] >= 0]
centroids = (
    assigned
    .groupby("cl_hdbscan")
    .agg(node_lon=("lon", "mean"), node_lat=("lat", "mean"), n_fixes=("lon", "count"))
    .reset_index()
    .rename(columns={"cl_hdbscan": "node_id"})
    .sort_values("node_id")
    .reset_index(drop=True)
)
print(f"\n✔  {len(centroids)} network nodes defined")
print(centroids.to_string(index=False))

# ── 5. Save outputs ────────────────────────────────────────────────────────────
clustered_path = CHECKPOINT_DIR / "foraging_clustered.csv"
foraging.to_csv(clustered_path, index=False)
print(f"\n✔  Clustered fixes saved to {clustered_path}")

centroids.to_csv("manx_shearwater_foraging_nodes.csv", index=False)
print("✔  manx_shearwater_foraging_nodes.csv saved")

print("\n" + "="*70)
print("CLUSTERING COMPLETE")
print("="*70)
print(f"  Input fixes      : {n:,}")
print(f"  HDBSCAN clusters : {n_clusters}  (noise: {n_noise:,}, {100*n_noise/n:.1f}%)")
print(f"  Network nodes    : {len(centroids)}")
print(f"\n  Next step: python plot.py")