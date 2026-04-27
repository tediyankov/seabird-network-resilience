#!/usr/bin/env python3
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("── build_network.py started ──────────────────────────────────────────────", flush=True)

"""
Manx Shearwater Foraging Network — Temporal Graph Construction
==============================================================

Inputs:
    ./manx_shearwater_foraging_nodes.csv   — foraging patch centroids (from cluster.py)
    ./data/processed_tracks_hmm.csv        — full tracking dataset with all behaviours

Method:
    1. Add colony lon/lat as additional nodes (one node per unique colony location).
    2. For each trip on each day, identify which nodes the trip visits (passes
       within VISIT_RADIUS_KM of a foraging node, or COLONY_RADIUS_KM of a
       colony node).
    3. For each ordered pair of nodes visited by the same trip on the same day,
       add a directed edge (or increment its weight).
    4. Each day becomes one snapshot graph in the temporal graph.

Edge weight = number of trips connecting node i → node j on that day.

VISIT RADIUS JUSTIFICATION:
    VISIT_RADIUS_KM = 20 km
        Matches the spatial scale of foraging patches identified by HDBSCAN
        (leaf selection, min_cluster_size=500). A trip is credited as visiting
        a patch if it passes within 20 km of the centroid — consistent with
        the typical within-patch movement range of Manx Shearwaters and
        ensuring that trips traversing a patch but not hitting the exact
        centroid are still counted.
    COLONY_RADIUS_KM = 5 km
        Colonies are point locations; GPS fixes near departure/arrival are
        tightly clustered. 5 km captures the full range of near-colony fixes
        without bleeding into the open sea.

Outputs (./network/):
    nodes.csv                — full node list (foraging patches + colonies)
    edges_temporal.csv       — one row per (day, node_i, node_j) with weight
    graph_snapshots/         — one .gpickle per day (networkx DiGraph)
    temporal_graph_summary.txt
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
import networkx as nx
import pickle

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TRACKS_PATH   = "./data/processed_tracks_hmm.csv"
NODES_PATH    = "./manx_shearwater_foraging_nodes.csv"
OUT_DIR       = Path("./network")
SNAP_DIR      = OUT_DIR / "graph_snapshots"
OUT_DIR.mkdir(exist_ok=True)
SNAP_DIR.mkdir(exist_ok=True)

VISIT_RADIUS_KM  = 20.0   # foraging node visit radius
COLONY_RADIUS_KM =  5.0   # colony node visit radius

# ── Haversine (vectorised) ─────────────────────────────────────────────────────
def haversine_km(lon1, lat1, lon2_arr, lat2_arr):
    """Distance in km from one point to an array of points."""
    R  = 6371.0
    φ1 = np.radians(lat1);  φ2 = np.radians(lat2_arr)
    dφ = φ2 - φ1
    dλ = np.radians(lon2_arr - lon1)
    a  = np.sin(dφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(dλ/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# ── 1. Load foraging nodes ─────────────────────────────────────────────────────
print("\n── Loading nodes ─────────────────────────────────────────────────────────", flush=True)
patch_nodes = pd.read_csv(NODES_PATH)
# Ensure consistent columns: node_id, node_lon, node_lat, n_fixes, node_type
patch_nodes = patch_nodes[["node_id", "node_lon", "node_lat", "n_fixes"]].copy()
patch_nodes["node_type"] = "foraging_patch"
patch_nodes["node_id"]   = patch_nodes["node_id"].astype(str).apply(lambda x: f"patch_{x}")
print(f"   {len(patch_nodes)} foraging patch nodes loaded")

# ── 2. Load tracks ─────────────────────────────────────────────────────────────
print("\n── Loading tracks ────────────────────────────────────────────────────────", flush=True)
tracks = pd.read_csv(TRACKS_PATH, low_memory=False)
print(f"   {len(tracks):,} fixes loaded")

# Parse datetime and extract date
tracks["datetime"] = pd.to_datetime(tracks["datetime"], utc=True, errors="coerce")
tracks["date"]     = tracks["datetime"].dt.date
tracks             = tracks.dropna(subset=["datetime", "lon", "lat", "trip_id"])
print(f"   {len(tracks):,} fixes after dropping missing datetime/lon/lat/trip_id")

# ── 3. Build colony nodes ──────────────────────────────────────────────────────
print("\n── Building colony nodes ─────────────────────────────────────────────────", flush=True)
colony_nodes = (
    tracks
    .dropna(subset=["colony", "colony_lon", "colony_lat"])
    .drop_duplicates(subset=["colony"])
    [["colony", "colony_lon", "colony_lat"]]
    .rename(columns={"colony": "colony_name",
                     "colony_lon": "node_lon",
                     "colony_lat": "node_lat"})
    .assign(
        node_id   = lambda df: "colony_" + df["colony_name"].astype(str),
        n_fixes   = 0,
        node_type = "colony",
    )
    [["node_id", "node_lon", "node_lat", "n_fixes", "node_type", "colony_name"]]
    .reset_index(drop=True)
)
print(f"   {len(colony_nodes)} colony nodes:")
for _, r in colony_nodes.iterrows():
    print(f"     {r['node_id']}  ({r['node_lon']:.3f}, {r['node_lat']:.3f})")

# ── 4. Merge into full node list ───────────────────────────────────────────────
nodes = pd.concat([patch_nodes, colony_nodes], ignore_index=True)
nodes["radius_km"] = np.where(nodes["node_type"] == "colony",
                               COLONY_RADIUS_KM, VISIT_RADIUS_KM)

# Arrays for fast distance computation
node_lons   = nodes["node_lon"].values
node_lats   = nodes["node_lat"].values
node_ids    = nodes["node_id"].values
node_radii  = nodes["radius_km"].values
N_NODES     = len(nodes)
print(f"\n   Total nodes: {N_NODES}  ({len(patch_nodes)} patches + {len(colony_nodes)} colonies)")

# ── 5. For each trip × day, find visited nodes ────────────────────────────────
print("\n── Identifying node visits per trip per day ──────────────────────────────", flush=True)

# Group by (date, trip_id) — each group is all fixes for one trip on one day
trip_day_groups = tracks.groupby(["date", "trip_id"], sort=False)
total_groups    = len(trip_day_groups)
print(f"   {total_groups:,} (date, trip_id) groups to process", flush=True)

edge_records = []   # list of dicts: {date, node_i, node_j}

processed = 0
for (date, trip_id), grp in trip_day_groups:

    # Sort fixes by time so visit order is chronological
    grp = grp.sort_values("datetime")
    lons_trip = grp["lon"].values
    lats_trip = grp["lat"].values
    n_fixes   = len(lons_trip)

    # For each node, find the index of the *first* fix within its radius.
    # This gives us the earliest moment the trip entered that node's area,
    # which we use to order visits chronologically.
    first_visit_idx = {}   # node_id → index of first fix within radius
    for ni in range(N_NODES):
        dists = haversine_km(node_lons[ni], node_lats[ni], lons_trip, lats_trip)
        within = np.where(dists <= node_radii[ni])[0]
        if len(within) > 0:
            first_visit_idx[node_ids[ni]] = within[0]

    # Sort visited nodes by the time they were first entered
    visited_ordered = sorted(first_visit_idx, key=lambda nid: first_visit_idx[nid])

    # Directed edges: for each consecutive pair in visit order, add source → target.
    # Also add all non-consecutive pairs (i → j for i before j) so that e.g.
    # colony → patch_A → patch_B produces edges colony→patch_A, colony→patch_B,
    # and patch_A→patch_B, capturing the full directed connectivity of the trip.
    if len(visited_ordered) >= 2:
        for idx_i, ni in enumerate(visited_ordered):
            for nj in visited_ordered[idx_i + 1:]:
                edge_records.append({
                    "date":   str(date),
                    "node_i": ni,
                    "node_j": nj,
                })

    processed += 1
    if processed % 5000 == 0:
        print(f"   ... {processed:,}/{total_groups:,} groups processed", flush=True)

print(f"   Done — {len(edge_records):,} raw edge events", flush=True)

# ── 6. Aggregate edges by (date, node_i, node_j) ──────────────────────────────
print("\n── Aggregating edge weights ──────────────────────────────────────────────", flush=True)
if len(edge_records) == 0:
    print("   ⚠  No edges found — check VISIT_RADIUS_KM and that trip_id is populated")
    sys.exit(1)

edges_df = (
    pd.DataFrame(edge_records)
    .groupby(["date", "node_i", "node_j"], sort=True)
    .size()
    .reset_index(name="weight")
)
print(f"   {len(edges_df):,} unique (date, node_i, node_j) edges")
print(f"   Dates spanned: {edges_df['date'].min()} → {edges_df['date'].max()}")
print(f"   Max edge weight on a single day: {edges_df['weight'].max()}")

# ── 7. Build per-day NetworkX graphs ──────────────────────────────────────────
print("\n── Building daily graph snapshots ────────────────────────────────────────", flush=True)
dates     = sorted(edges_df["date"].unique())
n_days    = len(dates)
print(f"   {n_days} days with at least one edge", flush=True)

graphs = {}
for date in dates:
    day_edges = edges_df[edges_df["date"] == date]
    G = nx.DiGraph()
    # Add all nodes (even isolated ones) with attributes
    for _, row in nodes.iterrows():
        G.add_node(row["node_id"],
                   lon       = row["node_lon"],
                   lat       = row["node_lat"],
                   node_type = row["node_type"],
                   n_fixes   = int(row["n_fixes"]))
    # Add edges
    for _, row in day_edges.iterrows():
        G.add_edge(row["node_i"], row["node_j"], weight=int(row["weight"]))
    graphs[date] = G

# Serialise snapshots
for date, G in graphs.items():
    out_path = SNAP_DIR / f"{date}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(G, f)

print(f"   Snapshots saved to {SNAP_DIR}/", flush=True)

# ── 8. Save outputs ────────────────────────────────────────────────────────────
print("\n── Saving outputs ────────────────────────────────────────────────────────", flush=True)
nodes.to_csv(OUT_DIR / "nodes.csv", index=False)
edges_df.to_csv(OUT_DIR / "edges_temporal.csv", index=False)
print(f"   nodes.csv          → {OUT_DIR}/nodes.csv")
print(f"   edges_temporal.csv → {OUT_DIR}/edges_temporal.csv")

# ── 9. Summary ────────────────────────────────────────────────────────────────
n_edges_total  = len(edges_df)
total_weight   = edges_df["weight"].sum()
mean_daily_edges = edges_df.groupby("date").size().mean()
mean_daily_weight = edges_df.groupby("date")["weight"].sum().mean()

# Aggregate edge weights across all days
edge_agg = (
    edges_df
    .groupby(["node_i", "node_j"])["weight"]
    .agg(total_weight="sum", n_days_active="count")
    .reset_index()
    .sort_values("total_weight", ascending=False)
)

summary_lines = [
    "Manx Shearwater Foraging Network — Temporal Graph Summary",
    "=" * 60,
    f"Nodes                  : {N_NODES}",
    f"  Foraging patches     : {len(patch_nodes)}",
    f"  Colonies             : {len(colony_nodes)}",
    f"",
    f"Temporal span          : {edges_df['date'].min()} → {edges_df['date'].max()}",
    f"Days with edges        : {n_days}",
    f"",
    f"Total edge events      : {n_edges_total:,}  (unique day×node_i×node_j)",
    f"Total trip connections : {total_weight:,}",
    f"Mean edges per day     : {mean_daily_edges:.1f}",
    f"Mean weight per day    : {mean_daily_weight:.1f}",
    f"",
    f"Visit radius (foraging): {VISIT_RADIUS_KM} km",
    f"Visit radius (colony)  : {COLONY_RADIUS_KM} km",
    f"",
    f"Top 10 edges by cumulative weight across all days:",
    edge_agg.head(10).to_string(index=False),
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(OUT_DIR / "temporal_graph_summary.txt", "w") as f:
    f.write(summary_text)

print(f"\n✔  Summary saved to {OUT_DIR}/temporal_graph_summary.txt")
print("\n" + "="*60)
print("NETWORK CONSTRUCTION COMPLETE")
print("="*60)
print(f"  Next step: python network_plots.py")