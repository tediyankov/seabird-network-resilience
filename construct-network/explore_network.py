#!/usr/bin/env python3
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("── network_plots.py started ──────────────────────────────────────────────", flush=True)

"""
Manx Shearwater Foraging Network — Plots
=========================================

Reads outputs from build_network.py:
    ./network/nodes.csv
    ./network/edges_temporal.csv
    ./network/graph_snapshots/*.pkl

Produces:
    ./network/plots/fig_network_aggregate.png
        All edges across all time steps. Edge opacity ∝ cumulative weight.

    ./network/plots/fig_network_snapshot_{date}.png  (3 random days)
        Per-day graph. Edge opacity ∝ daily weight.

    ./network/plots/fig_adjacency_{date}.png  (same 3 days)
        Adjacency matrix for each of the 3 snapshot days.
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
NET_DIR   = Path("./network")
SNAP_DIR  = NET_DIR / "graph_snapshots"
OUT_DIR   = NET_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

CELTIC_BBOX = dict(lon_min=-14, lon_max=-4, lat_min=47.5, lat_max=61)
RANDOM_SEED = 42
N_SNAPSHOTS = 3

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG        = "#fafaf8"
C_LAND      = "#e8e4dc"
C_BORDER    = "#b0a898"
C_GRID      = "#dedad4"
C_TEXT      = "#2c2c2c"
C_SUBTEXT   = "#6b6560"
C_CAPTION   = "#9e9890"
C_PATCH     = "#2a6496"     # foraging patch nodes — muted blue
C_COLONY    = "#c0392b"     # colony nodes — red
C_EDGE      = "#444444"     # edge base colour

plt.rcParams.update({
    "figure.facecolor":   C_BG,
    "axes.facecolor":     C_BG,
    "axes.edgecolor":     C_GRID,
    "axes.linewidth":     0.0,
    "axes.labelcolor":    C_TEXT,
    "axes.labelsize":     8,
    "xtick.color":        C_SUBTEXT,
    "ytick.color":        C_SUBTEXT,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "xtick.major.width":  0.0,
    "ytick.major.width":  0.0,
    "grid.color":         C_GRID,
    "grid.linewidth":     0.0,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color":         C_TEXT,
    "savefig.dpi":        300,
    "savefig.facecolor":  C_BG,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

# ── Basemap ────────────────────────────────────────────────────────────────────
print("\n── Loading basemap ───────────────────────────────────────────────────────", flush=True)
land_gdf = None
try:
    import cartopy.io.shapereader as shpreader
    ne_path  = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    land_gdf = gpd.read_file(ne_path)
    print("   Natural Earth 10m loaded", flush=True)
except Exception:
    try:
        land_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        print("   Natural Earth low-res loaded (fallback)", flush=True)
    except Exception:
        print("   ⚠  No land shapefile — maps without land", flush=True)

if land_gdf is not None:
    clip = box(CELTIC_BBOX["lon_min"]-1, CELTIC_BBOX["lat_min"]-1,
               CELTIC_BBOX["lon_max"]+1, CELTIC_BBOX["lat_max"]+1)
    land_gdf = land_gdf.clip(clip)

def draw_land(ax, alpha=0.35):
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.plot(ax=ax, color=C_LAND, edgecolor=C_BORDER,
                      linewidth=0.3, zorder=2, alpha=alpha)

def set_extent(ax):
    ax.set_xlim(CELTIC_BBOX["lon_min"], CELTIC_BBOX["lon_max"])
    ax.set_ylim(CELTIC_BBOX["lat_min"], CELTIC_BBOX["lat_max"])
    ax.set_xticks([])
    ax.set_yticks([])

def apply_map_style(ax, title=""):
    ax.set_facecolor(C_BG)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="semibold",
                     color=C_TEXT, loc="left", pad=10)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n── Loading network data ──────────────────────────────────────────────────", flush=True)

nodes_df   = pd.read_csv(NET_DIR / "nodes.csv")
edges_df   = pd.read_csv(NET_DIR / "edges_temporal.csv")

# Node position lookup
pos = {row["node_id"]: (row["node_lon"], row["node_lat"])
       for _, row in nodes_df.iterrows()}
node_type  = {row["node_id"]: row["node_type"] for _, row in nodes_df.iterrows()}

print(f"   {len(nodes_df)} nodes, {len(edges_df):,} temporal edge records")

# Aggregate edges across all days
edge_agg = (
    edges_df
    .groupby(["node_i", "node_j"])["weight"]
    .sum()
    .reset_index()
    .rename(columns={"weight": "total_weight"})
)
max_agg_weight = edge_agg["total_weight"].max()

# ── Helper: draw one network map ───────────────────────────────────────────────
def draw_network_map(ax, edges, weight_col, max_weight, title=""):
    draw_land(ax, alpha=0.3)

    for _, row in edges.iterrows():
        if row["node_i"] not in pos or row["node_j"] not in pos:
            continue
        x0, y0 = pos[row["node_i"]]
        x1, y1 = pos[row["node_j"]]
        alpha  = 0.08 + 0.72 * (row[weight_col] / max_weight)
        lw     = 0.3 + 1.4 * (row[weight_col] / max_weight)
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_EDGE,
                alpha=alpha,
                lw=lw,
                mutation_scale=6,
            ),
            zorder=3,
        )

    for _, nrow in nodes_df.iterrows():
        nid    = nrow["node_id"]
        if nid not in pos:
            continue
        x, y   = pos[nid]
        colour = C_COLONY if nrow["node_type"] == "colony" else C_PATCH
        size   = 60 if nrow["node_type"] == "colony" else 20
        zorder = 6 if nrow["node_type"] == "colony" else 5
        ax.scatter(x, y, c=colour, s=size, zorder=zorder,
                   edgecolors="white", linewidths=0.5)

    for _, nrow in nodes_df[nodes_df["node_type"] == "colony"].iterrows():
        x, y  = pos[nrow["node_id"]]
        label = nrow["node_id"].replace("colony_", "")
        ax.text(x + 0.15, y + 0.1, label,
                fontsize=6, color=C_COLONY, fontweight="bold",
                zorder=7, clip_on=True)

    set_extent(ax)
    apply_map_style(ax, title=title)

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_PATCH,
               markersize=5, markeredgecolor="white", markeredgewidth=0.5,
               label="Foraging patch node"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_COLONY,
               markersize=7, markeredgecolor="white", markeredgewidth=0.5,
               label="Colony node"),
        Line2D([0], [0], color=C_EDGE, linewidth=1.2, alpha=0.5,
               label="Directed edge (opacity ∝ weight)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True,
              framealpha=0.9, edgecolor=C_GRID, fontsize=7, facecolor=C_BG)

# ── Fig 1: Aggregate network ───────────────────────────────────────────────────
print("\n── Plotting aggregate network ────────────────────────────────────────────", flush=True)
fig1, ax1 = plt.subplots(figsize=(8, 9))
n_days = edges_df["date"].nunique()
draw_network_map(
    ax1, edge_agg, "total_weight", max_agg_weight,
    title="Manx Shearwater foraging network · all time steps",
)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "fig_network_aggregate.png")
plt.close(fig1)
print("   fig_network_aggregate.png saved")

# ── Choose 3 random snapshot days ─────────────────────────────────────────────
snap_files = sorted(SNAP_DIR.glob("*.pkl"))
all_dates  = [f.stem for f in snap_files]

np.random.seed(RANDOM_SEED)
chosen_dates = np.random.choice(all_dates, size=min(N_SNAPSHOTS, len(all_dates)),
                                 replace=False)
chosen_dates = sorted(chosen_dates)
print(f"\n── Snapshot dates chosen: {chosen_dates}", flush=True)

# ── Figs 2–4: Per-day snapshots + adjacency matrices ──────────────────────────
node_ids_ordered = nodes_df["node_id"].tolist()

for date in chosen_dates:
    snap_path = SNAP_DIR / f"{date}.pkl"
    with open(snap_path, "rb") as f:
        G = pickle.load(f)

    day_edges = edges_df[edges_df["date"] == date].copy()
    if len(day_edges) == 0:
        print(f"   ⚠  No edges for {date}, skipping")
        continue

    max_day_weight = day_edges["weight"].max()

    # ── Snapshot map ──────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 9))
    n_edges_day = len(day_edges)
    total_w_day = day_edges["weight"].sum()
    draw_network_map(
        ax2, day_edges, "weight", max_day_weight,
        title=f"Foraging network · {date}",
    )
    fig2.tight_layout()
    out_snap = OUT_DIR / f"fig_network_snapshot_{date}.png"
    fig2.savefig(out_snap)
    plt.close(fig2)
    print(f"   {out_snap.name} saved")

    # ── Adjacency matrix ──────────────────────────────────────────────────────
    # Only show nodes that have at least one edge on this day
    active_nodes = sorted(set(day_edges["node_i"]) | set(day_edges["node_j"]))
    n_active     = len(active_nodes)
    node_idx     = {nid: i for i, nid in enumerate(active_nodes)}

    adj = np.zeros((n_active, n_active), dtype=float)
    for _, row in day_edges.iterrows():
        if row["node_i"] in node_idx and row["node_j"] in node_idx:
            i = node_idx[row["node_i"]]
            j = node_idx[row["node_j"]]
            adj[i, j] = row["weight"]   # directed: i→j only, no transpose

    # Tick labels: shorten node IDs
    labels = [nid.replace("foraging_patch_", "p").replace("patch_", "p")
                  .replace("colony_", "c.") for nid in active_nodes]

    fig3, ax3 = plt.subplots(figsize=(max(6, n_active * 0.22 + 1),
                                       max(5, n_active * 0.22 + 1)))
    fig3.patch.set_facecolor(C_BG)
    ax3.set_facecolor(C_BG)

    im = ax3.imshow(adj, cmap="YlOrBr", aspect="auto",
                    interpolation="nearest",
                    vmin=0, vmax=max_day_weight)

    ax3.set_xticks(range(n_active))
    ax3.set_yticks(range(n_active))
    ax3.set_xticklabels(labels, rotation=90, fontsize=5, color=C_TEXT)
    ax3.set_yticklabels(labels, fontsize=5, color=C_TEXT)

    # Colour-code tick labels by node type
    for tick, nid in zip(ax3.get_xticklabels(), active_nodes):
        tick.set_color(C_COLONY if node_type.get(nid) == "colony" else C_PATCH)
    for tick, nid in zip(ax3.get_yticklabels(), active_nodes):
        tick.set_color(C_COLONY if node_type.get(nid) == "colony" else C_PATCH)

    cbar = fig3.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
    cbar.set_label("Edge weight (trips)", fontsize=7, color=C_TEXT)
    cbar.ax.tick_params(labelsize=6, colors=C_TEXT)

    ax3.set_title(f"Adjacency matrix · {date}",
                  fontsize=10, fontweight="semibold", color=C_TEXT,
                  loc="left", pad=10)
    # remove subtitle and caption text calls
    for spine in ax3.spines.values():
        spine.set_visible(False)

    fig3.tight_layout()
    out_adj = OUT_DIR / f"fig_adjacency_{date}.png"
    fig3.savefig(out_adj)
    plt.close(fig3)
    print(f"   {out_adj.name} saved")

print("\n" + "="*60)
print("PLOTTING COMPLETE")
print("="*60)
print(f"  Outputs → {OUT_DIR}/")