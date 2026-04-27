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
Manx Shearwater Foraging Patch — Plots
Celtic Seas – GPS Tracking Data
====================================================

Reads outputs written by cluster.py:
    ./checkpoints/foraging_clustered.csv
    ./manx_shearwater_foraging_nodes.csv

Outputs:
    ./plots/fig1_raw_foraging.png
    ./plots/fig2_dbscan.png
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("./checkpoints")
OUT_DIR        = Path("./plots")
OUT_DIR.mkdir(exist_ok=True)

CELTIC_BBOX               = dict(lon_min=-14, lon_max=-4, lat_min=47.5, lat_max=61)
MIN_CLUSTER_SIZE          = 150
MIN_SAMPLES               = 30
CLUSTER_SELECTION_EPSILON = 3.0

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG       = "#fafaf8"
C_LAND     = "#e8e4dc"
C_BORDER   = "#b0a898"
C_GRID     = "#dedad4"
C_TEXT     = "#2c2c2c"
C_SUBTEXT  = "#6b6560"
C_CAPTION  = "#9e9890"
C_NOISE    = "#c8c4bc"
C_RAW      = "#4a90c4"
C_CENTROID = "#c0392b"

def cluster_cmap(n):
    if n <= 10:
        base = plt.cm.tab10
    elif n <= 20:
        base = plt.cm.tab20
    else:
        base = plt.cm.turbo
    return [base(i / max(n - 1, 1)) for i in range(n)]

# ── Global matplotlib style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  C_BG,
    "axes.facecolor":    C_BG,
    "axes.edgecolor":    C_GRID,
    "axes.linewidth":    0.0,
    "axes.labelcolor":   C_TEXT,
    "axes.labelsize":    8,
    "xtick.color":       C_SUBTEXT,
    "ytick.color":       C_SUBTEXT,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "xtick.major.width": 0.0,
    "ytick.major.width": 0.0,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.0,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color":        C_TEXT,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.facecolor": C_BG,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

# ── Helpers ────────────────────────────────────────────────────────────────────
def apply_map_style(ax):
    ax.set_facecolor(C_BG)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def set_extent(ax):
    ax.set_xlim(CELTIC_BBOX["lon_min"], CELTIC_BBOX["lon_max"])
    ax.set_ylim(CELTIC_BBOX["lat_min"], CELTIC_BBOX["lat_max"])
    ax.set_xticks([])
    ax.set_yticks([])

def draw_land(ax):
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.plot(ax=ax, color=C_LAND, edgecolor=C_BORDER,
                      linewidth=0.4, zorder=2)

# ── Load basemap ───────────────────────────────────────────────────────────────
print("\n── Loading basemap ───────────────────────────────────────────────────────", flush=True)
land_gdf = None
try:
    import cartopy.io.shapereader as shpreader
    ne_path  = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    land_gdf = gpd.read_file(ne_path)
    print("   Natural Earth 10m land loaded (cartopy)", flush=True)
except Exception:
    try:
        land_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        print("   Natural Earth low-res land loaded (geopandas fallback)", flush=True)
    except Exception:
        print("   ⚠  No land shapefile available — maps will show fixes only", flush=True)

if land_gdf is not None:
    clip_box = box(CELTIC_BBOX["lon_min"] - 1, CELTIC_BBOX["lat_min"] - 1,
                   CELTIC_BBOX["lon_max"] + 1, CELTIC_BBOX["lat_max"] + 1)
    land_gdf = land_gdf.clip(clip_box)

# ── Load clustering outputs ────────────────────────────────────────────────────
print("\n── Loading clustering outputs ────────────────────────────────────────────", flush=True)

clustered_path = CHECKPOINT_DIR / "foraging_clustered.csv"
nodes_path     = Path("manx_shearwater_foraging_nodes.csv")

if not clustered_path.exists():
    sys.exit(f"✗  {clustered_path} not found — run cluster.py first")
if not nodes_path.exists():
    sys.exit(f"✗  {nodes_path} not found — run cluster.py first")

foraging  = pd.read_csv(clustered_path)
centroids = pd.read_csv(nodes_path)

lons      = foraging["lon"].values
lats      = foraging["lat"].values
cl_hdbscan = foraging["cl_hdbscan"].values
n          = len(foraging)
n_clusters = len(centroids)
n_noise    = int((cl_hdbscan == -1).sum())
noise_mask = cl_hdbscan == -1

print(f"   {n:,} fixes | {n_clusters} clusters | {n_noise:,} noise points")

# ── Figure 1 – Raw foraging fixes ─────────────────────────────────────────────
print("\n── Plotting ──────────────────────────────────────────────────────────────", flush=True)

fig1, ax1 = plt.subplots(figsize=(7, 8))
draw_land(ax1)
ax1.scatter(lons, lats, c=C_RAW, s=1.2, alpha=0.35, linewidths=0, zorder=3)
set_extent(ax1)
apply_map_style(ax1)
ax1.legend(
    handles=[Line2D([0], [0], marker="o", color="none",
                    markerfacecolor=C_RAW, markersize=5, alpha=0.7,
                    label=f"Foraging fix  (n = {n:,})")],
    loc="lower left", frameon=True, framealpha=0.9,
    edgecolor=C_GRID, fontsize=7, facecolor=C_BG)
fig1.tight_layout()
fig1.savefig(OUT_DIR / "fig1_raw_foraging.png")
plt.close(fig1)

# ── Figure 2 – DBSCAN clusters + nodes ────────────────────────────────────────
colours_cl = cluster_cmap(n_clusters)

fig2, ax2 = plt.subplots(figsize=(7, 8))
draw_land(ax2)

if noise_mask.any():
    ax2.scatter(lons[noise_mask], lats[noise_mask],
                c=C_NOISE, s=1, alpha=0.4, linewidths=0, zorder=3)

for ci, colour in enumerate(colours_cl):
    mask = cl_hdbscan == ci
    ax2.scatter(lons[mask], lats[mask],
                c=[colour], s=2.5, alpha=0.55, linewidths=0, zorder=4)

ax2.scatter(centroids["node_lon"], centroids["node_lat"],
            c=C_CENTROID, s=40, marker="o",
            edgecolors="white", linewidths=0.6, zorder=6)

set_extent(ax2)
apply_map_style(ax2)
ax2.legend(
    handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_NOISE,
               markersize=5, alpha=0.6, label=f"Unassigned  ({n_noise:,})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colours_cl[0],
               markersize=5, alpha=0.7, label=f"Cluster fix  ({(~noise_mask).sum():,})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CENTROID,
               markersize=6, markeredgecolor="white", markeredgewidth=0.6,
               label=f"Node centroid  ({len(centroids)})"),
    ],
    loc="lower left", frameon=True, framealpha=0.9,
    edgecolor=C_GRID, fontsize=7, facecolor=C_BG)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "fig2_hdbscan.png")
plt.close(fig2)
print("   fig2 saved")

print("\n" + "="*70)
print("PLOTTING COMPLETE")
print("="*70)
print(f"  fig1_raw_foraging.png → {OUT_DIR}/")
print(f"  fig2_hdbscan.png      → {OUT_DIR}/")