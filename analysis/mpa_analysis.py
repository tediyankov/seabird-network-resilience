#!/usr/bin/env python3
"""
MPA Intersection Analysis — Manx Shearwater Foraging Network
=============================================================

Inputs:
    ./data/uk.shp                          — JNCC UK MPA polygons
    ./data/ireland.shp                     — JNCC Ireland MPA polygons
    ./network/nodes.csv                    — network nodes (patches + colonies)
    ./data/processed_tracks_hmm.csv        — full tracking dataset

Outputs:
    ./network/mpa_analysis/
        mpa_union.shp                      — merged UK + Ireland MPA polygons
        nodes_mpa.csv                      — nodes with in_mpa flag + MPA name
        foraging_fixes_mpa.csv             — foraging fixes with in_mpa flag
        mpa_summary.txt                    — printed summary of all results
        fig_mpa_nodes.png                  — map: MPA polygons + nodes coloured by MPA status
        fig_mpa_fixes.png                  — map: MPA polygons + foraging fixes coloured by MPA status
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
UK_SHP       = "./data/uk.shp"
IRE_SHP      = "./data/ireland.shp"
NODES_PATH   = "./network/nodes.csv"
TRACKS_PATH  = "./data/processed_tracks_hmm.csv"

OUT_DIR = Path("./network/mpa_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELTIC_BBOX = dict(lon_min=-14, lon_max=-4, lat_min=47.5, lat_max=61)

# ── Palette ────────────────────────────────────────────────────────────────────
C_BG       = "#fafaf8"
C_LAND     = "#e8e4dc"
C_BORDER   = "#b0a898"
C_GRID     = "#dedad4"
C_TEXT     = "#2c2c2c"
C_SUBTEXT  = "#6b6560"
C_CAPTION  = "#9e9890"
C_MPA      = "#2171b5"      # blue — inside MPA
C_NO_MPA   = "#bdbdbd"      # grey — outside MPA
C_MPA_POLY = "#c6dbef"      # light blue fill for MPA polygons
C_COLONY   = "#c0392b"

plt.rcParams.update({
    "figure.facecolor":   C_BG,
    "axes.facecolor":     C_BG,
    "axes.edgecolor":     C_GRID,
    "axes.linewidth":     0.6,
    "axes.labelcolor":    C_TEXT,
    "axes.labelsize":     8,
    "xtick.color":        C_SUBTEXT,
    "ytick.color":        C_SUBTEXT,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "grid.color":         C_GRID,
    "grid.linewidth":     0.4,
    "grid.linestyle":     "--",
    "grid.alpha":         0.8,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color":         C_TEXT,
    "savefig.dpi":        300,
    "savefig.facecolor":  C_BG,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

def set_extent(ax):
    ax.set_xlim(CELTIC_BBOX["lon_min"], CELTIC_BBOX["lon_max"])
    ax.set_ylim(CELTIC_BBOX["lat_min"], CELTIC_BBOX["lat_max"])
    ax.set_xticks(range(-14, -3, 2))
    ax.set_yticks(range(48, 62, 2))
    ax.set_xticklabels([f"{abs(x)}°W" for x in range(-14, -3, 2)])
    ax.set_yticklabels([f"{y}°N"      for y in range(48, 62, 2)])

def apply_map_style(ax, title="", subtitle="", caption=""):
    ax.set_facecolor(C_BG)
    ax.grid(True, zorder=0)
    ax.set_xlabel("Longitude (°)", labelpad=4)
    ax.set_ylabel("Latitude (°)",  labelpad=4)
    if title:
        ax.set_title(title, fontsize=10, fontweight="semibold",
                     color=C_TEXT, loc="left", pad=10)
    if subtitle:
        ax.text(0.0, 1.025, subtitle, transform=ax.transAxes,
                color=C_SUBTEXT, fontsize=7.5, va="bottom", ha="left")
    if caption:
        ax.text(1.0, -0.06, caption, transform=ax.transAxes,
                color=C_CAPTION, fontsize=6.5, ha="right", style="italic")

# ── Basemap ────────────────────────────────────────────────────────────────────
land_gdf = None
try:
    import cartopy.io.shapereader as shpreader
    ne_path  = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    land_gdf = gpd.read_file(ne_path)
except Exception:
    try:
        land_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        pass

if land_gdf is not None:
    clip = box(CELTIC_BBOX["lon_min"]-1, CELTIC_BBOX["lat_min"]-1,
               CELTIC_BBOX["lon_max"]+1, CELTIC_BBOX["lat_max"]+1)
    land_gdf = land_gdf.clip(clip)

def draw_land(ax):
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.plot(ax=ax, color=C_LAND, edgecolor=C_BORDER,
                      linewidth=0.4, zorder=2)

# ── 1. Load & merge MPA shapefiles ────────────────────────────────────────────
print("\n── Loading MPA shapefiles ────────────────────────────────────────────────", flush=True)
uk  = gpd.read_file(UK_SHP)
ire = gpd.read_file(IRE_SHP)
print(f"   UK  : {len(uk):,} polygons  |  CRS: {uk.crs}")
print(f"   Eire: {len(ire):,} polygons  |  CRS: {ire.crs}")
print(f"   UK  bounds: {uk.total_bounds}")
print(f"   Eire bounds: {ire.total_bounds}")

# Bounds confirm WGS84 — set CRS label first, then no reprojection needed
if uk.crs is None:
    uk = uk.set_crs("EPSG:4326")
if ire.crs is None:
    ire = ire.set_crs("EPSG:4326")

# Align columns — keep only shared columns + geometry for a clean merge
shared_cols = list(set(uk.columns) & set(ire.columns))
mpa = pd.concat([uk[shared_cols], ire[shared_cols]], ignore_index=True)
mpa = gpd.GeoDataFrame(mpa, crs="EPSG:4326")

print(f"\n   Merged: {len(mpa):,} MPA polygons")
print(f"   Columns: {list(mpa.columns)}")

# Clip to Celtic Seas study area
clip_box = box(CELTIC_BBOX["lon_min"], CELTIC_BBOX["lat_min"],
               CELTIC_BBOX["lon_max"], CELTIC_BBOX["lat_max"])
mpa = mpa.clip(clip_box)
mpa = mpa[~mpa.geometry.is_empty].reset_index(drop=True)

# Clip can produce linestrings/points at bbox edges — keep only polygons
mpa = mpa[mpa.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
print(f"   After clipping to Celtic Seas bbox: {len(mpa):,} polygons")

# Save merged shapefile
mpa.to_file(OUT_DIR / "mpa_union.shp")
print(f"   Saved → {OUT_DIR}/mpa_union.shp")

# Single union geometry for point-in-polygon (faster than polygon-by-polygon)
mpa_union = mpa.geometry.union_all()

# ── 2. Node MPA analysis ──────────────────────────────────────────────────────
print("\n── Analysing nodes ───────────────────────────────────────────────────────", flush=True)
nodes = pd.read_csv(NODES_PATH)
nodes_gdf = gpd.GeoDataFrame(
    nodes,
    geometry=gpd.points_from_xy(nodes["node_lon"], nodes["node_lat"]),
    crs="EPSG:4326"
)

# Spatial join to get MPA name for each node (left join keeps all nodes)
# Identify the name column — common JNCC column names
name_col = next((c for c in mpa.columns
                 if c.upper() in ["NAME", "SITE_NAME", "MPA_NAME", "SITENAME"]), None)

if name_col:
    joined = gpd.sjoin(nodes_gdf, mpa[[name_col, "geometry"]],
                       how="left", predicate="within")
    # A node may match multiple MPA polygons — keep first match per node
    joined = joined[~joined.index.duplicated(keep="first")]
    nodes_gdf["in_mpa"]   = ~joined["index_right"].isna()
    nodes_gdf["mpa_name"] = joined[name_col].values
else:
    joined = gpd.sjoin(nodes_gdf, mpa[["geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    nodes_gdf["in_mpa"]   = ~joined["index_right"].isna()
    nodes_gdf["mpa_name"] = None

# Results
n_nodes        = len(nodes_gdf)
n_in_mpa       = nodes_gdf["in_mpa"].sum()
pct_nodes      = 100 * n_in_mpa / n_nodes

print(f"   Total nodes       : {n_nodes}")
print(f"   Nodes inside MPA  : {n_in_mpa}  ({pct_nodes:.1f}%)")
print(f"\n   By node type:")
node_summary = (
    nodes_gdf
    .groupby("node_type")["in_mpa"]
    .agg(total="count", in_mpa="sum")
    .assign(pct=lambda df: 100 * df["in_mpa"] / df["total"])
)
print(node_summary.to_string())

# Save
out_cols = ["node_id", "node_lon", "node_lat", "node_type", "n_fixes",
            "in_mpa", "mpa_name"]
nodes_gdf[[c for c in out_cols if c in nodes_gdf.columns]].to_csv(
    OUT_DIR / "nodes_mpa.csv", index=False)
print(f"\n   Saved → {OUT_DIR}/nodes_mpa.csv")

# ── 3. Foraging fixes MPA analysis ────────────────────────────────────────────
print("\n── Analysing foraging fixes ──────────────────────────────────────────────", flush=True)
tracks   = pd.read_csv(TRACKS_PATH, low_memory=False)
foraging = (
    tracks
    .query("behaviour == 'Foraging'")
    .dropna(subset=["lon", "lat"])
    .reset_index(drop=True)
)
print(f"   {len(foraging):,} foraging fixes to test")

foraging_gdf = gpd.GeoDataFrame(
    foraging,
    geometry=gpd.points_from_xy(foraging["lon"], foraging["lat"]),
    crs="EPSG:4326"
)

# sjoin is fast because geopandas uses an STRtree spatial index internally
# Same fix for foraging fixes join
joined_fixes = gpd.sjoin(foraging_gdf, mpa[["geometry"]],
                          how="left", predicate="within")
joined_fixes = joined_fixes[~joined_fixes.index.duplicated(keep="first")]
foraging_gdf["in_mpa"] = ~joined_fixes["index_right"].isna()

n_fixes       = len(foraging_gdf)
n_fixes_mpa   = foraging_gdf["in_mpa"].sum()
pct_fixes     = 100 * n_fixes_mpa / n_fixes

print(f"   Total foraging fixes    : {n_fixes:,}")
print(f"   Fixes inside MPA        : {n_fixes_mpa:,}  ({pct_fixes:.1f}%)")

# Save (just the key columns to keep file size reasonable)
save_cols = ["lon", "lat", "datetime", "trip_id", "colony", "behaviour", "in_mpa"]
foraging_gdf[[c for c in save_cols if c in foraging_gdf.columns]].to_csv(
    OUT_DIR / "foraging_fixes_mpa.csv", index=False)
print(f"   Saved → {OUT_DIR}/foraging_fixes_mpa.csv")

# ── 4. Summary text ───────────────────────────────────────────────────────────
summary_lines = [
    "MPA Intersection Analysis — Manx Shearwater Foraging Network",
    "=" * 60,
    f"MPA source           : JNCC (UK + Ireland), clipped to Celtic Seas",
    f"MPA polygons (merged): {len(mpa):,}",
    f"",
    f"── Nodes ──────────────────────────────────────────────────",
    f"Total nodes          : {n_nodes}",
    f"Inside MPA           : {n_in_mpa}  ({pct_nodes:.1f}%)",
    f"",
    node_summary.to_string(),
    f"",
    f"── Foraging fixes ─────────────────────────────────────────",
    f"Total fixes          : {n_fixes:,}",
    f"Inside MPA           : {n_fixes_mpa:,}  ({pct_fixes:.1f}%)",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(OUT_DIR / "mpa_summary.txt", "w") as f:
    f.write(summary_text)
print(f"\n   Saved → {OUT_DIR}/mpa_summary.txt")

# ── 5. Map: nodes coloured by MPA status ──────────────────────────────────────
print("\n── Plotting ──────────────────────────────────────────────────────────────", flush=True)

fig1, ax1 = plt.subplots(figsize=(8, 9))
draw_land(ax1)

# MPA polygons (faint background)
mpa.plot(ax=ax1, facecolor=C_MPA_POLY, edgecolor=C_MPA,
         linewidth=0.4, alpha=0.4, zorder=3)

# Nodes outside MPA
out_mask = ~nodes_gdf["in_mpa"]
ax1.scatter(nodes_gdf.loc[out_mask, "node_lon"],
            nodes_gdf.loc[out_mask, "node_lat"],
            c=C_NO_MPA, s=18, zorder=5,
            edgecolors="white", linewidths=0.4, label="Outside MPA")

# Nodes inside MPA
in_mask = nodes_gdf["in_mpa"]
ax1.scatter(nodes_gdf.loc[in_mask, "node_lon"],
            nodes_gdf.loc[in_mask, "node_lat"],
            c=C_MPA, s=25, zorder=6,
            edgecolors="white", linewidths=0.4, label="Inside MPA")

# Colony nodes on top
col_mask = nodes_gdf["node_type"] == "colony"
ax1.scatter(nodes_gdf.loc[col_mask, "node_lon"],
            nodes_gdf.loc[col_mask, "node_lat"],
            c=C_COLONY, s=60, zorder=7, marker="^",
            edgecolors="white", linewidths=0.5)
for _, r in nodes_gdf[col_mask].iterrows():
    ax1.text(r["node_lon"] + 0.15, r["node_lat"] + 0.1,
             r["node_id"].replace("colony_", ""),
             fontsize=6, color=C_COLONY, fontweight="bold", zorder=8)

set_extent(ax1)
apply_map_style(ax1,
    title="Network nodes · MPA overlap",
    subtitle=f"{n_in_mpa} of {n_nodes} nodes inside an MPA  ({pct_nodes:.1f}%)",
    caption="JNCC Marine Protected Areas (UK + Ireland)")

ax1.legend(handles=[
    Patch(facecolor=C_MPA_POLY, edgecolor=C_MPA, alpha=0.6, label="MPA"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=C_MPA,
           markersize=6, markeredgecolor="white", label=f"Node in MPA  ({n_in_mpa})"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=C_NO_MPA,
           markersize=6, markeredgecolor="white", label=f"Node outside MPA  ({n_nodes - n_in_mpa})"),
    Line2D([0],[0], marker="^", color="none", markerfacecolor=C_COLONY,
           markersize=7, markeredgecolor="white", label="Colony"),
], loc="lower left", frameon=True, framealpha=0.9,
   edgecolor=C_GRID, fontsize=7, facecolor=C_BG)

fig1.tight_layout()
fig1.savefig(OUT_DIR / "fig_mpa_nodes.png")
plt.close(fig1)
print("   fig_mpa_nodes.png saved")

# ── 6. Map: foraging fixes coloured by MPA status ─────────────────────────────
# Subsample fixes for plotting — 200k points is slow to render
PLOT_SAMPLE = 50_000
rng = np.random.default_rng(42)

outside_idx = foraging_gdf.index[~foraging_gdf["in_mpa"]]
inside_idx  = foraging_gdf.index[foraging_gdf["in_mpa"]]

# Sample each group proportionally
n_out_plot = min(len(outside_idx), int(PLOT_SAMPLE * (1 - pct_fixes/100)))
n_in_plot  = min(len(inside_idx),  int(PLOT_SAMPLE * (pct_fixes/100)))
n_out_plot = max(n_out_plot, min(1000, len(outside_idx)))
n_in_plot  = max(n_in_plot,  min(1000, len(inside_idx)))

plot_out = foraging_gdf.loc[rng.choice(outside_idx, n_out_plot, replace=False)]
plot_in  = foraging_gdf.loc[rng.choice(inside_idx,  n_in_plot,  replace=False)]

fig2, ax2 = plt.subplots(figsize=(8, 9))
draw_land(ax2)
mpa.plot(ax=ax2, facecolor=C_MPA_POLY, edgecolor=C_MPA,
         linewidth=0.4, alpha=0.4, zorder=3)

ax2.scatter(plot_out["lon"], plot_out["lat"],
            c=C_NO_MPA, s=0.8, alpha=0.25, linewidths=0, zorder=4)
ax2.scatter(plot_in["lon"], plot_in["lat"],
            c=C_MPA, s=0.8, alpha=0.5, linewidths=0, zorder=5)

set_extent(ax2)
apply_map_style(ax2,
    title="Foraging fixes · MPA overlap",
    subtitle=f"{n_fixes_mpa:,} of {n_fixes:,} foraging fixes inside an MPA  ({pct_fixes:.1f}%)",
    caption=f"Plotted: {n_in_plot:,} in-MPA + {n_out_plot:,} out-of-MPA fixes (random subsample for display)")

ax2.legend(handles=[
    Patch(facecolor=C_MPA_POLY, edgecolor=C_MPA, alpha=0.6, label="MPA"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=C_MPA,
           markersize=5, alpha=0.7, label=f"Fix inside MPA  ({n_fixes_mpa:,})"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=C_NO_MPA,
           markersize=5, alpha=0.7, label=f"Fix outside MPA  ({n_fixes - n_fixes_mpa:,})"),
], loc="lower left", frameon=True, framealpha=0.9,
   edgecolor=C_GRID, fontsize=7, facecolor=C_BG)

fig2.tight_layout()
fig2.savefig(OUT_DIR / "fig_mpa_fixes.png")
plt.close(fig2)
print("   fig_mpa_fixes.png saved")

print("\n" + "="*60)
print("MPA ANALYSIS COMPLETE")
print("="*60)
print(f"  Outputs → {OUT_DIR}/")