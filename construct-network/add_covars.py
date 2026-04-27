#!/usr/bin/env python3
"""
Add node-level environmental covariates to edges_temporal.csv.

For each edge (date, node_i, node_j), extracts covariate values at
node_i and node_j locations on the given date from .nc files.

Outputs:
    network/edges_with_covars.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
COVAR_DIR  = Path("./data/covars")
NODES_PATH = Path("./network/nodes.csv")
EDGES_PATH = Path("./network/edges_temporal.csv")
OUT_PATH   = Path("./network/edges_with_covars.csv")

# ── Load node and edge tables ──────────────────────────────────────────────────
print("── Loading nodes and edges ───────────────────────────────────────────────", flush=True)
nodes = pd.read_csv(NODES_PATH)
edges = pd.read_csv(EDGES_PATH)
edges["date"] = pd.to_datetime(edges["date"])

node_lookup = nodes.set_index("node_id")[["node_lon", "node_lat"]].to_dict("index")
print(f"   {len(nodes)} nodes, {len(edges):,} edges")

# ── Helper: extract values from one dataset for all node×date combos ──────────
def extract_covar(ds: xr.Dataset, var: str,
                  lons: np.ndarray, lats: np.ndarray,
                  dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Nearest-neighbour lookup in (time, lat, lon) space.
    For static variables (no time dimension), matches by location only.
    Returns an array of shape (N,) matching the length of lons/lats/dates.
    """
    da = ds[var]

    # Normalise dimension names to 'time', 'latitude', 'longitude'
    rename = {}
    for dim in da.dims:
        dl = dim.lower()
        if dl in ("lat", "latitude"):
            rename[dim] = "latitude"
        elif dl in ("lon", "longitude"):
            rename[dim] = "longitude"
        elif dl in ("time",):
            rename[dim] = "time"
    if rename:
        da = da.rename(rename)

    # Drop depth/level if present (take surface = index 0)
    for dim in list(da.dims):
        if dim not in ("time", "latitude", "longitude"):
            da = da.isel({dim: 0})

    has_time = "time" in da.dims

    if has_time:
        values = da.sel(
            time      = xr.DataArray(dates,     dims="points"),
            latitude  = xr.DataArray(lats,      dims="points"),
            longitude = xr.DataArray(lons,      dims="points"),
            method    = "nearest",
        ).values
    else:
        # Static variable — match by location only, broadcast across all edges
        values = da.sel(
            latitude  = xr.DataArray(lats, dims="points"),
            longitude = xr.DataArray(lons, dims="points"),
            method    = "nearest",
        ).values

    return values.astype(float)

# ── Collect lons/lats/dates for node_i and node_j ─────────────────────────────
print("── Preparing coordinate arrays ───────────────────────────────────────────", flush=True)

dates_arr = pd.to_datetime(edges["date"].values)

lons_i = np.array([node_lookup[n]["node_lon"] for n in edges["node_i"]])
lats_i = np.array([node_lookup[n]["node_lat"] for n in edges["node_i"]])
lons_j = np.array([node_lookup[n]["node_lon"] for n in edges["node_j"]])
lats_j = np.array([node_lookup[n]["node_lat"] for n in edges["node_j"]])

# ── Process each .nc file ─────────────────────────────────────────────────────
nc_files = sorted(COVAR_DIR.glob("*.nc"))
if not nc_files:
    print(f"   ⚠  No .nc files found in {COVAR_DIR}")
    sys.exit(1)

print(f"\n── Found {len(nc_files)} .nc file(s) ─────────────────────────────────────", flush=True)

for nc_path in nc_files:
    stem = nc_path.stem          # e.g. "chlorophyll"
    print(f"\n   Processing {nc_path.name} …", flush=True)

    ds = xr.open_dataset(nc_path)

    # Skip coordinate/bounds variables; only use true data variables
    skip = {"time", "time_bnds", "latitude", "longitude",
            "lat", "lon", "depth", "level"}
    data_vars = [v for v in ds.data_vars if v.lower() not in skip]
    print(f"   Variables: {data_vars}")

    for var in data_vars:
        col_i = f"{stem}_{var}_node_i"
        col_j = f"{stem}_{var}_node_j"

        print(f"     Extracting {var} for node_i …", flush=True)
        edges[col_i] = extract_covar(ds, var, lons_i, lats_i, dates_arr)

        print(f"     Extracting {var} for node_j …", flush=True)
        edges[col_j] = extract_covar(ds, var, lons_j, lats_j, dates_arr)

    ds.close()

# ── Save ───────────────────────────────────────────────────────────────────────
print(f"\n── Saving → {OUT_PATH}", flush=True)
edges.to_csv(OUT_PATH, index=False)
print(f"   {len(edges):,} rows × {len(edges.columns)} columns saved")
print("DONE")