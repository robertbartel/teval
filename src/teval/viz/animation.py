"""GIF animation of streamflow propagation through the river network."""

import numpy as np
import pandas as pd
import contextily as cx
import tempfile
import shutil
from pathlib import Path
from PIL import Image

# MUST BE SET BEFORE IMPORTING PYPLOT to ensure process safety in Joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from joblib import Parallel, delayed

def _render_frame(i, t, current_vals, gdf_sorted, cmap, norm, dynamic_lw, temp_dir, var_name, add_basemap):
    """Worker function to render a single animation frame."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    gdf_frame = gdf_sorted.copy()
    gdf_frame['val'] = current_vals

    # Plot with dynamic linewidth
    gdf_frame.plot(column='val', ax=ax, cmap=cmap, norm=norm, linewidth=dynamic_lw, alpha=0.9)
    
    if add_basemap:
        cx.add_basemap(ax, crs=gdf_sorted.crs, source=cx.providers.USGS.USTopo)
    
    ax.set_axis_off()
    time_str = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M')
    ax.set_title(f"{var_name.replace('_', ' ').title()} | {time_str}", fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, extend='both')

    frame_path = temp_dir / f"frame_{i:04d}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close(fig) # Prevent memory leak!
    
    return str(frame_path)


def animate_network(gdf, stats_ds, output_path, var_name="streamflow_mean", fps=8, add_basemap=True, log_scale=True, n_workers=1):
    """Render a time-lapse GIF of streamflow across the river network."""
    da = stats_ds[var_name]
    
    # Force a compute here if it's lazy so the workers don't trigger simultaneous reads
    da = da.compute()
    data_values = da.values 
    times = da.time.values

    # Colormap Setup
    global_min, global_max = np.nanmin(data_values), np.nanmax(data_values)
    if global_min <= 0: global_min = 0.001 
    
    if log_scale:
        norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)
    else:
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "hydro_flow", ["white", "sienna", "yellow", "limegreen", "lightskyblue", "skyblue", "deepskyblue", "dodgerblue", "blue"]
    )

    gdf_sorted = gdf.to_crs(epsg=3857)
    
    # ---------------------------------------------------------
    # DYNAMIC LINEWIDTH: Scale based on bounding box width
    # ---------------------------------------------------------
    minx, miny, maxx, maxy = gdf_sorted.total_bounds
    dx_meters = maxx - minx  # Width of domain in meters
    
    if dx_meters > 3_000_000:      # CONUS scale (>3000km)
        dynamic_lw = 0.5
    elif dx_meters > 1_000_000:    # Regional scale (>1000km)
        dynamic_lw = 1.0
    elif dx_meters > 500_000:      # Large Basin scale (>500km)
        dynamic_lw = 1.5
    else:                          # Small Basin scale (<500km)
        dynamic_lw = 2.5
        
    # ---------------------------------------------------------

    temp_dir = Path(tempfile.mkdtemp())
    total_frames = len(times)
    
    # Blast the frames to the CPU cores
    frame_files = Parallel(n_jobs=n_workers)(
        delayed(_render_frame)(
            i, t, np.maximum(data_values[i, :], global_min), 
            gdf_sorted, cmap, norm, dynamic_lw, temp_dir, var_name, add_basemap
        ) for i, t in enumerate(times)
    )

    # Sort files to guarantee chronological order before stitching
    frame_files.sort()

    images = [Image.open(f) for f in frame_files]
    if images:
        images[0].save(
            output_path, save_all=True, append_images=images[1:], 
            duration=int(1000 / fps), loop=0, optimize=True
        )
        
    shutil.rmtree(temp_dir)