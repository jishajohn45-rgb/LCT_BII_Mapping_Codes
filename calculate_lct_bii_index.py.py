"""
LCT-BII Index Generation and Global Mapping
-------------------------------------------
Calculates the Land Cover Transition Biodiversity Integrity Index (LCT-BII)
defined as the ratio of expected species richness (S_t+1) to 
observed richness (S_t).

Formula: LCT-BII = S_2023 / S_2017
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = Path("./data/richness_eco_global_5km.csv")
OUTPUT_DIR = Path("./output_lct_bii")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 1. Data Processing ---
def compute_lct_bii(path):
    """Loads data and computes the LCT-BII index with safety checks."""
    if not path.exists():
        print(f"Data not found at {path}. Generating synthetic dataset for workflow validation...")
        # Create mock data: concentrated richness in tropics, some loss/gain
        lons = np.linspace(-180, 180, 500)
        lats = np.linspace(-60, 80, 250)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        df = pd.DataFrame({
            'longitude': lon_grid.flatten(),
            'latitude': lat_grid.flatten(),
            'observed_richness_2017': np.random.randint(100, 500, len(lon_grid.flatten()))
        })
        # Simulate slight changes around 1.0
        df['expected_richness_2023'] = df['observed_richness_2017'] * np.random.uniform(0.92, 1.08, len(df))
    else:
        df = pd.read_csv(path)

    # Core Calculation: LCT-BII
    # Ensure we don't divide by zero
    df['lct_bii'] = np.where(
        df['observed_richness_2017'] > 0,
        df['expected_richness_2023'] / df['observed_richness_2017'],
        np.nan
    )
    
    return df.dropna(subset=['lct_bii'])

# --- 2. Grid Generation ---
def generate_raster_indices(df, res=0.05):
    """Bins point data into a regular 2D grid for mapping."""
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    # Use mean LCT-BII value per grid cell
    h, xedges, yedges = np.histogram2d(
        df['longitude'], df['latitude'],
        bins=[lon_bins, lat_bins],
        weights=df['lct_bii']
    )
    counts, _, _ = np.histogram2d(
        df['longitude'], df['latitude'],
        bins=[lon_bins, lat_bins]
    )
    
    # Avoid division by zero and transpose for plotting
    grid = np.divide(h, counts, where=counts > 0, out=np.full_like(h, np.nan))
    return grid.T, xedges, yedges

# --- 3. Visualization ---
def plot_lct_bii_map(grid, xedges, yedges):
    """Generates the final manuscript-quality diverging map."""
    
    # Diverging color palette: Burnt Red (Loss) -> White (Neutral) -> Forest Green (Gain)
    colors = ['#8b0000', '#ff4c4c', '#ffffff', '#4caf50', '#006400']
    cmap = LinearSegmentedColormap.from_list("lct_bii_diverging", colors, N=256)
    
    # Force 1.0 to be the center (White)
    norm = TwoSlopeNorm(vmin=0.90, vcenter=1.00, vmax=1.10)
    
    fig = plt.figure(figsize=(15, 8), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson()) # Robinson is better for global richness distributions
    
    # Backgrounds
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f4f8', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#333333', zorder=3)
    
    # Data layer
    im = ax.pcolormesh(
        xedges, yedges, grid,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=2
    )
    
    # Refined Colorbar
    cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.02])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
    cb.set_label("LCT-BII (Ratio of Richness 2023:2017)", fontsize=10, fontweight='bold')
    
    # Add clear Threshold markers
    cb.ax.axvline(1.0, color='black', lw=1.5, linestyle='--')
    fig.text(0.24, 0.12, "Loss", ha='right', fontsize=10, color='#8b0000', fontweight='bold')
    fig.text(0.76, 0.12, "Gain", ha='left', fontsize=10, color='#006400', fontweight='bold')

    plt.suptitle("Global Biodiversity Integrity Index (LCT-BII)", fontsize=16, y=0.95)
    ax.set_title(r"LCT-BII$_{(g,t)} = \frac{S_{(g,t+1)}}{S_{(g,t)}}$", fontsize=12, pad=10)

    # Save outputs
    plt.savefig(OUTPUT_DIR / "LCT_BII_Global_Map.png", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "LCT_BII_Global_Map.pdf", bbox_inches='tight') # Vector for publication
    print(f"✓ Maps saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    data = compute_lct_bii(DATA_PATH)
    grid, x, y = generate_raster_indices(data)
    plot_lct_bii_map(grid, x, y)