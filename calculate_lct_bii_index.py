"""
LCT-BII Index Generation and Global Mapping
-------------------------------------------
Source: Prepared for Nature Ecology & Evolution
Contact: [Your Name/Lab]

Description:
Calculates the Land Cover Transition Biodiversity Integrity Index (LCT-BII)
defined as the ratio of expected species richness (S_t+1) to 
observed richness (S_t).

Formula: LCT-BII = S_2023 / S_2017

Data Note: 
To comply with BirdLife International and IUCN licensing, raw distribution 
data are not included. This script generates a 'mock' dataset if the 
source CSV is missing to demonstrate the computational workflow.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path
import warnings

# Suppress technical coordinate system warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Configuration (Robust Path Handling) ---
# This ensures the script works regardless of where the terminal is opened
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR / "data" / "richness_eco_global_5km.csv"
OUTPUT_DIR = SCRIPT_DIR / "output_lct_bii"

# Create output directory (parents=True ensures nested folders work)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- 2. Data Processing & Index Calculation ---
def compute_lct_bii(path):
    """Loads richness data and computes the LCT-BII ratio."""
    if not path.exists():
        print(f"(!) Data not found at {path}")
        print("Generating synthetic 'mock' dataset for workflow validation...")
        
        # Create a representative global grid for demonstration
        lons = np.linspace(-180, 180, 600)
        lats = np.linspace(-60, 85, 300)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        df = pd.DataFrame({
            'longitude': lon_grid.flatten(),
            'latitude': lat_grid.flatten(),
            'observed_richness_2017': np.random.randint(50, 600, len(lon_grid.flatten()))
        })
        # Simulate index values (0.9 to 1.1)
        df['expected_richness_2023'] = df['observed_richness_2017'] * np.random.uniform(0.90, 1.10, len(df))
    else:
        print(f"✓ Loading dataset: {path.name}")
        df = pd.read_csv(path)

    # Core Calculation: LCT-BII (Ratio of 2023 to 2017)
    # Avoid division by zero where richness might be 0
    df['lct_bii'] = np.where(
        df['observed_richness_2017'] > 0,
        df['expected_richness_2023'] / df['observed_richness_2017'],
        np.nan
    )
    
    return df.dropna(subset=['lct_bii'])

# --- 3. Grid Generation (Rasterization) ---
def generate_raster_indices(df, res=0.1):
    """Bins point data into a regular 2D grid for spatial visualization."""
    lon_bins = np.arange(-180, 180 + res, res)
    lat_bins = np.arange(-90, 90 + res, res)

    # Use mean LCT-BII value per grid cell to handle point density
    h, xedges, yedges = np.histogram2d(
        df['longitude'], df['latitude'],
        bins=[lon_bins, lat_bins],
        weights=df['lct_bii']
    )
    counts, _, _ = np.histogram2d(
        df['longitude'], df['latitude'],
        bins=[lon_bins, lat_bins]
    )
    
    # Calculate mean and transpose for plotting (imshow/pcolormesh requirement)
    grid = np.divide(h, counts, where=counts > 0, out=np.full_like(h, np.nan))
    return grid.T, xedges, yedges

# --- 4. Visualization (Manuscript Quality) ---
def plot_lct_bii_map(grid, xedges, yedges):
    """Generates a high-resolution diverging global map using Robinson projection."""
    
    # Diverging Palette: Red (Loss) -> White (Neutral) -> Green (Gain)
    colors = ['#a50026', '#d73027', '#f7f7f7', '#1a9850', '#006837']
    cmap = LinearSegmentedColormap.from_list("lct_bii_diverging", colors, N=256)
    
    # Force 1.0 (no change) to be the exact center (White)
    norm = TwoSlopeNorm(vmin=0.90, vcenter=1.00, vmax=1.10)
    
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = plt.axes(projection=ccrs.Robinson()) 
    
    # Add geographical context
    ax.add_feature(cfeature.LAND, facecolor='#f4f4f4', zorder=1) # Background for empty areas
    ax.add_feature(cfeature.OCEAN, facecolor='#e0e8f0', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='#222222', zorder=3)
    
    # Data Layer
    im = ax.pcolormesh(
        xedges, yedges, grid,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=2
    )
    
    # Refined Colorbar
    cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.02])
    cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
    cb.set_label("LCT-BII (Ratio of Richness 2023:2017)", fontsize=11, fontweight='bold', labelpad=10)
    
    # Visual cues for Loss/Gain
    cb.ax.axvline(1.0, color='black', lw=1.2, linestyle='--')
    fig.text(0.29, 0.1, "Loss", ha='right', fontsize=10, color='#a50026', fontweight='bold')
    fig.text(0.71, 0.1, "Gain", ha='left', fontsize=10, color='#006837', fontweight='bold')

    # Titles and Formulation
    plt.suptitle("Global Biodiversity Integrity Index (LCT-BII)", fontsize=18, y=0.94, fontweight='bold')
    ax.set_title(r"Index Definition: $LCT\text{-}BII_{(g,t)} = \frac{S_{(g,t+1)}}{S_{(g,t)}}$", 
                 fontsize=13, pad=15, color='#444444')

    # Final Exports
    plt.savefig(OUTPUT_DIR / "LCT_BII_Global_Map.png", bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_DIR / "LCT_BII_Global_Map.pdf", bbox_inches='tight') 
    print(f"✓ Success: Maps saved to {OUTPUT_DIR}")

# --- 5. Execution ---
if __name__ == "__main__":
    try:
        data = compute_lct_bii(DATA_PATH)
        grid, x, y = generate_raster_indices(data)
        plot_lct_bii_map(grid, x, y)
    except Exception as e:
        print(f"Critical Error during execution: {e}")