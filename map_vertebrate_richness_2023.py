"""
Mapping Global Terrestrial Vertebrate Species Richness (2023)
------------------------------------------------------------
Source: Prepared for Nature Ecology & Evolution
Contact: [Your Name/Lab]

Description:
This script generates global and regional richness maps at a 5km resolution.
The workflow includes rasterization of point-based richness data and 
manuscript-quality visualization.

Note: Raw species data from BirdLife/IUCN is restricted. This script 
is configured to run on a 'mock' dataset for workflow demonstration if 
the source CSV is missing.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.features import rasterize
from rasterio.transform import from_origin
from matplotlib.colors import Normalize
from pathlib import Path
import warnings

# Suppress coordinate system transformation warnings
warnings.filterwarnings('ignore')

# --- 1. Configuration & Robust Path Handling ---
# Automatically find the directory where this script is saved
SCRIPT_DIR = Path(__file__).parent.resolve()

# Define paths relative to the script location (Prevents WinError 5 Access Denied)
OUTPUT_DIR = SCRIPT_DIR / "output_maps"
DATA_PATH = SCRIPT_DIR / "data" / "richness_5km_2023.csv"

# Create the output directory safely
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- 2. Data Loading & Preparation ---
def load_and_prepare_data(path):
    """Loads CSV and converts to GeoDataFrame, handling missing files with mock data."""
    if not path.exists():
        print(f"(!) Warning: {path.name} not found.")
        print("Generating mock dataset for workflow demonstration...")
        # Create dummy data for reproducibility testing (Global distribution)
        df = pd.DataFrame({
            'longitude': np.random.uniform(-180, 180, 5000),
            'latitude': np.random.uniform(-60, 85, 5000),
            'expected_richness_2023': np.random.randint(50, 800, 5000)
        })
    else:
        print(f"✓ Loading richness data: {path.name}")
        df = pd.read_csv(path)

    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    return gdf

# --- 3. Rasterization Engine ---
def create_richness_raster(gdf, resolution=0.05):
    """Converts point-based richness to a regular grid (0.05 deg ~ 5km)."""
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Expand bounds slightly for padding and edge safety
    minx, maxx = minx - 0.5, maxx + 0.5
    miny, maxy = miny - 0.5, maxy + 0.5
    
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    # Prepare shapes and values for the rasterizer
    shapes = [(geom, val) for geom, val in zip(gdf.geometry, gdf['expected_richness_2023'])]
    
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        all_touched=False,
        dtype=np.float32
    )
    
    return raster, (minx, maxx, miny, maxy)

# --- 4. Visualization Suite ---
def plot_richness_layer(ax, raster, extent, cmap='magma'):
    """Applies standardized cartographic styling to an axis."""
    im = ax.imshow(
        raster,
        origin='upper',
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=Normalize(vmin=np.nanmin(raster), vmax=np.nanmax(raster)),
        interpolation='bilinear' # Perceptually smoother for publication
    )
    
    # Add geographical features
    ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#666666', alpha=0.5, zorder=2)
    
    # Gridline configuration
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    return im

def main():
    print("--- Initiating Richness Mapping Workflow ---")
    
    # 1. Prepare Data
    gdf = load_and_prepare_data(DATA_PATH)
    
    # 2. Process Raster
    raster, bounds = create_richness_raster(gdf)
    
    # 3. Define Region Views
    regions = {
        "Global": {"extent": None, "is_global": True},
        "Africa": {"extent": [-25, 55, -35, 40], "is_global": False},
        "Tropical America": {"extent": [-115, -30, -35, 30], "is_global": False},
        "Southeast Asia": {"extent": [90, 160, -15, 30], "is_global": False}
    }

    # 4. Generate Figures
    for name, config in regions.items():
        fig = plt.figure(figsize=(12, 7), dpi=300)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        if config["is_global"]:
            ax.set_global()
        else:
            ax.set_extent(config["extent"], crs=ccrs.PlateCarree())
            
        im = plot_richness_layer(ax, raster, bounds)
        
        # Colorbar styling
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.03)
        cbar.set_label("Expected Species Richness (2023)", fontsize=10, labelpad=10)
        
        plt.title(f"Terrestrial Vertebrate Richness: {name}", fontsize=14, pad=20, fontweight='bold')
        
        # Exporting
        safe_name = name.replace(" ", "_").lower()
        output_file = OUTPUT_DIR / f"richness_{safe_name}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  ✓ Exported: {output_file.name}")

    print(f"\nWorkflow complete. Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()