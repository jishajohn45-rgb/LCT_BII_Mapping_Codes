"""
Mapping Global Terrestrial Vertebrate Species Richness (2023)
------------------------------------------------------------
This script generates global and regional richness maps at a 5km resolution.
The workflow includes rasterization of point-based richness data and visualization.

Note: Raw species data from BirdLife/IUCN is restricted. This script 
is configured to run on a 'mock' dataset for workflow demonstration.
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

# --- Configuration & Paths ---
OUTPUT_DIR = Path("./output_maps")
OUTPUT_DIR.mkdir(exist_ok=True)

# Use relative paths for portability
DATA_PATH = Path("./data/richness_5km_2023.csv") 

# --- 1. Data Loading & Preparation ---
def load_and_prepare_data(path):
    """Loads CSV and converts to GeoDataFrame, handling missing files with mock data."""
    if not path.exists():
        print(f"Warning: {path} not found. Generating mock dataset for demonstration...")
        # Create dummy data for reproducibility testing
        df = pd.DataFrame({
            'longitude': np.random.uniform(-180, 180, 1000),
            'latitude': np.random.uniform(-60, 80, 1000),
            'expected_richness_2023': np.random.randint(50, 500, 1000)
        })
    else:
        df = pd.read_csv(path)

    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    return gdf

# --- 2. Rasterization Engine ---
def create_richness_raster(gdf, resolution=0.05):
    """Converts point-based richness to a regular grid (0.05 deg ~ 5km)."""
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Expand bounds slightly for padding
    minx, maxx = minx - 0.5, maxx + 0.5
    miny, maxy = miny - 0.5, maxy + 0.5
    
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

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

# --- 3. Visualization Suite ---
def plot_richness_layer(ax, raster, extent, cmap='magma'):
    """Applies standardized cartographic styling to an axis."""
    im = ax.imshow(
        raster,
        origin='upper',
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=Normalize(vmin=np.nanmin(raster), vmax=np.nanmax(raster)),
        interpolation='bilinear' # Smoother for manuscript quality
    )
    
    # Cartographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333333')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#666666', alpha=0.5)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    return im

def main():
    # Setup
    print("Initiating richness mapping workflow...")
    gdf = load_and_prepare_data(DATA_PATH)
    raster, bounds = create_richness_raster(gdf)
    
    # Define Regions
    regions = {
        "Global": {"extent": None, "is_global": True},
        "Africa": {"extent": [-25, 55, -35, 38], "is_global": False},
        "Tropical America": {"extent": [-110, -30, -30, 25], "is_global": False},
        "Southeast Asia": {"extent": [90, 155, -12, 28], "is_global": False}
    }

    # Generate Individual Figures
    for name, config in regions.items():
        fig = plt.figure(figsize=(12, 7), dpi=300)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        if config["is_global"]:
            ax.set_global()
        else:
            ax.set_extent(config["extent"], crs=ccrs.PlateCarree())
            
        im = plot_richness_layer(ax, raster, bounds)
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, pad=0.03)
        cbar.set_label("Expected Species Richness (2023)", fontsize=10)
        
        plt.title(f"Terrestrial Vertebrate Richness: {name}", fontsize=14, pad=20)
        
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(OUTPUT_DIR / f"richness_{safe_name}.png", bbox_inches='tight')
        plt.close()
        print(f"  ✓ Exported: {name}")

    print(f"\nWorkflow complete. Maps available in: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()