import numpy as np
import matplotlib.pyplot as plt

from pystac_client import Client
import planetary_computer
import rasterio
from rasterio.plot import show

# ==========================================
# Farm Coordinates
# ==========================================

p1 = (31.10, 30.10)
p2 = (31.11, 30.10)
p3 = (31.11, 30.11)
p4 = (31.10, 30.11)

# ==========================================
# Bounding Box
# ==========================================

def get_bbox(p1, p2, p3, p4):

    lons = [p1[0], p2[0], p3[0], p4[0]]
    lats = [p1[1], p2[1], p3[1], p4[1]]

    return (
        min(lons),   # min longitude
        min(lats),   # min latitude
        max(lons),   # max longitude
        max(lats)    # max latitude
    )

bbox = get_bbox(p1, p2, p3, p4)

print("Bounding Box:", bbox)

# ==========================================
# Connect to Microsoft Planetary Computer
# ==========================================

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# ==========================================
# Search Sentinel-2 Images
# ==========================================

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-11-11/2025-12-31",
    query={
        "eo:cloud_cover": {
            "lt": 10
        }
    }
)

items = list(search.items())

print("Number of images found:", len(items))
# ==========================================
# Get Latest Image
# ==========================================

item = items[-1]

print("\nImage ID:")
print(item.id)

print("\nAvailable Bands:")
print(list(item.assets.keys()))
# ==========================================
# Read Red Band (B04)
# ==========================================

signed_item = planetary_computer.sign(item)

red_url = signed_item.assets["B04"].href

print("\nRed Band URL:")
print(red_url)

with rasterio.open(red_url) as src:
    red = src.read(1)

print("\nRed Band Shape:", red.shape)
print("Minimum:", red.min())
print("Maximum:", red.max())
print(list(item.assets.keys()))
# ==========================================
# Read Red Band (B04)
# ==========================================

signed_item = planetary_computer.sign(item)

red_url = signed_item.assets["B04"].href

print("\nRed Band URL:")
print(red_url)

with rasterio.open(red_url) as src:
    red = src.read(1)

print("\nRed Band Shape:", red.shape)
print("Minimum:", red.min())
print("Maximum:", red.max())
# ==========================================
# Read NIR Band (B08)
# ==========================================

nir_url = signed_item.assets["B08"].href

with rasterio.open(nir_url) as src:
    nir = src.read(1)

print("\nNIR Band Shape:", nir.shape)
print("Minimum:", nir.min())
print("Maximum:", nir.max())
# ==========================================
# Calculate NDVI
# ==========================================

red = red.astype("float32")
nir = nir.astype("float32")

ndvi = (nir - red) / (nir + red + 1e-10)

print("\nNDVI")
print("Min:", np.nanmin(ndvi))
print("Max:", np.nanmax(ndvi))
# ==========================================
# Display NDVI Heatmap
# ==========================================

plt.figure(figsize=(8, 8))

plt.imshow(ndvi, cmap="RdYlGn")

plt.colorbar(label="NDVI")

plt.title("NDVI Heatmap")

plt.axis("off")

plt.show()