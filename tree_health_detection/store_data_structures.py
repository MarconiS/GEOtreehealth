import geopandas as gpd
from pathlib import Path
import rasterio
import numpy as np
from rasterio.mask import mask
import laspy
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.mask import mask
# Define your paths
 
root_dir = Path('/media/smarconi/Gaia/Macrosystem_2/NEON_processed/data/')
rgb_dir = root_dir / "rgb"
hsi_dir = root_dir / "hsi"
png_dir = root_dir / "png"

lidar_dir = root_dir / "lidar"
polygon_mask_dir = root_dir / "polygon_mask"
labels_dir = root_dir / "labels"

# Ensure that all directories exist
for dir in [root_dir, rgb_dir, hsi_dir, lidar_dir, polygon_mask_dir, labels_dir]:
    dir.mkdir(parents=True, exist_ok=True)

# Load your polygon data (GeoDataFrame)
crowns = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Crowns/bboxesSERC_0.gpkg'
gdf = gpd.read_file(crowns)

# Function to extract data cube
from rasterio.mask import mask

def extract_data_cube(dataset_path, polygon, output_path, mask_path=None):
    with rasterio.open(dataset_path) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Create binary mask, where 1 is the tree and 0 is the background. The mask needs to be 1 channel
        binary_mask = np.zeros((out_image.shape[1], out_image.shape[2]))
        binary_mask[out_image[0, :, :] > 0] = 1

        # Save the data cube as a np array
        np.save(output_path, out_image)
        #with rasterio.open(output_path, "w", **out_meta) as dest:
        #    dest.write(out_image)

        if mask_path is not None:
            # Save the binary mask as a np array
            np.save(mask_path, binary_mask)
            #with rasterio.open(mask_path, "w", **out_meta) as mask_dest:
            #    mask_dest.write(binary_mask.astype('uint8'), indexes=1)
        return binary_mask
    
def extract_data_cube_lidar(dataset_path, polygon, output_path):
    # Read the .laz file
    lidar = laspy.read(dataset_path)

    # extract x,y, z and R,G,B values from the lidar file
    x = lidar.x
    y = lidar.y
    z = lidar.z
    r = lidar.red
    g = lidar.green
    b = lidar.blue

    #append the data in a geopandas array
    data = np.vstack((x,y,z,r,g,b)).transpose()

    # Mask the height map using the polygon
    masked = polygon.buffer(0)

    #remove rows with x < than mask.xmin
    data = data[(data[:,0] > masked.bounds[0]) & (data[:,0] < masked.bounds[2])]
    #remove rows with y < than mask.ymin
    data = data[(data[:,1] > masked.bounds[1]) & (data[:,1] < masked.bounds[3])]

    # Save the masked height map
    np.save(output_path, data)


#background_mask = binary_mask.copy()
def cumulative_linear_stretch(image, background_mask=None):
    if background_mask is not None:
        # Apply background mask
        background_mask = background_mask.astype(np.bool)        

    # Apply cumulative linear stretch to each band
    stretched_image = np.zeros_like(image, dtype=np.float)
    for band in range(3):
        # Select only the pixels where the mask is False for each band
        band_image = image[:, :, band][background_mask == True]
        # Calculate minimum and maximum values
        min_value = np.min(band_image)
        max_value = np.max(band_image)
        # Apply linear stretch
        stretched_image[:, :, band] = (image[:, :, band] - min_value) / (max_value - min_value) * 255


    stretched_image[stretched_image <0] = 0
    # Convert to 8-bit unsigned integers (uint8)
    stretched_image = stretched_image.astype(np.uint8)

    return stretched_image


def png_with_class(dataset_path, polygon, bool_mask, output_path, status_label):
    with rasterio.open(dataset_path) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Convert image to RGB with bands 28, 56, 115
        rgb_image = np.stack([
            out_image[12, :, :],  # Band 28 (0-indexed)
            out_image[45, :, :],  # Band 56 (0-indexed)
            out_image[69, :, :]  # Band 115 (0-indexed)
        ], axis=-1)

        # Create binary mask, where 1 is the tree and 0 is the background. The mask needs to be 1 channel
        binary_mask = np.zeros((out_image.shape[1], out_image.shape[2]))
        binary_mask[out_image[0, :, :] > 0] = 1

        # add a 0 mask to negative values
        rgb_image[rgb_image < 0] = 0
        rgb_image[rgb_image > 10000] = 10000

        # Normalize the image in range 0, 255
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255
        rgb_image = cumulative_linear_stretch(rgb_image, binary_mask)

        # Create PIL image from numpy array
        rgb_image = rgb_image.astype(np.uint8)
        pil_image = Image.fromarray(rgb_image)

        new_width = rgb_image.shape[1] *10
        new_height = rgb_image.shape[0] *10
        pil_image = pil_image.resize((new_width, new_height))
        # Add text label on the image
        #draw = ImageDraw.Draw(pil_image)
        #text = status_label
        #text_color = (255, 255, 255)  # White color
        #text_position = (20, 20)  # Position of the text label
        #draw.text(text_position, text, fill=text_color)
        # Save the modified image as PNG
        pil_image.save(output_path, format='PNG')


# Function to process each polygon
def process_polygon(polygon, root_dir, rgb_path, hsi_path, lidar_path, polygon_id, itcs):

    rgb_dir = root_dir / "rgb"
    hsi_dir = root_dir / "hsi"
    png_dir = root_dir / "png"

    lidar_dir = root_dir / "lidar"
    polygon_mask_dir = root_dir / "polygon_mask"
    labels_dir = root_dir / "labels"

    # Extract and save data cubes
    extract_data_cube(dataset_path = rgb_path, polygon = polygon.geometry, output_path = rgb_dir / f"{polygon_id}.npy")
                      
    bool_mask = extract_data_cube(dataset_path = hsi_path, polygon = polygon.geometry, output_path = hsi_dir / f"{polygon_id}.npy",
                      mask_path = polygon_mask_dir / f"{polygon_id}.npy")
    extract_data_cube_lidar(dataset_path = lidar_path, polygon =polygon.geometry, output_path = lidar_dir / f"{polygon_id}.npy")
    
    # from itcs, pick the rows that match the polygon StemTag
    label = itcs.loc[itcs['StemTag'] == polygon.StemTag] 
    # add polygon_id to the label and the path of the rgb, hsi and lidar extracted cubes
    label['polygon_id'] = polygon_id
    label['rgb_path'] = rgb_dir / f"{polygon_id}.npy"
    label['hsi_path'] = hsi_dir / f"{polygon_id}.npy"
    label['lidar_path'] = lidar_dir / f"{polygon_id}.npy"
    png_with_class(dataset_path = hsi_path, polygon = polygon.geometry, bool_mask = bool_mask,
                   output_path = png_dir / f"{label.Status.values[0]}_{polygon_id}.png", 
                   status_label = label.Status.values[0])
    # save the label
    #label.to_csv(labels_dir / f"{polygon_id}.csv", index=False)
    
