import geopandas as gpd
import numpy as np
import os
from rasterio.mask import mask
from tree_health_detection.src import delineation_utils
from tree_health_detection.src import get_itcs_polygons
import torch
import imageio
from shapely.affinity import translate
# Usage example
# Define a folder with remote sensing RGB data
folder = '../tree_mask_delineation/'
rgb_path = 'imagery/rgb_clip.tif'
stem_path = 'gradient_boosting_alignment.gpkg'
hsi_img  = 'imagery/hsi_clip.tif'
laz_path = 'imagery/LiDAR.laz'
# Loop through the files in the folder
#for file in os.listdir(folder):
# check if the tree_tops file exists. if not, launch get_tree_tops
def build_data_schema(folder, stem_path,rgb_path, hsi_path, laz_path):

    if os.path.exists(os.path.join(folder,)):
        itcs = gpd.read_file(os.path.join(folder+stem_path))
    else:
        print('itc = get_tree_tops(laz_path)')

    #get tree bounding boxes with deepForest for SAM
    import deepforest
    bbox = delineation_utils.extract_boxes(folder+rgb_path)
    torch.cuda.empty_cache()

    #use only points whose crwnpst is greater than 2
    itcs = itcs[itcs['Crwnpst'] > 1]
    image_file = os.path.join(folder, rgb_path)
    hsi_img = os.path.join(folder, hsi_img)
    # Split the image into batches of 40x40m
    batch_size = 400
    #image_file, hsi_img, itcs, bbox,  batch_size=40
    raster_batches, raster_hsi_batches, itcs_batches, itcs_boxes, affine = get_itcs_polygons.split_image(image_file, 
                                hsi_img, itcs, bbox, batch_size)


    # Make predictions of tree crown polygons using SAM
    for(i, batch) in enumerate(raster_hsi_batches):
        #skip empty batches
        if itcs_batches[i].shape[0] == 0: 
            continue

        # Make predictions of tree crown polygons using SAM
        predictions, _, _ = get_itcs_polygons.predict_tree_crowns(batch=batch[:3,:,:], input_points=itcs_batches[i],  
                                                                  input_boxes = itcs_boxes[i], neighbors=3, point_type = "euclidian") 
        # Apply the translation to the geometries in the GeoDataFrame
        x_offset, y_offset = affine[i][2], affine[i][5]
        y_offset = y_offset - batch_size
        #from a geopandas, remove all rows with a None geometry
        predictions = predictions[predictions['geometry'].notna()]
        predictions["geometry"] = predictions["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
        predictions.crs = "EPSG:32618"

        # Save the predictions as geopandas
        predictions.to_file(f'{folder}/outdir/itcs/itcs_{i}.gpkg', driver='GPKG')

        batch = batch[:3,:,:]
        batch = np.moveaxis(batch, 0, -1)
        imageio.imwrite(f'{folder}/outdir/clips/itcs_{i}.png', batch)

