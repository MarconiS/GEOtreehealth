import geopandas as gpd
import numpy as np
import os
from rasterio.mask import mask
from tree_health_detection.src import delineation_utils
from tree_health_detection.src import get_itcs_polygons
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry import box, Point, MultiPoint, Polygon
import imageio
import deepforest
from PIL import Image
from scipy.ndimage import zoom
import torch
import imageio
import rasterio
from shapely.affinity import translate

# Import the necessary libraries
import os
import rasterio
import numpy as np
import cv2
import geopandas as gpd
import torch
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import translate

from matplotlib import pyplot as plt
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import polygonize
from skimage.measure import find_contours

import numpy as np
import rasterio
from rasterio.features import shapes
from affine import Affine
import geopandas as gpd
from shapely.geometry import shape
import warnings

from segment_anything import sam_model_registry, SamPredictor
import geopandas as gpd
import pandas as pd
from skimage.transform import resize
from skimage.measure import label
from importlib import reload

# Usage example
# Define a folder with remote sensing RGB data
folder = '../tree_mask_delineation/'
rgb_path = 'imagery/rgb_clip.tif'
stem_path = 'gradient_boosting_alignment.gpkg'
hsi_img  = 'imagery/hsi_clip.tif'
laz_path = 'imagery/LiDAR.laz'
grid_space = 20
area_threshold = 100
# Loop through the files in the folder
#for file in os.listdir(folder):
# check if the tree_tops file exists. if not, launch get_tree_tops
def build_data_schema(folder, stem_path,rgb_path, hsi_path, laz_path, grid_space = 20, hierachical_predictions = False, threshold_score=0.8):



    if os.path.exists(os.path.join(folder,stem_path)):
        itcs = gpd.read_file(os.path.join(folder+stem_path))
    else:
        print('itc = get_tree_tops(laz_path)')


    #get affine of the raster used to extract the bounding boxes
    with rasterio.open(folder+rgb_path) as src: 
        rgb_transform = src.transform
        rgb_crs = src.crs
        rgb_side = src.width

    #get tree bounding boxes with deepForest for SAM
    import deepforest
    bbox = delineation_utils.extract_boxes(folder+rgb_path)
    #use bbox xmin,ymin,xmax,ymax to make corners of bounding box polygons
    bbox['geometry'] = bbox.apply(lambda row: delineation_utils.create_bounding_box(row, 1/rgb_transform[0]), axis=1)
    x_offset, y_offset = rgb_transform[2], rgb_transform[5]
    y_offset = y_offset - rgb_side*rgb_transform[0]
    #from a geopandas, remove all rows with a None geometry
    bbox = bbox[bbox['geometry'].notna()]

    bbox = gpd.GeoDataFrame(bbox, geometry='geometry')
    bbox["geometry"] = bbox["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
    bbox.crs = rgb_crs
    #bbox = bbox.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'])

    #use only points whose crwnpst is greater than 2
    itcs = itcs[itcs['Crwnpst'] > 1]
    itcs.crs = rgb_crs
    image_file = os.path.join(folder, rgb_path)
    hsi_img = os.path.join(folder, hsi_img)
    # Split the image into batches of 40x40m
    batch_size = 40
    #image_file, hsi_img, itcs, bbox,  batch_size=40
    raster_batches, raster_hsi_batches, itcs_batches, itcs_boxes, affine = get_itcs_polygons.split_image(image_file, 
                                hsi_img, itcs, bbox, batch_size)

    #reload dependencies
    reload(delineation_utils)
    reload(get_itcs_polygons)
    # Make predictions of tree crown polygons using SAM
    batch_ = raster_batches[0]
    i = 0
    for(i, batch_) in enumerate(raster_batches):
        #skip empty batches
        if itcs_boxes[i].shape[0] == 0: 
            continue
        
        torch.cuda.empty_cache()  
        input_points=itcs_batches[i].copy()
        input_boxes = itcs_boxes[i].copy()
        batch = batch_[:3,:,:].copy()

        #plot 1 band of batch array
        #plt.imshow(batch[0,:,:])
        # rescale  point and boxes coordiantes if using rgb as batch rather than hsi. 
        # please include this asap in the split_image function instead. Now just checking it out if it works
        # Make predictions of tree crown polygons using SAM
        predictions, _, _ = get_itcs_polygons.predict_tree_crowns(batch=batch, 
                input_points=input_points, #rescale_to =400, 
                grid_size = 20, rgb = True,
                neighbors=16,  mode = 'only_points', point_type = "grid",
                input_boxes = input_boxes )
        
        torch.cuda.empty_cache()    
        # Apply the translation to the geometries in the GeoDataFrame
        x_offset, y_offset = affine[i][2], affine[i][5]
        y_offset = y_offset - batch_size
        #from a geopandas, remove all rows with a None geometry
        predictions = gpd.GeoDataFrame(predictions, geometry='geometry')
        predictions = predictions[predictions['geometry'].notna()]
        predictions["geometry"] = predictions["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
        
        predictions.crs = rgb_crs
        #remove predictions that are clearly too large
        predictions.area
        predictions = predictions[predictions.area < 600]
        
        if hierachical_predictions:
            #if predictions are not empty, run the hierarchical predictions
            if predictions.shape[0] > 0:
                # using predictions, create a new variable with predictions bounding boxes
                p_bboxes = predictions.copy()
                #remove boxes with low score
                p_bboxes = p_bboxes[p_bboxes['score'] > threshold_score]
                p_bboxes = p_bboxes.geometry.bounds
                # append StemTag to p_bboxes
                p_bboxes['StemTag'] = predictions['StemTag'].copy()
                # turn coordinates back to clip coordinates
                p_bboxes['minx'] = p_bboxes['minx'] - x_offset
                p_bboxes['maxx'] = p_bboxes['maxx'] - x_offset
                p_bboxes['miny'] = p_bboxes['miny'] - y_offset
                p_bboxes['maxy'] = p_bboxes['maxy'] - y_offset
                #run the predictions using the bouding boxes
                batch_hsi = raster_hsi_batches[i].copy()
                batch_hsi = batch_hsi[:3,:,:]
                h_preds, scores, _ = get_itcs_polygons.predict_tree_crowns(batch=batch_hsi.copy(), 
                    rgb = False, input_points = input_points, #rescale_to =40, 
                    mode = 'bbox_and_centers', input_boxes = p_bboxes.copy())
                
                # Apply the translation to the geometries in the GeoDataFrame
                h_preds = gpd.GeoDataFrame(h_preds, geometry='geometry')
                h_preds = h_preds[h_preds['geometry'].notna()]
                h_preds["geometry"] = h_preds["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
                h_preds['score']=scores
                h_preds.crs = rgb_crs
                #remove predictions that are clearly too large
                h_preds.area
                h_preds = h_preds[h_preds.area < 600]
                h_preds.to_file(f'{folder}/outdir/HPoly/bboxes{i}.gpkg', driver='GPKG')
        else:
            # Save the predictions as geopandas
            predictions.to_file(f'{folder}/outdir/Polygons/bboxes{i}.gpkg', driver='GPKG')
            #clip boxes to the extent of predictions
        
        bbox.to_file(f'{folder}/outdir/deepForest/bboxes{i}.gpkg', driver='GPKG')

        #batch = batch_[:3,:,:].copy()
        batch = np.moveaxis(batch, 0, -1)
        flipped = np.flip(batch, axis=1) 
        flipped = np.flip(flipped, axis=1) 
        #flipped = np.transpose(flipped, (1, 0, 2)) # swap height and width
        imageio.imwrite(f'{folder}/outdir/clips/itcs_{i}.png', flipped)

