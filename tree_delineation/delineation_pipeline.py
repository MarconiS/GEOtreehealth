import geopandas as gpd
import numpy as np
import os
from rasterio.mask import mask
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

import tree_delineation
import config
# check if the tree_tops file exists. if not, launch get_tree_tops
def build_data_schema():

    if os.path.exists(os.path.join(config.data_path,config.stem_path)):
        itcs = gpd.read_file(os.path.join(config.data_path+config.stem_path))
    else:
        print('itc = get_tree_tops(laz_path)')


    #get affine of the raster used to extract the bounding boxes
    with rasterio.open(config.data_path+config.seg_rgb_path) as src: 
        rgb_transform = src.transform
        rgb_crs = src.crs
        rgb_side = src.width

    #get tree bounding boxes with deepForest for SAM
    if config.ttops == 'deepforest':
        bbox = tree_delineation.delineation_utils.extract_boxes(config.data_path+config.seg_rgb_path)
        #use bbox xmin,ymin,xmax,ymax to make corners of bounding box polygons
        bbox['geometry'] = bbox.apply(lambda row: tree_delineation.delineation_utils.create_bounding_box(row, 1/rgb_transform[0]), axis=1)
        x_offset, y_offset = rgb_transform[2], rgb_transform[5]
        y_offset = y_offset - rgb_side*rgb_transform[0]
        #from a geopandas, remove all rows with a None geometry
        bbox = bbox[bbox['geometry'].notna()]

        bbox = gpd.GeoDataFrame(bbox, geometry='geometry')
        bbox["geometry"] = bbox["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
        bbox.crs = rgb_crs
    
    elif config.ttops == 'lidR':
        bbox = gpd.read_file(os.path.join(config.data_path, '../legacy_polygons/'+config.legacy_polygons))
    #bbox = bbox.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'])
    elif config.ttops is None:
        bbox = None

    #use only points whose crwnpst is greater than 2
    itcs = itcs[itcs['Crwnpst'] > 2] #this comment it out soon. That kind of preprocessig should be done before the function is called
    itcs.crs = rgb_crs
    image_file = os.path.join(config.data_path, config.seg_rgb_path)
    hsi_path = os.path.join(config.data_path, config.seg_hsi_img)
    # Split the image into batches of 40x40m
    batch_size = config.seg_img_batch
    #image_file, hsi_img, itcs, bbox,  batch_size=40
    raster_batches, raster_hsi_batches, itcs_batches, itcs_boxes, affine = tree_delineation.get_polygons.split_image(image_file = image_file, 
                                hsi_img = hsi_path, itcs = itcs, bbox = bbox, 
                                batch_size = config.seg_img_batch, buffer_distance = config.seg_img_buffer)

    # Make predictions of tree crown polygons using SAM
    for(i, batch_) in enumerate(raster_batches):
        #skip empty batches
        if itcs_boxes[i].shape[0] == 0: 
            continue
        
        torch.cuda.empty_cache()  
        input_points=itcs_batches[i].copy()
        input_boxes = itcs_boxes[i].copy()
        batch = batch_[:3,:,:].copy()

        # rescale  point and boxes coordiantes if using rgb as batch rather than hsi. 
        # please include this asap in the split_image function instead. Now just checking it out if it works
        # Make predictions of tree crown polygons using SAM
        predictions, _, _ = tree_delineation.get_polygons.predict_tree_crowns(batch=batch, 
                input_points=input_points, rescale_to =config.rescale_to, 
                grid_size = config.grid_size, rgb = config.isrgb,
                neighbors=config.neighbors, mode = config.mode, point_type =  config.point_type,
                sam_checkpoint = config.sam_checkpoint, model_type=config.model_type,
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
        predictions.length / predictions.area
        #predictions = predictions[predictions.area < 600]
        #simplify geometries
        predictions['geometry'] = predictions['geometry'].simplify(0.1)
        predictions['geometry'] = predictions['geometry'].buffer(0)
        # Save the predictions as geopandas
        # check if path exists
        if not os.path.exists(os.path.join(config.data_path+'/Crowns/'+ config.siteID)):
            os.mkdir(os.path.join(config.data_path+'/Crowns/'+ config.siteID))

        predictions.to_file(f'{config.data_path}/Crowns/{config.siteID}/SAM_{i}.gpkg', driver='GPKG')

    return predictions

        




