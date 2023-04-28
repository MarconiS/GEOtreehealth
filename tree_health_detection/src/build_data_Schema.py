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
# Loop through the files in the folder
#for file in os.listdir(folder):
# check if the tree_tops file exists. if not, launch get_tree_tops
def build_data_schema(folder, stem_path,rgb_path, hsi_path, laz_path):

    #reload dependencies
    reload(delineation_utils)
    reload(get_itcs_polygons)

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
    torch.cuda.empty_cache()    

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

    # Make predictions of tree crown polygons using SAM
    for(i, batch_) in enumerate(raster_hsi_batches):
        #skip empty batches
        if itcs_boxes[i].shape[0] == 0: 
            continue
        
        #predictions = effe.copy()
        # Make predictions of tree crown polygons using SAM
        predictions, _, _ = get_itcs_polygons.predict_tree_crowns(batch=batch_[:3,:,:], input_points=itcs_batches[i], rescale_to =400, 
                                                                  input_boxes = itcs_boxes[i], neighbors=3, point_type = "euclidian") 
        # Apply the translation to the geometries in the GeoDataFrame
        x_offset, y_offset = affine[i][2], affine[i][5]
        y_offset = y_offset - batch_size
        #from a geopandas, remove all rows with a None geometry
        predictions = gpd.GeoDataFrame(predictions, geometry='geometry')
        predictions = predictions[predictions['geometry'].notna()]
        predictions["geometry"] = predictions["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
        predictions.crs = rgb_crs
        #divide by 3 to get the correct coordinates
        
        # Save the predictions as geopandas
        predictions.to_file(f'{folder}/outdir/Polygons/bboxes{i}.gpkg', driver='GPKG')
        #clip boxes to the extent of predictions
        
        bbox.to_file(f'{folder}/outdir/deepForest/bboxes{i}.gpkg', driver='GPKG')

        batch_ = batch_[:3,:,:]
        batch_ = np.moveaxis(batch_, 0, -1)
        imageio.imwrite(f'{folder}/outdir/clips/itcs_{i}.png', batch_)

