
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

def mask_to_polygons(mask, individual_point):
    # Find contours in the mask
    contours = np.array([mask], dtype=np.uint8)        #
    #np.array([mask], dtype=np.uint8)
    polygon_generator = shapes(contours)#, transform=transform)

    # Create a GeoDataFrame from the polygons
    geometries = []
    values = []

    for polygon, value in polygon_generator:
        geometries.append(shape(polygon))
        values.append(value)

    gdf_ = gpd.GeoDataFrame(geometry=geometries)
    gdf_['value'] = values

    #remove  polygons with value 0
    gdf_ = gdf_[gdf_['value'] != 0]

    # Check if there are any valid line segments
    if gdf_.shape[0] == 0:
        return None

    if individual_point is not None:
    # Filter the polygons to include only those that contain the individual_point
        containing_polygons = [polygon for polygon in gdf_['geometry']  if polygon.contains(individual_point)]
        # Choose the largest polygon based on its area
        largest_polygon = max(containing_polygons, key=lambda p: p.area)
    #calculate the area of the polygon
    # If there are no containing polygons, return None
    if not containing_polygons:
        return None

    # Choose the largest polygon based on its area
    #largest_polygon = max(containing_polygons, key=lambda p: p.area)
    return largest_polygon


def mask_to_delineation(mask):
    labeled_mask = label(mask, connectivity=1)
    #labeled_mask = labeled_mask.astype(np.uint16)
    # Create a dictionary with as many values as unique values in the labeled mask
    label_to_category = {label: category for label, category in zip(np.unique(labeled_mask), range(0, np.unique(labeled_mask).shape[0]))}

    category_mask = np.vectorize(label_to_category.get)(labeled_mask)
    
    # Turn each category into a polygon
    polygons = []
    
    for category in np.unique(category_mask):
        if category == 0:  # Skip the background
            continue

        # Create a binary mask for the current category
        binary_mask = (category_mask == category).astype(np.uint8)
        # Convert binary_mask to 8-bit single-channel image
        binary_mask = (binary_mask * 255).astype(np.uint8)
        # Find contours of the binary mask
        contours, _ = cv2.findContours(binary_mask[0,:,:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #skip if does not have enough dimensions
        if contours[0].shape[0] < 3:
            continue
        # Convert the contours to polygons
        for contour in contours:
            # Simplify the contour to a polygon
            poly = Polygon(shell=contour.squeeze())
            polygons.append(poly)

    return polygons


# sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth"
# Define a function to make predictions of tree crown polygons using SAM
def predict_tree_crowns(batch, input_points, neighbors = 10, 
                        input_boxes = None, point_type='random', 
                        onnx_model_path = None,  rescale_to = None, mode = 'bbox',
                        sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth",
                        model_type = "vit_h"):


    batch = np.moveaxis(batch, 0, -1)
    original_shape = batch.shape


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    #neighbors must be the minimum between the total number of input_points and the argument neighbors
    neighbors = min(input_points.shape[0]-2, neighbors)
    #rescale image to larger size if rescale_to is not null
    if rescale_to is not None:
        batch = resize(batch, (rescale_to, rescale_to), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
        input_points['x'] = input_points['x'] * rescale_to / original_shape[1]
        input_points['y'] = input_points['y'] * rescale_to / original_shape[0]

    # linstretch the image, normalize it to 0, 255 and convert to int8
    batch = np.uint8(255 * (batch - batch.min()) / (batch.max() - batch.min()))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    #flip rasterio to be h,w, channels
    predictor.set_image(batch)

    #turn stem points into a numpy array
    input_point = np.column_stack((input_points['x'], input_points['y']))
    input_crowns = input_points['StemTag']
    crown_mask = pd.DataFrame(columns=["geometry", "score"])
    crown_scores=[]
    crown_logits=[]
    crown_masks = []
    if input_boxes is not None and mode == 'bbox':# and onnx_model_path is None:
        # this part may be a GPU bottleneck. Better divide the boxes into batches of 1000 max
        # divede the boxes into batches of 1000 max if they are more than 1000
        if  input_boxes.shape[0] > 100:
            n_batches = int(input_boxes.shape[0] / 100)
            for i in range(n_batches):
                #transform the boxes into a torch.tensor
                transformed_boxes_batch = input_boxes[i*100:(i+1)*100, :]
                transformed_boxes_batch = torch.tensor(transformed_boxes_batch, device=predictor.device)
                transpformed_boxes_batch = predictor.transform.apply_boxes_torch(transformed_boxes_batch, batch.shape[:2])
                masks,scores, logits = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes_batch,
                    multimask_output=False,
                )
                #free space from GPU
                masks = masks.cpu().numpy()
                scores = scores.cpu().numpy()
                logits = logits.cpu().numpy()
                #append batch results to the list
                crown_masks.append(masks)
                crown_scores.append(scores)
                crown_logits.append(logits)

                del(transformed_boxes_batch, masks,scores, logits)
                torch.cuda.empty_cache()
            #predict the last batch
            transformed_boxes_batch = input_boxes[(i+1)*100:, :]
            transformed_boxes_batch = torch.tensor(transformed_boxes_batch, device=predictor.device)
            transformed_boxes_batch = predictor.transform.apply_boxes_torch(transformed_boxes_batch, batch.shape[:2])
            masks,scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes_batch,
                multimask_output=False,
            )
            #free space from GPU
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
            logits = logits.cpu().numpy()
            #append batch results to the list
            crown_masks.append(masks)
            crown_scores.append(scores)
            crown_logits.append(logits)
        else:
            transformed_boxes_batch = input_boxes
            transformed_boxes_batch = torch.tensor(transformed_boxes_batch, device=predictor.device)
            transformed_boxes_batch = predictor.transform.apply_boxes_torch(transformed_boxes_batch, batch.shape[:2])
            masks,scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes_batch,
                multimask_output=False,
            )
            #free space from GPU
            masks = masks.cpu().numpy()
            scores = scores.cpu().numpy()
            logits = logits.cpu().numpy()
            crown_scores.append(scores)
            crown_logits.append(logits)
            crown_masks.append(masks)

        # loop through the masks, polygonize their raster, and append them into a geopandas dataframe
        for msks in range(len(crown_masks)):
            for it in range(crown_masks[msks].shape[0]):
                #pick the mask with the highest score
                mask = crown_masks[msks][it]
                # Find the indices of the True values
                true_indices = np.argwhere(mask)
                #skip empty masks
                if true_indices.shape[0] < 3:
                    continue
                # Calculate the convex hull
                polygons = get_itcs_polygons.mask_to_delineation(mask)
                #likewise, if polygon is empty, skip
                if len(polygons) == 0:
                    continue    

                # Create a GeoDataFrame and append the polygon
                gdf_temp = gpd.GeoDataFrame(geometry=[polygons[0]], columns=["geometry"])
                gdf_temp["score"] = crown_scores[msks][it]
                gdf_temp["point_id"] = str(msks)+'_'+str(it)

                # Append the temporary GeoDataFrame to the main GeoDataFrame
                crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)
                crown_scores.append(scores)
                crown_logits.append(logits)

    if input_boxes is None or mode == 'only_points':# and onnx_model_path is None:
        #loop through each stem point, make a prediction, and save the prediction
        for it in range(0, input_point.shape[0]):
            #update input_label to be 0 everywhere except at position it       
            input_label = np.zeros(input_point.shape[0])
            input_label[it] = 1
            target_itc = input_point[it]
            # subset the input_points to be the current point and the 10 closest points
            # Calculate the Euclidean distance between the ith row and the other rows
            distances = np.linalg.norm(input_point - input_point[it], axis=1)
            if point_type == "euclidian":
            # Find the indices of the 10 closest rows
                closest_indices = np.argpartition(distances, neighbors+1)[:neighbors+1]  # We use 11 because the row itself is included
            elif point_type == "random":
                closest_indices = np.random.choice(np.arange(0, input_point.shape[0]), neighbors+1, replace=True)
            # Subset the array to the ith row and the 10 closest rows
            subset_point = input_point[closest_indices]
            subset_label = input_label[closest_indices]
            subset_label = subset_label.astype(np.int8)

            masks, scores, logits = predictor.predict(
                point_coords=subset_point,
                point_labels=subset_label,
                multimask_output=False,
            )
            #pick the mask with the highest score
            masks = masks[scores.argmax()]
            scores = scores[scores.argmax()]
            # Find the indices of the True values
            true_indices = np.argwhere(masks)
            #skip empty masks
            if true_indices.shape[0] < 3:
                continue

            # Calculate the convex hull
            individual_point = Point(input_point[it])
            polygons = mask_to_polygons(masks, individual_point)

            # Create a GeoDataFrame and append the polygon
            gdf_temp = gpd.GeoDataFrame(geometry=[polygons], columns=["geometry"])
            gdf_temp["score"] = scores
            gdf_temp["stemTag"] = input_crowns.iloc[it]

            # Append the temporary GeoDataFrame to the main GeoDataFrame
            crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)
            crown_scores.append(scores)
            crown_logits.append(logits)

    # Convert the DataFrame to a GeoDataFrame
    crown_mask = gpd.GeoDataFrame(crown_mask, geometry=crown_mask.geometry)
    #reshift crown mask coordiantes to original size
    if rescale_to is not None:
        crown_mask['geometry'] = crown_mask['geometry'].translate(xoff=0, yoff=0)
        crown_mask['geometry'] = crown_mask['geometry'].scale(xfact=original_shape[1]/rescale_to, yfact=original_shape[0]/rescale_to, origin=(0,0))

    return crown_mask, crown_scores, crown_logits


# Define a function to save the predictions as geopandas
def save_predictions(predictions, output_path):
    # Initialize an empty geopandas dataframe
    gdf = gpd.GeoDataFrame()
    # Loop through the predictions
    for prediction in predictions:
        # Convert the prediction to a polygon geometry
        polygon = gpd.GeoSeries(prediction).unary_union
        # Append the polygon to the dataframe
        gdf = gdf.append({'geometry': polygon}, ignore_index=True)
    # Save the dataframe as a shapefile
    gdf.to_file(output_path)


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

def transform_coordinates(geometry, x_offset, y_offset):
    if geometry.type == "Point":
        return Point(geometry.x - x_offset, geometry.y - y_offset)
    elif geometry.type == "MultiPoint":
        return MultiPoint([Point(p.x - x_offset, p.y - y_offset) for p in geometry])
    elif geometry.type == "Polygon":
        return Polygon([(p[0] - x_offset, p[1] - y_offset) for p in geometry.exterior.coords])
    else:
        raise ValueError("Unsupported geometry type")

import numpy as np
from scipy.ndimage import zoom

def upscale_array(input_array, reference_array, input_resolution, reference_resolution):
    # Calculate the ratio between the spatial resolutions
    ratio =  input_resolution /reference_resolution

    # Account for raster arrays with shape (bands, height, width)
    zoom_factors = (1,) + (ratio, ratio)
    upscaled_array = zoom(input_array, zoom_factors, order=3)

    # In case the upscaled dimensions are slightly larger than the reference dimensions,
    # truncate or pad the upscaled array to match the reference array's dimensions
    shape_diff = tuple(s - r for s, r in zip(upscaled_array.shape, reference_array.shape))
    upscaled_array = upscaled_array[:, :reference_array.shape[1], :reference_array.shape[2]]

    return upscaled_array

def split_image(image_file, hsi_img, itcs, bbox,  batch_size=40):
    # Open the raster image


    with rasterio.open(image_file) as src:
        # Get the height and width of the image in pixels
        height, width = src.shape
        # Convert the batch size from meters to pixels
        resolution =src.transform[0]
        batch_size_ = int(batch_size / resolution)
        # Initialize lists to store the raster batches and clipped GeoDataFrames
        raster_batches = []
        hsi_batches = []
        itcs_batches = []
        affines = []
        itcs_boxes = []
        # Loop through the rows and columns of the image
        for i in range(0, height, batch_size_):
            for j in range(0, width, batch_size_):
                # Define a window for the current batch
                window = Window(col_off=j, row_off=i, width=batch_size_, height=batch_size_)
                # Read the batch from the raster image
                batch = src.read(window=window)
                # Append the raster batch to the list
                raster_batches.append(batch)
                 
                # Convert the window to geospatial coordinates
                left, top = src.xy(i, j)
                right, bottom = src.xy(i+batch_size_, j+batch_size_)
                batch_bounds = box(left, bottom, right, top)

                #rasterio load hsi_img only within the bounds of batch_bounds
                with rasterio.open(hsi_img) as hsi: 
                    resolution_hsi = hsi.transform[0]
                    #modify window to account for hsi resolution
                    resolution_factor =  resolution_hsi /resolution 
                    batch_size_hsi = round(batch_size_ / resolution_factor)
                    window_hsi = Window(col_off=j/resolution_factor, row_off=i/resolution_factor, width=batch_size_hsi, height=batch_size_hsi)
                    hsi_batch = hsi.read(window=window_hsi)

                # upscale hsi_batch to the same size as batch
                hsi_batch = upscale_array(hsi_batch, batch, resolution_hsi, resolution)

                hsi_batches.append(hsi_batch)

                # Clip the GeoDataFrame using the batch bounds
                itcs_clipped = gpd.clip(itcs, batch_bounds)

                # Transform the coordinates relative to the raster batch's origin
                itcs_clipped["geometry"] = itcs_clipped["geometry"].apply(
                    transform_coordinates, x_offset=left, y_offset=bottom
                )
                
                bbox_clipped = deepforest.utilities.annotations_to_shapefile(bbox, transform=src.transform, crs = src.crs)
                #from bboxes, clip only those whose xmin, ymin, xmax, ymax fit within the batch bounds
                bbox_clipped = gpd.clip(bbox_clipped, batch_bounds)

                #remove boxes that are LINESTRING or POINT
                bbox_clipped = bbox_clipped[bbox_clipped.geometry.type == 'Polygon']
                # Transform the coordinates of each box polygin relative to the raster batch's origin
                bbox_clipped["geometry"] = bbox_clipped["geometry"].apply(
                                    transform_coordinates, x_offset=left, y_offset=bottom
                                )
                # Create a new DataFrame with stemTag, x, and y columns
                itcs_df = pd.DataFrame(
                    {
                        "StemTag": itcs_clipped["StemTag"],
                        "x": itcs_clipped["geometry"].x,
                        "y": itcs_clipped["geometry"].y,
                    }
                )
                # Create a new DataFrame with label, x, and y columns from bbox_clipped
                # Extract the bounding box coordinates for each polygon
                tmp_bx = []
                for geometry in bbox_clipped.geometry:
                    bounds = geometry.bounds  # Returns (minx, miny, maxx, maxy)
                    left, bottom, right, top = bounds
                    tmp_bx.append([left, bottom, right, top])

                tmp_bx = np.array(tmp_bx)

                # Append the DataFrame to the list
                itcs_batches.append(itcs_df)
                affines.append(src.window_transform(window))
                itcs_boxes.append(tmp_bx)
    # Return the lists of raster batches and clipped GeoDataFrames
    return raster_batches, hsi_batches, itcs_batches, itcs_boxes, affines

