
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
import os
import requests
import config


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


# sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth"
# Define a function to make predictions of tree crown polygons using SAM
def predict_tree_crowns(batch, input_points, neighbors = 3, 
                        input_boxes = None, point_type='grid', 
                        rescale_to = None, mode = 'only_points', rgb = True,
                        sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth",
                        model_type = "vit_h", grid_size = 6):

    if not os.path.exists(sam_checkpoint):
        response = requests.get(config.url, stream=True)
        if not os.path.exists('checkpoints'):
            # Create the folder
            os.mkdir('checkpoints')

        if response.status_code == 200:
            with open(sam_checkpoint, "wb") as file:
                for chunk in response.iter_content():
                    file.write(chunk)
            print("File downloaded successfully.")
        else:
            print("Failed to download the file.")
    else:
        print("File already exists in the target folder.")

    #from tree_health_detection.src.get_itcs_polygons import mask_to_delineation
    from skimage import exposure
    batch = np.moveaxis(batch, 0, -1)
    batch = np.flip(batch, axis=0) 
    original_shape = batch.shape

    #change input boxes values into integers
    if input_boxes is not None:
        input_boxes = input_boxes.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    #neighbors must be the minimum between the total number of input_points and the argument neighbors
    #neighbors = min(input_points.shape[0]-2, neighbors)
    #rescale image to larger size if rescale_to is not null
    if rescale_to is not None:
        batch = resize(batch, (rescale_to, rescale_to), order=3, mode='constant', cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        input_points['x'] = input_points['x'] * rescale_to / original_shape[1]
        input_points['y'] = input_points['y'] * rescale_to / original_shape[0]
        #rescale xmin, ymin, xmax, ymax of input_boxes
        if input_boxes is not None:
            input_boxes[:,0] = input_boxes[:,0] * rescale_to / original_shape[1]
            input_boxes[:,1] = input_boxes[:,1] * rescale_to / original_shape[0]
            input_boxes[:,2] = input_boxes[:,2] * rescale_to / original_shape[1]
            input_boxes[:,3] = input_boxes[:,3] * rescale_to / original_shape[0]

    # linear stretch of the batch image, between 0 and 255
    batch = exposure.rescale_intensity(batch, out_range=(0, 255))
    batch =  np.array(batch, dtype=np.uint8)

    sam.to(device=device)
    predictor = SamPredictor(sam)
    #flip rasterio to be h,w, channels
    predictor.set_image(np.array(batch))
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    #turn stem points into a numpy array
    input_point = np.column_stack((input_points['x'], input_points['y']))
    input_crowns = input_points['StemTag']
    crown_mask = pd.DataFrame(columns=["geometry", "score"])
    crown_scores=[]
    crown_logits=[]
    crown_masks = []

    if input_boxes is not None and mode == 'bbox':# and onnx_model_path is None:
        # if input_boxes is pandas, turn into numpy
        if isinstance(input_boxes, pd.DataFrame):
            input_boxes = input_boxes.to_numpy()
        transformed_boxes_batch = input_boxes[:,:4].copy()
        #if column StemTag is present, add it to the transformed_boxes_batch
        if input_boxes.shape[1] == 5:
            stemID = input_boxes[:,4].copy()

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
        torch.cuda.empty_cache()    
        crown_scores.append(scores)
        crown_logits.append(logits)
        crown_masks.append(masks)

        # loop through the masks, polygonize their raster, and append them into a geopandas dataframe
        for msks in range(len(crown_masks)):
            for it in range(crown_masks[msks].shape[0]):
                #pick the mask with the highest score
                mask = crown_masks[msks][it,0,:,:]
                # Find the indices of the True values
                true_indices = np.argwhere(mask)
                #skip empty masks
                if true_indices.shape[0] < 3:
                    continue
                # Calculate the convex hull
                polygons = mask_to_delineation(mask)
                #likewise, if polygon is empty, skip
                if len(polygons) == 0:
                    continue    

                # Create a GeoDataFrame and append the polygon
                gdf_temp = gpd.GeoDataFrame(geometry=[polygons[0]], columns=["geometry"])
                gdf_temp["score"] = crown_scores[msks][it]
                if input_boxes.shape[1] == 5:
                    gdf_temp["StemTag"] = stemID[it]
                else:    
                    gdf_temp["StemTag"] = str(msks)+'_'+str(it)

                # Append the temporary GeoDataFrame to the main GeoDataFrame
                crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)

    if input_boxes is not None and mode == 'bbox_and_centers':# and onnx_model_path is None:
                # if input_boxes is pandas, turn into numpy
        if isinstance(input_boxes, pd.DataFrame):
            input_boxes = input_boxes.to_numpy()
        transformed_boxes_batch = input_boxes[:,:4].copy()
        #if column StemTag is present, add it to the transformed_boxes_batch
        if input_boxes.shape[1] == 5:
            stemID = input_boxes[:,4].copy()
        else:
            stemID = np.arange(input_boxes.shape[0])

        # get the center of each box assuming that that is for sure a point in the crown
        centers = (transformed_boxes_batch[:,0] + transformed_boxes_batch[:,2])/2 
        #append a colum to center with y mean
        centers = np.column_stack((centers, (transformed_boxes_batch[:,1]+ transformed_boxes_batch[:,3])/2))
        #get the cardinal points of each box: north east
        cardinal_points_1 =   np.column_stack((transformed_boxes_batch[:,0], transformed_boxes_batch[:,1]))
        # get the cardinal points for each box: south west
        cardinal_points_2 =  np.column_stack((transformed_boxes_batch[:,2], transformed_boxes_batch[:,3]))
        # get the cardinal points for each box: south east
        cardinal_points_3 =  np.column_stack((transformed_boxes_batch[:,2], transformed_boxes_batch[:,1]))
        # get the cardinal points for each box: north west
        cardinal_points_4 =  np.column_stack((transformed_boxes_batch[:,0], transformed_boxes_batch[:,3]))

        #loop through each box
        for it in range(transformed_boxes_batch.shape[0]):
            input_label = [0,0,0,0,1]
            target_points = np.array([cardinal_points_1[it], cardinal_points_2[it], cardinal_points_3[it], cardinal_points_4[it], centers[it]])
            target_box = transformed_boxes_batch[it].copy()
            if rgb == True:
                target_box = target_box*10
                target_points = target_points*10
                
            masks,scores, logits = predictor.predict(
                point_coords=target_points,
                point_labels=input_label,
                box=target_box,
                multimask_output=True,
            )

            #pick the mask with the highest score
            if len(masks) > 1:
                masks = masks[scores.argmax()]
                scores = scores[scores.argmax()]
            # Find the indices of the True values
            true_indices = np.argwhere(masks)
            #skip empty masks
            if true_indices.shape[0] < 3:
                continue
            
            target_itc = centers[it].copy()
            # Calculate the convex hull
            if rgb:
                target_itc = target_itc*10 

            individual_point = Point(target_itc)
            polygons = mask_to_delineation(masks.copy(), rgb=rgb)
            #pick the polygon that intercepts with individual point
            #polygons = [poly for poly in polygons if poly.intersects(individual_point)]
            if len(polygons) ==0:
                continue

            # Create a GeoDataFrame and append the polygon
            gdf_temp = gpd.GeoDataFrame(geometry=[polygons[0]], columns=["geometry"])
            gdf_temp["score"] = scores
            gdf_temp["StemTag"] = stemID[it]

            # Append the temporary GeoDataFrame to the main GeoDataFrame
            crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)
            crown_scores.append(scores)
            crown_logits.append(logits)


    if input_boxes is None or mode == 'only_points':# and onnx_model_path is None:
        #loop through each stem point, make a prediction, and save the prediction
        if rgb == True:
            input_point[:,0] = input_point[:,0] * 10 
            input_point[:,1] = input_point[:,1] * 10 
            grid_size = grid_size * 10

        for it in range(input_point.shape[0]):
            #update input_label to be 0 everywhere except at position it       
            input_label = np.zeros(input_point.shape[0])
            input_label[it] = 1
            target_itc = input_point[it].copy()

            # Calculate the Euclidean distance between the ith row and the other rows
            distances = np.linalg.norm(input_point - input_point[it], axis=1)
            if point_type == "distance":
                # using distance, get the indexes of input_points, ordered by distances
                closest_indices = np.argsort(distances)[1:neighbors+1]
            # find the 4 closest points one in each cardinal direction
            elif point_type == "cardinal":
                closest_indices = np.argpartition(distances, 4)[:4]
            #pick a random sample of points, making sure that none is the target point
            
            elif point_type == "random":
                #pick a random sample of points, making sure that none is the target point
                closest_indices = np.random.choice(np.delete(np.arange(input_point.shape[0]), it), neighbors, replace=False)

            elif point_type == "grid":
                #create a grid of points around the target_itc, the grid is a squared box around target_itc. 
                # if neighbors = 4 get the corners of the box
                # if neighbors = 8 get the corners and the midpoints of the sides
                # if neighbors = 16 get the corners, the midpoints of the sides and the center
                # if points outside the image, pick the closest point inside the image

                points = []
                #get the coordinates of the target_itc
                x = target_itc[0]
                y = target_itc[1]
                #get the coordinates of the top left corner of the grid
                #make sure that the grid is inside the image
                #append to points
                points.append([max(x - grid_size, 0), min(y + grid_size, batch.shape[0])])
                #get the coordinates of the bottom right corner of the grid
                points.append([min(x + grid_size, batch.shape[1]), max(y - grid_size, 0)])
                #get the coordinates of the top right corner of the grid
                points.append([min(x + grid_size, batch.shape[1]), min(y + grid_size, batch.shape[0])])
                #get the coordinates of the bottom left corner of the grid
                points.append([max(x - grid_size, 0), max(y - grid_size, 0)])
                #get the coordinates of the midpoints of the sides of the grid
                points.append([x, min(y + grid_size, batch.shape[0])])
                points.append([x, max(y - grid_size, 0)])
                points.append([min(x + grid_size, batch.shape[1]), y])
                points.append([max(x - grid_size, 0), y])
                #get the coordinates of the midpoints of the sides of the grid
                points.append([min(x + grid_size, batch.shape[1]), min(y + grid_size, batch.shape[0])])
                points.append([max(x - grid_size, 0), min(y + grid_size, batch.shape[0])])
                points.append([min(x + grid_size, batch.shape[1]), max(y - grid_size, 0)])
                points.append([max(x - grid_size, 0), max(y - grid_size, 0)])
                #get the coordinates of the midpoints of the sides of the grid
                points.append([x, min(y + grid_size, batch.shape[0])])
                points.append([x, max(y - grid_size, 0)])
                points.append([min(x + grid_size, batch.shape[1]), y])
                points.append([max(x - grid_size, 0), y])

                points = np.array(points)[:neighbors,:]
            else:
                # define points as the index of all the points in the  input_point but input_point[it]
                closest_indices = np.delete(np.arange(input_point.shape[0]), it)

            # Subset the array to the ith row and the 10 closest rows

            if point_type != "grid":
                subset_point = input_point[closest_indices].copy()
                subset_label = input_label[closest_indices]
                subset_label = subset_label.astype(np.int8)
                #append subset_point with the target point
                subset_point = np.vstack((subset_point, target_itc))
                subset_label = np.append(subset_label, 1)
            else:
                subset_point = points.copy()
                subset_label = np.zeros(points.shape[0], dtype=np.int8)
                #append subset_point with the target point
                subset_point = np.vstack((subset_point, target_itc))
                subset_label = np.append(subset_label, 1)

            # set up a bounding box around the target point
            min_x = np.min(subset_point[:,0])
            max_x = np.max(subset_point[:,0])
            min_y = np.min(subset_point[:,1])
            max_y = np.max(subset_point[:,1])

            subset_point = np.vstack((subset_point, [0.0, 0.0]))
            subset_label = np.append(subset_label, -1)
            # Create an empty mask input and an indicator for no mask.
            #mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

            target_bbox = np.array([[min_x, min_y, max_x, max_y]])
            masks, scores, logits  = predictor.predict(
                point_coords=subset_point,
                point_labels=subset_label,
                box=target_bbox,
                multimask_output=True,
            )
            '''
            mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
            masks, scores, _  = predictor.predict(
                point_coords=subset_point,
                point_labels=subset_label,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            '''
            #uncomment if getting rid of ONNX
            '''
            masks, scores, logits = predictor.predict(
                point_coords=subset_point,
                point_labels=subset_label,
                multimask_output=False,
            )
            '''

            #pick the mask with the highest score
            if len(masks) > 1:
                masks = masks[scores.argmax()]
                scores = scores[scores.argmax()]
            else:
                masks = masks[0]
                scores = scores[0]
            # Find the indices of the True values
            true_indices = np.argwhere(masks)
            #skip empty masks
            if true_indices.shape[0] < 3:
                continue

            polygons = mask_to_delineation(mask = masks, center = target_itc,  buffer_size = 0)
            #pick the polygon that intercepts with individual point

            # Calculate the convex hull
            if rgb:
                target_itc = target_itc/10 
                scaled_polygons = []
                for poly in polygons:
                    # divide all cooridnates of polygons by 10 to get meters
                    x, y = poly.exterior.coords.xy
                    # Divide each vertex coordinate by 10
                    x = [coord / 10 for coord in x]
                    y = [coord / 10 for coord in y]

                    # Create a new polygon with the scaled coordinates
                    tmp = Polygon(zip(x, y))
                    scaled_polygons.append(tmp)

            individual_point = Point(target_itc)
            polygons = [poly for poly in scaled_polygons if poly.intersects(individual_point)]
            if len(polygons) ==0:
                continue

            # Create a GeoDataFrame and append the polygon
            gdf_temp = gpd.GeoDataFrame(geometry=[polygons[0]], columns=["geometry"])
            gdf_temp["score"] = scores
            gdf_temp["StemTag"] = input_crowns.iloc[it]

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

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
from rasterio.warp import reproject, Resampling

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

def split_image(image_file, hsi_img, itcs, bbox, batch_size=40, buffer_distance=10):
    with rasterio.open(image_file) as src:
        height, width = src.shape
        resolution = src.transform[0]
        batch_size_ = int(batch_size / resolution)
        
    buffer_distance_ = int(buffer_distance)

    raster_batches = []
    hsi_batches = []
    itcs_batches = []
    affines = []
    itcs_boxes = []
    dbox = []

    for i in range(0, height, batch_size_):
        for j in range(0, width, batch_size_):

            # get larger j and i
            i_ = max(0, i - buffer_distance_)
            j_ = max(0, j - buffer_distance_)
            i_ = min(height, i_)
            j_ = min(width, j_)

            # make sure +2*buffer_distance_ doesn't make the batch size  larger than the image size
            wdt = min(batch_size_+2*buffer_distance_, width-j_)
            hgt = min(batch_size_+2*buffer_distance_, height-i_)
            window = Window(col_off=j_, row_off=i_, width=wdt, height=hgt)

            with rasterio.open(image_file) as src:
                batch = src.read(window=window)
                raster_batches.append(batch)
                
                left, top = src.xy(i_, j_)
                right, bottom = src.xy(i_ + wdt, j_ + hgt)
                batch_bounds = box(left , bottom , right , top )

            with rasterio.open(hsi_img) as hsi: 
                resolution_hsi = hsi.transform[0]
                resolution_factor =  resolution_hsi / resolution 
                batch_size_hsi = round(batch_size_ / resolution_factor)
                hsi_height, hsi_width = hsi.shape
                window_hsi = Window(col_off=j_ / resolution_factor, row_off=i_ / resolution_factor, 
                            width=wdt/resolution_factor, height=hgt/resolution_factor)
                hsi_batch = hsi.read(window=window_hsi)
                hsi_batches.append(hsi_batch)

            itcs_clipped = gpd.clip(itcs, batch_bounds)
            itcs_clipped["geometry"] = itcs_clipped["geometry"].apply(
                transform_coordinates, x_offset=left, y_offset=bottom
            )

            bbox_clipped = gpd.clip(bbox, batch_bounds)
            bbox_clipped = bbox_clipped[bbox_clipped.geometry.type == 'Polygon']

            itcs_df = pd.DataFrame(
                {
                    "StemTag": itcs_clipped["StemTag"],
                    "x": itcs_clipped["geometry"].x,
                    "y": itcs_clipped["geometry"].y,
                }
            )

            tmp_bx = []
            left, bottom, right, top = batch_bounds.bounds
            for rows in range(bbox_clipped.shape[0]):
                bounds = bbox_clipped.geometry.bounds.iloc[rows]
                bbox_clipped['xmin'] = bounds['minx'] - left 
                bbox_clipped['ymin'] = bounds['miny'] - bottom
                bbox_clipped['xmax'] = bounds['maxx'] - left
                bbox_clipped['ymax'] = bounds['maxy'] - bottom
                bleft = bbox_clipped['xmin'].values[0] 
                bbottom = bbox_clipped['ymin'].values[0] 
                bright = bbox_clipped['xmax'].values[0] 
                btop = bbox_clipped['ymax'].values[0]
                tmp_bx.append([bleft, bbottom, bright, btop])

            tmp_bx = np.array(tmp_bx)

            # Append the DataFrame to the list
            itcs_batches.append(itcs_df)
            affines.append(src.window_transform(window))
            itcs_boxes.append(tmp_bx)
            dbox.append(bbox_clipped)
    # Return the lists of raster batches and clipped GeoDataFrames
    return raster_batches, hsi_batches, itcs_batches, itcs_boxes, affines


# merge bbox and tmp_bx, then subtract bbox from tmp_bx
def get_bbox_diff(bbox_clipped, tmp_bx):
    #subtract columnwise bbox_clipped from tmp_bx
    bbox_diff = tmp_bx - bbox_clipped[['xmin', 'ymin', 'xmax', 'ymax']].values


from multiprocessing import Pool

def create_polygon(category, category_mask, offset_x, offset_y):
    polygons = []
    # Create a binary mask for the current category
    binary_mask = (category_mask == category).astype(np.uint8)
    # Convert binary_mask to 8-bit single-channel image
    binary_mask = (binary_mask * 255).astype(np.uint8)
    # Find contours of the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # use a concave hull instead
    #contours = [cv2.convexHull(contour) for contour in contours]
    # if the data is rgb, rescale to meters by dividing by 10 each index
    # skip if does not have enough dimensions
    if contours and contours[0].shape[0] >= 3:
        # Convert the contours to polygons
        for contour in contours:
            # Simplify the contour to a polygon
            poly = Polygon(shell=contour.squeeze())
            # shift polygons coordinates to the position before clipping labelled mask
            poly = translate(poly, xoff=offset_x, yoff=offset_y)
            polygons.append(poly)

    return polygons


def mask_to_delineation(mask, center, rgb = True, buffer_size = 0):

    labeled_mask = label(mask, connectivity=1)
    # if 3d flatten to 2d
    if len(labeled_mask.shape) == 3:
        labeled_mask=labeled_mask[0,:,:]

    # Compute the smallest region containing all non-zero values
    non_zero_indices = np.nonzero(labeled_mask)
    min_x, min_y = np.min(non_zero_indices, axis=1)
    max_x, max_y = np.max(non_zero_indices, axis=1)
    # Define the center of the smallest region

    if buffer_size is not None:
        #identify buffer around the target to dramatically increase efficency
        submask = labeled_mask[int(min_x):int(max_x), int(min_y):int(max_y)]
    else:
        submask = labeled_mask

    # Create a dictionary with as many values as unique values in the labeled mask
    unique_labels = np.unique(submask)
    label_to_category = {label: category for label, category in zip(unique_labels, range(unique_labels.shape[0]))}

    category_mask = np.vectorize(label_to_category.get)(submask)
    
    # Turn each category into a polygon
    polygons = []
    for category in np.unique(category_mask):
        if category == 0:  # Skip the background
            continue
        # Create a binary mask for the current category
        poly = create_polygon(category, category_mask, min_y, min_x)
        polygons.append(poly)
    # flatten the list of lists
    polygons = [item for sublist in polygons for item in sublist]
    #check that center is in the polygon
    polygons = [poly for poly in polygons if poly.contains(Point(center))]

    return polygons
