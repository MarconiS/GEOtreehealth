
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
from shapely.affinity import translate, scale

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
from transformers import SamModel, SamProcessor


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
def predict_tree_crowns(batch, input_points, affine, neighbors = 3, first_neigh = 1,
                        input_boxes = None, point_type='grid', resolution = 0.1,
                        rescale_to = None, mode = 'only_points', rgb = True,
                        sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth",
                        model_type = "vit_h", grid_size = 6, grid_space =200):

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
    batch = np.transpose(batch, (2, 1, 0))

    # change x with y
    batch = np.flip(np.transpose(batch, (1, 0, 2)), 1)

    #batch = np.flip(batch, axis=0) 
    batch = np.flip(batch, axis=1) 
    # linear stretch of the batch image, between 0 and 255
    batch = exposure.rescale_intensity(batch, out_range=(0, 255))
    batch =  np.array(batch, dtype=np.uint8)
    original_shape = batch.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[config.model_type](checkpoint=config.sam_checkpoint)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

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
    
    sam.to(device=device)
    predictor = SamPredictor(sam)
    #flip rasterio to be h,w, channels
    predictor.set_image(batch)
    
    #reset index
    input_points = input_points.reset_index(drop=True)
    input_point = np.column_stack((input_points['x'], input_points['y']))
    input_crowns = input_points['StemTag']
    crown_mask = pd.DataFrame(columns=["geometry", "score","StemTag"])
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
        #turn transformed_boxes into numpy

        transformed_boxes_batch = torch.tensor(transformed_boxes_batch.astype('float32'), 
                                               device=predictor.device)
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
            input_boxes = input_boxes.geometry.to_numpy()

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
            input_point[:,0] = input_point[:,0] / resolution
            input_point[:,1] = input_point[:,1] / resolution
            grid_size = config.grid_size / resolution
            input_boxes = input_boxes / resolution


        #which_stem = np.where(input_points['StemTag']== "0123031")
        #predictor.set_image(np.array(batch))
        for it in range(input_point.shape[0]):            
            #update input_label to be 0 everywhere except at position it       
            target_itc = input_point[it].copy()
            # Calculate the Euclidean distance between the ith row and the other rows

            # if config.clip_each instance
            if config.clip_each_instance:
                # get xmin, ymin, xmax, ymax of the box around the target point
                xmin = max(0, target_itc[0] - grid_space)
                ymin = max(0, target_itc[1] - grid_space)
                xmax = min(batch.shape[0], target_itc[0] + grid_space)
                ymax = min(batch.shape[1], target_itc[1] + grid_space)

                # clip batch and update coordinates of subset+point
                batch_clipped = batch[int(xmin):int(xmax), int(ymin):int(ymax)]
                predictor.set_image(np.array(batch_clipped))

            if point_type == "distance":
                # using distance, get the indexes of input_points, ordered by distances
                distances = np.linalg.norm(input_point - input_point[it], axis=1)
                closest_indices = np.argsort(distances)[first_neigh:neighbors+first_neigh]
            elif point_type == "cardinal":
                closest_indices = np.argpartition(distances, 4)[:4]
            
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

                points = np.array(points)
            else:
                # define points as the index of all the points in the  input_point but input_point[it]
                closest_indices = np.delete(np.arange(input_point.shape[0]), it)

            # Subset the array to the ith row and the 10 closest rows

            if point_type != "grid":
                
                min_y = max(target_itc[1] - grid_size, 0)
                min_x = max(target_itc[0] - grid_size, 0)
                max_x = min(target_itc[0] + grid_size, batch.shape[0])
                max_y = min(target_itc[1] + grid_size, batch.shape[0])

                subset_point = input_point[closest_indices].copy()
                subset_label = np.zeros(subset_point.shape[0])
                subset_label = subset_label.astype(np.int8)
                #append subset_point with the target point
                subset_point = np.vstack(( target_itc, subset_point))
                subset_label = np.append(1, subset_label)
            else:
                subset_point = points.copy()[:neighbors,:]
                subset_label = np.zeros(subset_point.shape[0], dtype=np.int8)
                #append subset_point with the target point
                subset_point = np.vstack((subset_point, target_itc))
                subset_label = np.append(subset_label, 1)

                # set up a bounding box around the target point
                min_x = np.min(points[:,0])
                max_x = np.max(points[:,0])
                min_y = np.min(points[:,1])
                max_y = np.max(points[:,1])


            # Create an empty mask input and an indicator for no mask.
            #mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            if config.clip_each_instance:
                subset_point[:,0] = subset_point[:,0] - xmin
                subset_point[:,1] = subset_point[:,1] - ymin
                target_itc = target_itc - np.array([xmin, ymin])

            #subset_point = np.vstack((subset_point, [0.0, 0.0]))
            #subset_label = np.append(subset_label, -1)
            inBB = 0
            if input_boxes is not None:
                # extract the bounding box overlapping with the target point. 
                # which boxes have xmin < x < xmax and ymin < y < ymax
                which_box = np.where((input_boxes[:,0] < target_itc[0]) & (input_boxes[:,2] > target_itc[0]) & 
                                     (input_boxes[:,1] < target_itc[1]) & (input_boxes[:,3] > target_itc[1]))
                if len(which_box[0]) == 0:
                    target_bbox = np.array([[min_x, min_y, max_x, max_y]])
                else:
                    inBB = 1
                    # if length target_box is greater than 2, then pick the box whose center is slosest to the target itc
                    target_bbox = np.array(input_boxes[which_box[0], :4]).astype(np.float32)
                    if len(which_box[0]) > 1:
                        # get the center of the boxes
                        centers = np.array([(target_bbox[:,0] + target_bbox[:,2])/2, (target_bbox[:,1] + target_bbox[:,3])/2]).T
                        # get the distance between the centers and the target itc
                        dist = np.sqrt(np.sum((centers - target_itc)**2, axis=1))
                        # get the index of the box with the smallest distance
                        which_box = np.argmin(dist)
                        target_bbox = np.array([target_bbox[which_box, :]])

            else:
                target_bbox = np.array([[min_x, min_y, max_x, max_y]])
            
            # turn batch into an image
            
            #np.transpose(np.array(batch), (1, 0, 2))
            #predictor.set_image(np.array(batch))
            subset_point[:,1]  = batch.shape[0] -  subset_point[:,1] 
            target_bbox[0][[1,3]] = batch.shape[0] - target_bbox[0][[3,1]]
            masks, scores, logits  = predictor.predict(
                point_coords=subset_point,
                point_labels=subset_label,
                box=target_bbox,
                multimask_output=True,
            )
            #masks = np.transpose(masks,  (0, 2, 1))
            # Find the indices of the True values
            true_indices = np.argwhere(masks)
            # which of the true indices are the target point in dimensions 1 and 2
            mask_overlapping = true_indices[np.where((true_indices[:,2] == int(target_itc[0])) & (true_indices[:,1] == batch.shape[0] - int(target_itc[1])))]
            if mask_overlapping.shape[0] == 0:
                continue
            # from maks, keep only those that have mask_overlapping in the first dimension
            masks = masks[mask_overlapping[:,0]]
            scores = scores[mask_overlapping[:,0]]
            #pick the mask with the highest score
            if len(masks) > 1:
                #get only the mask whose value is closest to 1
                sc = abs(1 - scores)
                masks = masks[sc.argmax()]
                scores = scores[sc.argmax()]
            else:
                masks = masks[0]
                scores = scores[0]
            
            #skip empty masks
            if true_indices.shape[0] < 3:
                continue

            area = np.sum(masks)
            # from masks, subtract 650 from the y coordinates
            '''
            # save the mask as image
            mk= np.rot90(masks, 0, axes=(0, 1))
            mask_image = Image.fromarray(masks)
            mask_image.save(os.path.join('mask.png'))
            # save the batch as image
            batch_image = Image.fromarray(batch)
            batch_image.save(os.path.join('batch.png'))
            '''

            if area < 200**2:
                polygons = mask_to_delineation(mk = masks, 
                                               target_itc = target_itc,  
                                               buffer_size = 100)
            #calculate area of the polygon
            #pick the polygon that intercepts with individual point
            #individual_point = Point(target_itc)
            #polygons = [poly for poly in polygons if poly.intersects(individual_point)]

            # if if config.clip_each instance, shift the coordinates of the polygons to the original coordinates
            if config.clip_each_instance:
                # shift polygons coordinates to the original batch coordinates
                polygons = [translate(poly, xoff=xmin, yoff=ymin) for poly in polygons]
                # divide coordinates by 10 to get meters
            if rgb:
                polygons = [scale(poly, xfact= resolution, yfact= resolution, origin=(0,0,0)) for poly in polygons]

            if len(polygons) ==0:
                continue

            polygons = [translate(poly, xoff=affine[2], yoff=affine[5] + (batch.shape[0]*affine[4])) for poly in polygons]
            # Create a GeoDataFrame and append the polygon
            gdf_temp = gpd.GeoDataFrame(geometry=[polygons[0]], columns=["geometry"])
            gdf_temp["score"] = scores
            gdf_temp["inBB"] = inBB
            gdf_temp["StemTag"] = input_crowns.iloc[it]
            gdf_temp.to_file("temp.gpkg", driver="GPKG")
            # Append the temporary GeoDataFrame to the main GeoDataFrame
            crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)
            crown_scores.append(scores)
            crown_logits.append(logits)

    # Convert the DataFrame to a GeoDataFrame
    predictions = gpd.GeoDataFrame(crown_mask, geometry=crown_mask.geometry)
    predictions.to_file("temp.gpkg", driver="GPKG")
    # convert coordinates to utm suing affine
    #reshift crown mask coordiantes to original size
    # get bounding box of the geodataframe crown_mask
    
    if rescale_to is not None:
        predictions['geometry'] = predictions['geometry'].translate(xoff=0, yoff=0)
        predictions['geometry'] = predictions['geometry'].scale(xfact=original_shape[1]/rescale_to, yfact=original_shape[0]/rescale_to, origin=(0,0))

    return predictions, crown_scores, crown_logits



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

def split_image(image_file,itcs, bbox, batch_size=40, buffer_distance=10):
    with rasterio.open(image_file) as src:
        height, width = src.shape
        resolution = src.transform[0]
        batch_size_ = int(batch_size / resolution)
        
    buffer_distance_ = int(buffer_distance)/resolution

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
                right, bottom = src.xy(i_ + hgt, j_ + wdt)
                batch_bounds = box(left , bottom , right , top )

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
    return raster_batches, itcs_batches, itcs_boxes, affines


# merge bbox and tmp_bx, then subtract bbox from tmp_bx
def get_bbox_diff(bbox_clipped, tmp_bx):
    #subtract columnwise bbox_clipped from tmp_bx
    bbox_diff = tmp_bx - bbox_clipped[['xmin', 'ymin', 'xmax', 'ymax']].values


from multiprocessing import Pool

def create_polygon(category, category_mask, offset_x=0, offset_y=0):
    polygons = []
    # Create a binary mask for the current category
    binary_mask = (category_mask == category).astype(np.uint8)
    # Convert binary_mask to 8-bit single-channel image
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Find contours of the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Flip the y-coordinate of contour points
    for contour in contours:
        contour[:, 0, 1] = category_mask.shape[0] - contour[:, 0, 1]

    # use a concave hull instead
    #contours = [cv2.convexHull(contour) for contour in contours]
    # if the data is rgb, rescale to meters by dividing by 10 each index
    # skip if does not have enough dimensions
    if contours and contours[0].shape[0] >= 3:
        # Convert the contours to polygons
        for contour in contours:
            # Simplify the contour to a polygon
            if len(contour) < 4:
                continue
            poly = Polygon(shell=contour.squeeze())
            # shift polygons coordinates to the position before clipping labelled mask
            poly = translate(poly, xoff=offset_x, yoff=offset_y)
            polygons.append(poly)

    return polygons


def mask_to_delineation(mk, target_itc,  buffer_size = 100):
    # if mask is 3d flatten to 2d
    if len(mk.shape) == 3:
        mk=mk[0,:,:]

    # Convert mask to 8-bit single-channel image
    mask_uint8 = (mk * 255).astype(np.uint8)
    #np.savetxt('output.csv', mask_uint8, delimiter=',')

    #use the smallest region of the mask wich contains all 255s
    target_itc_bounds = np.array(np.where(mask_uint8 == 255))
    target_itc_bounds = np.array([target_itc_bounds[0,:].min(), target_itc_bounds[1,:].min(), 
                                    target_itc_bounds[0,:].max(), target_itc_bounds[1,:].max()])
    min_y = max(0, target_itc_bounds[0]-1)
    min_x = max(0, target_itc_bounds[1]-1)
    max_y = min(mask_uint8.shape[0], target_itc_bounds[2]+1)
    max_x = min(mask_uint8.shape[1], target_itc_bounds[3]+1)
    '''
    mask_uint8 = mask_uint8[min_y:max_y,min_x:max_x]
    #mask_uint8 = np.rot90(mask_uint8, 1, axes=(0, 1))
    np.savetxt('output.csv', mask_uint8, delimiter=',')

    # mask_unit8 has the coordinates starting in 0,0. cv2 wants the coordinates to start in the top left corner. 
    # So we need to shift the coordinates to the top left corner
    
  
    # Use connectedComponents function
    num_labels, labeled_mask = cv2.connectedComponents(mask_uint8, connectivity=8)
    # if 3d flatten to 2d
    if len(labeled_mask.shape) == 3:
        labeled_mask=labeled_mask[0,:,:]

    # Create a dictionary with as many values as unique values in the labeled mask
    unique_labels = np.unique(mask_uint8)
    label_to_category = {label: category for label, category in zip(unique_labels, range(unique_labels.shape[0]))}
    # pick labels with less than 3 pixels and remove them from label_to_category
    for label in unique_labels:
        if np.sum(mask_uint8 == label) < 3:
            del label_to_category[label]

    # Define a function to handle None values
    def get_category(label):
        return label_to_category.get(label, -9)

    # Apply the vectorized function to the submask with output data type as object (string)
    category_mask = np.vectorize(get_category, otypes=[object])(mask_uint8)
    #category_mask = np.roll(category_mask, -category_mask.shape[0], axis=0)
    np.savetxt('output.csv', category_mask, delimiter=',')
    '''
    category_mask = mask_uint8
    # Turn each category into a polygon
    polygons = []
    # Filter only numeric values
    for category in np.unique(category_mask):
        if category <= 0:  # Skip the background
            continue
        # Create a binary mask for the current category
        poly = create_polygon(category, category_mask)#, min_x, min_y)
        polygons.append(poly)
    # flatten the list of lists
    polygons = [item for sublist in polygons for item in sublist]
    if target_itc is not None:
        pp =   [poly for poly in polygons if poly.contains(Point(target_itc))]
        #check that center is in the polygon
        if len(polygons) > 1:
            if len(pp) ==1:
                polygons = pp
            else:
            #
            #   polygons = [poly for poly in polygons if poly.contains(Point(target_itc))]
            #above all polygons, pick the one whose centroid is closest to the target_itcs
                polygons = sorted(polygons, key=lambda poly: poly.centroid.distance(Point(target_itc)))
                polygons = [polygons[0]]
            

    return polygons
