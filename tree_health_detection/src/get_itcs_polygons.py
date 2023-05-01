
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
    # if 3d flatten to 2d
    if len(labeled_mask.shape) == 3:
        labeled_mask=labeled_mask[0,:,:]
    #labeled_mask = labeled_mask.astype(np.uint16)
    # Create a dictionary with as many values as unique values in the labeled mask
    label_to_category = {label: category for label, category in zip(np.unique(labeled_mask), range(0, np.unique(labeled_mask).shape[0]))}

    #to speed up, clip the labelled mask to the bounding box around non-zero values
    #get the bounding box
    #non_zero_indices = np.nonzero(labeled_mask)
    #min_x = max(np.min(non_zero_indices[0])-1,0)
    #max_x = min(np.max(non_zero_indices[0])+1, labeled_mask.shape[0])
    #min_y = max(np.min(non_zero_indices[1])-1,0)
    #max_y = min(np.max(non_zero_indices[1])+1, labeled_mask.shape[0])
    #clip the labelled mask
    #labeled_mask = labeled_mask[min_x:max_x, min_y:max_y]
    #get the category mask

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
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #skip if does not have enough dimensions
        if contours[0].shape[0] < 3:
            continue
        # Convert the contours to polygons
        for contour in contours:
            # Simplify the contour to a polygon
            poly = Polygon(shell=contour.squeeze())
            polygons.append(poly)

        # shift polygons coordinates to the position before clipping labelled mask
        #polygons = [translate(poly, xoff=min_x, yoff=min_y) for poly in polygons]

    return polygons


# sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth"
# Define a function to make predictions of tree crown polygons using SAM
def predict_tree_crowns(batch, input_points, neighbors = 5, 
                        input_boxes = None, point_type='grid', 
                        onnx_model_path = None,  rescale_to = None, mode = 'bbox',
                        sam_checkpoint = "../tree_mask_delineation/SAM/checkpoints/sam_vit_h_4b8939.pth",
                        model_type = "vit_h", meters_between_points = 50, threshold_distance =50):

    from tree_health_detection.src.get_itcs_polygons import mask_to_delineation
    from skimage import exposure
    batch = np.moveaxis(batch, 0, -1)
    #age = Image.fromarray(batch)
    #age.save('output.png')
    original_shape = batch.shape

    #change input boxes values into integers
    if input_boxes is not None:
        input_boxes = input_boxes.astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    if onnx_model_path == None:
        onnx_model_path = "tree_health_detection/tmp_data/sam_onnx_example.onnx"

        onnx_model = SamOnnxModel(sam, return_single_mask=True)

        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }

        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float).to('cpu'),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float).to('cpu'),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float).to('cpu'),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float).to('cpu'),
            "has_mask_input": torch.tensor([1], dtype=torch.float).to('cpu'),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float).to('cpu'),
        }
        output_names = ["masks", "iou_predictions", "low_res_masks"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(onnx_model_path, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )   
        


    #neighbors must be the minimum between the total number of input_points and the argument neighbors
    neighbors = min(input_points.shape[0]-2, neighbors)
    #rescale image to larger size if rescale_to is not null
    if rescale_to is not None:
        batch = resize(batch, (rescale_to, rescale_to), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
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
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

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
        # this part may be a GPU bottleneck. Better divide the boxes into batches of 1000 max
        # divede the boxes into batches of 1000 max if they are more than 1000
        transformed_boxes_batch = input_boxes
        transformed_boxes_batch = torch.tensor(transformed_boxes_batch, device=predictor.device)
        transformed_boxes_batch = predictor.transform.apply_boxes_torch(transformed_boxes_batch, batch.shape[:2])
        masks,scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes_batch.int(),
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
                gdf_temp["point_id"] = str(msks)+'_'+str(it)

                # Append the temporary GeoDataFrame to the main GeoDataFrame
                crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)


    if input_boxes is None or mode == 'only_points':# and onnx_model_path is None:
        #loop through each stem point, make a prediction, and save the prediction
        for it in range(0, input_point.shape[0]):
            #update input_label to be 0 everywhere except at position it       
            input_label = np.zeros(input_point.shape[0])
            input_label[it] = 1
            target_itc = input_point[it].copy()
            # subset the input_points to be the current point and the 10 closest points
            # Calculate the Euclidean distance between the ith row and the other rows
            distances = np.linalg.norm(input_point - input_point[it], axis=1)
            if point_type == "distance":
            # Find the indices of the 10 closest rows
                closest_indices = np.argpartition(distances, neighbors+1)[:neighbors+1]  # We use 11 because the row itself is included
            # find the 4 closest points one in each cardinal direction
            elif point_type == "cardinal":
                closest_indices = np.argpartition(distances, 4)[:4]
            #pick a random sample of points, making sure that none is the target point
            
            elif point_type == "random":
                #pick a random sample of points, making sure that none is the target point
                closest_indices = np.random.choice(np.delete(np.arange(input_point.shape[0]), it), neighbors, replace=False)

            elif point_type == "grid":
                #create a grid of points around the target point, that is within the bounds of the image
                points = []
                #loop through predefined distance between points
                for i in range(-neighbors, neighbors+1, meters_between_points):
                    for j in range(-neighbors, neighbors+1, meters_between_points):
                        points.append([target_itc[0]+i, target_itc[1]+j])
                points = np.array(points)
                #remove points that are outside the image
                points = points[(points[:,0] >= 0) & (points[:,0] < batch.shape[0]) & (points[:,1] >= 0) & (points[:,1] < batch.shape[1])]
                #remove the target point
                points = points[~np.all(points == target_itc, axis=1)]
                #remove points that are too close to the target point
                points = points[np.linalg.norm(points - target_itc, axis=1) > threshold_distance]
                #if there are not enough points, add random points

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
                subset_label = np.ones(points.shape[0], dtype=np.int8)
                #append subset_point with the target point
                subset_point = np.vstack((subset_point, target_itc))
                subset_label = np.append(subset_label, 1)

            #FROM HERE: IF USING ONNX 
            # #Add a batch index, concatenate a padding point, and transform.
            onnx_coord = np.concatenate([subset_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([subset_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
            onnx_coord = predictor.transform.apply_coords(onnx_coord, batch.shape[:2]).astype(np.float32)
            # Create an empty mask input and an indicator for no mask.
            onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)
            #Package the inputs to run in the onnx model
            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": onnx_coord.astype(np.float32),
                "point_labels": onnx_label,
                "mask_input": onnx_mask_input,
                "has_mask_input": onnx_has_mask_input,
                "orig_im_size": np.array(batch.shape[:2], dtype=np.float32)
            }
            masks, scores, logits = ort_session.run(None, ort_inputs)
            masks = masks > predictor.model.mask_threshold

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
            # Find the indices of the True values
            true_indices = np.argwhere(masks[0,0,:,])
            #skip empty masks
            if true_indices.shape[0] < 3:
                continue

            # Calculate the convex hull
            individual_point = Point(input_point[it])
            polygons = mask_to_delineation(masks[0,:,:,:])
            #pick the polygon that intercepts with individual point
            polygons = [poly for poly in polygons if poly.intersects(individual_point)]
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
    dbox= []
    # Loop through the rows and columns of the image
    for i in range(0, height, batch_size_):
        for j in range(0, width, batch_size_):
            # Define a window for the current batch
            window = Window(col_off=j, row_off=i, width=batch_size_, height=batch_size_)

            # Read the batch from the raster image
            with rasterio.open(image_file) as src:
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
                hsi_height, hsi_width = hsi.shape
                window_hsi = Window(col_off=j/resolution_factor, row_off=i/resolution_factor, width=batch_size_hsi, height=batch_size_hsi)
                hsi_batch = hsi.read(window=window_hsi)

            # upscale hsi_batch to the same size as batch
            #hsi_batch = upscale_array(hsi_batch, batch, resolution_hsi, resolution)

            hsi_batches.append(hsi_batch)

            # Clip the GeoDataFrame using the batch bounds
            itcs_clipped = gpd.clip(itcs, batch_bounds)

            # Transform the coordinates relative to the raster batch's origin
            itcs_clipped["geometry"] = itcs_clipped["geometry"].apply(
                transform_coordinates, x_offset=left, y_offset=bottom
            )
            bbox_clipped = bbox
            #from bboxes, clip only those whose xmin, ymin, xmax, ymax fit within the batch bounds
            bbox_clipped = gpd.clip(bbox_clipped, batch_bounds)


            #remove boxes that are LINESTRING or POINT
            bbox_clipped = bbox_clipped[bbox_clipped.geometry.type == 'Polygon']

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
            resolution_factor = resolution/ resolution_hsi

            tmp_bx = []
            left, bottom, right, top =  batch_bounds.bounds
            for rows in range(bbox_clipped.shape[0]):
                #from the polygon geometry, get xmin, xmax, ymin, ymax relative to image_rgb left, bottom, right, top

                bounds = bbox_clipped.geometry.bounds.iloc[rows]
                bbox_clipped['xmin'] = bounds['minx']-left 
                bbox_clipped['ymin'] = bounds['miny']-bottom
                bbox_clipped['xmax'] = bounds['maxx']-left
                bbox_clipped['ymax'] = bounds['maxy']-bottom
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

