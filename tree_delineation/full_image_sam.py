
import torch
from transformers import SamModel, SamProcessor
import rasterio 
import config
import geopandas as gpd
import os
import pandas as pd
from shapely.geometry import box, Point, MultiPoint, Polygon
from shapely.affinity import translate
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")


def transform_coordinates(geometry, x_offset, y_offset):
    if geometry.type == "Point":
        return Point(geometry.x - x_offset, geometry.y - y_offset)
    elif geometry.type == "MultiPoint":
        return MultiPoint([Point(p.x - x_offset, p.y - y_offset) for p in geometry])
    elif geometry.type == "Polygon":
        return Polygon([(p[0] - x_offset, p[1] - y_offset) for p in geometry.exterior.coords])
    else:
        raise ValueError("Unsupported geometry type")

#get affine of the raster used to extract the bounding boxes
with rasterio.open(config.data_path+config.seg_rgb_path) as src: 
        rgb_transform = src.transform
        rgb_crs = src.crs
        rgb_width = src.width
        rgb_height = src.height
        raw_image = src.read()

itcs = gpd.read_file(os.path.join(config.data_path+config.stem_path))
batch_bounds = box(src.transform[2], src.transform[5] + rgb_height * src.transform[4], src.transform[2]+ rgb_width * src.transform[0], src.transform[5] )
itcs = gpd.clip(itcs, batch_bounds)

# reset index of itcs
itcs = itcs.reset_index(drop=True)
input_points = pd.DataFrame(
    {
        "StemTag": itcs["StemTag"],
        "x": itcs["geometry"].x - src.transform[2],
        "y": itcs["geometry"].y - (src.transform[5] + rgb_height * src.transform[4]),
    }
)


inputs = processor(raw_image, return_tensors="pt").to(device)
image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

coords = input_points[['x','y']]*10
#nb_images, nb_predictions, nb_points_per_mask, 2


# Convert DataFrame to list of lists
points = coords.values.tolist()
points = [[point] for point in points]
ipoints = [points]

neighbors=3
for it in range(input_points.shape[0]):
    # create a vector of zeros of size input_points.shape[0]
    points = coords
    distances = np.linalg.norm(points - points.loc[it], axis=1)
    closest_indices = np.argsort(distances)[1:neighbors+1]
    points = points.iloc[closest_indices]
    input_labels = np.zeros(points.shape[0])
    points = np.vstack((points, coords.loc[it])).astype(int)
    input_labels = np.append(input_labels, 1)

    input_labels = [[lab] for lab in input_labels]
    points = [[point] for point in points]
    ipoints = [points]
    # inputs = processor(raw_image, input_boxes=input_boxes, input_points=[input_points], input_labels=[labels], return_tensors="pt").to(device)
    inputs = processor(raw_image, input_points=ipoints, input_labels=[input_labels], return_tensors="pt").to(device)
    # pop the pixel_values as they are not neded
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores


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


import cv2
def mask_to_delineation(msk):
    #if tensor, turn to a numpy on the cpu
    if isinstance(msk, torch.Tensor):
        msk = msk.cpu().numpy()

    mask_uint8 = (msk * 255).astype(np.uint8)

    # Use connectedComponents function
    num_labels, labeled_mask = cv2.connectedComponents(mask_uint8, connectivity=8)
    # if 3d flatten to 2d
    if len(labeled_mask.shape) == 3:
        labeled_mask=labeled_mask[0,:,:]

    # Compute the smallest region containing all non-zero values
    non_zero_indices = np.nonzero(labeled_mask)
    min_x, min_y = np.min(non_zero_indices, axis=1)
    max_x, max_y = np.max(non_zero_indices, axis=1)
    # Define the center of the smallest region
    if(max_x == min_x):
        max_x = max_x + 1

    if(max_y == min_y):
        max_y = max_y + 1
        
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

    return polygons


import torch
from transformers import SamModel, SamProcessor
import rasterio 
import config
import geopandas as gpd
import os
import pandas as pd
from shapely.geometry import box, Point, MultiPoint, Polygon
from shapely.affinity import translate
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Load the image
with rasterio.open(config.data_path+config.seg_rgb_path) as src: 
    rgb_transform = src.transform
    rgb_crs = src.crs
    rgb_width = src.width
    rgb_height = src.height
    raw_image = src.read()

# Define the dimensions of the subimages
sub_width = rgb_width // 3
sub_height = rgb_height // 3

# Define the bounds of the sub-images
bounds = [
    ((0, 0), (sub_height, sub_width)),  # upper left
    ((0, sub_width), (sub_height, 2*sub_width)),  # upper middle
    ((0, 2*sub_width), (sub_height, rgb_width)),  # upper right
    ((sub_height, 0), (2*sub_height, sub_width)),  # middle left
    ((sub_height, sub_width), (2*sub_height, 2*sub_width)),  # center
    ((sub_height, 2*sub_width), (2*sub_height, rgb_width)),  # middle right
    ((2*sub_height, 0), (rgb_height, sub_width)),  # lower left
    ((2*sub_height, sub_width), (rgb_height, 2*sub_width)),  # lower middle
    ((2*sub_height, 2*sub_width), (rgb_height, rgb_width))  # lower right
]

itcs = gpd.read_file(os.path.join(config.data_path+config.stem_path))
batch_bounds = box(src.transform[2], src.transform[5] + rgb_height * src.transform[4], src.transform[2]+ rgb_width * src.transform[0], src.transform[5] )
itcs = gpd.clip(itcs, batch_bounds)

# reset index of itcs
itcs = itcs.reset_index(drop=True)
input_points = pd.DataFrame(
    {
        "StemTag": itcs["StemTag"],
        "x": itcs["geometry"].x - src.transform[2],
        "y": itcs["geometry"].y - (src.transform[5] + rgb_height * src.transform[4]),
    }
)


# For each set of bounds...
for i, ((y_start, x_start), (y_end, x_end)) in enumerate(bounds):
    # Extract the subimage
    subimage = raw_image[:, y_start:y_end, x_start:x_end]
    
    # Select the points that fall within these bounds
    sub_points = input_points[(input_points['x']*10 >= x_start) & (input_points['x']*10 < x_end) & 
                              (input_points['y']*10 >= y_start) & (input_points['y']*10 < y_end)].copy()

    #rset the index of sub_points
    sub_points = sub_points.reset_index(drop=True)

    # Translate the points so they are correct relative to the sub-image
    sub_points['x'] = sub_points['x']*10 - x_start
    sub_points['y'] =sub_points['y']*10 - y_start

    # Process the subimage and sub_points here...
    inputs = processor(subimage, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    coords = sub_points[['x','y']]

    # Convert DataFrame to list of lists
    points = coords.values.tolist()
    points = [[point] for point in points]
    ipoints = [points]
    inputs = processor(subimage, input_points=ipoints, return_tensors="pt").to(device)
    inputs["input_points"].shape
    # pop the pixel_values as they are not needed
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores
    crown_mask = pd.DataFrame(columns=["geometry", "score","StemTag"])
    for it in range(len(masks[0])):

        msk = masks[0][it,:,:,:]
        scr = scores[0][it,:]
        # of the three options, get only msk corresponding to the highest scr
        msk = msk[scr.argmax(),:,:]
        scr = scr.max()
        # get the polygon
        pp = mask_to_delineation(msk)

        # divide all cooridnates of polygons by 10 to get meters
        x, y = pp[0].exterior.coords.xy
        # Divide each vertex coordinate by 10
        x = [coord / 10 for coord in x]
        y = [coord / 10 for coord in y]

        # Create a new polygon with the scaled coordinates
        tmp = Polygon(zip(x, y))

        # Create a GeoDataFrame and append the polygon
        gdf_temp = gpd.GeoDataFrame(geometry=[tmp], columns=["geometry"])
        gdf_temp["score"] = scr.cpu().numpy()
        gdf_temp["StemTag"] = sub_points.StemTag[it]

        # Append the temporary GeoDataFrame to the main GeoDataFrame
        crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)


    # Convert the DataFrame to a GeoDataFrame
    crown_mask_ = gpd.GeoDataFrame(crown_mask, geometry=crown_mask.geometry)
    crown_mask_.to_file(f'{config.data_path}/Crowns/{config.siteID}/SAM_{i}.gpkg', driver='GPKG')
            



