import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np
from transformers import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamModel,
)


import leafmap
from samgeo import tms_to_geotiff, split_raster
from samgeo.text_sam import LangSAM
import leafmap
from samgeo import tms_to_geotiff
import config
import rasterio
import glob
import os
import numpy as np
from skimage import exposure
import torch
from shapely.geometry import Polygon
import geopandas as gpd
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import pandas as pd
import geopandas as gpd
#from tree_health_detection.src.get_itcs_polygons import mask_to_delineation
from skimage import exposure

import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch
from samgeo import SamGeo, tms_to_geotiff


# TODO this goes to utils.py >>>

import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()



def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()
    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))
    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()

def show_single_mask_on_image(raw_image, mm, sc, points_=None, labels=None):
    if len(mm.shape) == 4:
        mm = mm.squeeze()
    if len(sc.shape) > 0:
        sc = [sc.squeeze()]
    else:
        sc = [sc.to.item()]
    nb_predictions = 1
    fig, axes = plt.subplots(1, 1)
    mask = mm.cpu().detach().numpy()
    axes.imshow(np.array(raw_image))
    # Assuming show_mask is a function you've defined to overlay the mask
    show_mask(mask, axes)
    if points_ is not None:
        points_ = np.array(points_)
        if labels is None:
            labels = np.ones_like(points_[:, 0])
        show_points(points_, labels, plt.gca())
    axes.set_title(f"Mask {0+1}, Score: {sc[0].cpu().item():.3f}")
    axes.axis("off")
    plt.show()


def calculate_overlap(poly1, poly2):
    return poly1.intersection(poly2).area / poly1.union(poly2).area

# <<<

import geopandas as gpd
import pandas as pd

def select_median_polygon(gdf):
    # Group by 'StemTag'
    grouped = gdf.groupby('StemTag')

    # For each group, find the polygon with the CA closest to the median CA of the group
    median_polygons = []
    for name, group in grouped:
        median_ca = group['CA'].median()
        closest_polygon = group.iloc[(group['CA'] - median_ca).abs().argsort()[:1]]
        median_polygons.append(closest_polygon)

        # Select only one closest polygon
        if len(closest_polygon) > 1:
            closest_polygon = closest_polygon.iloc[0]
        else:
            closest_polygon = closest_polygon

    # Concatenate all the selected polygons into a new geodataframe
    median_gdf = pd.concat(median_polygons)
    
    return median_gdf

# Use the function with your geodataframe
# selected_gdf = select_median_polygon(your_geodataframe)
# print(selected_gdf)


def remove_files_from_folder(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    # Iterate through and remove each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        # Check if it's a file (and not a folder)
        if os.path.isfile(file_path):
            os.remove(file_path)

def find_cardinal_direction(p1, p2):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    if x2 > x1:
        if y2 > y1:
            return 'SE'
        else:
            return 'NE'
    else:
        if y2 > y1:
            return 'SW'
        else:
            return 'NW'
        
import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union
import rtree
import shapely.speedups

# Enable shapely speedups if available for performance
if shapely.speedups.available:
    shapely.speedups.enable()

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import rtree

def remove_shared_region(polygons, lower_bound=0.1, upper_bound=0.5):
    index = rtree.index.Index()
    for pos, poly in enumerate(polygons):
        index.insert(pos, poly.bounds)

    modified_polygons = []

    for i, poly1 in enumerate(polygons):
        potential_overlaps = list(index.intersection(poly1.bounds))
        intersections_to_remove = []

        for j in potential_overlaps:
            poly2 = polygons[j]
            if poly1 != poly2 and poly1.intersects(poly2):
                intersection = poly1.intersection(poly2)
                perc1 = intersection.area / poly1.area
                perc2 = intersection.area / poly2.area
                if lower_bound <= perc1 <= upper_bound:
                    intersections_to_remove.append(intersection)

        if intersections_to_remove:
            # Calculate the combined intersection area to remove
            combined_intersections = unary_union(intersections_to_remove)
            new_poly = poly1.difference(combined_intersections)
            # Check if the operation results in a valid polygon
            if isinstance(new_poly, (Polygon, MultiPolygon)):
                if isinstance(new_poly, Polygon):
                    modified_polygons.append(new_poly)
                elif isinstance(new_poly, MultiPolygon):
                    # Keep the largest piece if it's a MultiPolygon
                    largest_piece = max(new_poly, key=lambda p: p.area)
                    modified_polygons.append(largest_piece)
        else:
            modified_polygons.append(poly1)

    return modified_polygons


from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import rtree


def remove_overlap_from_larger_polygon(polygons, lower_bound=0.1, upper_bound=0.5):
    # Create an R-tree index for the polygons
    index = rtree.index.Index()
    for idx, polygon in enumerate(polygons):
        index.insert(idx, polygon.bounds)

    # This list will store the resulting polygons
    modified_polygons = [None] * len(polygons)

    # Iterate over polygons to find and subtract overlaps
    for i, poly1 in enumerate(polygons):
        # Find candidate polygons for potential overlap with poly1
        candidates = list(index.intersection(poly1.bounds))
        for j in candidates:
            # Avoid comparing the same polygon with itself
            if i == j:
                continue

            poly2 = polygons[j]
            if poly1.intersects(poly2):
                # Calculate the intersection area
                intersection = poly1.intersection(poly2)
                intersection_area = intersection.area

                # Calculate overlap percentages relative to each polygon's area
                overlap_with_poly1 = intersection_area / poly1.area
                overlap_with_poly2 = intersection_area / poly2.area

                # Determine if the overlap is within the threshold and subtract from the larger polygon
                if lower_bound <= overlap_with_poly1 <= upper_bound:
                    # Check if we've already modified poly1 or if we should use the original
                    if modified_polygons[i]:
                        poly_to_subtract_from = modified_polygons[i]
                    else:
                        poly_to_subtract_from = poly1

                    # Subtract the intersection from the larger polygon and update the modified_polygons list
                    modified_polygons[i] = poly_to_subtract_from.difference(intersection)

                elif lower_bound <= overlap_with_poly2 <= upper_bound:
                    # Check if we've already modified poly2 or if we should use the original
                    if modified_polygons[j]:
                        poly_to_subtract_from = modified_polygons[j]
                    else:
                        poly_to_subtract_from = poly2

                    # Subtract the intersection from the larger polygon and update the modified_polygons list
                    modified_polygons[j] = poly_to_subtract_from.difference(intersection)

    # Replace None with original polygons if they weren't modified
    for idx, poly in enumerate(modified_polygons):
        if poly is None:
            modified_polygons[idx] = polygons[idx]

    # Clean up the resulting geometries to ensure they are all valid
    cleaned_polygons = [poly if poly.is_valid else poly.buffer(0) for poly in modified_polygons]

    return cleaned_polygons



def stretch_image(image_array):
    # Calculate the 2nd and 98th percentiles
    p2 = np.percentile(image_array, 2)
    p98 = np.percentile(image_array, 98)

    # Stretch the image to span the full range
    image_stretched = (image_array - p2) / (p98 - p2) * 255

    # Clip the values to be in the range [0, 255]
    image_stretched = np.clip(image_stretched, 0, 255)

    # Convert to an appropriate data type for images
    image_stretched = image_stretched.astype(np.uint8)

    return image_stretched

def batchsam(img_pth, itcs=None, tmptile = None, tmpdir = None, ttops = None, input_boxes = None, debug = False,flag_out_the_boundaries = True,max_suppression = True):
    from shapely.geometry import Polygon
    if tmpdir is None:
        tmpdir = "tmp/tree_crowns/"
    if tmptile is None:
        tmptile = "tmp/tiles"
    # remove content in the folders  tmp/tiles
    # if tmptile doesn't exist, create it
    if not os.path.exists(tmptile):
        os.makedirs(tmptile)
    # if tmpdir doesn't exist, create it
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    remove_files_from_folder(tmptile)
    remove_files_from_folder(tmpdir)
    split_raster(img_pth, out_dir=tmptile, tile_size=(1024, 1024), overlap = config.overlap)
    img = glob.glob(tmptile+"/*.tif")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # for each tile, extract points overlapping and run teh model tile = 'tmp/tiles/SERC/tile_1_4.tif'
    for tile in img:
        # tile = img[11]
        # load the tile using rasterio
        with rasterio.open(tile) as src:
            # get the tile bounds
            bounds = src.bounds
            crs = src.crs
            res = src.res
            height = src.height
            transform = src.transform
            rs = src.read()
        #
        rs = np.transpose(rs, (2, 1, 0))
        # flip image clockwise
        rs = np.rot90(rs, k=3)
        rs = np.rot90(rs, k=3)
        rs = np.rot90(rs, k=3)
        # flip image vertically
        rs = np.flip(rs, axis=0)
        # rescale image to 2 - 98 percentile
        rs = stretch_image(rs)
        rs = exposure.rescale_intensity(rs, out_range=(0, 255))
        rs =  np.array(rs, dtype=np.uint8)
        image = Image.fromarray(rs, 'RGB')
        #show_points_on_image(image,  points_, labels)
        if ttops is not None:
            boxes_coords = tree_detection(tile, ttops = ttops)
        # define tree points in the tile
        if itcs is not None:
            # from the field_tree_points geopandas, extract the points that are within the tile bounds
            # first create a geodatframe from tiles bounds
            # Convert bounding box to polygon
            tile_bds = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom)
            ])
            # Create a GeoDataFrame
            gdf_bounds = gpd.GeoDataFrame({'geometry': [tile_bds]})
            tile_points = itcs[itcs.geometry.within(gdf_bounds.unary_union)]
            # if there are spoints with the same stemTag, keep the one
            tile_points = tile_points.drop_duplicates(subset=['StemTag'])
            tile_points = tile_points[['StemTag', 'geometry']].copy()
            # reset index of tile_points
            tile_points = tile_points.reset_index(drop=True)
        else:
            # calculat the centroid of the boxes_coords
            tile_points = gpd.GeoDataFrame(geometry=[box.centroid for box in boxes_coords.geometry])
            tile_points['StemTag'] = [str(i) for i in range(len(boxes_coords))]
            tile_points = tile_points.reset_index(drop=True)
            tile_points = tile_points[['StemTag', 'geometry']].copy()
            # tile_points.to_file("tmp/tile_points.gpkg", driver="GPKG")
        #
        inputs = processor(image, return_tensors="pt").to(device)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
        # create an empty dataframe where to store the polygons
        gdf = gpd.GeoDataFrame(columns=['geometry', 'StemTag'], dtype=object)
        if tile_points.shape[0] < 2:
            continue
        for treeid in range(tile_points.shape[0]):
            #treeid =  1 for tile_points[tile_points['StemTag'] == '331510']
            all_indices = np.delete(np.arange(tile_points.shape[0]), treeid)
            if len(all_indices)>0:
                distances = tile_points.iloc[all_indices].geometry.distance(tile_points.iloc[treeid].geometry)
                if config.remove_too_close >0:
                    # remove points that are too close to the treeid
                    distances = distances[distances > config.remove_too_close]
                #
                if len(distances) ==1:
                    sampled_indices = [all_indices[0]]
                else:
                    sampled_indices = []
                    cardinal_buckets = {'NW': [], 'NE': [], 'SW': [], 'SE': []}
                    for idx, distance in zip(all_indices, distances):
                        direction = find_cardinal_direction(tile_points.iloc[treeid].geometry, tile_points.iloc[idx].geometry)
                        cardinal_buckets[direction].append((idx, distance))
                        #
                    for direction in cardinal_buckets.keys():
                        cardinal_buckets[direction].sort(key=lambda x: x[1])
                        if cardinal_buckets[direction]:
                            sampled_indices.append(cardinal_buckets[direction][0][0])
                            #
                    # If there are not enough neighbors in all cardinal directions, fill up with closest neighbors
                    if len(sampled_indices) < config.neighbors:
                        remaining_indices = [idx for bucket in cardinal_buckets.values() for idx, _ in bucket[1:]]
                        remaining_indices = sorted(remaining_indices, key=lambda idx: tile_points.iloc[idx].geometry.distance(tile_points.iloc[treeid].geometry))
                        sampled_indices += remaining_indices[:config.neighbors - len(sampled_indices)]
                # make an array where 1 is followed by r0 as many times as neighbors
                labels = np.zeros(len(sampled_indices)+1)
                labels[0] = 1
            else:
                sampled_indices = [treeid]
                labels = np.ones(1)
            # If you want to include the treeid at the beginning of your result array:
            tree_index = np.insert(sampled_indices, 0, treeid)
            points_ = tile_points.iloc[tree_index,1].copy()
            #from points_ POINT subtract bounds.left and bounds.bottom and multiply by resolution to get the coordinates in the image
            points_ = points_.apply(lambda point: ((point.x - bounds.left)/transform.a, (point.y - bounds.top)/transform.e))
            # convert to int
            points_ = points_.apply(pd.Series).astype(int)
            if debug == True:
                show_points_on_image(image,  points_, labels)
            points_3d = points_.values[np.newaxis, :]
            # Convert the 3D NumPy array to a list of lists of lists and make sure to convert integers to floats
            points_list_3d = [[[(coord) for coord in point] for point in points_group] for points_group in points_3d.tolist()]
            # convert labels in a list of lists of lists
            labels_ = labels[np.newaxis, :].astype(int)
            # make sure the points are in nb_images, nb_predictions, nb_points_per_mask, 2
            if input_boxes is not None:
                if points_list_3d is not None:
                    inputs = processor(image, input_boxes=[input_boxes], input_points=points_list_3d, input_labels=[labels_.tolist()], return_tensors="pt").to(device)
                else:
                    inputs = processor(image, input_boxes=[input_boxes], return_tensors="pt").to(device)
            else:
                inputs = processor(image, input_points=points_list_3d, input_labels=[labels_.tolist()], return_tensors="pt").to(device)
            # pop the pixel_values as they are not neded
            inputs.pop("pixel_values", None)
            inputs.update({"image_embeddings": image_embeddings})
            print(treeid)
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            # check sum of trues in masks
            scores = outputs.iou_scores
            if debug == True:
                show_single_mask_on_image(raw_image = image, mm = masks[0], sc =  scores, points_=points_, labels=labels)
            from skimage import measure
            from shapely.geometry import Polygon, MultiPolygon
            from shapely.affinity import translate
            from shapely.affinity import translate, scale
            # Loop through an array of masks 
            polygon_list = []
            for mask in masks:
                # Detect all individual regions in the (boolean) mask
                #check how many true pixels in the mask
                mask = mask.squeeze()
                all_labels = measure.label(mask) 
                candidate_pool = []
                for plg in range(1, all_labels.max()+1):
                    if plg ==0:
                        continue
                    # Create a mask for the current region
                    region_mask = all_labels == plg # true in mask 
                    #print(region_mask.sum())
                    # to speed up the process a bit, remove regions that are too small or too large
                    if region_mask.sum() < config.sam_min_area or region_mask.sum() > config.sam_max_area:
                        continue
                    # plot region_mask to check
                    if debug == True:
                        plt.imshow(region_mask)
                        plt.show()
                    # Detect contours in the mask
                    contours = measure.find_contours(region_mask, 0.5) 
                    for contour in contours:
                        if contour.shape[0] < 4:
                            continue  # skip the background
                        # Reverse the (row, column) order to (x, y) and create a polygon for this contour
                        polygon = Polygon(contour[:, ::-1]) 
                        polygon_list.append(polygon)  # save all polygons for the current mask
                    # get the polygon that overlaps with thee coordiantes of treeid
                    tree_polygon = tile_points.iloc[treeid].geometry
                    polygon_list = [p.buffer(0) for p in polygon_list]
                    # get coordinates of the top left corner of the tile and use it to translate the polygons
                    x_offset, y_offset = transform[2], transform[5]
                    #y_offset = y_offset - original_shape[0]*transform[0]
                    # Loop through each polygon in the original list
                    polygon_list_rp = []
                    for p in polygon_list:
                        # Scale the polygon coordinates by the pixel resolution (0.1)
                        scaled_polygon = scale(p, xfact=transform[0], yfact=transform[4], origin=(0, 0))
                        # Translate the scaled polygon
                        translated_polygon = translate(scaled_polygon, xoff=x_offset, yoff=y_offset)
                        # Append the translated polygon to the new list
                        polygon_list_rp.append(translated_polygon)
                    # remove empty polygons
                    polygon_list_rp = [p for p in polygon_list_rp if p.is_empty == False]
                    # select the best candidate polygon by extracting the one that overlaps with the treeid coordinates
                    candidate_polygons = [p for p in polygon_list_rp if p.intersects(tree_polygon)]
                    #if candidate is empty, select the polygon whose centroid is the closest to the treeid coordinates
                    if len(candidate_polygons) == 0:
                        candidate_polygons = [p for p in polygon_list_rp if p.distance(tree_polygon) == min([p.distance(tree_polygon) for p in polygon_list_rp])]
                    # add candidate polygon to canidate pool
                    candidate_pool.append(candidate_polygons[0])
                # among candidates, get the one overlapping with the treeid coordinates 
                candidate_polygons = [p for p in candidate_pool if p.intersects(tree_polygon)]
                if len(candidate_polygons) == 0:
                        candidate_polygons = [p for p in candidate_pool if p.distance(tree_polygon) == min([p.distance(tree_polygon) for p in candidate_pool])]
                # finally, append to geodataframe
                if len(candidate_polygons) > 0:
                    # to candidate polygons, append the tile_points.StemTag at index treeid
                    gdf.loc[treeid] = [candidate_polygons[0], tile_points.iloc[treeid].StemTag]
        gdf.set_geometry('geometry', inplace=True)
        gdf.crs = crs
        # left join gdf with itcs using stemTag if StemTag exists in itcs
        if (itcs is not None) & ('StemTag' in itcs.columns):
            itcs_df = pd.DataFrame(itcs).drop(columns=['geometry'])
            gdf = gdf.merge(itcs_df, on='StemTag', how='left')
        if flag_out_the_boundaries == True:
            # from gdf geodataframe, remove the rows that touch tile_bds
            gdf = gdf[~gdf.geometry.apply(lambda x: x.bounds).apply(lambda x: x[0] < bounds[0]+ 0.5
                                                                    or x[1] < bounds[1]+ 0.5
                                                                     or x[2] > bounds[2]- 0.5
                                                                     or x[3] > bounds[3]- 0.5)]
        # apply max suppression to remove overlapping polygons if requested
        if max_suppression == True:
            # Finally, before saving the gdf, apply non-max suppression
            original_polygons = gdf['geometry'].tolist()
            #original_polygons = remove_overlap_from_larger_polygon(original_polygons)
            stem_position = pd.to_numeric(gdf[config.stem_ranking_column], errors='coerce')
            stem_dbh = pd.to_numeric(gdf['DBH'], errors='coerce')
            # Now these lists will contain numeric values only
            stem_position = stem_position.tolist()
            stem_dbh = stem_dbh.tolist()
            filtered_polygons = weighted_non_max_suppression(original_polygons, 
                    [(stem_position, lambda x: x), (stem_dbh, lambda x: x)], overlap_threshold=  config.nms_sam )
            #filtered_polygons = non_max_suppression(original_polygons, stem_priority, overlap_threshold=  0.5)
            # Create a new GeoDataFrame with filtered polygons
            gdf_filtered = gdf[gdf['geometry'].isin(filtered_polygons)]
            # Save filtered GeoDataFrame to file
            gdf_filtered.to_file(tmpdir + tile.split("/")[-1].replace(".tif", ".gpkg"), driver="GPKG")
        #save gdf to file using the tile name
        # gdf.to_file("tile_sample.gpkg", driver="GPKG")
        if debug == True:
                # turn the list of polygons into a geopandas dataframe
                gdf = gpd.GeoDataFrame(geometry=candidate_polygons)
                gdf.crs = crs
                # save to file
                gdf.to_file("tmp_polygons.gpkg", driver="GPKG")
                #save tree_polygon to file
                tree_polygon = gpd.GeoDataFrame(geometry=[tree_polygon])
                tree_polygon.crs = crs
                tree_polygon.to_file("tmp_tree_polygon.gpkg", driver="GPKG")
                # save points_ to file
                points_gdf= gpd.GeoDataFrame(geometry=tile_points.geometry.copy())
                points_gdf.crs = crs
                points_gdf.to_file("outdir/tree_crowns/"+tile.split("/")[-1].replace(".tif", "stems.gpkg"), driver="GPKG")


def non_max_suppression(polygons, conditions, overlap_threshold=0.8):
    # Combine the polygons with their conditions
    combined_data = [tuple([poly] + [condition[i] for condition, _ in conditions]) for i, poly in enumerate(polygons)]

    # Sort the combined data based on the provided sorting functions in the conditions
    combined_data.sort(key=lambda x: tuple(sort_key(x[i + 1]) for i, (_, sort_key) in enumerate(conditions)))

    keep = []

    for i, data in enumerate(combined_data):
        poly = data[0]
        keep_flag = True
        for kept_data in keep:
            kept_poly = kept_data[0]
            if calculate_overlap(poly, kept_poly) > overlap_threshold:
                keep_flag = False
                break
        if keep_flag:
            keep.append(data)

    # Extracting the polygons to return
    return [data[0] for data in keep]



def sam_tile_composite(indir, outdir, outname, bounds_check = False):
    #indir = "outdir/tree_crowns/"
    # get all files in the directory
    files = os.listdir(indir)
    # remove files that are not geopackages
    files = [f for f in files if f.endswith(".gpkg")]
    # sort files by name
    files = sorted(files)
    # get the first file
    file = files[0]
    # read it
    gdf_tmp = gpd.read_file(os.path.join(indir, file))
    # get the boundaries of gdf_tmp
    bounds = gdf_tmp.total_bounds
    if bounds_check == True:
        # remove polygons that overlap with the edge of the plot (e.g. config.overlap)
        gdf = gdf_tmp[~gdf_tmp.geometry.apply(lambda x: x.bounds).apply(lambda x: x[0] < bounds[0]+ 0.5
                                                                        or x[1] < bounds[1]+ 0.5
                                                                        or x[2] > bounds[2]- 0.5
                                                                        or x[3] > bounds[3]- 0.5)]
    else:
        gdf = gdf_tmp.copy()
    # loop through the other files and append the polygons to the first file
    for file in files[1:]:
        gdf_tmp=gpd.read_file(os.path.join(indir, file))
        bounds = gdf_tmp.total_bounds
        if bounds_check == True:
            gdf_tmp = gdf_tmp[~gdf_tmp.geometry.apply(lambda x: x.bounds).apply(lambda x: x[0] < bounds[0]+ 0.5
                                                                        or x[1] < bounds[1]+ 0.5
                                                                        or x[2] > bounds[2]- 0.5
                                                                        or x[3] > bounds[3]- 0.5)]
        gdf = pd.concat([gdf, gdf_tmp], ignore_index=True)
    # save to file
    gdf.to_file(os.path.join(outdir, outname), driver="GPKG")


def tree_detection(rgb_path, ttops = "deepforest"):
    import tree_delineation
    from shapely.affinity import translate
        #get affine of the raster used to extract the bounding boxes
    with rasterio.open(rgb_path) as src: 
        rgb_transform = src.transform
        rgb_crs = src.crs
        rgb_width = src.width
        rgb_height = src.height
    #get tree bounding boxes with deepForest for SAM
    if ttops == 'deepforest':
        bbox = tree_delineation.delineation_utils.extract_boxes(rgb_path)
        # assume that y origin is at the top of the image: shift coordinates of y accordingly
        bbox['ymin'] = bbox['ymin'].apply(lambda y: rgb_height - y)
        bbox['ymax'] = bbox['ymax'].apply(lambda y: rgb_height - y)
        #use bbox xmin,ymin,xmax,ymax to make corners of bounding box polygons
        bbox['geometry'] = bbox.apply(lambda row: tree_delineation.delineation_utils.create_bounding_box(row), axis=1)
        x_offset, y_offset = rgb_transform[2], rgb_transform[5]
        y_offset = y_offset - rgb_height*rgb_transform[0]
        #from a geopandas, remove all rows with a None geometry
        bbox = bbox[bbox['geometry'].notna()]
        bbox = gpd.GeoDataFrame(bbox, geometry='geometry')
        bbox["geometry"] = bbox["geometry"].apply(lambda geom: translate(geom, x_offset, y_offset))
        bbox.crs = rgb_crs
        return bbox


def calculate_polygon_score(overlap_area, priority_values, priority_funcs):
    # Apply the priority functions to the corresponding values and sum them to get the priority score
    priority_score = sum(func(val) for val, func in zip(priority_values, priority_funcs))
    # The score favors polygons with a high priority score and low overlap.
    # Subtract the normalized overlap area to penalize for overlap
    return priority_score - overlap_area

def weighted_non_max_suppression(polygons, priority_conditions, overlap_threshold=0.5):
    # Calculate initial overlap areas for all polygon pairs
    overlap_areas = {poly: {other_poly: 0 for other_poly in polygons if other_poly != poly} for poly in polygons}

    for poly in polygons:
        for other_poly in polygons:
            if other_poly != poly:
                overlap = calculate_overlap(poly, other_poly)
                # Store the overlap area normalized by the area of the polygon being considered
                overlap_areas[poly][other_poly] = overlap if overlap > overlap_threshold else 0

    # Get priority functions from the priority_conditions
    priority_funcs = [cond[1] for cond in priority_conditions]

    # Calculate the score for each polygon
    polygon_scores = {}
    for i, poly in enumerate(polygons):
        # Extract priority values for the current polygon
        priority_values = [cond[0][i] for cond in priority_conditions]
        # Calculate the total overlap area for the current polygon
        total_overlap_area = sum(overlap_areas[poly].values())
        # Calculate the score
        polygon_scores[poly] = calculate_polygon_score(total_overlap_area, priority_values, priority_funcs)

    # Sort polygons by their score in descending order
    sorted_polygons = sorted(polygon_scores, key=polygon_scores.get, reverse=True)

    keep = []
    for poly in sorted_polygons:
        if all(calculate_overlap(poly, kept_poly) <= overlap_threshold for kept_poly in keep):
            keep.append(poly)

    return keep


import importlib
import config
importlib.reload(config)

img_pth = "/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Imagery/" + config.siteID +  "/RGB_ForestGeo.tif"
itcs = gpd.read_file(os.path.join(config.data_path+config.stem_path))
# if there are duplicates in the stemTag column, keep the one with the highest DBH
itcs = itcs.sort_values(by=['DBH'], ascending=False).drop_duplicates(subset=['StemTag'])
#itcs = itcs[itcs['Year'].notna()]
itcs = itcs.reset_index(drop=True)

# if column Crwnpst doesn't exists, rename crwnPstn to it
if 'Crwnpst' not in itcs.columns:
    itcs = itcs.rename(columns={'crwnPstn': 'Crwnpst'})
# check if crwnPst or Crwnpst is in the columns. 

# check how many have no crown position
itcs['Crwnpst'].isnull().sum()
if 'crwnPstn' in itcs.columns:
    # if not all crownPsnt are Nan, continue
    if not itcs['crwnPstn'].isnull().all():
        itcs = itcs[itcs['crwnPstn'] > 1]
elif 'Crwnpst' in itcs.columns:
    if not itcs['Crwnpst'].isnull().all():
        itcs = itcs[itcs['Crwnpst'] > 1]

if 'DBH' in itcs.columns:
    itcs = itcs[itcs['DBH'] > 10]


# pick only the last status
itcs = itcs.sort_values(by=['Year'], ascending=False).drop_duplicates(subset=['StemTag'])

# reset index
itcs = itcs.reset_index(drop=True)

batchsam(img_pth=img_pth, tmpdir = 'tmp/tree_crowns/'+config.siteID+"/", 
         tmptile =  'tmp/tiles/'+config.siteID+"/", itcs= itcs, 
         ttops = None, debug=False, flag_out_the_boundaries = True, max_suppression = True)


sam_tile_composite(indir = 'tmp/tree_crowns/'+config.siteID+"/", 
                   outdir=  "~/Documents/", outname=config.siteID+"_rgb_crowns.gpkg")



gdf = gpd.read_file("~/Documents/"+config.siteID+"_pan_crowns.gpkg")
# count the number of polygons per stemTag, sort and print the top 30
#ct = gdf.groupby('StemTag').size().sort_values(ascending=False)
# calculate area 
gdf['CA']= gdf['geometry'].area


# for stemTags with multiple polygons, get the one with median size
gdftmp = select_median_polygon(gdf)
'''
original_polygons = gdftmp['geometry'].tolist()
stem_position = pd.to_numeric(gdf[config.stem_ranking_column], errors='coerce')
stem_dbh = pd.to_numeric(gdf['DBH'], errors='coerce')
# Now these lists will contain numeric values only
stem_position = stem_position.tolist()
stem_dbh = stem_dbh.tolist()
stem_priority = [(stem_position, lambda x: x), (stem_dbh, lambda x: x)]
gdf_ = weighted_non_max_suppression(original_polygons, stem_priority, overlap_threshold=0.8)

'''

# keep only the polygons in gdf that are in gdf
gdf_fin = gdf[gdf['geometry'].isin(gdftmp)]
#remove polygons that are too big
gdftmp.to_file("/home/smarconi/Documents/"+config.siteID+"_pan_crowns_nodup_nms.gpkg", driver="GPKG")
