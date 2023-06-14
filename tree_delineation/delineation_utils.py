import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
#from arepl_dump import dump 
from geopandas.tools import sjoin
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection
import numpy as np
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
import laspy, laszip
import json
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import os
import config
def split_overlapping_polygons(gdf):
    def add_geometry(geom, subpolygons):
        if isinstance(geom, GeometryCollection):
            for g in geom:
                add_geometry(g, subpolygons)
        else:
            subpolygons.append(geom)

    # Initialize an empty list to store subpolygons
    subpolygons = []

    # Iterate through pairs of polygons
    for i, row_i in gdf.iterrows():
        polygon_i = row_i['geometry']
        for j, row_j in gdf.loc[i + 1:].iterrows():
            polygon_j = row_j['geometry']

            # Check if polygons overlap
            if polygon_i.intersects(polygon_j):
                # Split overlapping polygons into non-overlapping subpolygons
                split_polygon_i = polygon_i.difference(polygon_j)
                split_polygon_j = polygon_j.difference(polygon_i)
                intersection = polygon_i.intersection(polygon_j)

                # Replace the original polygons with the subpolygons
                polygon_i = split_polygon_i
                gdf.at[j, 'geometry'] = split_polygon_j

                # Add the intersection to the list of subpolygons
                add_geometry(intersection, subpolygons)

        # Replace the current polygon with the updated (non-overlapping) one
        gdf.at[i, 'geometry'] = polygon_i

    # Convert the list of subpolygons into a new GeoDataFrame
    subpolygons_gdf = gpd.GeoDataFrame(geometry=subpolygons, crs=gdf.crs)

    # Concatenate the original GeoDataFrame with the subpolygons GeoDataFrame
    result_gdf = gpd.GeoDataFrame(pd.concat([gdf, subpolygons_gdf], ignore_index=True))

    return result_gdf



def calculate_distances(gdf1, gdf2, tag = 'StemTag'):
    result = []

    # Iterate through rows of the first GeoDataFrame
    for index1, row1 in gdf1.iterrows():
        stemTag1 = row1[tag]
        point1 = row1['geometry']

        # Iterate through rows of the second GeoDataFrame
        for index2, row2 in gdf2.iterrows():
            stemTag2 = row2[tag]
            point2 = row2['geometry']

            # Check if stemTags match
            if stemTag1 == stemTag2:
                # Calculate distance between the two points
                distance = point1.distance(point2)

                # Save the result
                result.append({
                    'StemTag': stemTag1,
                    'distance': distance,
                    'point1_index': index1,
                    'point2_index': index2,
                })

    # Convert the result list into a pandas DataFrame
    result_df = pd.DataFrame(result)
    return result_df


def split_multipolygons_to_polygons(gdf):
    # Make sure the input is a GeoDataFrame
    if not isinstance(gdf, gpd.geodataframe.GeoDataFrame):
        raise ValueError("Input should be a GeoPandas GeoDataFrame.")

    # Use the explode function to split MultiPolygons into individual Polygons
    gdf_exploded = gdf.explode()

    # Reset the index and drop the old index column
    gdf_exploded = gdf_exploded.reset_index(drop=True)

    return gdf_exploded



def geopandas_overlay_partition(partition, gdf2, how='intersection'):
    return gpd.overlay(partition, gdf2, how=how)

from shapely.errors import GEOSException

def compute_difference(idx, non_overlapping_gdf, threshold):
    geom = non_overlapping_gdf.loc[idx].geometry

    if geom.geom_type not in ['Polygon', 'MultiPolygon']:
        return None

    spatial_index = non_overlapping_gdf.sindex
    possible_matches_index = list(spatial_index.intersection(geom.bounds))
    for idx2 in possible_matches_index:
        if idx == idx2:
            continue

        geom2 = non_overlapping_gdf.loc[idx2].geometry
        if geom2.geom_type not in ['Polygon', 'MultiPolygon']:
            continue
        try:
            intersection = geom.intersection(geom2)
            if intersection.area / geom2.area > threshold and intersection.is_valid:
                #try:
                if geom.area > geom2.area:
                    geom = geom.difference(intersection)
                    if geom.geom_type == 'LineString':
                        geom = None
                else:
                    geom2 = geom2.difference(intersection)
                    if geom2.geom_type == 'LineString':
                        geom2 = None
        except GEOSException:
            return None

    return geom


import geopandas as gpd
from shapely.ops import cascaded_union
import dask_geopandas as dgpd
import dask_geopandas as dgpd
import dask.bag as db

def clean_oversegmentations_by_overlap(gdf, itcs = None, threshold=0.05, buff = 0):
    # Cut overlapping polygons
    non_overlapping_gdf = gdf.copy()
    non_overlapping_gdf['geometry'] = non_overlapping_gdf['geometry'].buffer(buff)
    non_overlapping_gdf = split_multipolygons_to_polygons(non_overlapping_gdf)
    non_overlapping_gdf['geometry'] = non_overlapping_gdf['geometry'].buffer(buff)
    # remove all columns except geometry and StemTag
    #non_overlapping_gdf = non_overlapping_gdf[['geometry']]
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type.isin(['Polygon', 'MultiPolygon'])].reset_index(drop=True)
    non_overlapping_gdf = dgpd.from_geopandas(non_overlapping_gdf, npartitions=8)
    non_overlapping_gdf = non_overlapping_gdf.map_partitions(geopandas_overlay_partition, gdf2=non_overlapping_gdf, how='intersection')
    non_overlapping_gdf = non_overlapping_gdf.compute()

    #dgpd.overlay(non_overlapping_gdf, non_overlapping_gdf, how='intersection', keep_geom_type=True)
    non_overlapping_gdf = split_multipolygons_to_polygons(non_overlapping_gdf)

    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type.isin(['Polygon', 'MultiPolygon'])].reset_index(drop=True)
    non_overlapping_gdf['p_to_a'] =  non_overlapping_gdf.length /non_overlapping_gdf.area
    # Remove all polygons from non_overlapping_gdf that are less than 4m2
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf.p_to_a < 2].reset_index(drop=True)
    #non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf.area < 600].reset_index(drop=True)
    # non_overlapping_gdf.to_file(f'{data_path}/Crowns/bboxes{i}.gpkg', driver='GPKG')
    #remove all polygons that don't intercept with any itcs point
    if itcs is not None:
        non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf.intersects(itcs.unary_union)].reset_index(drop=True)
    # remove index before looping
    non_overlapping_gdf = non_overlapping_gdf.apply(
    lambda row: remove_linestring_from_geometrycollection(row) if isinstance(row.geometry, GeometryCollection) else row,
    axis=1)
    non_overlapping_gdf = non_overlapping_gdf.reset_index(drop=True)
    # Create a spatial index (R-tree) for non_overlapping_gdf
    spatial_index = non_overlapping_gdf.sindex
    
    # For polygons that are overlapping, remove the intersecting area from the bigger one
    modified_geoms = []

    for idx, row in non_overlapping_gdf.iterrows():
        #print(idx)
        geom = row.geometry.buffer(buff)
        
        #if geom tye is not polygon or multipolygon, skip
        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            continue
        
        # Get the indices of the polygons that intersect the bounding box of geom
        # Get the indices of the polygons that have a bounding box that intersects the bounding box of geom
        possible_matches_index = list(spatial_index.intersection(geom.bounds))
        for idx2 in possible_matches_index:
            if idx == idx2:
                continue

            geom2 = non_overlapping_gdf.loc[idx2].geometry.buffer(buff)
            if geom2.geom_type not in ['Polygon', 'MultiPolygon']:
                continue
            
            intersection = geom.intersection(geom2)
            #if geometry is a geometrycollection
            if intersection.area / geom2.area > threshold and intersection.is_valid:
                if geom.area > geom2.area:
                    geom = geom.difference(intersection)
                else:
                    geom2 = geom2.difference(intersection)
        modified_geoms.append(geom)


    

    # if 2 polygons overlapmore than 50%, remove the smaller one
    #new_gdf = remove_smaller_overlapping_polygons(new_gdf, overlap_threshold=0.5)
    # remove empty geometries
    new_gdf = new_gdf[~new_gdf.is_empty].reset_index(drop=True)

    #if you keep it, make it a function
    # Create a spatial index (R-tree) for non_overlapping_gdf

    
    dgdf = dgpd.from_geopandas(new_gdf, npartitions=8)
    indices_bag = db.from_sequence(new_gdf.index)

    modified_geoms_bag = indices_bag.map(compute_difference, non_overlapping_gdf=new_gdf, threshold=threshold)
    modified_geoms = modified_geoms_bag.compute()

    modified_geoms = [geom for geom in modified_geoms if geom is not None]
    new_gdf = gpd.GeoDataFrame(geometry=modified_geoms, crs=non_overlapping_gdf.crs)


    #new_gdf = gpd.GeoDataFrame(geometry=modified_geoms, crs=non_overlapping_gdf.crs)
    new_gdf = split_multipolygons_to_polygons(new_gdf)
    #remove polygons with area less than 4m2
    new_gdf['p_to_a'] =  new_gdf.length /new_gdf.area
    new_gdf = new_gdf[new_gdf.p_to_a < 1].reset_index(drop=True)
    new_gdf = new_gdf[new_gdf.area > 4].reset_index(drop=True)

    # if 2 polygons overlapmore than 50%, remove the smaller one
    #new_gdf = remove_smaller_overlapping_polygons(new_gdf, overlap_threshold=0.5)
    # remove empty geometries
    new_gdf = new_gdf[~new_gdf.is_empty].reset_index(drop=True)

    #new_gdf.to_file(f'{folder}/itcs_combined.gpkg', driver='GPKG')
    return new_gdf


from shapely.geometry import GeometryCollection, LineString

from shapely.geometry import GeometryCollection, LineString

def remove_linestring_from_geometrycollection(geometry_collection):
    if not isinstance(geometry_collection, GeometryCollection):
        raise ValueError("Input should be a shapely GeometryCollection.")

    # Filter out LINESTRING geometries
    filtered_geometries = [geom for geom in geometry_collection.geoms if not isinstance(geom, LineString)]

    # Create a new GeometryCollection without LINESTRING geometries
    new_geometry_collection = GeometryCollection(filtered_geometries)
    # Filter out LINESTRING geometries
    filtered_geometries = [geom for geom in new_geometry_collection.geoms if not isinstance(geom, Point)]

    # Create a new GeometryCollection without LINESTRING geometries
    new_geometry_collection = GeometryCollection(filtered_geometries)
    return new_geometry_collection


#remove_perc_of_smallest_polygons
def remove_perc_of_smallest_polygons(gdf, perc=0.1):
    # sort by area
    gdf['area'] = gdf['geometry'].area
    gdf = gdf.sort_values(by='area')
    # remove perc percent of smallest polygons
    gdf = gdf.iloc[int(perc * len(gdf)):]
    return gdf


# remove_polygons_without_itcs
def remove_polygons_without_itcs(gdf, itcs):
    #remove all polygons that don't intercept with any itcs point
    gdf = gdf[gdf.intersects(itcs.unary_union)].reset_index(drop=True)




def remove_smaller_overlapping_polygons(gdf, overlap_threshold=0.5):
    # Create a spatial index (R-tree) for the GeoDataFrame
    spatial_index = gdf.sindex

    # Set to store indices of polygons to remove
    to_remove = set()

    for idx, row in gdf.iterrows():
        if idx in to_remove:
            continue

        geom = row.geometry
        # Get the indices of the polygons that have a bounding box that intersects the bounding box of geom
        possible_matches_index = list(spatial_index.intersection(geom.bounds))

        for idx2 in possible_matches_index:
            if idx == idx2 or idx2 in to_remove:
                continue

            geom2 = gdf.loc[idx2].geometry
            intersection = geom.intersection(geom2)
            overlap_area = intersection.area
            if geom.area > 0 and geom2.area > 0 and intersection.area > 0:
                if overlap_area / geom.area > overlap_threshold or overlap_area / geom2.area > overlap_threshold:
                    if geom.area < geom2.area:
                        to_remove.add(idx2)
                    else:
                        to_remove.add(idx)
                        break

    # Remove smaller overlapping polygons
    gdf_cleaned = gdf.drop(index=list(to_remove)).reset_index(drop=True)

    return gdf_cleaned


def create_bounding_box(row, resolution_ratio=10):
    from shapely.geometry import Polygon
    
    xmin, xmax, ymin, ymax = row['xmin']/resolution_ratio, row['xmax']/resolution_ratio, row['ymin']/resolution_ratio, row['ymax']/resolution_ratio
    return Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])



def extract_features(gdf, hsi, rgb, lidar_path):
    features = []

    # Read LiDAR data using laspy
    lidar_data = laspy.read(lidar_path)
    lidar_data = laspy.read(lidar_path, laz_backend=laszip.LaszipBackend())

    for index, row in gdf.iterrows():
        geom = row.geometry
        geom_json = json.loads(geom.to_json())

        # Extract hyperspectral and RGB data for the point
        hsi_data, _ = mask(hsi, [geom_json], crop=True)
        rgb_data, _ = mask(rgb, [geom_json], crop=True)
        hsi_mean = np.mean(hsi_data)
        rgb_mean = np.mean(rgb_data)

        # Extract LiDAR data for the point using laspy
        bounds = geom.bounds
        x_mask = np.logical_and(lidar_data.x >= bounds[0], lidar_data.x <= bounds[2])
        y_mask = np.logical_and(lidar_data.y >= bounds[1], lidar_data.y <= bounds[3])
        mask = np.logical_and(x_mask, y_mask)
        lidar_points = lidar_data[mask]

        if len(lidar_points) > 0:
            lidar_mean_z = np.mean(lidar_points.z)
        else:
            lidar_mean_z = np.nan

        # Append the extracted features to the features list
        features.append([hsi_mean, rgb_mean, lidar_mean_z])

    return np.array(features)


def align_data(forestGEO_gdf, field_gdf, forestGEO_features, field_features, threshold=0.1):
    aligned_data = []

    for idx_f, row_f in forestGEO_gdf.iterrows():
        tag_f = row_f['Tag']
        feature_f = forestGEO_features[idx_f]

        min_distance = float('inf')
        min_idx = None

        for idx_fd, row_fd in field_gdf.iterrows():
            tag_fd = row_fd['Tag']

            if tag_f != tag_fd:
                continue

            feature_fd = field_features[idx_fd]
            distance = euclidean_distances([feature_f], [feature_fd])[0][0]

            if distance < min_distance:
                min_distance = distance
                min_idx = idx_fd

        if min_distance < threshold:
            aligned_data.append((row_f, field_gdf.loc[min_idx]))

    return aligned_data

def create_aligned_gdf(aligned_data):
    aligned_gdf = gpd.GeoDataFrame(columns=forestGEO_gdf.columns)

    for row_f, row_fd in aligned_data:
        new_row = row_f.copy()
        new_row['geometry'] = row_fd['geometry']
        aligned_gdf = aligned_gdf.append(new_row, ignore_index=True)

    return aligned_gdf


#aligned_gdf = create_aligned_gdf(aligned_data)
#create a function to plot the polygons of a geopandas data
def plot_polygons(gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax)
    #extract x and y min max from the geometry column
    xmin, xmax, ymin, ymax = gdf.geometry.total_bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # Save the figure as a PNG file
    fig.savefig('itcs.png', dpi=300, bbox_inches='tight')


# create a function to extract deepforest boxes from rgb images
def extract_boxes(rgb_path):

    from deepforest import main
    model = main.deepforest()
    model.use_release()
    boxes = model.predict_tile(raster_path=rgb_path, return_plot=False)
    return boxes


# assign StemTag to each box overlapping with itc
def assign_itc_to_boxes(bbox, itcs, fields):
    # Perform a spatial join between the two GeoDataFrames
    joined_gdf = gpd.sjoin(bbox, itcs, how="inner", predicate='contains')
    joined_gdf = joined_gdf[['StemTag', 'geometry']]
    fields_dat = pd.read_csv(fields)
    fields_dat = fields_dat[['StemTag', 'SiteID', 'Species', 'Status',
                        'DBH', 'Crown position', 'Percentage of crown intact',
        'Percentage of crown living', 'Lean angle if greater than 15 degrees',
        'FAD', 'Wounded main stem', 'Canker; swelling, deformity', 'Rotting trunk']]
    
    # merge the joined_gdf with fields_dat
    joined_gdf = joined_gdf.merge(fields_dat, on='StemTag', how='left')
    updated_itcs = itcs[['StemTag', 'geometry']]
    updated_itcs = updated_itcs.merge(fields_dat, on='StemTag', how='left')
    updated_itcs = gpd.GeoDataFrame(updated_itcs, geometry='geometry')
    updated_itcs.to_file(f'{config.data_path}/Stems/{config.siteID}_itcs.gpkg', driver='GPKG')
    # Group by the bounding box identifier and find the tree with the highest 'CrwnPst' value in each box
    # If there are multiple trees with the same 'CrwnPst' value, pick the one with the highest 'dbh' value
    result = joined_gdf.sort_values(['Crown position', 'DBH'], ascending=[False, False]).groupby('StemTag').first()
    result = gpd.GeoDataFrame(result, geometry='geometry')
    result.to_file(f'{config.data_path}/Crowns/{config.siteID}/SAM_pp_{0}.gpkg', driver='GPKG')
    return result
