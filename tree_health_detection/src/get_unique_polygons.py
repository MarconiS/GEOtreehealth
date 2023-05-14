import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
#from arepl_dump import dump 
from geopandas.tools import sjoin
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection

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


#read gdf from fil
gdf = gpd.read_file('../tree_mask_delineation/outdir/combined_gdf.gpkg')
points_gdf = gpd.read_file('tree_health_detection/indir/SERC/gradient_boosting_alignment.gpkg')
# gdf = gpd.read_file('tree_health_detection/indir/SERC/itcs_cleaned_oversegmentations.gpkg')
# generate a function that cleans the oversegmentations and overlappings 
def clean_oversegmentations(gdf, points_gdf, tag = 'StemTag'):
    # Cut overlapping polygons
    non_overlapping_gdf = gpd.overlay(gdf, gdf, how='union', keep_geom_type=False)
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type.isin(['Polygon', 'MultiPolygon'])].reset_index(drop=True)
    non_overlapping_gdf = split_multipolygons_to_polygons(non_overlapping_gdf)
    # when stemTag_1 is empty,  replace it with stemTag_2
    non_overlapping_gdf['StemTag_1'] = non_overlapping_gdf['StemTag_1'].fillna(non_overlapping_gdf['StemTag_2'])
    non_overlapping_gdf['StemTag_2'] = non_overlapping_gdf['StemTag_2'].fillna(non_overlapping_gdf['StemTag_1'])

    #remove nans from non overlapping gdf
    non_overlapping_gdf = non_overlapping_gdf.dropna(subset=['StemTag_1', 'StemTag_2'])

    #turn StemTag_1 into StemTag, and drop StemTag_2
    non_overlapping_gdf['StemTag'] = non_overlapping_gdf['StemTag_1']
    non_overlapping_gdf = non_overlapping_gdf.drop(columns=['StemTag_1', 'StemTag_2'])

    # remove rows if theyr geometry is one the following types: 'MultiLineString' 'Point' 'LineString' 'MultiPoint' from non_overlapping_gdf
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type.isin(['Polygon', 'MultiPolygon', 'GeometryCollection'])].reset_index(drop=True)


    # Calculate centroids
    centroids_polygons = non_overlapping_gdf.geometry.centroid
    #assign index as column to be able to link back to non_overlapping_gdf later
    centroids_polygons.index = non_overlapping_gdf.index

    #turn centroids_polygons into a gdf
    centroids_polygons = gpd.GeoDataFrame(centroids_polygons, columns=['geometry'], crs=non_overlapping_gdf.crs)
    # assign stemtag to centroids_polygons
    centroids_polygons['StemTag'] = non_overlapping_gdf['StemTag']
    centroids_polygons['polyID'] = non_overlapping_gdf.index
    non_overlapping_gdf['polyID'] = non_overlapping_gdf.index

    # Set the index of each GeoDataFrame to the 'stemTag' column
    centroids_polygons = centroids_polygons.set_index('StemTag')
    points_gdf = points_gdf.set_index('StemTag')
    points_gdf.crs = non_overlapping_gdf.crs

    # select only StemTag, geometry, DBH, Crwnpst, FAD, Status, Species 
    points_gdf = points_gdf[['geometry', 'DBH', 'Crwnpst', 'FAD', 'Status', 'Species']]
    # Perform the spatial join
    # Merge the two GeoDataFrames based on the 'stemTag' column
    merged_gdf = centroids_polygons.merge(points_gdf, on='StemTag', suffixes=('_gdf1', '_gdf2'))
    # Define a function to calculate the distance between two points
    def calculate_distance(row):
        point1 = row['geometry_gdf1']
        point2 = row['geometry_gdf2']
        return point1.distance(point2)

    # Calculate the distances between the matching points
    merged_gdf['distance'] = merged_gdf.apply(calculate_distance, axis=1)
    # Perform the spatial join
    # Select the 'stemTag' and 'distance' columns for the final result
    result = merged_gdf.copy()[['polyID','DBH','Crwnpst','FAD','Status','Species','distance']]
    result['StemTag']=result.index

    #remove index to make it easier to join
    result = result.reset_index(drop=True)
    #left_join result to non_overlapping_gdf based on StemTag
    result = result.merge(non_overlapping_gdf, how='left', on='polyID')
    #drop rows with nan in StemTag
    result = result.dropna(subset=['StemTag_y'])

    # Initialize an empty list to store indices of polygons to remove
    to_remove = []

    # Iterate through pairs of polygons
    for i, row_i in result.iterrows():
        for j, row_j in result.loc[i + 1:].iterrows():
            # Check if polygons overlap
            if row_i['geometry'].intersects(row_j['geometry']):
                # Calculate the overlapping area
                overlapping_area = row_i['geometry'].intersection(row_j['geometry']).area
                
                # Check if the overlapping area is larger than 10 m²
                if overlapping_area > 10:
                    # Compare the 'distance' values and mark the polygon with the larger value for removal
                    if row_i['distance'] <= row_j['distance']:
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
                        break  # No need to check the current polygon (row_i) against the remaining polygons

    # Remove duplicates from the 'to_remove' list
    to_remove = list(set(to_remove))

    # Remove the marked polygons from the GeoDataFrame
    gdf_cleaned = gdf.copy().drop(to_remove)

    #reiterate to reintroduce StemTags that have been removed from the gdf

    # Write the cleaned GeoDataFrame to a file
    gdf_cleaned.to_file('tree_health_detection/indir/SERC/itcs_combined.gpkg', driver='GPKG')



def clean_oversegmentations_by_overlap(gdf, itcs = None, threshold=0.05):
    # Cut overlapping polygons
    non_overlapping_gdf = gpd.overlay(gdf, gdf, how='union', keep_geom_type=False)
    non_overlapping_gdf = split_multipolygons_to_polygons(non_overlapping_gdf)

    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type.isin(['Polygon', 'MultiPolygon'])].reset_index(drop=True)
    # Remove all polygons from non_overlapping_gdf that are less than 4m2
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf.area > 4].reset_index(drop=True)
    # remove perc percent of smallest polygons
    #non_overlapping_gdf = remove_perc_of_smallest_polygons(non_overlapping_gdf, perc=0.2)

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
        geom = row.geometry.buffer(0)
        
        #if geom tye is not polygon or multipolygon, skip
        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            continue
        
        # Get the indices of the polygons that intersect the bounding box of geom
        # Get the indices of the polygons that have a bounding box that intersects the bounding box of geom
        possible_matches_index = list(spatial_index.intersection(geom.bounds))
        
        for idx2 in possible_matches_index:
            if idx == idx2:
                continue
            geom2 = non_overlapping_gdf.loc[idx2].geometry.buffer(0)
            if geom2.geom_type not in ['Polygon', 'MultiPolygon']:
                continue
            intersection = geom.intersection(geom2)
            #if geometry is a geometrycollection
            if intersection.geom_type in ['GeometryCollection']:
                intersection = remove_linestring_from_geometrycollection(intersection)
            if intersection.area / geom2.area > threshold and geom.is_valid and geom2.is_valid:
                if geom.area > geom2.area:
                    geom = geom.difference(intersection)
                    if geom.geom_type in ['GeometryCollection']:
                        geom = remove_linestring_from_geometrycollection(geom)
                else:
                    geom2 = geom2.difference(intersection)
                    geom2.geom_type
                    if geom2.geom_type in ['GeometryCollection']:
                        geom2 = remove_linestring_from_geometrycollection(geom2)
        modified_geoms.append(geom)

    new_gdf = gpd.GeoDataFrame(geometry=modified_geoms, crs=non_overlapping_gdf.crs)
    
    #remove polygons with area less than 4m2
    new_gdf = new_gdf[new_gdf.area > 4].reset_index(drop=True)

    # if 2 polygons overlapmore than 50%, remove the smaller one
    new_gdf = remove_smaller_overlapping_polygons(new_gdf, overlap_threshold=0.5)
    # remove empty geometries
    new_gdf = new_gdf[~new_gdf.is_empty].reset_index(drop=True)

    #new_gdf.to_file(f'{folder}/tmp/itcs_combined.gpkg', driver='GPKG')
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

            if overlap_area / geom.area > overlap_threshold or overlap_area / geom2.area > overlap_threshold:
                if geom.area > geom2.area:
                    to_remove.add(idx2)
                else:
                    to_remove.add(idx)
                    break

    # Remove smaller overlapping polygons
    gdf_cleaned = gdf.drop(index=list(to_remove)).reset_index(drop=True)

    return gdf_cleaned
