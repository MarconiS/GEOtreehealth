import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
#from arepl_dump import dump 
from geopandas.tools import sjoin



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


#read gdf from fil
gdf = gpd.read_file('indir/SERC/itcs.gpkg')
non_overlapping_gdf.read_file('tmp/gradient_boosting_alignment.gpkg')

# generate a function that cleans the oversegmentations and overlappings 
def clean_oversegmentations(gdf, tag = 'StemTag'):
    # Cut overlapping polygons
    non_overlapping_gdf = gpd.overlay(gdf, gdf, how='union', keep_geom_type=False)

    # when stemTag_1 is empty,  replace it with stemTag_2
    non_overlapping_gdf['StemTag_1'] = non_overlapping_gdf['StemTag_1'].fillna(non_overlapping_gdf['StemTag_2'])
    non_overlapping_gdf['StemTag_2'] = non_overlapping_gdf['StemTag_2'].fillna(non_overlapping_gdf['StemTag_1'])

    #remove nans from non overlapping gdf
    non_overlapping_gdf = non_overlapping_gdf.dropna(subset=['StemTag_1', 'StemTag_2'])

    #turn StemTag_1 into StemTag, and drop StemTag_2
    non_overlapping_gdf['StemTag'] = non_overlapping_gdf['StemTag_1']
    non_overlapping_gdf = non_overlapping_gdf.drop(columns=['StemTag_1', 'StemTag_2'])

    #keep only polygons and reset index
    #non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type == 'Polygon'].reset_index(drop=True)
    #write to file for debugging
    #non_overlapping_gdf.to_file('indir/SERC/tmp.gpkg', driver='GPKG')
    # load field data


    #non_overlapping_gdf = get_copy.copy()
    # remove rows if theyr geometry is one the following types: 'MultiLineString' 'Point' 'LineString' 'MultiPoint' from non_overlapping_gdf
    non_overlapping_gdf = non_overlapping_gdf[non_overlapping_gdf['geometry'].type.isin(['Polygon', 'MultiPolygon', 'GeometryCollection'])].reset_index(drop=True)

    points_gdf = gpd.read_file('indir/SERC/gradient_boosting_alignment.gpkg')

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
    gdf_cleaned = gdf.drop(to_remove)
    # Write the cleaned GeoDataFrame to a file
    gdf_cleaned.to_file('indir/SERC/itcs_cleaned.gpkg', driver='GPKG')