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

