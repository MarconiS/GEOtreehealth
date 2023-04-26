import geopandas as gpd
import numpy as np
from rasterio.mask import mask
import laspy, laszip
import json
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import os
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

# create a function that loops through all gpkg in a folder, turn them into a geodataframe, and append them together in a single geodataframe

import os
import geopandas as gpd
import pandas as pd

def gpkg_to_gdf(folder_path):
    #create an empty list to store the GeoDataFrames
    gdfs = []
    
    #loop through all the gpkg in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".gpkg"):
            #read the gpkg as a geodataframe
            temp_gdf = gpd.read_file(os.path.join(folder_path, file))
            #append the geodataframe to the list
            gdfs.append(temp_gdf)
    
    #concatenate all the geodataframes in the list
    gdf = pd.concat(gdfs, ignore_index=True)
    
    return gdf

