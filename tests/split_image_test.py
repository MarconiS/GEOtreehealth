import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
import pytest
from shapely.geometry import box

def test_split_image():
    # path to your image file
    image_file = 'path_to_file'

    # path to your HSI image
    hsi_img = 'path_to_file'
    
    # Creating GeoDataFrame for itcs and bbox
    itcs = gpd.GeoDataFrame({'geometry': []})
    bbox = gpd.GeoDataFrame({'geometry': []})

    batch_size = 40
    buffer_distance = 10

    raster_batches, hsi_batches, itcs_batches, itcs_boxes, affines = split_image(image_file, hsi_img, itcs, bbox, batch_size, buffer_distance)

    # Check types of the returned values
    assert isinstance(raster_batches, list)
    assert isinstance(hsi_batches, list)
    assert isinstance(itcs_batches, list)
    assert isinstance(itcs_boxes, list)
    assert isinstance(affines, list)

    # Check length of the returned lists
    assert len(raster_batches) == len(hsi_batches) == len(itcs_batches) == len(itcs_boxes) == len(affines)

    # Check types of the elements in the list
    assert all(isinstance(x, np.ndarray) for x in raster_batches)
    assert all(isinstance(x, np.ndarray) for x in hsi_batches)
    assert all(isinstance(x, pd.DataFrame) for x in itcs_batches)
    assert all(isinstance(x, np.ndarray) for x in itcs_boxes)
    assert all(isinstance(x, rasterio.Affine) for x in affines)

