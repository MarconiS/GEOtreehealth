import unittest
from shapely.geometry import Polygon
from tree_delineation import remove_overlapping_polygons
import geopandas as gpd
import pandas as pd

class TestRemoveOverlappingPolygons(unittest.TestCase):
    
    def test_remove_overlapping_polygons(self):
        polygon1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        polygon2 = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])
        polygon3 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])

        gdf = gpd.GeoDataFrame(pd.Series([polygon1, polygon2, polygon3]), columns=['geometry'])

        merged_gdf = remove_overlapping_polygons(gdf)
        
        expected_gdf = gpd.GeoDataFrame(pd.Series([Polygon([(0, 0), (5, 0), (8, 0), (8, 3), (5, 5), (0, 5), (3, 3), (3, 0)]), 
                                                   polygon3]), columns=['geometry'])

        pd.testing.assert_frame_equal(merged_gdf, expected_gdf)

if __name__ == '__main__':
    unittest.main()
