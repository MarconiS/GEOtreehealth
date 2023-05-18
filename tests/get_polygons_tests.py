import unittest
import numpy as np
from shapely.geometry import Point
from tree_delineation import mask_to_polygons
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from shapely.geometry import Point, Polygon

class TestMaskToPolygons(unittest.TestCase):
    
    def test_mask_to_polygons(self):
        # Create a binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1

        individual_point = Point(50, 50)

        # Convert mask to polygons
        largest_polygon = mask_to_polygons(mask, individual_point)

        # Check if the polygon is not None
        self.assertIsNotNone(largest_polygon)

        # Check if the polygon is indeed a Polygon
        self.assertTrue(isinstance(largest_polygon, Polygon))

        # Check if the polygon's area is as expected (2500 = 50*50)
        self.assertEqual(largest_polygon.area, 2500)

        # Check if the individual_point is contained within the polygon
        self.assertTrue(largest_polygon.contains(individual_point))

if __name__ == '__main__':
    unittest.main()


