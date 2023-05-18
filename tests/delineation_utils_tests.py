import unittest
import geopandas as gpd
import pandas as pd
from unittest.mock import patch
from tree_delineation import FieldAlignment

class TestFieldAlignment(unittest.TestCase):
    
    @patch("geopandas.read_file")
    def setUp(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({'StemTag': ['1', '2'], 'Crwnpst': [2, 3]}, 
                                                  geometry=gpd.points_from_xy([3, 4], [5, 6]))
        self.field_alignment = FieldAlignment('dummy_path1', 'dummy_path2')

    @patch("geopandas.read_file")
    @patch("pandas.DataFrame.merge")
    @patch("pandas.DataFrame")
    def test_prepare_data(self, mock_df, mock_merge, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({'StemTag': ['1', '2']}, 
                                                  geometry=gpd.points_from_xy([3, 4], [5, 6]))
        mock_merge.return_value = gpd.GeoDataFrame({'StemTag': ['1', '2'], 'geometry_field': [None, None], 
                                                    'geometry_reference': [None, None]})
        mock_df.return_value = pd.DataFrame({'east': [3, 4], 'north': [5, 6]})
        self.field_alignment.prepare_data()
        mock_df.assert_called()
        mock_merge.assert_called()

    @patch("sklearn.model_selection.train_test_split")
    @patch("sklearn.ensemble.GradientBoostingRegressor.fit")
    def test_train(self, mock_fit, mock_split):
        mock_split.return_value = [None, None, None, None]
        self.field_alignment.train()
        mock_split.assert_called()
        mock_fit.assert_called()

    @patch("sklearn.ensemble.GradientBoostingRegressor.predict")
    @patch("geopandas.GeoDataFrame.drop")
    def test_predict(self, mock_drop, mock_predict):
        mock_drop.return_value = gpd.GeoDataFrame({'StemTag': ['1', '2']}, 
                                                  geometry=gpd.points_from_xy([3, 4], [5, 6]))
        mock_predict.return_value = pd.DataFrame({'east': [3, 4], 'north': [5, 6]})
        self.field_alignment.predict()
        mock_drop.assert_called()
        mock_predict.assert_called()

if __name__ == '__main__':
    unittest.main()
