import unittest
import os
import shutil
import geopandas as gpd
import config
from tree_delineation import build_data_schema

class TestBuildDataSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_path = config.data_path
        cls.old_stem_path = config.stem_path
        config.data_path = '/tmp/'
        config.stem_path = 'test_stem_path.geojson'

    @classmethod
    def tearDownClass(cls):
        # Restore original values
        config.data_path = cls.old_path
        config.stem_path = cls.old_stem_path
        # Clean up created directories and files
        shutil.rmtree('/tmp/legacy_polygons')
        os.remove(os.path.join(config.data_path, config.stem_path))

    def test_no_file_exists(self):
        with self.assertRaises(FileNotFoundError):
            _ = build_data_schema()

    def test_file_exists(self):
        gpd.GeoDataFrame().to_file(os.path.join(config.data_path, config.stem_path), driver='GeoJSON')
        try:
            _ = build_data_schema()
        except FileNotFoundError:
            self.fail("build_data_schema raised FileNotFoundError unexpectedly!")

    def test_directory_creation(self):
        if os.path.exists(os.path.join(config.data_path, '../legacy_polygons/')):
            shutil.rmtree(os.path.join(config.data_path, '../legacy_polygons/'))
        gpd.GeoDataFrame().to_file(os.path.join(config.data_path, config.stem_path), driver='GeoJSON')
        _ = build_data_schema()
        self.assertTrue(os.path.exists(os.path.join(config.data_path, '../legacy_polygons/')))

if __name__ == '__main__':
    unittest.main()
