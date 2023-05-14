import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


class FieldAlignment:
    def __init__(self, gdf_field, gdf_reference, identifier=['Tag', 'StemTag']):
        gdf_field = gpd.read_file(gdf_field)
        gdf_field = gdf_field[gdf_field['Crwnpst'] > 1]
        gdf_reference = gpd.read_file(gdf_reference)
        identifier = identifier
        self.prepare_data()

    def prepare_data(self):
        gdf_field = gdf_field.rename(columns={self.identifier[0]: 'StemTag'})
        gdf_reference = gdf_reference.rename(columns={self.identifier[0]: 'StemTag'})

        gdf_reference['StemTag'] = gdf_reference['StemTag'].astype(int)
        gdf_reference['StemTag'] = gdf_reference['StemTag'].astype(str)
        gdf_field['StemTag'] = gdf_field['StemTag'].astype(str)

        matched_points = pd.merge(gdf_field, gdf_reference, on='StemTag', suffixes=('_field', '_reference'))

        control_points_source_x = matched_points['geometry_field'].x.values
        control_points_source_y = matched_points['geometry_field'].y.values
        control_points_dest_x = matched_points['geometry_reference'].x.values
        control_points_dest_y = matched_points['geometry_reference'].y.values

        X = pd.DataFrame({'east': control_points_source_x, 'north': control_points_source_y})
        y = pd.DataFrame({'east': control_points_dest_x, 'north': control_points_dest_y})

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        y_train_lat = y_train['east']
        y_train_lon = y_train['north']

        gbm_lat = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, max_depth=3, random_state=89)
        gbm_lon = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, max_depth=3, random_state=89)

        gbm_lat.fit(X_train, y_train_lat)
        gbm_lon.fit(X_train, y_train_lon)

        gbm_lat_ = gbm_lat.predict(X_test)
        gbm_lon_ = gbm_lon.predict(X_test)
        stacking_pred = np.column_stack((gbm_lat_, gbm_lon_))

        mae = mean_absolute_error(y_test, stacking_pred)
        mse = mean_squared_error(y_test, stacking_pred)
        rmse = np.sqrt(mse)

        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        gbm_lat.fit(self.X, self.y.east)
        gbm_lon.fit(self.X, self.y.north)

    def predict(self, outdir=None):
        Xf = pd.DataFrame({'east': gdf_field['geometry'].x.values, 'north': gdf_field['geometry'].y.values})
        stacking_pred_lat = self.gbm_lat.predict(Xf)
        stacking_pred_lon = self.gbm_lon.predict(Xf)

        gdf_field = gdf_field.drop(columns=['geometry'])
        gdf_field['geometry'] = gpd.points_from_xy(stacking_pred_lat, stacking_pred_lon)

        gdf_field = gpd.GeoDataFrame(gdf_field, geometry='geometry')
        gdf_field.crs = gdf_reference.crs

        if outdir is not None:
            gdf_field.to_file(outdir+'/data_field_reprojected.gpkg', driver='GPKG')

        return gdf_field

