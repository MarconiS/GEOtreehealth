import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import config

class FieldAlignment:
    def __init__(self, gdf_field, gdf_reference, identifier=['Tag', 'StemTag']):
        self.gdf_field_path = gdf_field
        self.gdf_reference_path = gdf_reference
        self.gdf_field = gpd.read_file(gdf_field)
        #self.gdf_field = self.gdf_field[self.gdf_field['Crwnpst'] > 1]
        self.gdf_reference = gpd.read_file(gdf_reference)
        self.identifier = identifier
        self.prepare_data()
        self.train()
        self.predict()

    def prepare_data(self):
        self.gdf_field = self.gdf_field.rename(columns={self.identifier[0]: 'StemTag'})
        self.gdf_reference = self.gdf_reference.rename(columns={self.identifier[0]: 'StemTag'})

        self.gdf_reference['StemTag'] = self.gdf_reference['StemTag'].astype(int)
        self.gdf_reference['StemTag'] = self.gdf_reference['StemTag'].astype(str)
        self.gdf_field['StemTag'] = self.gdf_field['StemTag'].astype(str)

        matched_points = pd.merge(self.gdf_field, self.gdf_reference, on='StemTag', suffixes=('_field', '_reference'))

        control_points_source_x = matched_points['geometry_field'].x.values
        control_points_source_y = matched_points['geometry_field'].y.values
        control_points_dest_x = matched_points['geometry_reference'].x.values
        control_points_dest_y = matched_points['geometry_reference'].y.values

        self.X = pd.DataFrame({'east': control_points_source_x, 
                               'north': control_points_source_y, 
                               'Quad' : matched_points['Quad'], 
                               'QX' : matched_points['QX'], 
                               'QY' : matched_points['QY'],
                               'DBH' : matched_points['DBH'],
                               })
        self.y = pd.DataFrame({'east_offset': control_points_dest_x - control_points_source_x, 
                                'north_offset': control_points_dest_y - control_points_source_y})
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        y_train_lat = y_train['east_offset']
        y_train_lon = y_train['north_offset']

        gbm_lat = GradientBoostingRegressor( learning_rate=0.1,  random_state=89)
        gbm_lon = GradientBoostingRegressor(learning_rate=0.1,  random_state=89)

        gbm_lat.fit(X_train.drop('DBH', axis=1), y_train_lat)
        gbm_lon.fit(X_train.drop('DBH', axis=1), y_train_lon)

        gbm_lat_ = gbm_lat.predict(X_test.drop('DBH', axis=1))
        gbm_lon_ = gbm_lon.predict(X_test.drop('DBH', axis=1))

        mae = mean_absolute_error(y_test, np.column_stack((gbm_lat_, gbm_lon_)))
        mse = mean_squared_error(y_test, np.column_stack((gbm_lat_, gbm_lon_)))
        rmse = np.sqrt(mse)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        # Random Forest Regressor
        rfr_lat = RandomForestRegressor(random_state=89)
        rfr_lon = RandomForestRegressor(random_state=89)

        rfr_lat.fit(X_train.drop('DBH', axis=1), y_train_lat)
        rfr_lon.fit(X_train.drop('DBH', axis=1), y_train_lon)

        rfr_lat_ = rfr_lat.predict(X_test.drop('DBH', axis=1))
        rfr_lon_ = rfr_lon.predict(X_test.drop('DBH', axis=1))

        mae = mean_absolute_error(y_test, np.column_stack((rfr_lat_, rfr_lon_)))
        mse = mean_squared_error(y_test, np.column_stack((rfr_lat_, rfr_lon_)))
        rmse = np.sqrt(mse)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")


        # Store models for later prediction
        self.models_lat = [gbm_lat, rfr_lat]
        self.models_lon = [gbm_lon, rfr_lon]

        stacking_pred = np.column_stack((gbm_lat_, gbm_lon_))

        mae = mean_absolute_error(y_test, stacking_pred)
        mse = mean_squared_error(y_test, stacking_pred)
        rmse = np.sqrt(mse)

        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        #gbm_lat.fit(self.X, self.y.east)
        #gbm_lon.fit(self.X, self.y.north)

    def predict(self):
        Xf = pd.DataFrame({'east': self.gdf_field['geometry'].x.values, 'north': self.gdf_field['geometry'].y.values,
                        'Quad' : self.gdf_field['Quad'], 'QX' : self.gdf_field['QX'], 'QY' : self.gdf_field['QY'],
                        'DBH' : self.gdf_field['DBH']})
        # Predict latitude and longitude with each model
        predictions_lat = [model.predict(Xf.drop('DBH', axis=1)) for model in self.models_lat]
        predictions_lon = [model.predict(Xf.drop('DBH', axis=1)) for model in self.models_lon]

        # Ensemble by averaging predictions
        stacking_pred_lat = Xf.east + np.mean(predictions_lat, axis=0)
        stacking_pred_lon = Xf.north + np.mean(predictions_lon, axis=0)

        self.gdf_field = self.gdf_field.drop(columns=['geometry'])
        self.gdf_field['geometry'] = gpd.points_from_xy(stacking_pred_lat, stacking_pred_lon)

        self.gdf_field = gpd.GeoDataFrame(self.gdf_field, geometry='geometry')
        self.gdf_field.crs = self.gdf_reference.crs

        self.gdf_field.to_file(config.outdir+'_stems_ref.gpkg', driver='GPKG')

        return self.gdf_field

#self = FieldAlignment(gdf_field = config.unaligned_stems, gdf_reference = config.aligned_stems)


