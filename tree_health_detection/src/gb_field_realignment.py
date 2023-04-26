import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

def gb_field_realignment(outdir, gdf_field, gdf_reference):

    # Load your GeoDataFrame (assuming it's a shapefile)
    gdf_field =  gpd.read_file(gdf_field)
    gdf_field = gdf_field[gdf_field['Crwnpst'] > 1]

    gdf_reference = gpd.read_file('/home/smarconi/Documents/GitHub/tree_mask_delineation/indir/data_field.shp')
    #in gdf_reference, rename Tag in StemTag and turn values into character
    gdf_reference = gdf_reference.rename(columns={'Tag': 'StemTag'})
    gdf_reference['StemTag'] = gdf_reference['StemTag'].astype(int)

    gdf_reference['StemTag'] = gdf_reference['StemTag'].astype(str)
    gdf_field['StemTag'] = gdf_field['StemTag'].astype(str)

    #find matching trees between field and reference
    matched_points = pd.merge(gdf_field, gdf_reference, on='StemTag', suffixes=('_field', '_reference'))

    # Extract source and destination control points from matched_points
    control_points_source_x = matched_points['geometry_field'].x.values
    control_points_source_y = matched_points['geometry_field'].y.values
    control_points_dest_x = matched_points['geometry_reference'].x.values
    control_points_dest_y = matched_points['geometry_reference'].y.values

    #define X as the pandas dataframe with control_points_source_x, control_points_source_y
    X = pd.DataFrame({'east': control_points_source_x, 'north': control_points_source_y})
    #define y as the pandas dataframe with control_points_dest_x, control_points_dest_y
    y = pd.DataFrame({'east': control_points_dest_x, 'north': control_points_dest_y})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Separate the target variables
    y_train_lat = y_train['east']
    y_train_lon = y_train['north']

    # Initialize the base models
    gbm_lat = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbm_lon = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    rf_lat = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf_lon = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

    # Initialize the StackingRegressor
    stacking_lat = StackingRegressor(estimators=[('gbm_lat', gbm_lat)], final_estimator=LinearRegression())
    stacking_lon = StackingRegressor(estimators=[('gbm_lon', gbm_lon)], final_estimator=LinearRegression())

    # Train the StackingRegressor
    stacking_lat.fit(X_train, y_train_lat)
    stacking_lon.fit(X_train, y_train_lon)

    # Make predictions
    stacking_pred_lat = stacking_lat.predict(X_test)
    stacking_pred_lon = stacking_lon.predict(X_test)

    stacking_pred = np.column_stack((stacking_pred_lat, stacking_pred_lon))

    # Evaluate the stacked model
    mae = mean_absolute_error(y_test, stacking_pred)
    mse = mean_squared_error(y_test, stacking_pred)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    # Now retrain the model with all coordinates, and use it for prediction of all gdf_field crowms
    stacking_lat.fit(X, y['east'])
    stacking_lon.fit(X, y['north'])

    # Make predictions
    Xf = pd.DataFrame({'east': gdf_field['geometry'].x.values, 'north': gdf_field['geometry'].y.values})
    stacking_pred_lat = stacking_lat.predict(Xf)
    stacking_pred_lon = stacking_lon.predict(Xf)
    #drop geometry from gdf_field and stack new coordinates
    gdf_field = gdf_field.drop(columns=['geometry'])
    gdf_field['geometry'] = gpd.points_from_xy(stacking_pred_lat, stacking_pred_lon)

    #turn gdf_field back into a geodataframe
    gdf_field = gpd.GeoDataFrame(gdf_field, geometry='geometry')
    #set the crs of gdf_field to the crs of gdf_reference
    gdf_field.crs = gdf_reference.crs
    #write gdf_field to a shapefile if outdir is not null
    if outdir is not None:
        gdf_field.to_file(outdir+'/data_field_reprojected.shp')
    
    return gdf_field


