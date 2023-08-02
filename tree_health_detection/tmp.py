import pandas as pd
import geopandas as gpd
import config
import os
# load geopandas and append metadata based on StemTag
all_info = pd.read_csv("/home/smarconi/Documents/GitHub/Macrosystems_analysis/Output/full_survey.csv")
all_info.columns
all_info = all_info[all_info['SiteID'] == 'HARV']
all_info.Year.unique()
all_info = all_info[all_info['Year'] == 'Status 2021']
#all_info = all_info[all_info['Status'] == 'A']
#if all_info is not nan then status is AU
all_info.Status[(all_info['FAD'].notna()) & (all_info['Status'] == 'A')] = 'AU'
# get the delineated trees
delineated_trees = gpd.read_file("/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Crowns/full_site/HARV_500m.gpkg")
important_data = all_info[['StemTag','FAD','Year',"Status", 'Crown position', 'Percentage of crown living']]
# turn StemTag into a string
important_data['StemTag'] = important_data['StemTag'].astype(int).astype(str)
delineated_trees['StemTag'] = delineated_trees['StemTag'].astype(int).astype(str)

# remove Status from delineated trees
delineated_trees = delineated_trees.drop(columns=['Status'])
delineated_trees = delineated_trees[delineated_trees['selected'] == True]

# left_join the delineated trees with the important data
delineated_trees = delineated_trees.merge(important_data, how='left', on='StemTag')
delineated_trees.Status
delineated_trees.to_file("/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Crowns/full_site/HARV_clean.gpkg", driver='GPKG')