from comet_ml import Experiment
from PIL import Image
from rasterio.windows import from_bounds
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import comet_ml
import cv2
import geopandas as gpd
import laspy
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
import torch

import field_data_alignment
import tree_health_detection
import tree_delineation

from importlib import reload
reload(tree_health_detection)
reload(tree_delineation)
data_path = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/'
rgb_path = 'Imagery/SERC/RGB_364000_4305000.tif'
stem_path = 'Stems/SERC_stems_legacy.gpkg'
stem_path = 'Stems/SERC_reprojected_legacy.gpkg'
hsi_img  = 'Imagery/SERC/HSI_364000_4305000.tif'
laz_path = 'Imagery/SERC/LAS_364000_4305000.laz'
crowns = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Crowns/bboxesSERC_0.gpkg'
boxes_pt = "/media/smarconi/Gaia/Macrosystem_2/legacy_polygons/dp_SERC_.gpkg"
folder = 'Crowns'
require_alignment = True
field_dir = 'Stems'
gdf_reference = '/home/smarconi/Documents/GitHub/GEOtreehealth/indir/field_data/SERC/data_field.shp'

def __main__(get_clips = False, noGUI = True):
    if noGUI:
        import matplotlib
        matplotlib.use('Agg')
        
    import matplotlib.pyplot as plt

    # if required alignment, launch field_data_alignment using gradient boosting
    # stem_positions = gpd.read_file("/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Stems//data_field_reprojected.gpkg")
    #stem_positions = gpd.read_file(data_path+stem_path)
    if require_alignment:
        if gdf_reference is None:
            print("need reference data for alignment: continuing without alignment")
        else:
            model = field_data_alignment.FieldAlignment(gdf_field=data_path+stem_path, gdf_reference=gdf_reference, identifier=['Tag', 'StemTag'])
            model.train()
            model.predict(outdir=data_path+field_dir)
 
    #get the crowns that match with stem_path
    if crowns is not None:
        crowns = gpd.read_file(crowns)
    else:
        # get crowns from tree_delineation pipeline
        crowns = tree_delineation.build_data_schema(data_path, stem_path,rgb_path, hsi_path, 
            laz_path, grid_space = 20, threshold_score=0.5)         
    


    if 'StemTag' not in crowns.columns:
        print("original polygons do not have crown information, assigning crowns to stems by spatial join")
        # left join crowns and stem_positions by StemTag, assigning a polygon to each stem
        stem_positions = gpd.sjoin(stem_positions, crowns, how="left", op='within')
    else:
        # left_join crowns and stem_positions by StemTag, keep only crowns polygon geometries
        stem_positions = gpd.sjoin(stem_positions, crowns[['StemTag', 'geometry']], how="left", op='within')
    # remove points not visible from sky
    stem_positions = stem_positions[stem_positions['Crwnpst'] >2]
    # just for testing, get some per class
    sample_A = stem_positions.loc[stem_positions['Status'] == 'A'].sample(n=280, random_state=42)
    sample_B = stem_positions.loc[stem_positions['Status'] == 'AU'].sample(n=280,random_state=42)
    sample_C = stem_positions.loc[stem_positions['Status'] == 'DS'].sample(n=280, random_state=42)
    # concatenate the three samples into one GeoDataFrame
    stem_positions = gpd.GeoDataFrame(pd.concat([sample_A, sample_B, sample_C]))
    # optionally reset the index of the sample GeoDataFrame
    stem_positions = stem_positions.reset_index(drop=True)

    health_classes = stem_positions.Status
    le = LabelEncoder()
    # Fit the LabelEncoder to the 'labels' column and transform it
    health_classes = le.fit_transform(pd.DataFrame(stem_positions['Status']).values.ravel())

    # Save the dataset
    bounding_boxes = tree_health_detection.generate_bounding_boxes(stem_positions, buffer_size=2)

    if get_clips == True:
        hyperspectral_data = tree_health_detection.extract_hyperspectral_data(hsi_path, bounding_boxes)
        tree_health_detection.save_dataset(hyperspectral_data, 'tree_health_detection/loaders/hyperspectral_SERC.pkl')

        rgb_data = tree_health_detection.extract_hyperspectral_data(rgb_path, bounding_boxes)
        tree_health_detection.save_dataset(rgb_data, 'tree_health_detection/loaders/rgb_SERC.pkl')

        las_data = tree_health_detection.extract_pointcloud_data(laz_path, bounding_boxes)
        tree_health_detection.save_dataset(las_data, 'tree_health_detection/loaders/las_SERC.pkl')

    # Load the saved dataset
    if 'hyperspectral_data' not in locals():
        hyperspectral_data = tree_health_detection.load_dataset('tree_health_detection/loaders/hyperspectral_SERC.pkl')
    if 'rgb_data' not in locals():
        rgb_data = tree_health_detection.load_dataset('tree_health_detection/loaders/rgb_SERC.pkl')
    if 'las_data' not in locals():
        las_data = tree_health_detection.load_dataset('tree_health_detection/loaders/las_SERC.pkl')

    # Split the dataset
    #las_data = [np.expand_dims(x, axis=0) for x in las_data]

    species = stem_positions.Status # List of species corresponding to each data point
    geography = stem_positions.SiteID # List of geography information corresponding to each data point

    #hsi = CustomDataset(hyperspectral_data, health_classes, label_to_int)
    train_data, val_data, test_data = tree_health_detection.split_dataset(rgb_data, hyperspectral_data, las_data, health_classes)

    train_dataset = tree_health_detection.MultiModalDataset(*train_data) #transform = CustomResize((128, 128))
    val_dataset = tree_health_detection.MultiModalDataset(*val_data)
    test_dataset = tree_health_detection.MultiModalDataset(*test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    num_epochs = 10
    lidar_output_size = 93
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=tree_health_detection.custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=tree_health_detection.custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=tree_health_detection.custom_collate_fn)

    #train model
    # Define the model, loss function, and optimizer
    num_bands = train_dataset.hs_data[0].shape[0]
    num_classes = len(set(health_classes))
    #model = SpectralAttentionClassifier(num_bands, num_classes)

    model = tree_health_detection.MultiModalModel(num_classes, num_bands)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    #set up comet experiment
    experiment = Experiment(project_name="treehealth", workspace="marconis")
    # Starting the experiment
    for epoch in range(num_epochs):
        train_loss, train_accuracy = tree_health_detection.train(model, train_loader, criterion, optimizer, device, experiment)
        val_loss, val_accuracy, true_labels, predicted_labels = tree_health_detection.validate(model, val_loader, criterion, device, experiment)
        # Calculate the confusion matrix for the validation set
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        # Log the confusion matrix, accuracy, and loss to Comet.ml
        experiment.log_current_epoch(epoch)
        experiment.log_confusion_matrix(labels=list(range(num_classes)), matrix=conf_matrix, step=epoch, title="Validation Confusion Matrix")
        experiment.log_metric("val_loss", val_loss, step=epoch)
        experiment.log_metric("val_accuracy", val_accuracy, step=epoch)
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Testing the model
    true_labels, predicted_labels = tree_health_detection.test(model, test_loader, device, experiment, noGUI)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Log the confusion matrix to Comet.ml
    experiment.log_confusion_matrix(labels=list(range(num_classes)), matrix=conf_matrix)

    # Calculate and log the test accuracy
    test_accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    experiment.log_metric("test_accuracy", test_accuracy)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Ending the experiment
    experiment.end()

    return(model)

