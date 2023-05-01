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

from tree_health_detection.src import multimodal as mti
from tree_health_detection.src import train_val_t as tvt
from tree_health_detection.src.spectral_attention import *
from tree_health_detection.src.utils import *

folder = '../tree_mask_delineation/'
rgb_tile = 'imagery/rgb_clip.tif'
stem_path = 'gradient_boosting_alignment.gpkg'
hyperspectral_tile  = 'imagery/hsi_clip.tif'
las_file = 'imagery/LiDAR.laz'

def __main__(get_clips = False, noGUI = True):
    if noGUI:
        import matplotlib
        matplotlib.use('Agg')
        
    import matplotlib.pyplot as plt

    stem_positions = gpd.read_file("/home/smarconi/Documents/GitHub/Macrosystems_analysis/Data/geolocations/SERC/field.shp")
    hyperspectral_tile = '/home/smarconi/Documents/DAT_for_health/SERC/SERC/HSI.tif'
    rgb_tile = '/home/smarconi/Documents/DAT_for_health/SERC/SERC/2021_SERC_5_364000_4305000_image.tif'
    las_file = '/home/smarconi/Documents/GitHub/Macrosystems_analysis/Data/remotesensing/DP1.30003.001/neon-aop-products/2021/FullSite/D02/2021_SERC_5/L1/DiscreteLidar/ClassifiedPointCloud/NEON_D02_SERC_DP1_364000_4305000_classified_point_cloud_colorized.laz'
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
    bounding_boxes = generate_bounding_boxes(stem_positions, buffer_size=2)

    if get_clips == True:
        hyperspectral_data = extract_hyperspectral_data(hyperspectral_tile, bounding_boxes)
        save_dataset(hyperspectral_data, 'tree_health_detection/loaders/hyperspectral_SERC.pkl')

        rgb_data = extract_hyperspectral_data(rgb_tile, bounding_boxes)
        save_dataset(rgb_data, 'tree_health_detection/loaders/rgb_SERC.pkl')

        las_data = extract_pointcloud_data(las_file, bounding_boxes)
        save_dataset(las_data, 'tree_health_detection/loaders/las_SERC.pkl')

    # Load the saved dataset
    if 'hyperspectral_data' not in locals():
        hyperspectral_data = load_dataset('tree_health_detection/loaders/hyperspectral_SERC.pkl')
    if 'rgb_data' not in locals():
        rgb_data = load_dataset('tree_health_detection/loaders/rgb_SERC.pkl')
    if 'las_data' not in locals():
        las_data = load_dataset('tree_health_detection/loaders/las_SERC.pkl')

    # Split the dataset
    #las_data = [np.expand_dims(x, axis=0) for x in las_data]

    species = stem_positions.Status # List of species corresponding to each data point
    geography = stem_positions.SiteID # List of geography information corresponding to each data point

    #hsi = CustomDataset(hyperspectral_data, health_classes, label_to_int)
    train_data, val_data, test_data = split_dataset(rgb_data, hyperspectral_data, las_data, health_classes)

    train_dataset = MultiModalDataset(*train_data) #transform = CustomResize((128, 128))
    val_dataset = MultiModalDataset(*val_data)
    test_dataset = MultiModalDataset(*test_data)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    num_epochs = 10
    lidar_output_size = 93
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    #train model
    # Define the model, loss function, and optimizer
    num_bands = train_dataset.hs_data[0].shape[0]
    num_classes = len(set(health_classes))
    #model = SpectralAttentionClassifier(num_bands, num_classes)

    model = mti.MultiModalModel(num_classes, num_bands)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)

    #set up comet experiment
    experiment = Experiment(project_name="treehealth", workspace="marconis")
    # Starting the experiment
    for epoch in range(num_epochs):
        train_loss, train_accuracy = tvt.train(model, train_loader, criterion, optimizer, device, experiment)
        val_loss, val_accuracy, true_labels, predicted_labels = tvt.validate(model, val_loader, criterion, device, experiment)
        # Calculate the confusion matrix for the validation set
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        # Log the confusion matrix, accuracy, and loss to Comet.ml
        experiment.log_current_epoch(epoch)
        experiment.log_confusion_matrix(labels=list(range(num_classes)), matrix=conf_matrix, step=epoch, title="Validation Confusion Matrix")
        experiment.log_metric("val_loss", val_loss, step=epoch)
        experiment.log_metric("val_accuracy", val_accuracy, step=epoch)
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Testing the model
    true_labels, predicted_labels = tvt.test(model, test_loader, device, experiment, noGUI)

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

