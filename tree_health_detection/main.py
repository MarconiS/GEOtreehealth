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
import matplotlib.pyplot as plt
import importlib


import tree_delineation
import tree_health_detection
import field_data_alignment

import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from pathlib import Path
import shutil
#import config.py
import config

def stratified_subset_indices(labels, subset_size):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_label_count = min(label_counts)
    subset_indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        np.random.shuffle(label_indices)
        subset_indices.extend(label_indices[:subset_size])
    return subset_indices

def linStretch(input_img):
    img_min = input_img.min()
    img_max = input_img.max()
    return (input_img - img_min) / (img_max - img_min)


def percentileStretch(img, low=2, high=98):
    # Compute the percentiles
    
    low_p, high_p = np.percentile(img[img > 0], [low, high])
    
    # Stretch the image
    img_stretched = np.clip((img - low_p) / (high_p - low_p), 0, 1)
    
    return img_stretched


def __main__():
    if config.noGUI:
        import matplotlib
        matplotlib.use('Agg')
        
    #get the crowns that match with stem_path
    if config.crowns is not None:
        crowns = gpd.read_file(config.data_path + config.crowns+config.siteID+'_subset.gpkg', driver='GPKG')
    else:
        #loop through geopackages in custom folder and append polygons in a unique geodataframe
        if config.get_tree_crowns:
            tree_delineation.build_data_schema()

        else:
            sub_seg_dir = os.path.join(config.data_path+'/Crowns/'+ config.siteID)
            crowns = gpd.GeoDataFrame()
            for file in os.listdir(sub_seg_dir):
                if file.endswith('.gpkg'):
                    # skip file if empty
                    if os.stat(sub_seg_dir +'/'+ file).st_size == 0:
                        continue
    
                    crowns = pd.concat([crowns, gpd.read_file(sub_seg_dir +'/'+ file)],ignore_index=True)
            
            crowns.to_file(config.data_path + config.crowns + '/SAM_'+config.siteID+'.gpkg', driver='GPKG') 

        if crowns is None or crowns == []:
            print("No tree crowns found. Please run tree_mask_delineation.py first. You can run it as a standalone or by setting get_tree_crown as True in config.py")
            exit() 
    


    # if data storage required
    if config.store_clips:
        if config.clean_dataset:
            # remove directories and their content if they exist
            for dir in [ config.root_dir + "rgb", config.root_dir + "hsi", config.root_dir + "lidar", config.root_dir + "polygon_mask", config.root_dir + "labels"]:
                if os.path.exists(dir):
                    shutil.rmtree(dir)

        rgb_dir = config.root_dir + "rgb"
        hsi_dir = config.root_dir + "hsi"
        png_dir = config.root_dir + "png"
        lidar_dir = config.root_dir + "lidar"
        polygon_mask_dir = config.root_dir + "polygon_mask"
        labels_dir = config.root_dir + "labels"

        # Ensure that all directories exist. if directories do not exist, create them
        for dir in [config.root_dir, rgb_dir, hsi_dir, png_dir, lidar_dir, polygon_mask_dir, labels_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # Load your polygon data (GeoDataFrame)
        gdf = gpd.read_file(config.data_path + config.crowns+config.siteID+'_subset.gpkg', driver='GPKG')
        
        # remove rows with area < 4 m2
        gdf = gdf[gdf['geometry'].area > 8]
        gdf = gdf[gdf['selected'] == True]
        #if gdf doesn't have colunm 'SiteID', add it
        if 'SiteID' not in gdf.columns:
            gdf["SiteID"] = config.siteID
        # Process each polygon
        for idx, row in gdf.iterrows():
            try:
                tree_health_detection.store_data_structures.process_polygon(polygon = row.copy(),
                                root_dir = config.root_dir,  rgb_path = config.data_path+config.rgb_path, 
                                hsi_path = config.data_path+config.hsi_img, 
                                lidar_path = config.data_path+config.laz_path, polygon_id=  idx)
            except Exception as e:
                print(f"Error processing polygon {idx}: {str(e)}")
                continue
    ### move to main script
    # list all files in folder data_pt
    data_pt = config.root_dir +"labels/"

    files = os.listdir(data_pt)
    # llop through all files and append to a list
    labels = []
    for file in files:
        if file.endswith('.csv'):
            tmp = pd.read_csv(data_pt + file)
            labels.append(tmp)
    # concatenate all dataframes in the list
    labels_df = pd.concat(labels, ignore_index=True)

    # if siteID is not null, select only rows with siteID
    if config.site_training is not None:
        labels_df = labels_df[labels_df['SiteID'] == config.siteID]
    # only select rows where selected is True
    if config.siteID is not "OSBS":
        labels_df = labels_df[labels_df['selected'] == True]
    # remove all AU labels
    #labels_df = labels_df[~labels_df[config.response].isin(['AU'])]

    #if response is D chenge in DS
    labels_df[config.response] = labels_df[config.response].replace(['D'], 'DS')
    # count the number of unique responses, and remove entries occurring less than 10 times
    #loop through all siteID and append the labels
    sites = labels_df.SiteID.unique()

    for site in sites:
        counts = labels_df[labels_df['SiteID'] == site][config.response].value_counts()
        # get tmp_labels, that is the labels that occur more than 10 times and have siteID == site
        tmp_labels = labels_df[(labels_df[config.response].isin(counts[counts > 10].index)) & (labels_df['SiteID'] == site)]
        tmp_labels = tmp_labels.reset_index(drop=True)
        # count the number of unique labels
        rebalanced_to = tmp_labels[config.response].value_counts().min()

        # downsample more frequent classes to balance dataset
        tmp_labels = tmp_labels.groupby(config.response).apply(lambda x: x.sample(rebalanced_to)).reset_index(drop=True)
        # append tmp_labels to labels_df
        labels_df = labels_df[labels_df['SiteID'] != site]
        labels_df = pd.concat([labels_df, tmp_labels], ignore_index=True)

    # if labels_df['response'] is character, convert it into a numeric class
    if labels_df[config.response].dtype == 'object':
        labels_df[config.response] = pd.factorize(labels_df[config.response])[0]

    labels_df = labels_df.reset_index(drop=True)


    # Split the data into train-test-validation (70%-15%-15%) with stratification. Stratify by response, but also by siteID

    labels_df['combined_stratify'] = labels_df[config.response].astype(str) + "_" + labels_df['SiteID'].astype(str)
    train_data, temp_data = train_test_split(labels_df, test_size=0.3, random_state=42, stratify=labels_df['combined_stratify'])
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['combined_stratify'])

    #reset index of train, validation, test
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    train_dataset = tree_health_detection.MultiModalDataset(data = train_data, response=config.response, max_points=config.max_points)
    test_dataset = tree_health_detection.MultiModalDataset(test_data, response=config.response, max_points=config.max_points)
    val_dataset = tree_health_detection.MultiModalDataset(val_data, response=config.response, max_points=config.max_points)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train model
    # from train_dataset, get the number of bands in a hsi image
    num_bands = train_dataset[0]['hsi'].shape[0]
    
    # from train_dataset, get the number of classes
    num_classes = len(np.unique(train_data[config.response]))
 
    model = tree_health_detection.MultiModalNet(config.in_channels, num_classes, num_bands,
                 config.hsi_out_features, config.rgb_out_features, config.lidar_out_features)
    
    model = model.to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.00001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)  #L2 regularization


     # Create an experiment with your api key
    #set up comet experiment
    comet_api_key = os.environ.get('COMET_API_KEY')
    if comet_api_key is None:
        raise ValueError("COMET_API_KEY environment variable not set")
    experiment = Experiment(project_name=config.comet_name, workspace=config.comet_workspace, api_key=comet_api_key)
    # Starting the experiment
    # torch.cuda.empty_cache()

    for epoch in range(config.num_epochs):
        # Training
        model = model.train()
        running_loss = 0
        for batch in train_loader:
            hsi = batch['hsi'].float().to(device)
            # turn nan into 0
            hsi[hsi != hsi] = 0
            rgb = batch['rgb'].float().to(device)
            lidar = batch['lidar'].float().to(device)
            labels = batch['label'].long().to(device)
            optimizer.zero_grad()      
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(hsi, rgb, lidar)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        
        # Log training loss
        experiment.log_metric('train_loss', running_loss / len(train_loader), epoch=epoch)
            
        # Validation
        model = model.eval()
        running_val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            correct = 0
            total = 0

            for batch in val_loader:
                hsi_ = batch['hsi'].float().to(device)
                rgb_ = batch['rgb'].float().to(device)
                lidar_ = batch['lidar'].float().to(device)
                labels_ = batch['label'].long().to(device)
                outputs = model(hsi_, rgb_, lidar_)
                _, predicted = torch.max(outputs.data, 1)
                val_loss = criterion(outputs, labels_)
                running_val_loss += val_loss.item()
                total += labels_.size(0)
                correct += (predicted == labels_).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels_.cpu().numpy())
            
                # Log validation loss
                experiment.log_metric('val_loss', running_val_loss / len(val_loader), epoch=epoch)
                # Log validation accuracy
                experiment.log_metric('val_accuracy', 100 * correct / total, epoch=epoch)
                # Log confusion matrix
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels_.cpu().numpy())
                c_matrix = confusion_matrix(val_labels, val_preds)
                # Generate the class names (assumes that labels are integers starting from 0)
                class_names = [str(i) for i in range(c_matrix.shape[0])]

                # Convert the confusion matrix to the Comet.ml format
                matrix = {}
                for i, row in enumerate(c_matrix):
                    matrix[class_names[i]] = dict(zip(class_names, row.tolist()))

                
        # Log the confusion matrix
        experiment.log_confusion_matrix(matrix=c_matrix, labels=['A', 'AU', 'DS'])
        print(f'Epoch {epoch+1}, Accuracy: {100 * correct / total}%')

    # Transform function to resize the image
    resize = transforms.Resize((config.hsi_resize, config.hsi_resize))
    del hsi, rgb, lidar, labels
    del  hsi_, rgb_, lidar_, labels_
    torch.cuda.empty_cache()
    # Test Set Evaluation
    model = model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            hsi_ = batch['hsi'].float().to(device)
            rgb_ = batch['rgb'].float().to(device)
            lidar_ = batch['lidar'].float().to(device)
            labels_ = batch['label'].long().to(device)
            outputs = model(hsi_, rgb_, lidar_)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_.size(0)
            correct += (predicted == labels_).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels_.cpu().numpy())
            #save hsi as raster
        c_matrix = confusion_matrix(test_labels, test_preds)
        # Generate the class names (assumes that labels are integers starting from 0)
        class_names = [str(i) for i in range(c_matrix.shape[0])]

        # Convert the confusion matrix to the Comet.ml format
        matrix = {}
        for i, row in enumerate(c_matrix):
            matrix[class_names[i]] = dict(zip(class_names, row.tolist()))

        # Plotting RGB and HSI Images
        for i in range(len(rgb_)):
            fig, ax = plt.subplots(1, 2)

            # Plot RGB Image
            rgb_img = (rgb_[i].cpu()).permute(1, 2, 0).numpy()
            ax[0].imshow(rgb_img)
            ax[0].set_title(f'Prd: {test_preds[i]}' f', Obs: {test_labels[i]}')
            ax[0].axis('off')

            # Plot HSI Image
            hsi_img = hsi_[i, [28, 87, 115]].permute(1, 2, 0).cpu().numpy()
            hsi_img = percentileStretch(hsi_img)
            ax[1].imshow(hsi_img)
            ax[1].set_title(f'Prd: {test_preds[i]}' f', Obs: {test_labels[i]}')
            ax[1].axis('off')

            # Save figure to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                fig.savefig(temp_file.name, format='png')
                # Log Image to Comet.ml
                experiment.log_image(temp_file.name, image_format='png')

        # Calculate test accuracy
        test_accuracy = 100 * correct / total
        print(f'Test Accuracy: {test_accuracy}%')

        # Convert the confusion matrix to the Comet.ml format
        matrix = {}
        for i, row in enumerate(c_matrix):
            matrix[class_names[i]] = dict(zip(class_names, row.tolist()))

    # Log test accuracy
    experiment.log_metric('test_accuracy', test_accuracy)

    # Ending the experiment
    experiment.end()

    return(model)


