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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
import torch

from tree_health_detection.src import multimodal as mti
from tree_health_detection.src import train_val_test as tvt
from tree_health_detection.src.spectral_attention import *

def clean_hsi_to_0_255_range(hsi):
    band_ranges = [(0, 18), (92, 114), (154, 190), (279, 314), (400, 425)]

    # Use list comprehension to generate the list of band indices to remove
    bands_to_remove = [band for start, end in band_ranges for band in range(start, end+1)]
    # Assuming hsi_image is already loaded
    num_bands = hsi.shape[0]

    # Create a boolean array for the bands to keep
    bands_to_keep = np.ones(num_bands, dtype=bool)
    bands_to_keep[bands_to_remove] = False

    # Remove the specified bands from hsi_image
    hsi = hsi[bands_to_keep, :, :]

    #assume pixels out of range to be 0 10000
    hsi[hsi > 10000] = 0
    hsi[hsi < 0] = 0

    #normalize to 0-255
    hsi = hsi / 10000 * 255
    return hsi


def generate_bounding_boxes(stem_positions, buffer_size):
    stem_positions['geometry'] = stem_positions.buffer(buffer_size)
    return stem_positions

import numpy as np

def normalize_rgb(rgb_array):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    # First, convert the RGB array to float in the range [0, 1]
    rgb_array = rgb_array.astype(np.float32) / 255.0
    
    # Normalize each channel
    normalized_rgb_array = (rgb_array - mean) / std
    
    return normalized_rgb_array


def extract_hyperspectral_data(hyperspectral_tile, bounding_boxes):
    extracted_data = []
    with rasterio.open(hyperspectral_tile) as src:
        for _, row in bounding_boxes.iterrows():
            bounds = row.geometry.bounds
            window = from_bounds(*bounds, src.transform)
            data = src.read(window=window)
            if data.shape[0] >3:
                data = clean_hsi_to_0_255_range(data)
            elif data.shape[0] == 3:
                data = normalize_rgb(data)
            
            extracted_data.append(data)
    return extracted_data

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_data, health_classes):
        self.hyperspectral_data = hyperspectral_data
        self.health_classes = health_classes

    def __len__(self):
        return len(self.hyperspectral_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'data': torch.from_numpy(self.hyperspectral_data[idx]), 'label': self.health_classes[idx]}
        return sample

class MultiModalTransform:
    def __init__(self, rgb_transform=None, hs_transform=None, lidar_transform=None):
        self.rgb_transform = rgb_transform
        self.hs_transform = hs_transform
        self.lidar_transform = lidar_transform

    def __call__(self, rgb_sample, hs_sample, lidar_sample):
        if self.rgb_transform:
            rgb_sample = self.rgb_transform(Image.fromarray(rgb_sample))

        if self.hs_transform:
            hs_sample = self.hs_transform(Image.fromarray(hs_sample))

        if self.lidar_transform:
            lidar_sample = self.lidar_transform(lidar_sample)

        return rgb_sample, hs_sample, lidar_sample
    


class CustomResize:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, rgb, hs, lidar, label):
        # Resize the RGB image
        rgb_resized = cv2.resize(rgb, self.output_size)

        # Resize the Hyperspectral image
        hs_resized = cv2.resize(hs, self.output_size, interpolation=cv2.INTER_LINEAR)

        # Resize the LiDAR image
        lidar_resized = cv2.resize(lidar, self.output_size, interpolation=cv2.INTER_LINEAR)

        return rgb_resized, hs_resized, lidar_resized, label

class MultiModalDataset(Dataset):
    def __init__(self, rgb_data, hs_data, lidar_data, labels, transform=None):
        self.rgb_data = rgb_data
        self.hs_data = hs_data
        self.lidar_data = lidar_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        rgb, hs, lidar, label = self.rgb_data[index], self.hs_data[index], self.lidar_data[index], self.labels[index]
        if self.transform:
            rgb, hs, lidar, label = self.transform(rgb, hs, lidar, label)

        # Convert the data to Float tensors
        rgb = torch.from_numpy(rgb).float()
        hs = torch.from_numpy(hs).float()
        lidar = torch.from_numpy(lidar).float()
        label = torch.tensor(label).long()

        return rgb, hs, lidar, label


# Save the dataset to a pickle file
def save_dataset(dataset, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)

# Load the dataset from a pickle file
def load_dataset(file_name):
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def create_subset(data, indices):
    return [data[i] for i in indices]


class CustomDataset(Dataset):
    def __init__(self, data, labels, label_to_int):
        self.data = data
        self.labels = [label_to_int[label] for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
    

def split_dataset(rgb_data, hs_data, lidar_data, health_classes, stratify_1=None, stratify_2=None, train_ratio=0.7, val_ratio=0.2, seed=42):
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_len = len(health_classes)
    train_val_ratio = train_ratio + val_ratio
    if stratify_2 is None:
        if stratify_1 is None:
            stratify = health_classes
        else:
            stratify = stratify_1
    else:
        stratify = [f"{s}-{g}" for s, g in zip(stratify_1, stratify_2)]

    train_val_indices, test_indices, _, _ = train_test_split(
        range(total_len), stratify, train_size=train_val_ratio, stratify=stratify, random_state=seed
    )

    train_indices, val_indices, = train_test_split(
        train_val_indices, stratify=[stratify[i] for i in train_val_indices], train_size=train_ratio / train_val_ratio, random_state=seed
    )

    train_rgb = create_subset(rgb_data, train_indices)
    val_rgb = create_subset(rgb_data, val_indices)
    test_rgb = create_subset(rgb_data, test_indices)

    train_hs = create_subset(hs_data, train_indices)
    val_hs = create_subset(hs_data, val_indices)
    test_hs = create_subset(hs_data, test_indices)

    train_lidar = create_subset(lidar_data, train_indices)
    val_lidar = create_subset(lidar_data, val_indices)
    test_lidar = create_subset(lidar_data, test_indices)

    train_labels = create_subset(health_classes, train_indices)
    val_labels = create_subset(health_classes, val_indices)
    test_labels = create_subset(health_classes, test_indices)

    return (train_rgb, train_hs, train_lidar, train_labels), (val_rgb, val_hs, val_lidar, val_labels), (test_rgb, test_hs, test_lidar, test_labels)


def __main__(get_clips = True):
    stem_positions = gpd.read_file("Data/geolocations/SERC/field.shp")
    hyperspectral_tile = '/home/smarconi/Documents/DAT_for_health/SERC/SERC/HSI.tif'
    rgb_tile = '/home/smarconi/Documents/DAT_for_health/SERC/SERC/2021_SERC_5_364000_4305000_image.tif'

    # remove points not visible from sky
    stem_positions = stem_positions[stem_positions['Crwnpst'] >2]
    #stem_positions = stem_positions[stem_positions['target_column'].isin(['value_1', 'value_2', 'value_3'])]

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

    # Load the saved dataset
    if 'hyperspectral_data' not in locals():
        hyperspectral_data = load_dataset('tree_health_detection/loaders/hyperspectral_SERC.pkl')
    if 'rgb_data' not in locals():
        rgb_data = load_dataset('tree_health_detection/loaders/rgb_SERC.pkl')

    # Split the dataset
    species = stem_positions.Species # List of species corresponding to each data point
    geography = stem_positions.SiteID # List of geography information corresponding to each data point
    
    #hsi = CustomDataset(hyperspectral_data, health_classes, label_to_int)
    train_data, val_data, test_data = split_dataset(rgb_data, hyperspectral_data, rgb_data, health_classes)

    train_dataset = MultiModalDataset(*train_data) #transform = CustomResize((128, 128))
    val_dataset = MultiModalDataset(*val_data)
    test_dataset = MultiModalDataset(*test_data)
 
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    num_epochs = 50
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #train model
    # Define the model, loss function, and optimizer
    num_bands = train_dataset.hs_data[0].shape[0]
    num_classes = len(set(health_classes))
    #model = SpectralAttentionClassifier(num_bands, num_classes)
    from tree_health_detection.src import multimodal as mti
    from tree_health_detection.src import train_val_test as tvt

    model = mti.MultiModalModel(num_classes, num_bands)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #set up comet experiment
    experiment = Experiment(project_name="treehealth", workspace="marconis")
    # Starting the experiment
    for epoch in range(num_epochs):
        experiment.log_current_epoch(epoch)

        train_loss, train_accuracy = tvt.train(model, train_loader, criterion, optimizer, device, experiment)
        val_loss, val_accuracy, true_labels, predicted_labels = tvt.validate(model, val_loader, criterion, device, experiment)

        # Calculate the confusion matrix for the validation set
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # Log the confusion matrix, accuracy, and loss to Comet.ml
        experiment.log_confusion_matrix(labels=list(range(num_classes)), matrix=conf_matrix, step=epoch, title="Validation Confusion Matrix")
        experiment.log_metric("val_loss", val_loss, step=epoch)
        experiment.log_metric("val_accuracy", val_accuracy, step=epoch)

        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Testing the model
    true_labels, predicted_labels = tvt.test(model, test_loader, device)

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
