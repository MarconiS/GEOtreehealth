from comet_ml import Experiment
from PIL import Image
from rasterio.windows import from_bounds
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch_geometric.data import Data, Batch

import comet_ml
import cv2
import geopandas as gpd
import torch_geometric as geo

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
import torch
import laspy

from tree_health_detection.src import multimodal as mti
from tree_health_detection.src import train_val_t as tvt
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


def extract_pointcloud_data(lidar_file, bounding_boxes):
    extracted_data = []
    src =  laspy.read(lidar_file) 
    # Get point cloud data as structured array
    points = src.xyz.copy()  #.view(np.dtype((np.void, src.points.dtype.itemsize)))

    # Get X, Y, and Z coordinates
    x_coords = src.x
    y_coords = src.y
    z_coords = src.z

    for _, row in bounding_boxes.iterrows():
        bounds = row.geometry.bounds
        min_x, min_y, max_x, max_y = bounds

        # Filter points within the bounding box
        mask = (x_coords >= min_x) & (x_coords <= max_x) & (y_coords >= min_y) & (y_coords <= max_y)
        filtered_points = points[mask]
        extracted_data.append(filtered_points)

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

        return rgb, hs, lidar, label



def custom_collate_fn(batch):
    # Separate the input data
    rgb_batch, hs_batch, lidar_batch, labels_batch = zip(*batch)

    # Convert to tensors and stack RGB and hyperspectral data
    rgb_batch = torch.stack([torch.from_numpy(rgb) for rgb in rgb_batch], dim=0)
    hs_batch = torch.stack([torch.from_numpy(hs) for hs in hs_batch], dim=0)

    # For LiDAR data with varying sizes, use a list instead of stacking
    # No need to use Data class from torch_geometric
    #lidar_batch = torch.stack([torch.tensor(l) for l in lidar_batch], dim=0)

    # TODO for now accept padding. next week figure how to make it graph without padding and breaking things
    max_points = max(l.shape[0] for l in lidar_batch)
    lidar_batch = torch.stack([torch.tensor(np.pad(l, ((0, max_points - l.shape[0]), (0, 0)), mode='constant')) for l in lidar_batch], dim=0)

    # Convert labels to tensors and stack
    labels_batch = torch.stack([torch.tensor(label) for label in labels_batch], dim=0)

    return rgb_batch, hs_batch, lidar_batch, labels_batch




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


def plot_validation_images(images, true_labels, predicted_labels, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    axes = axes.ravel()

    for i in np.arange(0, num_images):
        axes[i].imshow(images[i], cmap=plt.cm.binary)
        axes[i].set_title(
            f"True: {true_labels[i]}, Predicted: {predicted_labels[i]}", fontsize=12
        )
        axes[i].axis("off")

    plt.subplots_adjust(wspace=1)
    return fig
