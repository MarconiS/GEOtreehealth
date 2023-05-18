import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, data, response, max_points):
        self.data = data
        self.response = response
        self.max_points = max_points

    def __len__(self):
        return len(self.data)
    
    def pad_image(self, image, target_shape):
        pad_dims = [(0, target_shape[i] - image.shape[i]) for i in range(image.ndim)]
        return np.pad(image, pad_dims, mode='constant')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obj_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx][self.response]

        hsi_path = os.path.join(self.data.iloc[idx]['hsi_path'][:-4]+ '.npy')
        rgb_path = os.path.join(self.data.iloc[idx]['rgb_path'][:-4]+ '.npy')
        lidar_path = os.path.join(self.data.iloc[idx]['lidar_path'][:-4]+ '.npy')

        hsi = np.load(hsi_path)
        rgb = np.load(rgb_path)
        lidar = np.load(lidar_path)

        # Pad images
        max_shape = [426,40,40]
        hsi = self.pad_image(hsi, max_shape)
        max_shape = [3,400,400]
        rgb = self.pad_image(rgb, max_shape)

        # Ensure lidar data has a fixed size
        if lidar.shape[0] > self.max_points :
            # Subsample lidar data
            indices = np.random.choice(lidar.shape[0], self.max_points , replace=False)
            lidar = lidar[indices]
        elif lidar.shape[0] < self.max_points :
            # Pad lidar data
            padding = np.zeros((self.max_points  - lidar.shape[0], lidar.shape[1]))
            lidar = np.concatenate([lidar, padding], axis=0)


        hsi = self.preprocess(hsi) #, mask = None) #, np.load(mask_hsi_path))
        rgb = self.preprocess_rgb(rgb)#, mask = None) #, np.load(mask_rgb_path))
        lidar = self.normalize_point_cloud(lidar)
        
        sample = {'hsi': hsi, 'rgb': rgb, 'lidar': lidar,  'label': label}

        return sample


    def preprocess(self, img_masked):#, mask):
        # Normalize the image
        #if mask is not None:
        #    img_masked = img[mask > 0]  # Assuming mask has values of 0 and 1
        #else:
        #    img_masked = img

        img_masked[img_masked < 0] = 0
        img_masked[img_masked > 10000] = 10000

        # remove bands that are water absorption bands [0:14,190:219, 274:320, 399:425]
        bad_bands = np.concatenate([np.arange(0,14), np.arange(190,219), np.arange(274,320), np.arange(399,425)])  
        img_masked = np.delete(img_masked, bad_bands, axis=0)

        # Calculate the L2 norm for each row
        l2_norms = np.linalg.norm(img_masked, axis=0)

        # Normalize each row
        img_masked = img_masked / l2_norms[None,:,:]
        img_masked[np.isnan(img_masked)] = 0
        img_masked = (img_masked - img_masked.min()) / (img_masked.max() - img_masked.min()) 

        # Create PIL image from numpy array
        img_masked = img_masked.astype('float32')

        # create a 2 pixels wide padding only in dimensions 1 and 2 (height and width)
        img_masked = np.pad(img_masked, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)

        return img_masked
    
    # points = lidar.copy()
    def normalize_point_cloud(self, points):        
        # Calculate the centroid of the point cloud
        centroid = np.mean(points, axis=0)
        # Translate points to move centroid to the origin
        points -= centroid
        
        # Scale points to fit within a unit sphere
        #furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
        #points /= furthest_distance
        #print the summary statistics of the normalized point cloud
        #turn the normalized points into a dataframe
        #pd.DataFrame(points).describe()
        
        return points

    def preprocess_rgb(self, img_rgb):
        # Normalize the image
        img_rgb[img_rgb < 0] = 0
        img_rgb[img_rgb > 255] = 255

        # Create PIL image from numpy array
        img_rgb = Image.fromarray(img_rgb.astype('uint8'), 'RGB')

        # Resize the image to the expected model size and convert to tensor
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_rgb = transform(img_rgb)

        return img_rgb

