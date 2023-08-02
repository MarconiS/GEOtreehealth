import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
from torchvision import transforms
import config
import numpy as np
import cv2

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

        #obj_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx][self.response]

        hsi_path = os.path.join(self.data.iloc[idx]['hsi_path'][:-4]+ '.npy')
        rgb_path = os.path.join(self.data.iloc[idx]['rgb_path'][:-4]+ '.npy')
        lidar_path = os.path.join(self.data.iloc[idx]['lidar_path'][:-4]+ '.npy')

        hsi = np.load(hsi_path)
        rgb = np.load(rgb_path)
        lidar = np.load(lidar_path)

        # Pad images
        max_shape = [426,config.hsi_shape,config.hsi_shape]
        hsi = self.pad_image(hsi, max_shape)
        max_shape = [3,config.rgb_shape,config.rgb_shape]
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

        # rearrange hsi from x,y, z to z,y,x
        hsi = np.transpose(hsi, (2, 0, 1))

        sample = {'hsi': hsi, 'rgb': rgb, 'lidar': lidar,  'label': label}

        return sample


    def preprocess(self, img_masked):#, mask):
        # Normalize the image
        #    img_masked = hsi

        img_masked[img_masked < 0] = 0
        img_masked[img_masked > 10000] = 10000
        # remove nan from the image by turning it to 0s
        img_masked = np.nan_to_num(img_masked)

        # remove bands that are water absorption bands [0:14,190:219, 274:320, 399:425]
        bad_bands = np.concatenate([np.arange(0,14), np.arange(190,219), np.arange(274,320), np.arange(399,426)])  
        img_masked = np.delete(img_masked, bad_bands, axis=0)
        # Assuming `hyperspectral_data` is your hyperspectral data with shape (channels, height, width)
        #img_masked = img_masked / np.linalg.norm(img_masked, axis=0)

        img_masked = self.normalize_hsi(img_masked)

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
        # img_rgb = rgb
        img_rgb[img_rgb < 0] = 0
        img_rgb[img_rgb > 255] = 255
        img_rgb = np.nan_to_num(img_rgb)

        # Step 1: Check if there are non-zero pixels outside the 224x224 top-left corner
        non_zero_pixels = np.any(img_rgb[1,224:, :] != 0) or np.any(img_rgb[1,:, 224:] != 0)

        # Step 2: If there are, resize the image so that all valid pixels fit within the 224x224 boundary
        if non_zero_pixels:
            # Find the largest non-zero pixel coordinate in the x and y directions
            x_max, y_max = np.where(img_rgb[1,:,:] != 0)
            max_coord = max(x_max.max(), y_max.max())

            # Calculate the scaling factor required to fit these pixels into a 224x224 image
            scale = 224 / (max_coord + 1)

            # Resize the image, preserving all non-zero pixels
            #img_rgb = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Step 3: Crop the top-left corner of the resized image to the required size
        img_rgb = img_rgb[:, :224, :224]

        # normalize the [3,224,224] over each channel (dim 0) using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        img_rgb = img_rgb / 255.0
        img_rgb[0,:,:] = (img_rgb[0,:,:] - 0.485) / 0.229
        img_rgb[1,:,:] = (img_rgb[1,:,:] - 0.456) / 0.224
        img_rgb[2,:,:] = (img_rgb[2,:,:] - 0.406) / 0.225

        return img_rgb
    
    # data = img_masked
    def normalize_hsi(self, data):
        # calculate the l2 norm along the 0-th dimension
        l2_norms = np.linalg.norm(data, axis=0)
        # Transpose data to have shape (height, width, channels)
        data = np.transpose(data, (1, 2, 0))
        '''
        # add a small constant to avoid division by zero
        epsilon = 1e-8
        # normalize
        data = data / (l2_norms + epsilon)
        # create a mask for values bigger than 0
        mask = data > 0

        # find the min value bigger than 0 for each band
        min_vals = np.min(np.where(mask, data, np.inf), axis=(1, 2), keepdims=True)
        max_vals = np.max(data, axis=(1, 2), keepdims=True)

        # apply min-max scaling
        x_scaled = (data - min_vals) / (max_vals - min_vals + epsilon)

        mask_zero = data == 0
        # set the values back to 0 where the initial data was 0
        x_scaled = np.where(mask_zero, 0, x_scaled)
        '''
        return data

