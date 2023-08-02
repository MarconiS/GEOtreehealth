import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter


class TreeTopDetector:
    def __init__(self, lidar_data, rgb_data):
        self.lidar_data = lidar_data
        self.rgb_data = rgb_data
        self.tree_points = None
        
    def detect_tree_tops(self, eps=0.3, min_samples=10):
        # apply dbscan clustering on lidar data
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(self.lidar_data)
        labels = dbscan.labels_
        
        # filter tree points
        self.tree_points = self.lidar_data[labels != -1]
        
        return self.tree_points
    
    def filter_by_color(self, low_range, high_range):
        # apply rgb filtering 
        mask = cv2.inRange(self.rgb_data, low_range, high_range)
        rgb_filtered = cv2.bitwise_and(self.rgb_data, self.rgb_data, mask=mask)
        
        return rgb_filtered
    
    def tree_detection_pipeline(self, eps=0.3, min_samples=10, low_range=(25, 50, 20), high_range=(100, 255, 100)):
        self.detect_tree_tops(eps, min_samples)
        rgb_filtered = self.filter_by_color(low_range, high_range)
        
        return self.tree_points, rgb_filtered


def get_tree_tops(lidar_data, rgb_data, eps=0.3, min_samples=10, low_range=(25, 50, 20), high_range=(100, 255, 100)):
    # apply dbscan clustering on lidar data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(lidar_data)
    labels = dbscan.labels_
    
    # filter tree points
    tree_points = lidar_data[labels != -1]
    
    # apply rgb filtering 
    mask = cv2.inRange(rgb_data, low_range, high_range)
    rgb_filtered = cv2.bitwise_and(rgb_data, rgb_data, mask=mask)
    
    return tree_points, rgb_filtered