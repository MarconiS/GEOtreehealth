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

import tree_delineation
import tree_health_detection
import field_data_alignment

import torch
import torch.nn as nn
import torch.optim as optim

rgb_tile = 'indir/SERC/rgb_clip.tif'
stem_path = 'indir/SERC/gradient_boosting_alignment.gpkg'
hyperspectral_tile  = 'indir/SERC/hsi_clip.tif'
las_file = 'indir/SERC/LiDAR.laz'
crowns = 'Crowns/bboxesSERC_0.gpkg'
SAM_outputs = '/home/smarconi/Documents/GitHub/tree_mask_delineation/outdir/itcs/backup/'
root_dir = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/data/'
# Paths to your datasets
hsi_img  = 'Imagery/SERC/HSI_364000_4305000.tif'
laz_path = 'Imagery/SERC/LAS_364000_4305000.laz'
rgb_path = 'Imagery/SERC/RGB_364000_4305000.tif'
data_path = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/'
stem_path = 'Stems/SERC_stems_legacy.gpkg'
store_clips = False
response='Status'
hsi_out_features = 512
rgb_out_features = 1024
lidar_out_features = 256
in_channels = 6

import numpy as np

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


def __main__(get_clips = False, noGUI = True):
    if noGUI:
        import matplotlib
        matplotlib.use('Agg')
        
    #get the crowns that match with stem_path
    if crowns is not None:
        crowns = gpd.read_file(crowns)
    else:
        #loop through geopackages in custom folder and append polygons in a unique geodataframe
        crowns = gpd.GeoDataFrame()
        for file in os.listdir(SAM_outputs):
            if file.endswith('.gpkg'):
                # skip file if empty
                if os.stat(SAM_outputs + file).st_size == 0:
                    continue
  
                crowns = pd.concat([crowns, gpd.read_file(SAM_outputs + file)],ignore_index=True)
                crowns.to_file('indir/SERC/itcs.gpkg') 
    
    stem_positions = gpd.read_file(root_dir+stem_path)
    if 'StemTag' not in crowns.columns:
        # left join crowns and stem_positions by StemTag, assigning a polygon to each stem
        stem_positions = gpd.sjoin(stem_positions, crowns, how="left", op='within')

    # if data storage required
    if store_clips:
        rgb_dir = root_dir / "rgb"
        hsi_dir = root_dir / "hsi"
        png_dir = root_dir / "png"

        lidar_dir = root_dir / "lidar"
        polygon_mask_dir = root_dir / "polygon_mask"
        labels_dir = root_dir / "labels"

        # Ensure that all directories exist
        for dir in [root_dir, rgb_dir, hsi_dir, lidar_dir, polygon_mask_dir, labels_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # Load your polygon data (GeoDataFrame)
        gdf = gpd.read_file(data_path+crowns)
        itcs = gpd.read_file( data_path+stem_path)

        # select only the columns needed and turn into dataframe
        itcs = itcs[['StemTag','Crwnpst', 'SiteID','Species','DBH', 'Status', 'FAD']]
        itcs = pd.DataFrame(itcs)

        # Process each polygon
        for idx, row in gdf.iterrows():
            tree_health_detection.store_data_structures.process_polygon(polygon = row, root_dir = root_dir,  rgb_path = data_path+rgb_path, 
                            hsi_path = data_path+hsi_img, 
                            lidar_path = data_path+laz_path, polygon_id=  idx, itcs=itcs)



    import matplotlib
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

    import tree_delineation
    import tree_health_detection
    import field_data_alignment

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from comet_ml import Experiment
    from sklearn.metrics import confusion_matrix
    from tempfile import NamedTemporaryFile
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import numpy as np

    def linStretch(input_img):
        img_min = input_img.min()
        img_max = input_img.max()
        return (input_img - img_min) / (img_max - img_min)

    rgb_tile = 'indir/SERC/rgb_clip.tif'
    stem_path = 'indir/SERC/gradient_boosting_alignment.gpkg'
    hyperspectral_tile  = 'indir/SERC/hsi_clip.tif'
    las_file = 'indir/SERC/LiDAR.laz'
    crowns = 'Crowns/bboxesSERC_0.gpkg'
    SAM_outputs = '/home/smarconi/Documents/GitHub/tree_mask_delineation/outdir/itcs/backup/'
    root_dir = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/data/'
    # Paths to your datasets
    hsi_img  = 'Imagery/SERC/HSI_364000_4305000.tif'
    laz_path = 'Imagery/SERC/LAS_364000_4305000.laz'
    rgb_path = 'Imagery/SERC/RGB_364000_4305000.tif'
    data_path = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/'
    stem_path = 'Stems/SERC_stems_legacy.gpkg'
    store_clips = False
    response='Status'
    hsi_out_features = 512
    rgb_out_features = 1024
    lidar_out_features = 256
    in_channels = 6
    max_points = 3000
    num_epochs = 20
    batch_size = 32
    ### move to main script
    # Read the csv file
    # list all files in folder data_pt
    data_pt = root_dir +"labels/"

    files = os.listdir(data_pt)
    # llop through all files and append to a list
    labels = []
    for file in files:
        if file.endswith('.csv'):
            labels.append(pd.read_csv(data_pt + file))
    # concatenate all dataframes in the list
    labels_df = pd.concat(labels, ignore_index=True)

    # count the number of unique labels
    rebalanced_to = labels_df[response].value_counts().min()

    # downsample more frequent classes to balance dataset
    labels_df = labels_df.groupby(response).apply(lambda x: x.sample(rebalanced_to)).reset_index(drop=True)

    # if labels_df['response'] is character, convert it into a numeric class
    if labels_df[response].dtype == 'object':
        labels_df[response] = pd.factorize(labels_df[response])[0]

    labels_df = labels_df.reset_index(drop=True)
    # Split the data into train-test-validation (70%-15%-15%) with stratification
    train_data, temp_data = train_test_split(labels_df, test_size=0.3, random_state=42, stratify=labels_df[response])
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data[response])

    # from train_dataset, get the max size of the pointclod
 

    #reset index of train, validation, test
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)


    # self = MultiModalDataset(train_data)
    train_dataset = tree_health_detection.MultiModalDataset(train_data, response=response, max_points=max_points)
    test_dataset = tree_health_detection.MultiModalDataset(test_data, response=response, max_points=max_points)
    val_dataset = tree_health_detection.MultiModalDataset(val_data, response=response, max_points=max_points)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #train model
    # Define the model, loss function, and optimizer
    # from train_dataset, get the number of bands in a hsi image
    num_bands = train_dataset[0]['hsi'].shape[0]
    
    # from train_dataset, get the number of classes
    num_classes = len(np.unique(train_data[response]))
 
    model = tree_health_detection.MultiModalNet(in_channels, num_classes, num_bands,
                 hsi_out_features, rgb_out_features, lidar_out_features)
    
    model.to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create an experiment with your api key
    #set up comet experiment
    experiment = Experiment(project_name="treehealth", workspace="marconis")
    # Starting the experiment

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0
        for batch in train_loader:
            hsi = batch['hsi'].float().to(device)
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
        model.eval()
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
    resize = transforms.Resize((40, 40))

    # Test Set Evaluation
    model.eval()
    test_preds = []
    test_labels = []
    matplotlib.use('Agg')
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
            hsi_img = linStretch(hsi_img)
            ax[1].imshow(hsi_img)
            ax[1].set_title(f'Prd: {test_preds[i]}' f', Obs: {test_labels[i]}')
            ax[1].axis('off')

            # Save figure to a temporary file
            with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                fig.savefig(temp_file.name, format='png')

            # Log Image to Comet.ml
            experiment.log_image(temp_file.name, image_format='png')


        # Calculate test accuracy
        test_accuracy = 100 * correct / total

    # Log test accuracy
    experiment.log_metric('test_accuracy', test_accuracy)

    # Ending the experiment
    experiment.end()

    return(model)

