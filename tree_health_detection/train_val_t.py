import comet_ml
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
#import matplotlib.pyplot as plt

def train(model, dataloader, criterion, optimizer, device, experiment):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (rgb, hs, lidar, labels) in enumerate(dataloader):
        # Pass the LiDAR data to the model one at a time
        rgb, hs, labels = rgb.float().to(device), hs.float().to(device), labels.to(device)
        
        # Convert lidar data to PyTorch tensors and move to device
        lidar = [l.float().to(device) for l in lidar]
        rgb, hs, labels = rgb.to(device), hs.to(device), labels.to(device)

        # Train the model
        optimizer.zero_grad()
        outputs = model(rgb, hs, lidar)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy



def validate(model, dataloader, criterion, device, experiment):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in dataloader:
            rgb, hs, lidar, labels = data
            rgb, hs, labels =  rgb.float().to(device), hs.float().to(device), labels.to(device)
            lidar = [l.float().to(device) for l in lidar]

            outputs = model(rgb, hs, lidar)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy, true_labels, predicted_labels



def test(model, dataloader, device, experiment, noGUI = True, bands_to_plot = [18,56,135]):
    #make sure you can plot tests
    if noGUI:
        import matplotlib
        matplotlib.use('Agg')
        
    import matplotlib.pyplot as plt
    model.eval()
    true_labels = []
    predicted_labels = []
    logged_images = 0

    with torch.no_grad():
        for data in dataloader:
            rgb, hs, lidar, labels = data
            rgb, hs, labels = rgb.float().to(device), hs.float().to(device), labels.to(device)
            lidar = [l.float().to(device) for l in lidar]

            outputs = model(rgb, hs, lidar)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Log images and predictions
            sampling_rate =5
            if logged_images % sampling_rate == 0:
                for i in range(len(rgb)):
                    # Convert tensors to numpy arrays and transpose the axes
                    rgb_np = rgb[i].cpu().numpy().transpose(1, 2, 0)
                    # TODO retro
                    hs_np = hs[i].cpu().numpy()[bands_to_plot]

                    # Normalize to [0, 1] range for floats
                    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())

                    # sepect 3 bands and normalize hs
                    hs_np = (hs_np - hs_np.min()) / (hs_np.max() - hs_np.min())

                    # Plot and log images
                    fig = tree_health_detection.plot_validation_images([rgb_np, hs_np], [f"True: {true_labels[-1]}, Predicted: {predicted_labels[-1]}"] * 2, noGUI, cmap="gray" if hs_np.ndim == 2 else None)
                    experiment.log_figure("Test Images", fig)
                    plt.close(fig)

    return true_labels, predicted_labels
