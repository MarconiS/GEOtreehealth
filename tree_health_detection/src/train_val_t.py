import comet_ml
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

def train(model, dataloader, criterion, optimizer, device, experiment):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(dataloader):
        rgb, hs, lidar, labels = data
        rgb, hs, lidar, labels = rgb.to(device), hs.to(device), lidar.to(device), labels.to(device)

        # Log images to Comet.ml
        if batch_idx % 10 == 0:  # Log every 10th batch
            img_dir = "temp_images"
            os.makedirs(img_dir, exist_ok=True)

            for i in range(min(3, rgb.size(0))):  # Log 3 images per batch
                # Save and log RGB images
                save_image(rgb[i].cpu(), f"{img_dir}/rgb_{batch_idx}_{i}.png")
                experiment.log_image(f"{img_dir}/rgb_{batch_idx}_{i}.png", step=batch_idx, image_channels="first")

                # Save and log HSI images
                save_image(hs[i].cpu(), f"{img_dir}/hsi_{batch_idx}_{i}.png")
                experiment.log_image(f"{img_dir}/hsi_{batch_idx}_{i}.png", step=batch_idx, image_channels="first")

                # Save and log LiDAR images
                save_image(lidar[i].cpu(), f"{img_dir}/lidar_{batch_idx}_{i}.png")
                experiment.log_image(f"{img_dir}/lidar_{batch_idx}_{i}.png", step=batch_idx, image_channels="first")

            # Clean up the temporary images directory
            for file in os.listdir(img_dir):
                os.remove(os.path.join(img_dir, file))

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
            rgb, hs, lidar, labels = rgb.to(device), hs.to(device), lidar.to(device), labels.to(device)

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



def test(model, dataloader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in dataloader:
            rgb, hs, lidar, labels = data
            rgb, hs, lidar, labels = rgb.to(device), hs.to(device), lidar.to(device), labels.to(device)

            outputs = model(rgb, hs, lidar)
            _, predicted = outputs.max(1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return true_labels, predicted_labels

