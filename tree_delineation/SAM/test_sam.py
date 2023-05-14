from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)

# Read image with Rasterio
with rasterio.open('indir/smaler_clip.tif') as src:
    image = src.read()

# Reorder the bands from (bands, rows, columns) to (rows, columns, bands)
image = np.transpose(image, (1, 2, 0))
# Convert the image array to 8-bit data type if necessary
if image.dtype != np.uint8:
    image = np.uint8(image * 255)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

#other otions
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks2 = mask_generator_2.generate(image)
