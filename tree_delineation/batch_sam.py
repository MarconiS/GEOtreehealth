import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

from transformers import (
    SamVisionConfig,
    SamPromptEncoderConfig,
    SamMaskDecoderConfig,
    SamModel,
)

#from tree_health_detection.src.get_itcs_polygons import mask_to_delineation
from skimage import exposure
batch = np.transpose(batch, (2, 1, 0))
# change x with y
batch = np.flip(batch, axis=1) 
batch = np.flip(batch, axis=0) 
original_shape = batch.shape
batch = exposure.rescale_intensity(batch, out_range=(0, 255))
batch =  np.array(batch, dtype=np.uint8)
image = Image.fromarray(batch, 'RGB')

itcs_stems = input_points.loc[:,['x','y']] / resolution

# divide the image in 400x400 pixels batches, and extract all the boxes contained in each batch
img_clip_resize = 1200


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# clip the batch image into subimages of 1200x1200 pixels 

for i in range(0, original_shape[0], img_clip_resize):
    for j in range(0, original_shape[1], img_clip_resize):
        clipped_image = (image.crop((i, j, i+img_clip_resize, j+img_clip_resize)))
        # get the boxes contained in the clipped image
        clipped_boxes = itcs_boxes.loc[(itcs_boxes['xmin'] >= i) & (itcs_boxes['xmax'] <= i+img_clip_resize) & 
                                       (itcs_boxes['ymin'] >= j) & (itcs_boxes['ymax'] <= j+img_clip_resize),:]
        clipped_image.save('test.jpg')

        # convert the boxes to the format expected by the model
        input_boxes = clipped_boxes.loc[:,['xmin','ymin','xmax','ymax']]
        #shift boxes coorinates to clipped image origin
        input_boxes['xmin'] = input_boxes['xmin'] - i
        input_boxes['xmax'] = input_boxes['xmax'] - i
        input_boxes['ymin'] = input_boxes['ymin'] - j
        input_boxes['ymax'] = input_boxes['ymax'] - j
        input_boxes = input_boxes.to_numpy()

        input_boxes = input_boxes.tolist()
        inputs = processor(clipped_image, input_boxes=[input_boxes], return_tensors="pt").to(device)
        image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores

        # for all masks, turn the mask into a polygon
        itcs_polygons = []
        for msk in masks[0]:
            #make mask a numpy
            msk = msk.cpu().numpy()
            area = np.sum(msk)
            if area == 0:
                continue

            polygons = mask_to_delineation(mask = msk, center = None)
            if len(polygons) == 0:
                continue

            polygons = [translate(poly, xoff=j - buffer, yoff=i - buffer) for poly in polygons]
            polygons = [translate(poly, xoff=affine[2], yoff=affine[5] + (batch.shape[1]*affine[4])) for poly in polygons]
            itcs_polygons.append(polygons)
































new_width, new_height = original_shape[0] // img_clip_resize, original_shape[1] // img_clip_resize
image = image.resize((new_width, new_height))

#write the image to disk
image.save('test.jpg')

#turn array into RGB image
# convert the boxes to the format expected by the model
input_boxes = itcs_boxes.loc[:,['xmin','ymin','xmax','ymax']]
input_boxes = input_boxes.iloc[0:100,:]
input_boxes = input_boxes.to_numpy()

# rescale transformed_boxes_batch x and y using the scale factor
scale_factor_x = new_width / original_shape[0]
scale_factor_y = new_height / original_shape[1]

input_boxes[:,0] = input_boxes[:,0] * scale_factor_x
input_boxes[:,1] = input_boxes[:,1] * scale_factor_y
input_boxes[:,2] = input_boxes[:,2] * scale_factor_x
input_boxes[:,3] = input_boxes[:,3] * scale_factor_y
input_boxes = input_boxes.tolist()
inputs = processor(image, input_boxes=[input_boxes], return_tensors="pt").to(device)
image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = model(**inputs, multimask_output=False)

masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores
# for all masks, turn the mask into a polygon
itcs_polygons = []       
for mask in masks[0]:
    #make mask a numpy
    mask = mask.cpu().numpy()
    area = np.sum(mask)
    if area == 0:
        continue

    polygons = mask_to_delineation(mask = mask, center = None)
    if len(polygons) == 0:
        continue

    polygons = [translate(poly, xoff=j - buffer, yoff=i - buffer) for poly in polygons]
    polygons = [translate(poly, xoff=affine[2], yoff=affine[5] + (batch.shape[1]*affine[4])) for poly in polygons]
    
    gdf_temp = gpd.GeoDataFrame(geometry=polygons, columns=["geometry"])


gdf_temp.to_file("temp.gpkg", driver="GPKG")
crown_mask = pd.concat([crown_mask, gdf_temp], ignore_index=True)

# Convert the DataFrame to a GeoDataFrame
predictions = gpd.GeoDataFrame(crown_mask, geometry=crown_mask.geometry)
predictions.to_file("temp.gpkg", driver="GPKG")

transformed_boxes_batch = torch.tensor(input_boxes, device=predictor.device)
transformed_boxes_batch = predictor.transform.apply_boxes_torch(transformed_boxes_batch, batch.shape[:2])
masks,scores, logits = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes_batch,
    multimask_output=False,
)
#free space from GPU
masks = masks.cpu().numpy()
scores = scores.cpu().numpy()
logits = logits.cpu().numpy()
torch.cuda.empty_cache()    
crown_scores.append(scores)
crown_logits.append(logits)
crown_masks.append(masks)










device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

inputs = processor(raw_image, input_boxes=[input_boxes], return_tensors="pt").to(device)

image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

