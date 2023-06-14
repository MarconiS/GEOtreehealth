# Paths for classification
#rgb_tile = 'indir/SERC/rgb_clip.tif'
#stem_path = 'indir/SERC/gradient_boosting_alignment.gpkg'
siteID = "OSBS"
stem_path = "Stems/"+siteID+"_stems_legacy.gpkg"
hyperspectral_tile  = 'indir/'+siteID+'/hsi_clip.tif'
las_file = 'indir/'+siteID+'/LiDAR.laz'
crowns = 'Crowns'
SAM_outputs = '/home/smarconi/Documents/GitHub/tree_mask_delineation/outdir/itcs/backup/'
root_dir = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/data/'
hsi_img  = 'Imagery/'+siteID+'/HSI_ForestGeo.tif'
laz_path = 'Imagery/'+siteID+'/LAS_ForestGeo.laz'
rgb_path = 'Imagery/'+siteID+'/RGB_ForestGeo.tif'
data_path = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/'
stem_path = 'Stems/'+siteID+'_stems_legacy.gpkg'

# configs for alignment
aligned_stems = '/media/smarconi/Gaia/Macrosystem_2/data/'+siteID+'/itcs/data_field.shp'
unaligned_stems = '/media/smarconi/Gaia/Macrosystem_2/data/'+siteID+'/itcs/stems_legacy.gpkg'
outdir = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/Stems/'+siteID

#paths for segmentation
folder = '../tree_mask_delineation/'
seg_data_path = 'data/'
seg_rgb_path = 'Imagery/'+siteID+'/PAN_ForestGeo.tif' 
seg_pan_path = 'Imagery/'+siteID+'/RGB_ForestGeo.tif' 
seg_laz_path = 'Imagery/'+siteID+'/LAS_ForestGeo.laz'
seg_hsi_img = 'Imagery/'+siteID+'/HSI_ForestGeo.tif'
legacy_polygons = 'dp_'+siteID+'_404000_3285000.gpkg'
mode = 'only_points'
point_type = "distance"

ttops = 'deepforest' # or lidR. For lidR, you need to install it and run the R script separately
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
url = url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

detelineation = 'SAM'
grid_space = 100 # pixels

grid_size = 4 # meters from which set u the negative point mask
seg_img_batch = 100
seg_img_buffer = 10
neighbors =4
first_neigh = 1
rescale_to = None
resolution = 0.1

# Flags
clip_each_instance = False
store_clips = False
clean_dataset=False
get_clips = False
noGUI = True
get_tree_crowns = False
# Flags for segmentation
isrgb = True

# Response variable
response = 'Status'

# Hyperparameters
hsi_out_features = 512
rgb_out_features = 512
lidar_out_features = 128
in_channels = 6
max_points = 2000
num_epochs = 8
batch_size = 32

#comet experiment
comet_name = "treehealth"
comet_workspace = "marconis"
hsi_resize = 20
hsi_shape = 23
rgb_shape = 224