# Paths for classification
rgb_tile = 'indir/SERC/rgb_clip.tif'
stem_path = 'indir/SERC/gradient_boosting_alignment.gpkg'
hyperspectral_tile  = 'indir/SERC/hsi_clip.tif'
las_file = 'indir/SERC/LiDAR.laz'
crowns = 'Crowns'
SAM_outputs = '/home/smarconi/Documents/GitHub/tree_mask_delineation/outdir/itcs/backup/'
root_dir = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/data/'
hsi_img  = 'Imagery/SERC/HSI_364000_4305000.tif'
laz_path = 'Imagery/SERC/LAS_364000_4305000.laz'
rgb_path = 'Imagery/SERC/RGB_364000_4305000.tif'
data_path = '/media/smarconi/Gaia/Macrosystem_2/NEON_processed/'
stem_path = 'Stems/SERC_stems_legacy.gpkg'

#paths for segmentation
folder = '../tree_mask_delineation/'
seg_data_path = 'data/'
seg_rgb_path = 'Imagery/SERC/RGB_364000_4305000.tif'
seg_hsi_img  = 'Imagery/SERC/HSI_364000_4305000.tif'
seg_laz_path = 'Imagery/SERC/LAS_364000_4305000.laz'
legacy_polygons = 'dp_SCBI_DP1_747000_4309000_.gpkg'
mode = 'only_points'
point_type = "grid"
siteID = "SERC"
ttops = 'deepforest' # or lidR. For lidR, you need to install it and run the R script separately
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
url = url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

grid_space = 20
grid_size = 6 # meters from which set u the negative point mask
seg_img_batch = 400
seg_img_buffer = 10
neighbors =3
rescale_to = None

# Flags
store_clips = False
get_clips = False
noGUI = True
get_tree_crowns = False
# Flags for segmentation
isrgb = True

# Response variable
response = 'Status'

# Hyperparameters
hsi_out_features = 256
rgb_out_features = 256
lidar_out_features = 256
in_channels = 6
max_points = 2000
num_epochs = 20
batch_size = 32

#comet experiment
comet_name = "treehealth"
comet_workspace = "marconis"
hsi_resize = 20
hsi_shape = 30
rgb_shape = 300