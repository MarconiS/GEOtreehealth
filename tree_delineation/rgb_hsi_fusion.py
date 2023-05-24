import rasterio
import numpy as np
import config
from skimage import color
from skimage import transform

# read hyperspectral image
with rasterio.open(config.data_path+config.hsi_img) as src:
    hyperspectral_img = src.read()
    hyperspectral_profile = src.profile

# read RGB image
with rasterio.open(config.data_path+config.rgb_path) as src:
    rgb_img = src.read()
    rgb_profile = src.profile

# ensure we have float data
hyperspectral_img = hyperspectral_img.astype(float)
rgb_img = rgb_img.astype(float)

# rescale RGB to match the range of hyperspectral image
for band in range(rgb_img.shape[0]):
    rgb_img[band] /= rgb_img[band].max()

# rescale hyperspectral to match the range of 0-1
for band in range(hyperspectral_img.shape[0]):
    hyperspectral_img[band] /= hyperspectral_img[band].max()

# convert the RGB image to grayscale
rgb_gray = color.rgb2gray(np.moveaxis(rgb_img, 0, -1))

hyperspectral_img = hyperspectral_img[[24,65,90],:,:]

# linstrect the hyperspectral_img
for band in range(hyperspectral_img.shape[0]):
    hyperspectral_img[band] = np.interp(hyperspectral_img[band], (hyperspectral_img[band].min(), hyperspectral_img[band].max()), (0, 1))

# normalize the hyperspectral_img
for band in range(hyperspectral_img.shape[0]):
    hyperspectral_img[band] = (hyperspectral_img[band] - hyperspectral_img[band].min()) / (hyperspectral_img[band].max() - hyperspectral_img[band].min())

# convert the hyperspectral image to LAB color space
hyperspectral_lab = color.rgb2lab(np.moveaxis(hyperspectral_img, 0, -1))

# upscale the hyperspectral image to match the high-resolution grayscale image
hyperspectral_lab = transform.resize(hyperspectral_lab, rgb_gray.shape + (3,))

# replace the luminance channel with the high-resolution grayscale image
hyperspectral_lab[:,:,0] = rgb_gray*100 # L channel in LAB is in range 0-100

# convert back to RGB
fusion = color.lab2rgb(hyperspectral_lab)

# move the color channel back to the first dimension
fusion = np.moveaxis(fusion, -1, 0)

# update the hyperspectral profile to match the higher resolution
hyperspectral_profile.update({
    'height': fusion.shape[1],
    'width': fusion.shape[2],
    'count': fusion.shape[0],
    'transform': rgb_profile['transform']
})

# write the output
with rasterio.open(config.data_path+config.seg_rgb_path, 'w', **hyperspectral_profile) as dst:
    dst.write(fusion)
