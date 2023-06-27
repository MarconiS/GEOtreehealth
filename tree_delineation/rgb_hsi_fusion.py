import rasterio
import numpy as np
import config
from skimage import color
from skimage import transform, exposure
def pansharpened_hsi_rgb():
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
    hyperspectral_img[hyperspectral_img<0]=0
    hyperspectral_img[hyperspectral_img>10000]=10000

    hyperspectral_img = hyperspectral_img/10000 *255
    # turn nan into 0
    hyperspectral_img = np.nan_to_num(hyperspectral_img)
    rgb_img = rgb_img.astype(float)

    # rescale RGB to match the range of hyperspectral image
    for band in range(rgb_img.shape[0]):
        # calculate the max ignoring nan values
        rgb_img[band] /= np.nan_to_num(rgb_img[band]).max() #rgb_img[band].max()

    # rescale hyperspectral to match the range of 0-1
    for band in range(hyperspectral_img.shape[0]):
        hyperspectral_img[band] /= np.nan_to_num(hyperspectral_img[band]).max() # hyperspectral_img[band].max()

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

    # linstretch each band of fusion to 2-98 percentile
    for band in range(fusion.shape[0]):
        p2 =  np.nan_to_num(fusion[band]).min() # np.min(fusion[band])
        p98 = np.nan_to_num(fusion[band]).max() #np.max(fusion[band])
        fusion[band] = np.interp(fusion[band], (p2, p98), (0, 255))

    #rescale fusion to 1-255 and turn into int8
    #fusion = fusion*255
    fusion = fusion.astype(np.uint8)#    


    # update the hyperspectral profile to match the higher resolution
    hyperspectral_profile.update({
        'height': fusion.shape[1],
        'width': fusion.shape[2],
        'count': fusion.shape[0],
        'transform': rgb_profile['transform']
    })

    # write the output
    with rasterio.open(config.data_path+config.seg_pan_path, 'w', **hyperspectral_profile) as dst:
        dst.write(fusion)