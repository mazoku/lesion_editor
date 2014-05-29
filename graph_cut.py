# SOMETHING LIKE UNIDENTIFIED PYTHON OBJECT
# ONLY FOR TESTING, SCRIBBLING ETC.

__author__ = 'tomas'


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as scindimor

import skimage.segmentation as skiseg
import skimage.morphology as skimor
import skimage.filter as skifil

import cv2
import pygco

import tools

#-------------------------------------------------------------
def load_data(slice_idx=33):
    # 33 ... hypodense
    # 41 ... small tumor
    # 138 ... hyperdense

    # labels = np.load('label_im.npy')
    data = np.load('input_data.npy')
    o_data = np.load('input_orig_data.npy')
    mask = np.load('mask.npy')

    data_s = data[slice_idx, :, :]
    o_data_s = o_data[slice_idx, :, :]
    mask_s = mask[slice_idx, :, :]

    data_bbox, _ = tools.crop_to_bbox(data_s, mask_s)
    o_data_bbox, _ = tools.crop_to_bbox(o_data_s, mask_s)
    mask_bbox, _ = tools.crop_to_bbox(mask_s, mask_s)

    # plt.figure()
    # plt.subplot(131), plt.imshow(data_bbox, 'gray')
    # plt.subplot(132), plt.imshow(o_data_bbox, 'gray')
    # plt.subplot(133), plt.imshow(mask_bbox, 'gray')
    # plt.show()

    return data_bbox, o_data_bbox, mask_bbox

#-------------------------------------------------------------
def run(slice_idx, sigma):
    data, data_o, mask = load_data(slice_idx)

    data_s = skifil.gaussian_filter(data_o, sigma)

    #imd = np.absolute(im - ims)
    data_d = data_o - data_s

    data_d = np.where(data_d < 0, 0, data_d)


    plt.figure()
    plt.subplot(131), plt.imshow(data_o, 'gray', vmin=0, vmax=255)
    # plt.subplot(132), plt.imshow(imd, 'gray', vmin=0)
    plt.subplot(132), plt.imshow(data_d, 'gray')
    plt.subplot(133), plt.imshow(data_s, 'gray', vmin=0, vmax=255)

    # plt.figure()
    # plt.imshow(ims, 'gray')
    plt.show()



#-------------------------------------------------------------
if __name__ == '__main__':
    # 33 ... hypodense
    # 41 ... small tumor
    # 138 ... hyperdense
    slice_idx = 33

    sigma = 100  # sigma for gaussian blurr

    run(slice_idx, sigma)