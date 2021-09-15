# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:46:22 2021

@author: n9784471
"""

from sklearn import preprocessing
import random
from random import randint

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

import matplotlib.pyplot as plt

test_mode = False

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform 
    # raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data
        
    
def plot_img(img, size=(7, 7), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

def plot_imgs(imgs, cols=4, size=7, is_rgb=True, title="", 
                          cmap='gray', img_size=(500,500)):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

def get_bbox_area(row):
    return (row['x_max']-row['x_min'])*(row['y_max']-row['y_min'])


# ============================================================================

if test_mode:
    # test plot for a dummy dicom file 
    test_path = '013893a5fa90241c65c3efcdbdd2cec1.dicom'
    img = read_xray(test_path)
    plt.figure(figsize = (12,12))
    plt.imshow(img, 'gray')
    
    
    # compare 
    IMG_SIZE = 40
    plt.Figure(figsize = (IMG_SIZE,IMG_SIZE))
    plt.subplot(1, 2, 1)
    
    img = read_xray( test_path )
    plt.imshow(img, 'gray')
    
    plt.subplot(1, 2, 2)
    img = read_xray( test_path, fix_monochrome = False)
    plt.imshow(img, 'gray')
    
    plt.tight_layout()






