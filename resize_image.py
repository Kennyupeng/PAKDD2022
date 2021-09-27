# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:10:05 2021

@author: n9784471
"""
import numpy as np
from glob import glob
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
# resize image to 256x256

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

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

error_files = []
files = glob(f'test/PNEUMONIA/*.dicom')    
for file in files[:]:
    try:
        img = read_xray(file)
        re_img = resize(img,256)
        re_img.save('test/Resized/' + file[15:-6] + '.png')
    except:
        print(file)
        error_files.append(file)
        