# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:56:23 2021

@author: n9784471
"""

''' this is how you can read back the hull
from numpy import genfromtxt
my_data = genfromtxt(r'hull_0.csv', delimiter=',')
'''
import cv2
from glob import glob
import numpy as np
from numpy import genfromtxt
from shapely.geometry import Polygon
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd

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


def find_new_coordinate(old_shape,  hull):
    
    new_hull= hull / 256
    

    for coordinate in new_hull:
        coordinate[0] = coordinate[0] * old_shape[0]
        coordinate[1] = coordinate[1] * old_shape[1]
        
    return new_hull



test_file_path = r'annotations/annotations_test.csv'        
files = glob(f'test/PNEUMONIA/*.dicom')    
IOUS = []

for file in files[:]:
    # get image id 
    image_id = file[15:-6]
    
    # find hull path
    hull_path = r'test/BBOX2/' + image_id +'/'
    
    # glob hulls
    hulls = glob(hull_path + '*.csv')
    
    # read old dicom image and get shape
    dicom = read_xray(file)
    
    p_cords = []
    
    for hull in hulls:
    
    # -------------------------------
    # LIME Convex Hull
        
        # get hull from saved csv
        np_hull = genfromtxt(hull, delimiter=',')
        
        if np_hull.shape == (2,) or np_hull.shape == (2,2): 
            # control the hull that only has 1 or 2 value- 
            # cannot draw polygon with 1 or 2 
            print(hull)
        else:
            # resize the 256x256 to the target dicom image shape
            new_hull = find_new_coordinate(dicom.shape, np_hull)
            
            # make each hull tuple 
            p_cord = [tuple(h) for h in new_hull.squeeze()] 
            
            p_cords.append(p_cord)

    
    # -------------------------------
    # Ground Truth
     # read the df 
             
    # read test annotatin file
    df = pd.read_csv(test_file_path)
    
    # filter the rows that we need
    mask = df['image_id'] == image_id
    this_df = df[mask]
    
    q_cords = []
    
    
    for i in range(len(this_df)):

        '''
        xmin,ymax ------xmax,ymax
        |            |
        |            |
        |            |
        xmin,ymin--------xmax,ymin
        
        '''
        xmin = this_df.iloc[i]['x_min']
        xmax = this_df.iloc[i]['x_max']
        ymin = this_df.iloc[i]['y_min']
        ymax = this_df.iloc[i]['y_max']
        
        q_cord = [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)]
        q_cords.append(q_cord)
        


    # -------------------------------
    # Calculate the IoU by considering them polygon
    IoUs = []
    for p_cord in p_cords:
        for q_cord in q_cords:
            
            p = Polygon(p_cord)
            q = Polygon(q_cord)
        
            if p.intersects(q):
                
                IoU = p.intersection(q).area / (p.area + q.area - p.intersection(q).area )
                #print(IoU)
                
                IoUs.append(IoU)
    if len(IoUs) != 0:
        print('='*50)
        print('max IoU')
                  
        print(max(IoUs))
        IOUS.append(max(IoUs))
    print('end of ' + image_id)
    print('='*50)
                
max(IOUS)             
                
                
                