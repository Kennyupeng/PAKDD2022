# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 12:36:57 2021

@author: n9784471
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage
from glob import glob
    

files = glob(f'test/Resized_P/*.png')    

for file in files[:]:
    image_id = file[15:-4]
    PATH = r'test/BBOX2/' + image_id +'/'
    # read image
    img = cv2.imread(PATH + 'LIME_.png')
    
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # threshold
    thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)[1]
    
    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        img = cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0,0 ), 2)
        print("x,y,w,h:",x,y,w,h)
     
    image = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    skimage.io.imsave(PATH + 'contour_bbox.png',image)
    # skimage.io.imsave('test/BBOX2/' + image_id + '/LIME_.png',perturb_image(image,mask,superpixels) )
    # show the image with the drawn contours
    plt.imshow(image)
    plt.show()
    
    # read image
    img = cv2.imread(PATH + 'LIME_.png')
    
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # threshold
    thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)[1]
    
    hulls = [cv2.convexHull(c) for c in contours]
    final = cv2.drawContours(img, hulls, -1, (255,0,0) )
    plt.imshow(img)
    plt.show()
    skimage.io.imsave(PATH + 'polygon.png',final)
    
    counter = 0
    for hull in hulls:
        
        np_hull = np.array(hull)
        np.savetxt(PATH + "hull_"+str(counter)+".csv", np_hull.squeeze(), delimiter=",")
        counter += 1
    
''' this is how you can read back the hull
from numpy import genfromtxt
my_data = genfromtxt(r'hull_0.csv', delimiter=',')
'''
