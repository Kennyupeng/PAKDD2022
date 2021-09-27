# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:31:52 2021

@author: n9784471
"""
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings
print('Notebook running: keras ', keras.__version__)
np.random.seed(222)

# models will be trained from another notebook and saved in my google drive
# load classifier
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

PATH = r'models'
METRICS = ['accuracy' , tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

IMAGE_SIZE = 256

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

def load_model_history( model_name, path):
    model_hist_loaded = {}
    values = []

    # load dictionary
    r = open( path + model_name + "_hist.csv", "r").read()
    for line in r.split("\n"):
        if(len(line) == 0):
            continue
  
        metric = line.split(",\"[")[0]                                    # extract metrics
        values_str = line.split(",\"[")[1].replace("]\"","").split(", ")  # extract validation values
        values = [float(val_str) for val_str in values_str]
        model_hist_loaded.update( {metric : values} )
    
    return model_hist_loaded

def load_model( model_name, path ):
    json_file = open( path + model_name +  "_DUO.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # load weights into new model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + model_name +  "_DUO.h5")
    print("Loaded model from disk")
    
    return loaded_model

def perturb_image(img,perturbation,segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
      
    for active in active_pixels:
        mask[segments == active] = 1 
     
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    return perturbed_image


def get_feature_num(file_path, image_id):
    
    # read the df 
    df = pd.read_csv(file_path)
    
    # filter the rows that we need
    mask = df['image_id'] == image_id
    this_df = df[mask]
    
    return len(this_df)

from keras.models import model_from_json
model_history = load_model_history('/final_hist_DenseNet_'+str(IMAGE_SIZE), PATH)
model_dense = load_model('/final_DenseNet_'+str(IMAGE_SIZE), PATH)
model_dense.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
model_dense.summary()


# ------------ ------------ ------------ ------------ ------------ -----------
from glob import glob
    
np.random.seed(222)
files = glob(f'test/Resized_P/*.png')    
file_path = r'annotations/annotations_test.csv'

for file in files[:]:
    image_id = file[15:-4]
    # change the file path and plot it
    image = Image.open(file).convert("RGB")
    image = image.resize((256, 256)) 

    image = np.array(image) / 255.
    image_raw = image[ np.newaxis, ... ]
    Xi = image_raw

    preds = model_dense.predict(Xi)
    preds[0][0] 


    # generate the superpixels for cxr
    superpixels = skimage.segmentation.quickshift(image, kernel_size=3,max_dist=200, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]
    num_superpixels


    skimage.io.imshow(skimage.segmentation.mark_boundaries(image, superpixels))
    num_perturb = 150
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
    perturbations[0] #Show example of perturbation


    predictions = []
    for pert in perturbations:
      perturbed_img = perturb_image(image,pert,superpixels)
      pred = model_dense.predict(perturbed_img[ np.newaxis, ... ])
      predictions.append(pred)
    
    predictions = np.array(predictions)
    predictions.shape

    original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
    distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
    distances.shape



    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    weights.shape

    top_pred_classes = preds[0].argsort()[-5:][::-1]
    top_pred_classes 


    class_to_explain = top_pred_classes[0]
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_[0]
    coeff

    num_top_features = get_feature_num(file_path, image_id)
    top_features = np.argsort(coeff)[-num_top_features:] 
    top_features


    mask = np.zeros(num_superpixels) 
    mask[top_features]= True #Activate top superpixels
    perturb_image(image,mask,superpixels).shape
    #skimage.io.imshow(perturb_image(image,mask,superpixels) )
    skimage.io.imsave('test/BBOX2/' + image_id + '/LIME_.png',perturb_image(image,mask,superpixels) )
