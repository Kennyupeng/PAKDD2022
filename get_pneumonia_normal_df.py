# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:09:05 2021

@author: n9784471
"""

import pandas as pd 
from sklearn import preprocessing
import random
from random import randint
import cv2
import numpy as np


from draw_ground_truth_bounding_box import read_xray, plot_imgs
# paths
train_file_path = r'annotations/image_labels_train.csv'
test_file_path = r'annotations/image_labels_test.csv'
train_annotation_path = r'annotations/annotations_train.csv'
test_annotation_path = r'annotations/annotations_test.csv'



test_mode = True

def get_random_pneu_normal(file_path):
    '''
    randomly get pneumonia regardless of the radiologist id 
        1. this function first read the all label image data 
        2. then filter out the global label that we want to retrieve
        eg. If we are getting Pneumonia 
            Then 
                COPD                   0
                Lung tumor	           0
                Pneumonia	           1
                Tuberculosis	       0
                Other diseases	       0
                No finding             0
        3. randomly drop duplicates by using subset image_id
            it will randomly (drop the from second duplicate row that contains 
                              the duplicated image ids)
    return 2 list 
        pneumonia df and normal df
    NOTE: In test set other diseases column name is other disease, manually 
    chaged the column name for test csv
    '''
    
    # read the training data 
    df = pd.read_csv(file_path)
    
    # THE GLOBAL LABELS:
    '''
        COPD	
        Lung tumor	
        Pneumonia	
        Tuberculosis	
        Other diseases	
        No finding
    '''
    
    # filter out the ones that are Pneumonia & No findings only 
    temp_df = df.loc[df['Pneumonia'] == 1]
    temp_df = temp_df.loc[temp_df['No finding'] == 0]
    temp_df = temp_df.loc[temp_df['COPD'] == 0]
    temp_df = temp_df.loc[temp_df['Lung tumor'] == 0]
    temp_df = temp_df.loc[temp_df['Tuberculosis'] == 0]
    pneu_df = temp_df.loc[temp_df['Other diseases'] == 0]
    
    # filter out the ones that are Pneumonia & No findings only 
    temp_df = df.loc[df['No finding'] == 1]
    temp_df = temp_df.loc[temp_df['Pneumonia'] == 0]
    temp_df = temp_df.loc[temp_df['COPD'] == 0]
    temp_df = temp_df.loc[temp_df['Lung tumor'] == 0]
    temp_df = temp_df.loc[temp_df['Tuberculosis'] == 0]
    normal_df = temp_df.loc[temp_df['Other diseases'] == 0]
    
    # drop duplicates
    pneu_df_no_dup = pneu_df.drop_duplicates(subset=['image_id'])
    normal_df_no_dup = normal_df.drop_duplicates(subset=['image_id'])
    
    
    return pneu_df_no_dup, normal_df_no_dup



def get_coordinate_by_file(file_path, image_id):
    
    # read the df 
    df = pd.read_csv(file_path)
    
    # filter the rows that we need
    mask = df['image_id'] == image_id
    this_df = df[mask]
    
    # label the radiologist as they may make different bboxes
    le = preprocessing.LabelEncoder()  # encode rad_id
    this_df['rad_label'] = list(le.fit_transform(this_df['rad_id']))
    this_df['class_id'] = list(le.fit_transform(this_df['class_name']))
    return this_df    





# ============================================================================
if test_mode:   
    
    # get dfs 
    # will use this later to get all the files and upload to google drive 
    pneu_df_train, normal_df_train = get_random_pneu_normal(train_file_path)
    pneu_df_test, normal_df_test = get_random_pneu_normal(test_file_path)
    
    # 
        

    # test 
    test_image_id = '013893a5fa90241c65c3efcdbdd2cec1'    
    test_path = r'013893a5fa90241c65c3efcdbdd2cec1.dicom'
    this_df = get_coordinate_by_file(train_annotation_path, test_image_id)
    img = read_xray(test_path)
    image_size = img.shape
    
    img = np.stack([img, img, img], axis=-1)
    
    # class id 
    class_ids = this_df['class_id'].unique()
    class_name = this_df['class_name'].unique()
    label2color = {class_id:[randint(0,255) for i in range(3)] for class_id in class_ids}
    
    
    img_id =  this_df['image_id'].iloc[0]
    
    boxes = this_df.loc[this_df['image_id'] == img_id, ['x_min', 'y_min', 'x_max', 'y_max']].values
    labels = this_df.loc[this_df['image_id'] == img_id, ['class_id']].values.squeeze()
    names = this_df.loc[this_df['image_id'] == img_id, ['class_name']].values.squeeze()
    
    for label_id, box in zip(labels, boxes):
        color = label2color[label_id]
        img = cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 15
            )
        img = cv2.putText(img, class_name[label_id], 
                          (int(box[0]), int(box[1])-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 4, (36,255,12),
                          10)
    #img = cv2.resize(img, (500,500))
    imgs = []
    imgs.append(img)
    
    plot_imgs(imgs, cmap=None, img_size=image_size )















