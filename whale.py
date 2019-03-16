# -*- coding:utf-8 -*-
import os
import sys
import time
import pickle
import random
import numpy as np
from skimage import io
import cv2

class_num = 5005
image_size = 96
img_channels = 3
train_sample_num = 213604
test_sample_num = 7960

#-----------------------------------------------------------------------------
#load image and label
#-----------------------------------------------------------------------------

def loadTrainData(image_dir):
    print("load train data.....")
    train_csv = os.path.join(image_dir,"train_aug_2.csv")
    f = open(train_csv,"r")
    lines = f.readlines()
    f.close()
    #np.random.shuffle(lines)
    images = []
    labels = []
    labels_all = []
    labels_id = []
    for i,line in enumerate(lines):
        if(i==0): continue
        temp = line.strip().split(",")
        image_name = temp[0]
        image_name = os.path.join(image_dir,"train_aug_2",image_name)
        if(not os.path.exists(image_name)): continue
        label=temp[1]
        images.append(image_name)
        labels.append(label)
        
    for label in labels:
        if(label not in labels_all):
            labels_all.append(label)
    for label in labels:
        labels_id.append(labels_all.index(label))
    cls_txt = open(os.path.join(image_dir,"cls.txt"),"w")
    for cls in labels_all:
        print(cls,file=cls_txt)
    cls_txt.close()
    
    return images,labels_id

def loadTestData(image_dir):
    test_csv = os.path.join(image_dir,"test.csv")
    f = open(test_csv,"r")
    lines = f.readlines()
    #np.random.shuffle(lines)
    images = []
    for i,line in enumerate(lines):
        #if(i==0): continue
        temp = line.strip().split(",")
        image_name = temp[0]
        image_name = os.path.join(image_dir,"test",image_name)
        images.append(image_name)
    return images


def loadTrainDataBatch(images,labels,batch,iters):
    #iters from 0 to ,,,,
    if((iters+1)*batch<train_sample_num):
        batch_images = images[iters*batch:(iters+1)*batch]
        batch_labels = labels[iters*batch:(iters+1)*batch]
    else:
        batch_images = images[iters*batch:]
        batch_labels = labels[iters*batch:]
    Imgs = []
    Labels = []
    for id,image_name in enumerate(batch_images):
        label = [0]*5005
        label_id = batch_labels[id]
        label[label_id]=1
        Labels.append(label)
        image = io.imread(image_name)
        if(image.ndim==2):
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=-1)
        re_image = cv2.resize(image,(96,96))
        Imgs.append(re_image)
    print("p6")
    Imgs = np.array(Imgs)
    Labels = np.array(Labels)
    print(Imgs.shape,Labels.shape)
    return Imgs,Labels

def loadTestDataBatch(images,batch,iters):
    #iters from 0 to ,,,,
    if((iters+1)*batch<test_sample_num):
        batch_images = images[iters*batch:(iters+1)*batch]
    else:
        batch_images = images[iters*batch:]
    Imgs = []
    only_image_name = []
    for id,image_name in enumerate(batch_images):
        only_image_name.append(os.path.split(image_name)[1])
        image = io.imread(image_name)
        if(image.ndim==2):
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=-1)
        re_image = cv2.resize(image,(96,96))
        Imgs.append(re_image)
    Imgs = np.array(Imgs)
    return Imgs,only_image_name




        








