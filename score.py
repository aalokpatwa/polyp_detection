'''
score.py calculates multiple scores for images in the test dataset with trained model
'''

import numpy as np
import os
import cv2
import pandas as pd

basedir = "/unet_polyp/"
dataset = basedir+"datasets/cvc612/"
pred = "prpolyp_768drop/"
gtpath = dataset+"gtpolyp/"
prpath = dataset+pred

def jaccard_idx(y_true, y_pred):
    intersection = np.sum(y_true*y_pred)
    sumtotal = np.sum(y_true + y_pred)
    smooth = 1
    jac = (intersection + smooth)/(sumtotal-intersection+smooth)
    return np.mean(jac)

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true*y_pred)
    sumtotal = np.sum(y_true + y_pred)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    return np.mean(dice)

def dice_loss(y_true, y_pred):
    intersection = np.sum(y_true*y_pred)
    sumtotal = np.sum(y_true + y_pred)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice
    return np.mean(diceloss)

def intersect(y_true, y_pred):
    intersection = np.sum(y_true*y_pred)
    return np.mean(intersection)
def falsepos(y_true, y_pred):
    allone = np.ones_like(y_true)
    fp = np.sum((allone-y_true)*y_pred)
    return np.mean(fp)

results = []

outfile = open("scores.csv", 'w')
fpcount = 0
tpcount = 0
fncount = 0
for mask in os.listdir(gtpath):
    y_true = cv2.imread(gtpath+mask, cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.imread(prpath+mask, cv2.IMREAD_GRAYSCALE)
    y_true = y_true/255.0
    y_pred = y_pred/255.0
    #print("image:", mask, y_true.shape, y_pred.shape, y_true.dtype, y_pred.dtype)
    #print("max", np.max(y_true), np.max(y_pred), "min", np.min(y_true), np.min(y_pred))
    dice = dice_score(y_true, y_pred)
    jacc = jaccard_idx(y_true, y_pred)
    tp = intersect(y_true, y_pred)
    fp = falsepos(y_true, y_pred)
    thresh = 100
    
    if tp > thresh:
        tpcount += 1
        if (tp < 225):
            print("tp", mask)
    else:
        fncount += 1
        #print("fn", mask)
        if fp > thresh:
            fpcount += 1
            #print ("fp", mask)
    outfile.write(mask+ ','+str(dice)+','+str(jacc)+','+str(tp)+','+str(fp)+'\n')
print ("tpCount: ", tpcount, "fpCount", fpcount, "fnCount", fncount)
#df = pd.DataFrame(data=results)
#print(df['dice_score'].mean)
