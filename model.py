'''
model.py has the unet model that is used by main.py
loss and evaluation metric is defined here as well
'''

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

# paper did not use padding=same.
# paper used weighted binary cross-entropy loss, with weights being calculated
# paper required all 2x2 max-pooing operations to be applied on a layer with even x and y sizes
# 
# J(A,B) = (A&B)/(A+B-A&B)
# D(A,B) = (2*A&B)/(A+B)

def jaccard_idx(y_true, y_pred):
    threshold = 0.25
    #y_pred = keras.cast(keras.greater(y_pred,threshold), dtype='float32')
    intersection = keras.sum(y_true*y_pred)
    sumtotal = keras.sum(y_true + y_pred)
    smooth = 1
    jac = (intersection + smooth)/(sumtotal-intersection+smooth)
    return keras.mean(jac)

def dice_score(y_true, y_pred):
    threshold = 0.25
    #y_pred = keras.cast(keras.greater(y_pred,threshold), dtype='float32')
    intersection = keras.sum(y_true*y_pred)
    sumtotal = keras.sum(y_true + y_pred)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    return keras.mean(dice)

def dice_loss(y_true, y_pred):
    threshold = 0.25
    #y_pred = keras.cast(keras.greater(y_pred,threshold), dtype='float32')
    intersection = keras.sum(y_true*y_pred)
    sumtotal = keras.sum(y_true + y_pred)
    smooth = 1
    dice  = (2*intersection + smooth)/(sumtotal+smooth)
    diceloss = 1 - dice
    return keras.mean(diceloss)

def intersect(y_true, y_pred):
    threshold = 0.25
    #y_pred = keras.cast(keras.greater(y_pred,threshold), dtype='float32')
    intersection = keras.sum(y_true*y_pred)
    return keras.mean(intersection)
def falsepos(y_true, y_pred):
    threshold = 0.25
    #y_pred = keras.cast(keras.greater(y_pred,threshold), dtype='float32')
    fp = keras.sum((1-y_true)*y_pred)
    return keras.mean(fp)

# b_ce: if y=1  b_ce = -log(y_pred) else b_ce = -log(1-y_pred)
# wt_b_ce: if y=1  b_ce = -log(y_pred)*w1 else b_ce = -log(1-y_pred)*w0
def weighted_binary_crossentropy(y_true, y_pred):
    w1 = 2
    w0 = 1
    wt_b_ce = -1*w1*y_true*keras.log(y_pred) - w0*(1-y_true)*keras.log(1-y_pred)
    #b_ce = keras.binary_crossentropy(y_true, y_pred)
    return keras.mean(wt_b_ce, axis=-1)

def unet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2), interpolation='nearest')(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2), interpolation='nearest')(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2), interpolation='nearest')(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2), interpolation='nearest')(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    print(inputs.shape, conv10.shape)
    model = Model(input = inputs, output = conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss ='binary_crossentropy', metrics = [jaccard_idx, dice_score, 'accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss ='binary_crossentropy', metrics = ['accuracy', jaccard_idx, dice_loss])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


