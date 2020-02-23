'''
evaluate.py is used to compare multiple models that were trained earlier
It is useful when the model architecture is being defined
'''
from model import *
from data import *
from keras.models import load_model
import os
import time

# evaluation is done on two different test datasets

basedir = "/unet_polyp/"
eval_root = basedir+"datasets/etislarib"
eval_images_path = eval_root + "/images"
num_eval_images = 196
'''
eval_root = basedir+"datasets/cvc612"
eval_images_path = eval_root + "/images"
num_eval_images = 607
'''

threshold = 0.1

def jaccard_idx(y_true, y_pred):
    #threshold = 0.3
    #y_pred = keras.cast(keras.greater(y_pred,threshold), dtype='float32')
    intersection = keras.sum(y_true*y_pred)
    sumtotal = keras.sum(y_true + y_pred)
    smooth = 1
    jac = (intersection + smooth)/(sumtotal-intersection+smooth)
    return keras.mean(jac)

def dice_score(y_true, y_pred):
    #threshold = 0.3
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


model1 = load_model('unet_o90.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_score':dice_score})
model2 = load_model('unet_bi.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_score':dice_score})
model3 = load_model('unet_768.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_score':dice_score})
model4 = load_model('unet_3l.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_score':dice_score})

model5 = load_model('unet_768dice.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_loss':dice_loss})
model6 = load_model('unet_3ldice.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_loss':dice_loss})


# evaluation
batch_size = 1
eval_gen_args = dict(data_format="channels_last",
                     fill_mode='nearest') #not reflect
print("Create evaluation pairs")
evalGene1 = trainGenerator(batch_size,eval_root, "images","gtpolyp",eval_gen_args,save_to_dir = None)
evalGene2 = trainGenerator(batch_size,eval_root, "images","gtpolyp",eval_gen_args,save_to_dir = None)
evalGene3 = trainGenerator(batch_size,eval_root, "images","gtpolyp",eval_gen_args,save_to_dir = None)
evalGene4 = trainGenerator(batch_size,eval_root, "images","gtpolyp",eval_gen_args,save_to_dir = None)
evalGene5 = trainGenerator(batch_size,eval_root, "images","gtpolyp",eval_gen_args,save_to_dir = None)
evalGene6 = trainGenerator(batch_size,eval_root, "images","gtpolyp",eval_gen_args,save_to_dir = None)

'''
print("Going to model evaluate generator")
print("start:",time.time())
results = model1.evaluate_generator(evalGene1, steps=num_eval_images, verbose=1)
print("finish:", time.time())
print('o90', model1.metrics_names, results)

print("start:",time.time())
results = model2.evaluate_generator(evalGene2, steps=num_eval_images, verbose=1)
print("finish:", time.time())
print('bi', model2.metrics_names, results)

'''
print("start:",time.time())
results = model3.evaluate_generator(evalGene4, steps=num_eval_images, verbose=1)
print("finish:", time.time())
print('768', model3.metrics_names, results)

print("start:",time.time())
results = model4.evaluate_generator(evalGene3, steps=num_eval_images, verbose=1)
print("finish:", time.time())
print('3l', model4.metrics_names, results)

print("start:",time.time())
results = model5.evaluate_generator(evalGene5, steps=num_eval_images, verbose=1)
print("finish:", time.time())
print('768dice', model5.metrics_names, results)

print("start:",time.time())
results = model6.evaluate_generator(evalGene5, steps=num_eval_images, verbose=1)
print("finish:", time.time())
print('3ldice', model6.metrics_names, results)
