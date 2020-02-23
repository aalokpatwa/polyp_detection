'''
predict.py predicts polyp segmentation on the test images
'''

from model import *
from data import *
from keras.models import load_model
import os
import time

basedir = "/unet_polyp/"
test_path = basedir+"datasets/cvc612/images"
save_path = basedir+"datasets/cvc612/prpolyp_3ldice"
num_test_images = 607
'''
test_path = basedir+"datasets/etislarib/images"
save_path = basedir+"datasets/etislarib/prpolyp_768dice"
num_test_images = 196
'''

if (os.path.isdir(save_path)):
    print(save_path+" exists")
else:
    try:
        os.mkdir(save_path)
    except OSError:
        print(save_path+"  could not be created")
    else:
        print(save_path+" created")
'''
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
'''

model = load_model('unet_768dice.hdf5', custom_objects={'jaccard_idx':jaccard_idx, 'dice_loss':dice_loss})
#print("going to test generator")
testGene = testGenerator(test_path, num_image=num_test_images)
#remember original image names and sizes
original_shape = []
original_name = []
for filename in sorted(os.listdir(test_path)):
    img = io.imread(os.path.join(test_path, filename), as_gray = False)
    imgshape = img.shape    
    original_shape.append((imgshape[0], imgshape[1]))
    original_name.append(filename.split('.')[0])
#print(original_shape, original_name)
print("going to predict generator")
print(time.time(), time.clock())
results = model.predict_generator(testGene, steps = num_test_images, verbose=1)
print(time.time(), time.clock())
saveResult(save_path, results, original_shape, original_name)
print(time.time(), time.clock())
