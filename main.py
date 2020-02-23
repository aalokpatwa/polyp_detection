'''
unet_polyp

Segmentation of Colon Polyps with UNET model
Aalok Patwa 2019

main.py has the main training/evaluate loop with support for checkpointing the best model.
It relies on model.py for definition of the unet model.
It also relies on data.py for train and evaluate generators.

replace basedir with your directory name

'''

from model import *
from data import *
import json,codecs
import numpy as np

basedir = "/unet_polyp/"

def saveHist(path,history):

    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if  type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))
                
    print(new_hist)
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)
                
def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#augmentation strategy

data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     data_format="channels_last",
                     fill_mode='nearest') #not reflect

#batches, steps and epochs
#batch_size controls how many images are given at a time to calculate the gradient and update the weight. Updates are averaged over a batch_size
#steps control how many batches are fed
#epochs control when to save the model and if validating, when to validate the model

# training
batch_size = 1

myGene = trainGenerator(batch_size,basedir+"datasets/cvc300","augimages","auggtpolyp",data_gen_args,save_to_dir = basedir+"datasets/cvc300/train/aug")

# evaluation
eval_gen_args = dict(data_format="channels_last",
                     fill_mode='nearest') #not reflect
eval_batch_size = 1
myEval = trainGenerator(eval_batch_size,basedir+"datasets/etislarib","images","gtpolyp",eval_gen_args,save_to_dir = None)

#model = unet(pretrained_weights='unet_fl.hdf5')
model = unet()
print(model.summary())
#model_checkpoint = ModelCheckpoint('unet_polyp.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint = ModelCheckpoint('unet_polyp.hdf5', monitor='val_dice_loss', mode='min', verbose=1, save_best_only=True)
# class_weight{0:.1, 1:.9} not supported for 3+ dimentional targets
fit_hist = model.fit_generator(myGene,steps_per_epoch=1000,epochs=60,callbacks=[model_checkpoint], validation_data=myEval, validation_steps=196, max_queue_size=batch_size*2) # steps_per_epoch = #images/batch_size 
saveHist('./fit_hist.json', fit_hist)

# Test few images
test_path = basedir+"datasets/cvc300/test/images"
save_path = basedir+"datasets/cvc300/test/prpolyp"
# code to check and create save_path
if (os.path.isdir(save_path)):
    print(save_path+" exists")
else:
    try:
        os.mkdir(save_path)
    except OSError:
        print(save_path+"  could not be created")
    else:
        print(save_path+" created")


num_test_images = 7
print("going to test generator")
testGene = testGenerator(test_path, num_image=num_test_images)
#remember original image names and sizes
original_shape = []
original_name = []
for filename in sorted(os.listdir(test_path)):
    img = io.imread(os.path.join(test_path, filename), as_gray = False)
    imgshape = img.shape    
    original_shape.append((imgshape[0], imgshape[1]))
    original_name.append(filename.split('.')[0])
print(original_shape, original_name)
print("going to predict generator")
results = model.predict_generator(testGene, steps = num_test_images, verbose=1)
saveResult(save_path, results, original_shape, original_name)

# 
