from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import clear_session
import os
import numpy as np
import time
from tqdm import tqdm
import cv2
import TrainStep
import models
from tensorflow import keras
import h5py
from libtiff import TIFF


img_path = '..\\Input'
msk_path = '..\\GroundTruth'
tst_path = '..\\test'

model_results_dir = '..\\results'
if not os.path.exists(model_results_dir):
    os.makedirs(model_results_dir)


pretrained_weights = None

img_resize = (256,192)
input_size = (192,256,3)

epochs = 350
batchSize = 8
learning_rate = 1e-3


img_files = next(os.walk(img_path))[2]
msk_files = next(os.walk(msk_path))[2]
tst_files = next(os.walk(tst_path))[2]

img_files.sort() 
msk_files.sort()
tst_files.sort()

X = []
Y = []
Z = []

for img_fl in tqdm(img_files): 
    image = cv2.imread(img_path + '\\' + img_fl, cv2.IMREAD_COLOR)
    # image = cv2.imread(img_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE) # Grayscale input images
    resized_img = cv2.resize(image, img_resize)
    X.append(resized_img)

for img_fl in tqdm(msk_files): 
    mask = cv2.imread(msk_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
    resized_msk = cv2.resize(mask, img_resize)
    Y.append(resized_msk)

for img_fl in tqdm(tst_files):    
    test = cv2.imread(tst_path + '\\' + img_fl, cv2.IMREAD_COLOR)
    # test = cv2.imread(tst_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)  # Grayscale input images
    resized_tst = cv2.resize(test, img_resize)
    Z.append(resized_tst)


X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

# X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))  # Grayscale input images
Y = Y.reshape((Y.shape[0],Y.shape[1],Y.shape[2],1))

np.random.seed(2000)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
y_shuffled = Y[shuffle_indices]

x_shuffled = x_shuffled / 255
y_shuffled = y_shuffled / 255
y_shuffled = np.round(y_shuffled,0)
Z = Z / 255

print(x_shuffled.shape)
print(y_shuffled.shape)
print(Z.shape)

length = int(float(len(x_shuffled))/5)


for i in range(0,5):
    tic = time.ctime()
    fp = open(model_results_dir +'\\jaccard-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\dice-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\acc-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\se-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\sp-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\pre-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-acc-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-se-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-sp-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-pre-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()

    index = int(float(len(x_shuffled))*(i+1)/5)
    x_train = np.concatenate((x_shuffled[:index-length], x_shuffled[index:]), axis=0)
    x_val = x_shuffled[index-length:index]
    y_train = np.concatenate((y_shuffled[:index-length],y_shuffled[index:]), axis=0)
    y_val = y_shuffled[index-length:index]

    model = models.AFE_W_Net(pretrained_weights = pretrained_weights, input_size = input_size)

    print ('iter: %s' % (str(i)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    TrainStep.trainStep(model, x_train, y_train, x_val, y_val, epochs=epochs, batchSize=batchSize, iters = i, results_save_path = model_results_dir, losshistory=None, reverse = False)

    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-jaccard.txt','a')
    tic = time.ctime()
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-dice.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-pre-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-pre.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-acc-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-acc.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-se-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-se.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-sp-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-sp.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    clear_session()
    tf.compat.v1.reset_default_graph()

