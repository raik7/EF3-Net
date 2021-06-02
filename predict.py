from tensorflow.keras.models import load_model
import tensorflow as tf
import models
import skimage.io as io
import skimage.transform as trans
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json



tst_path = '..\\test'
image_resize = (256, 192)

pre_model = models.EF3_Net(input_size = (192,256,3))

model = pre_model.load_weights('..\\model.h5')

predicted_results_dir = "..\\predict_results"
if not os.path.exists(predicted_results_dir):
    os.makedirs(predicted_results_dir)
    

tst_files = next(os.walk(tst_path))[2]

tst_files.sort()

Z = []
file_name = []

for img_fl in tst_files:    
    # tst = TIFF.open(tst_path + '\\' + img_fl, mode='r')
    # test = tst.read_image()
    # test = cv2.imread(tst_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(tst_path + '\\' + img_fl, cv2.IMREAD_COLOR)
    resized_tst = cv2.resize(test,image_resize)
    Z.append(resized_tst)
    img_name = img_fl.split('.')[0]
    file_name.append(img_name)

Z = np.array(Z)

# Z = Z.reshape((Z.shape[0],Z.shape[1],Z.shape[2],1)) 

Z = Z / 255

yp = pre_model.predict(x=Z, batch_size=8, verbose=1)
yp = yp * 255
for i in range(yp.shape[0]):
    cv2.imwrite(predicted_results_dir + '\\' + file_name[i] + '.png',yp[i])

