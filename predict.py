from tensorflow.keras.models import load_model
import tensorflow as tf
import MModel
import skimage.io as io
import skimage.transform as trans
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

def testGenerator(test_path,num_image,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = cv2.imread(os.path.join(test_path,"%d.png"%i),cv2.IMREAD_GRAYSCALE)
        
        img = img / 255
        img = cv2.resize(img,target_size)
        img = np.array(img)
        img = img.reshape((img.shape[0],img.shape[1],img.shape[2],1))
        # img = trans.resize(img,target_size)
        # # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        # img = np.reshape(img,(img.shape[0],img.shape[1],img.shape[2],1))
        
        yield img

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.experimental.set_visible_devices(devices=gpus[0:1], device_type='GPU')


new_model = MModel.MultiResUNet(pretrained_weights =None, input_size = (256,256,1))
# new_model.load_weights('E:\\ProjectFiles\\models\\Warwick QU\\AFE_W_Net-0522-0\\modelW-0.h5')
model = new_model.load_weights('E:\\ProjectFiles\\models\\ISBI-2012\\MultiResUNet\\model-4-0317.h5')
# net = load_model('E:\\ProjectFiles\\models\\ISIC2018\\W-Net\\W_AFE_MultiResUNet_F-2\\modelW2-jaccard-1220.h5', 
#                 custom_objects={'tf': tf, 
#                                 'conv2d_bn': MModel.conv2d_bn, 
#                                 'MultiResBlock': MModel.MultiResBlock, 
#                                 'Channel_wise_FE': MModel.Channel_wise_FE, 
#                                 'Spatial_wise_FE': MModel.Spatial_wise_FE, 
#                                 'AFE': MModel.AFE, 
#                                 }
#                 ) 

# model = MModel.W_AFE_MultiResUNet_F(pretrained_weights =None, input_size = (256, 256,3))




predicted_results_dir = "E:\\ProjectFiles\\Data\\ISBI2012\\mrunet-4"
if not os.path.exists(predicted_results_dir):
    os.makedirs(predicted_results_dir)
    
# testGene = testGenerator("E:\\ProjectFiles\\Data\\2009_ISBI_2DNuclei_code_data\\test",6,as_gray = True)
# results = new_model.predict_generator(testGene,8,verbose=1)
# # results = np.round(results,0)
# saveResult(predicted_results_dir,results)
tst_path = 'E:\\ProjectFiles\\Data\\ISBI2012\\test'

tst_files = next(os.walk(tst_path))[2]

tst_files.sort()

Z = []
save_name = []

for img_fl in tst_files:    
    # tst = TIFF.open('E:\\ProjectFiles\\data\\CVC_ClinicDB\\CVC_ClinicDB\\test\\{}'.format(img_fl), mode='r')
    # test = tst.read_image()
    test = cv2.imread(tst_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
    # test = cv2.imread(tst_path + '\\' + img_fl, cv2.IMREAD_COLOR)
    resized_tst = cv2.resize(test,(256, 256))
    Z.append(resized_tst)
    img_name = img_fl.split('.')[0]
    save_name.append(img_name)

Z = np.array(Z)

Z = Z.reshape((Z.shape[0],Z.shape[1],Z.shape[2],1))

Z = Z / 255

yp = new_model.predict(x=Z, batch_size=8, verbose=1)
yp = yp * 255
for i in range(yp.shape[0]):

    cv2.imwrite(predicted_results_dir + '\\' + save_name[i] + '.png',yp[i])
    # print(save_name[i])
    # cv2.imshow('img',yp[i])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    