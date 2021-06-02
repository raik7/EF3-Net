from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
import skimage.transform as trans
import numpy as np
import h5py
import tensorflow as tf

class Mish(Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def conv2d_bn(x, filters, kernel_size=(3, 3), padding='same', strides=(1, 1), activation=None, BN=False, name=None):

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(x)
    
    if BN == True:
        x = BatchNormalization(axis=3, scale=False)(x)
    else:
        pass

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

def DepthwiseConv2D_bn(x, kernel_size, padding='same', strides=(1, 1), activation='relu', BN = False, name=None):

    x = DepthwiseConv2D(kernel_size=kernel_size, strides = strides, padding=padding, activation=None, use_bias=True, depthwise_initializer = 'he_normal', bias_initializer = 'he_normal')(x)
    if BN == True:
        x = BatchNormalization(axis=3, scale=False)(x)
    else:
        pass

    if activation == 'mish':
        x = Mish()(x)
        return x
    elif activation == None:
        return x
    else:
        x = Activation(activation, name=name)(x)
        return x

def MultiResBlock(inp, U, activation = 'relu', BN=True):
    # U = U *1.67
    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(U*0.167) + int(U*0.333) + int(U*0.5), (1, 1), activation=None, padding='same', BN=BN)

    conv3x3 = conv2d_bn(inp, int(U*0.167), (3, 3), activation=activation, padding='same', BN=BN)

    conv5x5 = conv2d_bn(conv3x3, int(U*0.333), (3, 3), activation=activation, padding='same', BN=BN)

    conv7x7 = conv2d_bn(conv5x5, int(U*0.5), (3, 3), activation=activation, padding='same', BN=BN)

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    
    if BN == True:
        out = BatchNormalization(axis=3)(out)
    else:
        pass
    
    out = add([shortcut, out])

    if activation == 'mish':
        out = Mish()(out)
    elif activation == None:
        pass
    else:
        out = Activation(activation)(out)
    
    if BN == True:
        out = BatchNormalization(axis=3)(out)
    else:
        pass

    return out

def ResPath(filters, length, inp, activation = 'relu', BN=False):

    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, (1, 1), activation=None, padding='same', BN=BN)

    out = conv2d_bn(inp, filters, (3, 3), activation=activation, padding='same', BN=BN)

    out = add([shortcut, out])

    if activation == 'mish':
        out = Mish()(out)
    elif activation == None:
        pass
    else:
        out = Activation(activation)(out)

    if BN == True:
        out = BatchNormalization(axis=3)(out)
    else:
        pass

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, (1, 1), activation=None, padding='same', BN=BN)

        out = conv2d_bn(out, filters, (3, 3), activation=activation, padding='same', BN=BN)

        out = add([shortcut, out])

        if activation == 'mish':
            out = Mish()(out)
        elif activation == None:
            pass
        else:
            out = Activation(activation)(out)
        
        if BN == True:
            out = BatchNormalization(axis=3)(out)
        else:
            pass

    return out

def Channel_wise_FE(x, filters, reshape_size = (2,2,2)):

    cw = GlobalAveragePooling2D()(x)
    cw = Dense(filters, activation='relu', kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(cw)
    cw = Dense(filters, activation='sigmoid', kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(cw)
    cw = Reshape(reshape_size)(cw)
    out = multiply([x,cw])

    return out

def Spatial_wise_FE(x, filters):

    sw = conv2d_bn(x, filters, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', BN=False)
    sw = DepthwiseConv2D_bn(sw, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid', BN = False)
    out = multiply([x,sw])

    return out

def AFE(x, filters, reshape_size = (2,2,2)):

    Sr = x
    Sc = Channel_wise_FE(Sr, filters, reshape_size = reshape_size)
    Ss = Spatial_wise_FE(Sr, filters)
    So = concatenate([Sr,Sc,Ss], axis = 3)
    So = conv2d_bn(So, filters, kernel_size=(1, 1), activation='mish', BN=False)
    So = add([x, So])
    So = BatchNormalization(axis=-1, center=True, scale=False)(So)

    return So

def AFE_W_Net(pretrained_weights = None,input_size = (256,256,1)):
    kn = 32
    km1 = int(kn*0.167) +    int(kn*0.333) +    int(kn*0.5)
    km2 = int(kn*2*0.167) +  int(kn*2*0.333) +  int(kn*2*0.5)
    km3 = int(kn*4*0.167) +  int(kn*4*0.333) +  int(kn*4*0.5)
    km4 = int(kn*8*0.167) +  int(kn*8*0.333) +  int(kn*8*0.5)
    km5 = int(kn*14*0.167) + int(kn*14*0.333) + int(kn*14*0.5)


    size0 = input_size[0]
    size1 = input_size[1]
    size2 = input_size[2]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = Spatial_wise_FE(mresblock1, km1)
    mresblock1 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = AFE(mresblock2, km2, reshape_size = (1, 1, km2))

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = AFE(mresblock3, km3, reshape_size = (1, 1, km3))

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = Channel_wise_FE(mresblock4, km4, reshape_size = (1, 1, km4))
    mresblock4 = BatchNormalization(axis=-1, scale=False)(mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*14)
    mresblock5_ccn = Channel_wise_FE(mresblock5, km5, reshape_size = (1, 1, km5))
    mresblock5_ccn = BatchNormalization(axis=-1, scale=False)(mresblock5_ccn)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)
    mresblock6_ccn = Channel_wise_FE(mresblock6, km4, reshape_size = (1, 1, km4))
    mresblock6_ccn = BatchNormalization(axis=-1, scale=False)(mresblock6_ccn)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)
    mresblock7_ccn = AFE(mresblock7, km3, reshape_size = (1, 1, km3))

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)
    mresblock8_ccn = AFE(mresblock8, km2, reshape_size = (1, 1, km2))

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)
    mresblock9 = conv2d_bn(mresblock9, 3, (1, 1), activation='relu')



    inputs2 = concatenate([inputs, mresblock9], axis=3)
    mresblock10 = MultiResBlock(inputs2, kn)
    pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
    mresblock10 = Spatial_wise_FE(mresblock10, km1)
    mresblock10 = BatchNormalization(axis=-1, scale=False)(mresblock10)
    merge10 = concatenate([mresblock8_ccn, pool10], axis=3)
    
    mresblock11 = MultiResBlock(merge10, kn*2)
    pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    mresblock11 = AFE(mresblock11, km2, reshape_size = (1, 1, km2))
    merge11 = concatenate([mresblock7_ccn, pool11], axis=3)

    mresblock12 = MultiResBlock(merge11, kn*4)
    pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
    mresblock12 = AFE(mresblock12, km3, reshape_size = (1, 1, km3))
    merge12 = concatenate([mresblock6_ccn, pool12], axis=3)

    mresblock13 = MultiResBlock(merge12, kn*8)
    pool13 = MaxPooling2D(pool_size=(2, 2))(mresblock13)
    mresblock13 = Channel_wise_FE(mresblock13, km4, reshape_size = (1, 1, km4))
    mresblock13 = BatchNormalization(axis=-1, scale=False)(mresblock13)
    merge13 = concatenate([mresblock5_ccn, pool13], axis=3)

    mresblock14 = MultiResBlock(merge13, kn*16)

    up15 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock14), mresblock13], axis=3)
    mresblock15 = MultiResBlock(up15, kn*8)

    up16 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock15), mresblock12], axis=3)
    mresblock16 = MultiResBlock(up16, kn*4)

    up17 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock16), mresblock11], axis=3)
    mresblock17 = MultiResBlock(up17, kn*2)

    up18 = concatenate([Conv2DTranspose(kn,   (2, 2), strides=(2, 2), padding='same')(mresblock17), mresblock10], axis=3)
    mresblock18 = MultiResBlock(up18, kn)
    
    conv19 = conv2d_bn(mresblock18, 1, (1, 1), activation='sigmoid')

    model = Model(inputs = inputs, outputs = conv19)
    
    # model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def six_fold_net(pretrained_weights = None ,input_size = (256,256,1)):
    kn = 17
    input_channel = input_size[2]

    inputs = Input(input_size)

    conv1 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
   
    conv3 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(kn*12, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(kn*12, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv2D(kn*8, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(kn*4, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(kn*2, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(kn, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(input_channel, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    inputs2 = concatenate([inputs, conv9], axis=3)

    conv10 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs2)
    conv10 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
    merge10 = concatenate([pool10, conv8], axis = 3)
    
    conv11 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv11 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    merge11 = concatenate([pool11, conv7], axis = 3)
    

    conv12 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv12 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    merge12 = concatenate([pool12, conv6], axis = 3)
    
    conv13 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge12)
    conv13 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)
    pool13 = MaxPooling2D(pool_size=(2, 2))(conv13)
    merge13 = concatenate([pool13, conv5], axis = 3)

    conv14 = Conv2D(kn*16, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge13)
    conv14 = Conv2D(kn*16, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv14)

    up15 = Conv2D(kn*8, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv14))
    merge15 = concatenate([conv13,up15], axis = 3)
    conv15 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge15)
    conv15 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)

    up16 = Conv2D(kn*4, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv15))
    merge16 = concatenate([conv12,up16], axis = 3)
    conv16 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge16)
    conv16 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv16)

    up17 = Conv2D(kn*2, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv16))
    merge17 = concatenate([conv11,up17], axis = 3)
    conv17 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge17)
    conv17 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv17)

    up18 = Conv2D(kn, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv17))
    merge18 = concatenate([conv10,up18], axis = 3)
    conv18 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge18)
    conv18 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv18) 
    conv18 = Conv2D(input_channel, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv18)



    inputs3 = concatenate([inputs, conv18], axis=3)
    conv19 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs3)
    conv19 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv19)
    pool19 = MaxPooling2D(pool_size=(2, 2))(conv19)
    merge19 = concatenate([pool19, conv17], axis = 3)
    
    conv20 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge19)
    conv20 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv20)
    pool20 = MaxPooling2D(pool_size=(2, 2))(conv20)
    merge20 = concatenate([pool20, conv16], axis = 3)

    conv21 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge20)
    conv21 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv21)
    pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
    merge21 = concatenate([pool21, conv15], axis = 3)
    
    conv22 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge21)
    conv22 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv22)
    pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
    merge22 = concatenate([pool22, conv14], axis = 3)

    conv23 = Conv2D(kn*16, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge22)
    conv23 = Conv2D(kn*16, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv23)

    up24 = Conv2D(kn*8, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv23))
    merge24= concatenate([conv22,up24], axis = 3)
    conv24 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge24)
    conv24 = Conv2D(kn*8, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv24)

    up25 = Conv2D(kn*4, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv24))
    merge25 = concatenate([conv21,up25], axis = 3)
    conv25 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge25)
    conv25 = Conv2D(kn*4, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv25)

    up26 = Conv2D(kn*2, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv25))
    merge26 = concatenate([conv20,up26], axis = 3)
    conv26 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge26)
    conv26 = Conv2D(kn*2, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv26)

    up27 = Conv2D(kn, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv26))
    merge27 = concatenate([conv19,up27], axis = 3)
    conv27 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge27)
    conv27 = Conv2D(kn, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv27) 
    conv27 = Conv2D(input_channel, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv27)

    conv28 = Conv2D(1, (1, 1), activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv27)

    model = Model(inputs = inputs, outputs = conv28)
    
    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def SFE_W_Net(pretrained_weights = None,input_size = (256,256,1)):
    kn = 27
    km1 = int(kn*0.167) +    int(kn*0.333) +    int(kn*0.5)
    km2 = int(kn*2*0.167) +  int(kn*2*0.333) +  int(kn*2*0.5)
    km3 = int(kn*4*0.167) +  int(kn*4*0.333) +  int(kn*4*0.5)
    km4 = int(kn*8*0.167) +  int(kn*8*0.333) +  int(kn*8*0.5)
    km5 = int(kn*15*0.167) + int(kn*15*0.333) + int(kn*15*0.5)


    size0 = input_size[0]
    size1 = input_size[1]
    size2 = input_size[2]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = Spatial_wise_FE(mresblock1, km1)
    mresblock1 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = Spatial_wise_FE(mresblock2, km2)
    mresblock2 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock2)

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = Spatial_wise_FE(mresblock3, km3)
    mresblock3 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock3)

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = Spatial_wise_FE(mresblock4, km4)
    mresblock4 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*15)
    mresblock5_ccn = Spatial_wise_FE(mresblock5, km5)
    mresblock5_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock5_ccn)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)
    mresblock6_ccn = Spatial_wise_FE(mresblock6, km4)
    mresblock6_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock6_ccn)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)
    mresblock7_ccn = Spatial_wise_FE(mresblock7, km3)
    mresblock7_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock7_ccn)

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)
    mresblock8_ccn = Spatial_wise_FE(mresblock8, km2)
    mresblock8_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock8_ccn)

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)
    mresblock9 = conv2d_bn(mresblock9, 3, (1, 1), activation='relu')


    inputs2 = concatenate([inputs, mresblock9], axis=3)
    mresblock10 = MultiResBlock(inputs2, kn)
    pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
    mresblock10 = Spatial_wise_FE(mresblock10, km1)
    mresblock10 = BatchNormalization(axis=-1, scale=False)(mresblock10)
    merge10 = concatenate([mresblock8_ccn, pool10], axis=3)
    
    mresblock11 = MultiResBlock(merge10, kn*2)
    pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    mresblock11 = Spatial_wise_FE(mresblock11, km2)
    mresblock11 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock11)
    merge11 = concatenate([mresblock7_ccn, pool11], axis=3)

    mresblock12 = MultiResBlock(merge11, kn*4)
    pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
    mresblock12 = Spatial_wise_FE(mresblock12, km3)
    mresblock12 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock12)
    merge12 = concatenate([mresblock6_ccn, pool12], axis=3)

    mresblock13 = MultiResBlock(merge12, kn*8)
    pool13 = MaxPooling2D(pool_size=(2, 2))(mresblock13)
    mresblock13 = Spatial_wise_FE(mresblock13, km4)
    mresblock13 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock13)
    merge13 = concatenate([mresblock5_ccn, pool13], axis=3)

    mresblock14 = MultiResBlock(merge13, kn*16)

    up15 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock14), mresblock13], axis=3)
    mresblock15 = MultiResBlock(up15, kn*8)

    up16 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock15), mresblock12], axis=3)
    mresblock16 = MultiResBlock(up16, kn*4)

    up17 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock16), mresblock11], axis=3)
    mresblock17 = MultiResBlock(up17, kn*2)

    up18 = concatenate([Conv2DTranspose(kn,   (2, 2), strides=(2, 2), padding='same')(mresblock17), mresblock10], axis=3)
    mresblock18 = MultiResBlock(up18, kn)
    
    conv19 = conv2d_bn(mresblock18, 1, (1, 1), activation='sigmoid')

    model = Model(inputs = inputs, outputs = conv19)
    
    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def CFE_W_Net(pretrained_weights = None,input_size = (256,256,1)):
    kn = 33
    km1 = int(kn*0.167) +    int(kn*0.333) +    int(kn*0.5)
    km2 = int(kn*2*0.167) +  int(kn*2*0.333) +  int(kn*2*0.5)
    km3 = int(kn*4*0.167) +  int(kn*4*0.333) +  int(kn*4*0.5)
    km4 = int(kn*8*0.167) +  int(kn*8*0.333) +  int(kn*8*0.5)
    km5 = int(kn*16*0.167) + int(kn*16*0.333) + int(kn*16*0.5)


    size0 = input_size[0]
    size1 = input_size[1]
    size2 = input_size[2]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = Channel_wise_FE(mresblock1, km1, reshape_size = (1, 1, km1))
    mresblock1 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = Channel_wise_FE(mresblock2, km2, reshape_size = (1, 1, km2))
    mresblock2 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock2)

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = Channel_wise_FE(mresblock3, km3, reshape_size = (1, 1, km3))
    mresblock3 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock3)

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = Channel_wise_FE(mresblock4, km4, reshape_size = (1, 1, km4))
    mresblock4 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*16)
    mresblock5_ccn = Channel_wise_FE(mresblock5, km5, reshape_size = (1, 1, km5))
    mresblock5_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock5_ccn)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)
    mresblock6_ccn = Channel_wise_FE(mresblock6, km4, reshape_size = (1, 1, km4))
    mresblock6_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock6_ccn)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)
    mresblock7_ccn = Channel_wise_FE(mresblock7, km3, reshape_size = (1, 1, km3))
    mresblock7_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock7_ccn)

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)
    mresblock8_ccn = Channel_wise_FE(mresblock8, km2, reshape_size = (1, 1, km2))
    mresblock8_ccn = BatchNormalization(axis=-1, center=True, scale=False)(mresblock8_ccn)

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)
    mresblock9 = conv2d_bn(mresblock9, 3, (1, 1), activation='relu')


    inputs2 = concatenate([inputs, mresblock9], axis=3)
    mresblock10 = MultiResBlock(inputs2, kn)
    pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
    mresblock10 = Channel_wise_FE(mresblock10, km1, reshape_size = (1, 1, km1))
    mresblock10 = BatchNormalization(axis=-1, scale=False)(mresblock10)
    merge10 = concatenate([mresblock8_ccn, pool10], axis=3)
    
    mresblock11 = MultiResBlock(merge10, kn*2)
    pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    mresblock11 = Channel_wise_FE(mresblock11, km2, reshape_size = (1, 1, km2))
    mresblock11 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock11)
    merge11 = concatenate([mresblock7_ccn, pool11], axis=3)

    mresblock12 = MultiResBlock(merge11, kn*4)
    pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
    mresblock12 = Channel_wise_FE(mresblock12, km3, reshape_size = (1, 1, km3))
    mresblock12 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock12)
    merge12 = concatenate([mresblock6_ccn, pool12], axis=3)

    mresblock13 = MultiResBlock(merge12, kn*8)
    pool13 = MaxPooling2D(pool_size=(2, 2))(mresblock13)
    mresblock13 = Channel_wise_FE(mresblock13, km4, reshape_size = (1, 1, km4))
    mresblock13 = BatchNormalization(axis=-1, center=True, scale=False)(mresblock13)
    merge13 = concatenate([mresblock5_ccn, pool13], axis=3)

    mresblock14 = MultiResBlock(merge13, kn*16)

    up15 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock14), mresblock13], axis=3)
    mresblock15 = MultiResBlock(up15, kn*8)

    up16 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock15), mresblock12], axis=3)
    mresblock16 = MultiResBlock(up16, kn*4)

    up17 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock16), mresblock11], axis=3)
    mresblock17 = MultiResBlock(up17, kn*2)

    up18 = concatenate([Conv2DTranspose(kn,   (2, 2), strides=(2, 2), padding='same')(mresblock17), mresblock10], axis=3)
    mresblock18 = MultiResBlock(up18, kn)
    
    conv19 = conv2d_bn(mresblock18, 1, (1, 1), activation='sigmoid')

    model = Model(inputs = inputs, outputs = conv19)
    
    # model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def AFE_W_Net_A(pretrained_weights = None,input_size = (256,256,1)):
    kn = 24
    km1 = int(kn*0.167) +    int(kn*0.333) +    int(kn*0.5)
    km2 = int(kn*2*0.167) +  int(kn*2*0.333) +  int(kn*2*0.5)
    km3 = int(kn*4*0.167) +  int(kn*4*0.333) +  int(kn*4*0.5)
    km4 = int(kn*8*0.167) +  int(kn*8*0.333) +  int(kn*8*0.5)
    km5 = int(kn*16*0.167) + int(kn*16*0.333) + int(kn*16*0.5)


    size0 = input_size[0]
    size1 = input_size[1]
    size2 = input_size[2]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = AFE(mresblock1, km1, reshape_size = (1, 1, km1))

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = AFE(mresblock2, km2, reshape_size = (1, 1, km2))

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = AFE(mresblock3, km3, reshape_size = (1, 1, km3))

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = AFE(mresblock4, km4, reshape_size = (1, 1, km4))

    mresblock5 = MultiResBlock(pool4, kn*16)
    mresblock5_ccn = AFE(mresblock5, km5, reshape_size = (1, 1, km5))

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)
    mresblock6_ccn = AFE(mresblock6, km4, reshape_size = (1, 1, km4))

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)
    mresblock7_ccn = AFE(mresblock7, km3, reshape_size = (1, 1, km3))

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)
    mresblock8_ccn = AFE(mresblock8, km2, reshape_size = (1, 1, km2))

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)
    mresblock9 = conv2d_bn(mresblock9, 3, (1, 1), activation='relu')


    inputs2 = concatenate([inputs, mresblock9], axis=3)
    mresblock10 = MultiResBlock(inputs2, kn)
    pool10 = MaxPooling2D(pool_size=(2, 2))(mresblock10)
    mresblock10 = AFE(mresblock10, km1, reshape_size = (1, 1, km1))
    merge10 = concatenate([mresblock8_ccn, pool10], axis=3)
    
    mresblock11 = MultiResBlock(merge10, kn*2)
    pool11 = MaxPooling2D(pool_size=(2, 2))(mresblock11)
    mresblock11 = AFE(mresblock11, km2, reshape_size = (1, 1, km2))
    merge11 = concatenate([mresblock7_ccn, pool11], axis=3)

    mresblock12 = MultiResBlock(merge11, kn*4)
    pool12 = MaxPooling2D(pool_size=(2, 2))(mresblock12)
    mresblock12 = AFE(mresblock12, km3, reshape_size = (1, 1, km3))
    merge12 = concatenate([mresblock6_ccn, pool12], axis=3)

    mresblock13 = MultiResBlock(merge12, kn*8)
    pool13 = MaxPooling2D(pool_size=(2, 2))(mresblock13)
    mresblock13 = AFE(mresblock13, km4, reshape_size = (1, 1, km4))
    merge13 = concatenate([mresblock5_ccn, pool13], axis=3)

    mresblock14 = MultiResBlock(merge13, kn*15)

    up15 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock14), mresblock13], axis=3)
    mresblock15 = MultiResBlock(up15, kn*8)

    up16 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock15), mresblock12], axis=3)
    mresblock16 = MultiResBlock(up16, kn*4)

    up17 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock16), mresblock11], axis=3)
    mresblock17 = MultiResBlock(up17, kn*2)

    up18 = concatenate([Conv2DTranspose(kn,   (2, 2), strides=(2, 2), padding='same')(mresblock17), mresblock10], axis=3)
    mresblock18 = MultiResBlock(up18, kn)
    
    conv19 = conv2d_bn(mresblock18, 1, (1, 1), activation='sigmoid')

    model = Model(inputs = inputs, outputs = conv19)
    
    # model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def AFE_MRUNet(pretrained_weights = None,input_size = (256,256,1)):
    kn = 50
    km1 = int(kn*0.167) + int(kn*0.333) + int(kn*0.5)
    km2 = int(kn*2*0.167) + int(kn*2*0.333) + int(kn*2*0.5)
    km3 = int(kn*4*0.167) + int(kn*4*0.333) + int(kn*4*0.5)
    km4 = int(kn*8*0.167) + int(kn*8*0.333) + int(kn*8*0.5)


    size0 = input_size[0]
    size1 = input_size[1]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = Spatial_wise_FE(mresblock1, km1)
    mresblock1 = BatchNormalization(axis=3, center=True, scale=False)(mresblock1)

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = AFE(mresblock2, km2, reshape_size = (1, 1, km2))

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = AFE(mresblock3, km3, reshape_size = (1, 1, km3))

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = conv2d_bn(mresblock4, kn*8, kernel_size=(1, 1), BN=False)
    mresblock4 = Channel_wise_FE(mresblock4, kn*8, reshape_size = (1, 1, kn*8))
    mresblock4 = BatchNormalization(axis=3, center=True, scale=False)(mresblock4)

    mresblock5 = MultiResBlock(pool4, kn*16)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)

    conv10 = conv2d_bn(mresblock9, 1, (1, 1), activation='sigmoid')
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def AFE_MRUNet_A(pretrained_weights = None,input_size = (256,256,1)):
    kn = 45
    km1 = int(kn*0.167) + int(kn*0.333) + int(kn*0.5)
    km2 = int(kn*2*0.167) + int(kn*2*0.333) + int(kn*2*0.5)
    km3 = int(kn*4*0.167) + int(kn*4*0.333) + int(kn*4*0.5)
    km4 = int(kn*8*0.167) + int(kn*8*0.333) + int(kn*8*0.5)


    size0 = input_size[0]
    size1 = input_size[1]

    inputs = Input(input_size)

    mresblock1 = MultiResBlock(inputs, kn)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = AFE(mresblock1, km1, reshape_size = (1, 1, km1))

    mresblock2 = MultiResBlock(pool1, kn*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = AFE(mresblock2, km2, reshape_size = (1, 1, km2))

    mresblock3 = MultiResBlock(pool2, kn*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = AFE(mresblock3, km3, reshape_size = (1, 1, km3))

    mresblock4 = MultiResBlock(pool3, kn*8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = AFE(mresblock4, km4, reshape_size = (1, 1, km4))

    mresblock5 = MultiResBlock(pool4, kn*16)

    up6 = concatenate([Conv2DTranspose(kn*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(up6, kn*8)

    up7 = concatenate([Conv2DTranspose(kn*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(up7, kn*4)

    up8 = concatenate([Conv2DTranspose(kn*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(up8, kn*2)

    up9 = concatenate([Conv2DTranspose(kn, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(up9, kn)

    conv10 = conv2d_bn(mresblock9, 1, (1, 1), activation='sigmoid')
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.summary()
    

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
