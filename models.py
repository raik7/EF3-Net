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

def EF3_Net(pretrained_weights = None,input_size = (256,256,1)):
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

