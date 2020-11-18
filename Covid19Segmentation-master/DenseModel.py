#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:54:11 2020

@author: tarek
"""


import os
import numpy as np
import matplotlib.pyplot as plt 
import glob
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize
from sklearn.preprocessing import normalize
import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from skimage.io import imread, imshow
from sklearn.model_selection import train_test_split
from keras.callbacks import  ModelCheckpoint
import matplotlib.pyplot as plt

# import segmentation_models as sm
from keras.layers import LeakyReLU
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.merge import concatenate, add

from function import *
import segmentation_models as sm

 


IMG_Width  = 128
IMG_Height = 128
IMG_Channels = 1


inputs=tf.keras.Input((IMG_Width,IMG_Height,IMG_Channels)) # this layer is in core layer of keras documentation
def dens_model(inputs):

    skip_connection_list=[]
    stack=[]
    num_layers_per_block = 3
    n_pool = 3
    n_feat_first_layer = [12, 12, 12]
    growth_rate = 12
    n_layers_per_block = [2, 3, 4, 5, 4, 3, 2]
    dropout_rate = 0.2 
    num_classes=4


    #########
    #First layer#
    ###########

    out_of_inception= inceptionx_block(inputs,12,12,12) 

    stack = tf.keras.layers.Concatenate()(out_of_inception)

    print(stack.shape)

    n_filters = sum(n_feat_first_layer)


    #######################
    #   Downsampling path   #
    #######################
    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ELU_Conv(stack, growth_rate, dropout_rate=dropout_rate)
            print("J",l.shape)
            stack = tf.keras.layers.Concatenate()([stack, l])
            n_filters += growth_rate
        print("DB_Down:", i, " shape ", stack.shape)        

        proj_l = BN_ELU_Conv(stack, growth_rate*n_layers_per_block[i+1],
                    filter_size=1, dropout_rate=dropout_rate)
        print("DB Projection Layer:", " shape ", proj_l.shape)
        skip_connection_list.append(proj_l)
        stack = TransitionDown(stack, n_filters, dropout_rate=dropout_rate)
        print("TD:", i, " shape ", stack.shape) 

    skip_connection_list = skip_connection_list[::-1]
    block_to_upsample = []


    #*********** Bottle-Neck Layer **********#
    proj_l = BN_ELU_Conv(stack, growth_rate*n_layers_per_block[n_pool],
                        filter_size=1, dropout_rate=dropout_rate)
    print("Bottleneck Projection Layer:", " shape ", proj_l.shape)

    for j in range(n_layers_per_block[n_pool]):
        l = BN_ELU_Conv(stack, growth_rate, dropout_rate=dropout_rate)
        block_to_upsample.append(l)
        stack = tf.keras.layers.Concatenate()([stack, l])
        
    block_to_upsample = tf.keras.layers.Concatenate()(block_to_upsample)
    block_to_upsample = tf.add(block_to_upsample, proj_l)
    print("Bottleneck:", " shape ", block_to_upsample.shape)


    #######################
    #   Upsampling path   #
    #######################
    print (skip_connection_list[0].shape)
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = ResidualTransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)
        print("TU:", i, " shape ", stack.get_shape().as_list())  

        proj_l = BN_ELU_Conv(stack, growth_rate*n_layers_per_block[n_pool + i +1],
                            filter_size=1, dropout_rate=dropout_rate)
        print("Transition Up Projection Layer:", " shape ", proj_l.shape)
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ELU_Conv(stack, growth_rate, dropout_rate=dropout_rate)
            block_to_upsample.append(l)
            stack = tf.keras.layers.Concatenate()([stack, l])

            # stack = tf.concat([stack, l],3)
        block_to_upsample = tf.keras.layers.Concatenate()(block_to_upsample)    
        block_to_upsample = tf.add(block_to_upsample, proj_l)    
        print("DB_Up:", i, " shape ", block_to_upsample.shape)
    print("Final Stack:", i, " shape ", stack.shape)



    #####################
    #       Outputs     #
    #####################
    logits = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1),padding="same", use_bias=False)(block_to_upsample)
    outputs = tf.keras.layers.Activation( 'softmax' )(logits)

    print("Final Softmax Layer:", " shape ", outputs.shape)
    # self.predictions = tf.cast(tf.argmax(self.logits,3), tf.float32)
    # predictions = tf.argmax(logits,3)
    # print("Predictions:", " shape ", predictions.get_shape().as_list())

    # IMG_Width  = 128
    # IMG_Height = 128
    # IMG_Channels = 1


    # input=tf.keras.Input((IMG_Width,IMG_Height,IMG_Channels)) # this layer is in core layer of keras documentation
    # modelLastLayer= DenseModel.dens_model(input)

    optim=tf.keras.optimizers.Adam(learning_rate=0.001)
    model= tf.keras.Model(inputs=[inputs],outputs=[outputs])
    loss= tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=0,
        reduction="auto",
        name="categorical_crossentropy",
    )
    model.compile(optim,loss=loss ,metrics=[sm.metrics.f1_score])

    print("done ")

    return model

# dens_model(inputs)





