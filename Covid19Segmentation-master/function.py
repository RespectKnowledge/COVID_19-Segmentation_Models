#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:37:15 2020

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

eps = 1.1e-5
dropout_rate = 0.2

n_layers_per_block = [2, 3, 4, 5, 4, 3, 2]
growth_rate = 12

concat=[]





def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)    

def H(  inputs, num_filters , dropout_rate ):

    x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    # print("L1: ",x.shape)
  
    x = tf.keras.layers.Activation('relu')(x)
    # print("L2: ",x.shape)
  
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    # print("L3: ",x.shape)
  
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), use_bias=False , kernel_initializer='he_normal' )(x)
    print("L4: ",x.shape)
  
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    # print("L5: ",x.shape)
  
    return x


def BN_ELU_Conv(inputs, n_filters, filter_size=3,dropout_rate=0.2):

   
    l = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
    # print("L1: ",l.shape)

    l = tf.keras.layers.Activation('relu')(l)
    # print("L2: ",l.shape)

    l = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3),padding="same", use_bias=False , kernel_initializer='he_normal' )(l)
    # print("L4: ",l.shape)
  
    l = tf.keras.layers.Dropout(rate=0.2)(l)
    # print("L5: ",x.shape)                         
    return l

def TransitionDown(inputs, n_filters, dropout_rate=0.2):
    l = BN_ELU_Conv(inputs, n_filters, filter_size=1, dropout_rate=dropout_rate)
    l = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2)(l)

    return l

def ResidualTransitionUp(skip_connection, block_to_upsample, n_filters_keep,
                     is_training=tf.constant(False,dtype=tf.bool)):
    """
    Performs upsampling on block_to_upsample by a factor 2 and adds it with the skip_connection 
    """

    l = tf.keras.layers.Conv2DTranspose(filters=n_filters_keep,
                                     kernel_size = (3,3), 
                                     strides = (2,2), 
                                     padding = 'SAME',)(block_to_upsample)

    l = tf.add(l, skip_connection)

    return l



def dense_block( inputs, num_layers, num_filters, growth_rate , dropout_rate ):

  for i in range(n_layers_per_block): # num_layers is the value of 'l'
      conv_outputs = H(inputs, growth_rate , dropout_rate )
      inputs = tf.keras.layers.Concatenate()([conv_outputs, inputs])
      print("cnocatenate:",i,inputs.shape )

      num_filters += growth_rate # To increase the number of filters for each layer.
      
  return inputs, num_filters

def transition(inputs, num_filters , compression_factor , dropout_rate ):

  # compression_factor is the 'θ'
  x = tf.keras.layers.BatchNormalization(epsilon=eps)(inputs)
  x = tf.keras.layers.Activation('relu')(x)
  num_feature_maps = inputs.shape[1] # The value of 'm'

  x = tf.keras.layers.Conv2D( np.floor( compression_factor * num_feature_maps ).astype( np.int ) ,
                            kernel_size=(1, 1), use_bias=False, padding='same' , kernel_initializer='he_normal' , kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ) )(x)
  x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  print("after pool:",x.shape )

  return x


def project(x,num_filters):
    x = tf.keras.layers.BatchNormalization(epsilon=eps)(x)
    # print("L1: ",x.shape)
  
    x = tf.keras.layers.Activation('relu')(x)
    # print("L2: ",x.shape)

    x = tf.keras.layers.Conv2D(num_filters, kernel_size=(1, 1), use_bias=False , kernel_initializer='he_normal' )(x)

    x = tf.keras.layers.Dropout(rate=0.2)(x)
    # print("L5: ",x.shape)



def inceptionx_block(inputs,numf1,numf2,numf3,name=None):


    conv1 = tf.keras.layers.Conv2D( numf1 ,padding="same" ,kernel_size=( 3 , 3 ) , use_bias=False,
                                kernel_initializer='he_normal' ,
                                kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ) )( inputs )
    concat.append(conv1)

    conv1 = tf.keras.layers.Conv2D( numf2 , padding="same",kernel_size=( 5 , 5 ) , use_bias=False,
                                    kernel_initializer='he_normal' ,
                                    kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ) )( inputs )
    concat.append(conv1)

    conv1 = tf.keras.layers.Conv2D( numf2 ,padding="same", kernel_size=( 7 , 7 ) , use_bias=False,
                                    kernel_initializer='he_normal' ,
                                    kernel_regularizer=tf.keras.regularizers.l2( 1e-4 ) )( inputs )
    concat.append(conv1)

  
    return concat



def data():
    
    # images_Path = 'drive/My Drive/DataSet/images/'
    # masks_Path = 'drive/My Drive/DataSet//masks/'
    
    images_Path = 'DataSet/images/'
    masks_Path = 'DataSet/masks/'
    
    # List of files
    #-------------------------------
    images = os.listdir(images_Path)
    masks = os.listdir(masks_Path)
    #----------------------
    IMG_Width  = 128
    IMG_Height = 128
    IMG_Channels = 1
    #----------------------
    X = np.zeros((len(images),IMG_Height,IMG_Width,IMG_Channels),dtype=np.uint8)
    Y = np.zeros((len(masks),IMG_Height,IMG_Width,4),dtype=np.bool)
    
    for n, id_ in tqdm(enumerate(images), total=len(images)):   
        img = imread(images_Path + id_)  
        X[n][:,:,0] = img  #Fill empty X_train with values from img
        mask = imread(masks_Path+ id_)   
        Y[n] = mask #Fill empty Y_train with values from mask
    
    # Split Train Test Validate
    ratio=0.1
    X_, X_val, Y_, Y_val = train_test_split(X, Y, test_size=ratio,random_state= 42)
    X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=ratio/(1-ratio),random_state=42)
    
    
    # X_train = tf.convert_to_tensor(X_train)
    # Y_train = tf.convert_to_tensor(Y_train)
    # X_val = tf.convert_to_tensor(X_val)
    # Y_val = tf.convert_to_tensor(Y_val)
    # X_tset = tf.convert_to_tensor(X_tset)
    # Y_test = tf.convert_to_tensor(Y_test)
    
    # print(X_val.shape)
    # print(Y_val.shape)
    
    print('\nData Rreading, and Splitting Done!\n')
    
    
    return X_train, X_test, Y_train, Y_test,X_val,Y_val






  
