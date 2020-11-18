import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
import sys
import glob
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.colors



import keras.backend as K
#########################################################################################################################


#np.set_printoptions(threshold=sys.maxsize)
seed = 4558
np.random.seed = seed



train_image_dir = 'dataset_seperated_converted_384x384/train'   
test_image_dir = 'dataset_seperated_converted_384x384/test'
val_image_dir = 'dataset_seperated_converted_384x384/validate'

img_fname       = 'images'  # folder_name train images
mask_fname      = 'masks'  # folder_name of train masks


def get_train_imgs():
    img_path = os.path.join(train_image_dir,img_fname)
    images = glob.glob(os.path.join(img_path,'*.*'))
    mask_path = os.path.join(train_image_dir,mask_fname)
    masks = glob.glob(os.path.join(mask_path,'*.*'))
    return [os.path.basename(image) for image in images],[os.path.basename(mask) for mask in masks]

# print(get_tain_imgs())


def get_validate_imgs():
    val_path = os.path.join(val_image_dir,img_fname)
    imagesval = glob.glob(os.path.join(val_path,'*.*'))
    mask_pathval = os.path.join(val_image_dir,mask_fname)
    masksval = glob.glob(os.path.join(mask_pathval,'*.*'))
    return [os.path.basename(imageval) for imageval in imagesval],[os.path.basename(maskval) for maskval in masksval]


'''
def get_test_imgs():
	test_img_path = os.path.join(test_image_dir,img_fname)
	test_img = glob.glob(os.path.join(test_img_path,'*.*'))
	return[os.path.basename(testimage) for testimage in test_img],[]
'''

TRAIN_IMGS = get_train_imgs()
#TEST_IMGS = get_test_imgs()
VAL_IMGS=get_validate_imgs()

all_batches = TRAIN_IMGS
#all_test = TEST_IMGS
all_val=VAL_IMGS
# print(all_test)

img_path  = os.path.join(train_image_dir,img_fname)
mask_path = os.path.join(train_image_dir,mask_fname)
#test_img_path = os.path.join(test_image_dir,img_fname)
val_path = os.path.join(val_image_dir,img_fname)
mask_pathval = os.path.join(val_image_dir,mask_fname)


height= 384
width=384
channels=1


BATCH_SIZE=4
LR = 0.001 
EPOCHS = 25



X_train = np.zeros((len(all_batches[0]),height,width,channels),dtype=np.float32) 
Y_train = np.zeros((len(all_batches[1]),height,width,4), dtype=np.bool)
X_val = np.zeros((len(all_val[0]),height,width,channels),dtype=np.float32) 
Y_val = np.zeros((len(all_val[1]),height,width,4), dtype=np.bool)
#X_test = np.zeros((len(all_test[0]),height,width,channels), dtype=np.float32)

# loading train images
c_img = np.zeros((len(all_batches[0]),height,width,channels),dtype=np.float32)
for num in range(len(all_batches[0])):
	img1 = os.path.join(img_path,all_batches[0][num])
	c_img  = cv2.imread(img1,0)
	#c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
	norm_image = cv2.normalize(c_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	norm_image  = np.expand_dims(norm_image,axis=-1)
	X_train[num] = norm_image



for mask_file in range(len(all_batches[1])):
	img2 = os.path.join(mask_path,all_batches[1][mask_file])
	mask = cv2.imread(img2,0)
	#mask = np.expand_dims(mask, axis=-1)
	#held= cv2.imread('coronacases_007.ni_z121.png',0)
	masker=tf.keras.utils.to_categorical(mask, num_classes=4, dtype='uint8')*255
	masker=masker.astype(np.bool)
	Y_train[mask_file] = masker




# loading validation images
c_img = np.zeros((len(all_val[0]),height,width,channels),dtype=np.float32)
for num in range(len(all_val[0])):
	img1 = os.path.join(val_path,all_val[0][num])
	c_img  = cv2.imread(img1,0)
	#c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
	norm_image = cv2.normalize(c_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	norm_image  = np.expand_dims(norm_image,axis=-1)
	X_val[num] = norm_image



for mask_file in range(len(all_val[1])):
	img2 = os.path.join(mask_pathval,all_val[1][mask_file])
	mask = cv2.imread(img2,0)
	#mask = np.expand_dims(mask, axis=-1)
	#held= cv2.imread('coronacases_007.ni_z121.png',0)
	masker=tf.keras.utils.to_categorical(mask, num_classes=4, dtype='uint8')*255
	Y_val[mask_file] = masker



'''
# loading test images
test_img1=np.zeros((len(all_batches[0]),height,width,channels),dtype=np.float32)
for test in range(len(all_test[0])):
	img3 = os.path.join(test_img_path,all_test[0][test])
	test_img1  = cv2.imread(img3,0)
	test_img1  = np.expand_dims(test_img1,axis=-1)
	test_img1/255
	X_test[test] = test_img1
'''

#imshow(np.squeeze(X_val[24]))	
#print('Train image', X_train[121].min(), X_train[121].max(), X_train[121].mean(), X_train[121].std())
#print('train mask', Y_train[121].min(), Y_train[121].max())
#plt.show()
# loading train mask images	


# print(Y_train)
print('img',X_train.shape,'mask',Y_train.shape,'img',X_val.shape,'mask',Y_val.shape)
# print(Y_train.dtype)

'''
# Display above compiled data
image_x = random.randint(0,len(all_batches[0]))
image_y = random.randint(0,len(all_test[0]))
fig, axes = plt.subplots(1, 6)
axes[0].imshow(np.squeeze(X_train[image_x]),cmap='gray')
axes[1].imshow(Y_train[image_x,...,0].squeeze(),cmap='gray')
axes[2].imshow(Y_train[image_x,...,1].squeeze(),cmap='gray')
axes[3].imshow(Y_train[image_x,...,2].squeeze(),cmap='gray')
axes[4].imshow(Y_train[image_x,...,3].squeeze(),cmap='gray')
axes[5].imshow(np.squeeze(X_test[image_y]),cmap='gray')                        
#axes[3].imshow(np.squeeze(Y_test[image_x]),cmap='gray')
print('Y_train data',Y_train.shape,Y_train.dtype)
plt.show()
'''


#########################################################################################################################


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
    vertical_flip=False, rescale=None, preprocessing_function=None,
    data_format=None, validation_split=0.1, dtype=float 
)


datagen.fit(X_train)


validgen=tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
    vertical_flip=False, rescale=None, preprocessing_function=None,
    data_format=None, validation_split=0.1, dtype=float 
)


'''
validation_generator = validgen.flow_from_directory(
        'dataset_seperated_converted_384x384/validate',
        batch_size=BATCH_SIZE,
        class_mode=None)
'''


print('heyyyyyyyyyyyyyyyyyyyyyyyyyy',len(X_val))

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def Tversky_Loss(y_true, y_pred, smooth = 1, alpha = 0.3, beta = 0.7, flatten = False):
    
    if flatten:
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
    
    TP = K.sum(y_true * y_pred)
    FP = K.sum((1-y_true) * y_pred)
    FN = K.sum(y_true * (1-y_pred))
    
    tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    
    return 1 - tversky_coef

def Focal_Loss(y_true, y_pred, alpha = 0.8, gamma = 2.0, flatten = False):
    
    if flatten:
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)    
    
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)
    
    loss = K.mean(alpha * K.pow((1-bce_exp), gamma) * bce)
    return loss

def weighted_bce(weight = 0.6):
    
    def convert_2_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.log(y_pred / (1-y_pred))
    
    def weighted_binary_crossentropy(y_true, y_pred):
        y_pred = convert_2_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits = y_pred, targets = y_true, pos_weight = weight)
        return loss
    
    return weighted_binary_crossentropy

def Combo_Loss(y_true, y_pred, a = 0.4, b = 0.2, c= 0.4):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    return a*weighted_bce()(y_true, y_pred) + b*Focal_Loss(y_true_f, y_pred_f) + c*Tversky_Loss(y_true_f, y_pred_f)



def UNET(input_shape=(height,width,channels)):
	#Build the model
	inputs = tf.keras.layers.Input(input_shape)
	#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)## converting image into float points by division

	#Contraction path
	c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
	## 16 is the number of filters and 3x3 their size, kernel initialiser initialises whats inside comv kernal
	## multiplied by s means what the layer is applied on
	c1 = tf.keras.layers.Dropout(0.1)(c1)
	c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
	p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

	c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
	c2 = tf.keras.layers.Dropout(0.1)(c2)
	c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
	p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
	 
	c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
	c3 = tf.keras.layers.Dropout(0.2)(c3)
	c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
	p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
	 
	c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
	c4 = tf.keras.layers.Dropout(0.2)(c4)
	c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
	p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
	 
	c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
	c5 = tf.keras.layers.Dropout(0.3)(c5)
	c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

	#Expansive path 
	u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
	u6 = tf.keras.layers.concatenate([u6, c4])
	c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
	c6 = tf.keras.layers.Dropout(0.2)(c6)
	c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
	 
	u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
	u7 = tf.keras.layers.concatenate([u7, c3])
	c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
	c7 = tf.keras.layers.Dropout(0.2)(c7)
	c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
	 
	u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
	u8 = tf.keras.layers.concatenate([u8, c2])
	c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
	c8 = tf.keras.layers.Dropout(0.1)(c8)
	c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
	 
	u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
	u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
	c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
	c9 = tf.keras.layers.Dropout(0.1)(c9)
	c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
	 
	outputs= tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(c9)
	 
	model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

	return model



model = UNET(input_shape=(height,width,channels))
model.summary()


# Define callbacks for learning rate scheduling, logging and best checkpoints saving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('Model COVID19.h5',save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1, patience=2, mode='min'), ## new_lr = lr * factor # 5
    tf.keras.callbacks.EarlyStopping(monitor='dice_coef', min_delta=0.005, verbose=1, patience=4, mode='max', restore_best_weights=True),    
    tf.keras.callbacks.CSVLogger('training.csv'),
    tf.keras.callbacks.TensorBoard(log_dir='logs',write_graph=True),
    tf.keras.callbacks.TerminateOnNaN()
]



model.fit(datagen.flow(X_train, Y_train,batch_size=BATCH_SIZE,shuffle=True),validation_steps=len(X_val)/BATCH_SIZE,
	validation_data=validgen.flow(X_val,Y_val,batch_size=BATCH_SIZE),steps_per_epoch=len(X_train)/BATCH_SIZE, epochs=EPOCHS,callbacks=callbacks)

model.save('U-Net COVID19 Segment model')



