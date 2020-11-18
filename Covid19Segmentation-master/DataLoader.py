import os
import numpy as np
import matplotlib.pyplot as plt 
import glob
from skimage.io import imsave
from skimage.transform import resize
from sklearn.preprocessing import normalize
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow




def nifToPng(imagesPath, masksPath):
	# Exract the real cases nifti files and save them to 'DataSet\images\',
	# temporal google server for speed  
	Dataset = glob.glob( os.path.join(imagesPath, '*.gz') )
	ctr = 0
	for image in Dataset:
	    images = nib.load(image).get_fdata()
	    ctr+=1
	    for _id in range(images.shape[2]):
	    	# Resize & Normalize 
	        imgProcessed = preProcess(images[:,:,_id], 128)         
	        imsave(os.path.join('DataSet/images',
	                            str(ctr)+'_'+str(_id+1)+'.png'),imgProcessed)
	# Extract the labeled nifti files (masks) and save them to 'DataSet\images\',
	# temporal google server for speed       
	Dataset = glob.glob( os.path.join(masksPath, '*.gz') )
	ctr = 0
	for image in Dataset:
	    images = nib.load(image).get_fdata()
	    ctr+=1
	    for _id in range(images.shape[2]):
	        resizedImgs = resize(images[:,:,_id],(128,128),preserve_range=True).astype(np.uint8)
	        htImg = OneHotEncoding(resizedImgs,4)
	        masks_path = os.path.join('DataSet/masks',str(ctr)+'_'+str(_id+1)+'.png')
	        imsave(masks_path, htImg )
	        

def OneHotEncoding(im,n_classes):
            one_hot = np.zeros((im.shape[0], im.shape[1], n_classes),dtype=np.uint8)
            for i, unique_value in enumerate(np.unique(im)):
                one_hot[:, :, i][im == unique_value] = 1
            return one_hot



# Resize and Normalize the image 		
def preProcess(image, size):
	imgRes = resize(image, (size,size))
	imgNorm = normalize(imgRes)
	return imgNorm

# Split The Data to Test and Train and Validate		
def splitData():
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

	#  Train Test Validate
	ratio=0.1
	# train data ratio: 0.9, validation data ratio: 0.1. for images(X) sand masks(y)
	X_, X_val, Y_, Y_val = train_test_split(X, Y, test_size=ratio,random_state= 42)
	# another split for train data(X_, Y_) : 0.1 / (0.9) ~ 0.11, approx : 0.8 train, 0.1 val , 0.1 test
	X_train, X_tset, Y_train, Y_test = train_test_split(X_, Y_, test_size=ratio/(1-ratio),random_state=42)
	print('\nSplit Done..\n')
	return X_train, X_tset, Y_train, Y_test, X_, X_val, Y_, Y_val,X

	def savePredictions(pred): 
		path = 'drive/My Drive/Project/'
		for i in range(pred.shape[0]):
			imsave(path,pred[i])


def DataGenerator(Sequence): 
	pass