#important Packages 
import os
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2 
from nibabel.testing import data_path 
import pickle as pk 
import PIL.Image as im
import SimpleITK as sitk

#........................Loading the MRI images From Nifti Format..................................

def loadNifti(path): # this class returns nifti image array numpy type
    img1 = nib.load(path)
    trainingImg = img1.get_fdata() # 4d, to show the animated cardiac image
    return trainingImg

def loadSimpleITK(path):
    result =  sitk.ReadImage(path)
    return result

def loadNiftSimpleITK(root , file):
    path =  root + '/' + file 
    result =  sitk.ReadImage(path)
    return result


def loadAllNifti(root , file): # this class returns nifti image array numpy type
    path =  root + '/' + file 
    img1 = nib.load(path)
    trainingImg = img1.get_fdata() # 4d, to show the animated cardiac image
    return trainingImg

#.........................Get Separate Slices.............................................

def getSlice(image,numOfSlice):
    if(numOfSlice >= image.shape[2]):
        print("number of slices is only", image.shape[2])
        return 0
    else:    
        return image[:,:,numOfSlice]
    
def getNotNumpySliceITK(image,numOfSlice):
    
    if(numOfSlice >= image.GetSize()[2]):
        print("number of slices is only", image.GetSize()[2])
        return 0
    else:    
    
        return image[:,:,numOfSlice]
    
def getSliceITK(image,numOfSlice):
    if(numOfSlice >= image.GetSize()[2]):
        print("number of slices is only", image.GetSize()[2])
        return 0
    else:    
        return sitk.GetArrayFromImage(image[:,:,numOfSlice] )
    
def displaySlices(imageGT,sliceNum):
    plt.imshow(getSlice(imageGT,sliceNum))
    plt.show()
"""
imageGT : image loaded fron GT file 
sliceNum : it has 10 slices , from 0-9 
"""        

#........................ Display the images...................................................

def displayAnimatedNifti(niftiImage, sliceNum1):
    slicePat1 = [] # initalizing a list 
    for t in range(niftiImage.shape[3]): # 30 frames, the image shape (width, heigh, # of slices , 30 frame for each slice), thats why we put 30 parameter for our array that holds 30 frame for slice 0 to animate
        slicePat1.append(niftiImage[:, :, sliceNum1,t])    
    fig = plt.figure() # make figure
    axes = fig.subplots(1,2) # make two subplots in the figure
    im = axes[1].imshow(slicePat1[0], vmin=0, vmax=255,cmap="gray", origin="lower") #show 3d animated image
    def updatefig(j):
        im.set_array(slicePat1[j])
        return [im]
    ani = animation.FuncAnimation(fig, updatefig, frames=range(np.array(slicePat1).shape[0]), 
                              interval=50, blit=True)
    plt.show()   
    
    

"""
niftiImage : image loaded from nii.gz file 4d, it has 10 slices, from 0-9 , each slice with 30 frame animation.
sliceNum1 : number of slice from 0 - 9
"""

