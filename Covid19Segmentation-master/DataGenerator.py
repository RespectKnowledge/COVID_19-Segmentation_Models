from tensorflow.keras.utils import Sequence
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as gen

# Generate Batches (32 size) with Augmentation 
def GenerateBatches(dir,imagePath, maskPath):
    # Generate batches (32 images) by iteration 
    trainImages = gen(rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest").flow_from_directory(directory = dir,classes = [imagePath], batch_size=32, target_size=(128, 128),color_mode = 'rgb')

    maskImages = gen(rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest").flow_from_directory(directory = dir,classes = [maskPath],batch_size=32, target_size=(128, 128),color_mode = 'grayscale')

    train_generator = zip(trainImages, maskImages)

    for (img,mask) in train_generator:
        yield (img,mask)
    


# # class for datalader into batch of data
# class DataGenerator(Sequence):
#     # input : (X_train, T_train)
#     """Load data from dataset and form batches
    
#     Args:
#         dataset: instance of Dataset class for image loading and preprocessing.
#         batch_size: Integet number of images in batch.
#         shuffle: Boolean, if `True` shuffle image indexes each epoch.
#     """
#     def __init__(self, dataset, batch_size=1, shuffle=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indexes = np.arange(len(dataset))

#         self.on_epoch_end()

#     def __getitem__(self, i):
        
#         # collect batch data
#         start = i * self.batch_size
#         stop = (i + 1) * self.batch_size
#         data = []
#         for j in range(start, stop):
#             data.append(self.dataset[j])
        
#         # Transpose list of lists
#         batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
#         return batch
    
#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return len(self.indexes) // self.batch_size
    
#     def on_epoch_end(self):
#         """Callback function to shuffle indexes each epoch"""
#         if self.shuffle:
#             self.indexes = np.random.permutation(self.indexes)