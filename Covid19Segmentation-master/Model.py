from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.layers import Input, Conv2D
from keras.models import Model
import keras
import segmentation_models as sm
import numpy as np


model = 0
def Network(input):
    global model
    #BACKBONE = 'efficientnetb7' # Accuracy : 0.8771349 with 20 epoch, lr =1e-4, loss: bce_dice_loss, metrices: iou_score
    #BACKBONE = 'densenet201' # Accuracy : 0.86626434 with 20 epoch, lr = 1e-4, loss: bce_dice_loss, metrices: iou_score
    #BACKBONE = 'inceptionresnetv2' # Accuracy: 0.60136926, 20 epoch, lr = le-5, ----
    BACKBONE = 'mobilenetv2' # Adam(lr=1e-3)
    base_model = Unet(BACKBONE, encoder_weights='imagenet',classes=4, activation='softmax')
    N = input.shape[-1]
    inp = Input(shape=(None, None, N )) # input layer
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1) # output 

    model = Model(inp, out, name=base_model.name)

    # Choose Adam Optimizer 
    optim = keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optim,
        loss=sm.losses.bce_dice_loss,
        metrics=[sm.metrics.iou_score],
    )   

def Fit(X_train,Y_train,X_val,Y_val ):
    global model
    # treain the model 
    history = model.fit(
       x=X_train,
       y=Y_train,
       batch_size=16,
       epochs=20,
       validation_data=(X_val, Y_val)
    )
    # Save
    model.save('drive/My Drive/Project/mobilenetv2.h5')

    np.save('drive/My Drive/Project/historymobilenetv2.npy',history.history)
    print('Training done.')

    return history, model

