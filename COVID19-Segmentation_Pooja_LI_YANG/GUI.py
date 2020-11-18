from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QSlider, QApplication
from PyQt5.QtCore import Qt
import os
import sys
import numpy as np
from PIL import Image
import matplotlib as plt
import h5py
from keras.models import load_model
import keras.backend as K
import cv2
import keras
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setObjectName("MainWindow")
        self.resize(1100, 900)
        self.setWindowTitle("Scene Segmentation Project by Yunke LI, Qi YANG, Kumari Pooja，MSCV1，June，2020")
        font_1 = QtGui.QFont()
        font_1.setFamily("Arial")
        font_1.setPointSize(12)

        font_2 = QtGui.QFont()
        font_2.setFamily("Microsoft YaHei")
        font_2.setPointSize(22)

        self.ResultDisplay = QtWidgets.QLabel(self)
        self.ResultDisplay.setText("Welcome!")
        self.ResultDisplay.setFont(font_1)
        self.ResultDisplay.move(70, 820)

        self.Title = QtWidgets.QLabel(self)
        self.Title.setText("Scene Segmentation Project DEMO")
        self.Title.setFont(font_2)
        self.Title.move(70, 40)
        self.Title.adjustSize()

        self.LoadRawButton = QtWidgets.QPushButton(self)
        self.LoadRawButton.setGeometry(70, 120, 256, 50)
        self.LoadRawButton.setObjectName("LoadButton")
        self.LoadRawButton.setText("Load File")
        self.LoadRawButton.setFont(font_1)
        self.LoadRawButton.clicked.connect(self.LoadFile)

        self.LoadModelButton = QtWidgets.QPushButton(self)
        self.LoadModelButton.setObjectName("LoadModel")
        self.LoadModelButton.setText("Load Model")
        self.LoadModelButton.clicked.connect(self.LoadModel)
        self.LoadModelButton.setFont(font_1)
        self.LoadModelButton.setGeometry(70, 200, 256, 50)

        self.LoadGTButton = QtWidgets.QPushButton(self)
        self.LoadGTButton.setObjectName("LoadGT")
        self.LoadGTButton.setText("Load GT")
        self.LoadGTButton.clicked.connect(self.LoadGT)
        self.LoadGTButton.setFont(font_1)
        self.LoadGTButton.setGeometry(70, 280, 256, 50)

        self.ShowMap = QtWidgets.QLabel(self)
        # self.ShowMap.move(350, 110)
        self.ShowMap.move(70, 520)
        self.ShowMap.setObjectName("ShowMap")
        self.ShowMap.setScaledContents(True)
        # self.ShowMap.setPixmap(QtGui.QPixmap('logo.png'))

        self.disMap = QtWidgets.QLabel(self)
        self.disMap.move(400, 520)
        self.disMap.setObjectName("DiseaseShowMap")
        self.disMap.setScaledContents(True)
        # self.disMap.setPixmap(QtGui.QPixmap('logo.png'))

        self.llMap = QtWidgets.QLabel(self)
        self.llMap.move(400, 150)
        self.llMap.setObjectName("LeftLungShowMap")
        self.llMap.setScaledContents(True)
        # self.llMap.setPixmap(QtGui.QPixmap('logo.png'))

        self.rlMap = QtWidgets.QLabel(self)
        self.rlMap.move(720, 150)
        self.rlMap.setObjectName("RightLungShowMap")
        self.rlMap.setScaledContents(True)
        # self.rlMap.setPixmap(QtGui.QPixmap('logo.png'))

        self.bgMap = QtWidgets.QLabel(self)
        self.bgMap.move(720, 520)
        self.bgMap.setObjectName("BackGroundShowMap")
        self.bgMap.setScaledContents(True)
        # self.bgMap.setPixmap(QtGui.QPixmap('logo.png'))

        self.SegButton = QtWidgets.QPushButton(self)
        self.SegButton.setObjectName("Segment")
        self.SegButton.setText("Segment")
        self.SegButton.setGeometry(70, 360, 256, 50)
        self.SegButton.setFont(font_1)
        self.SegButton.clicked.connect(self.Segment)

        self.QuitButton = QtWidgets.QPushButton(self)
        self.QuitButton.setObjectName("Quit")
        self.QuitButton.setText("Quit")
        self.QuitButton.setGeometry(70, 440, 256, 50)
        self.QuitButton.setFont(font_1)
        self.QuitButton.clicked.connect(QApplication.quit)

        self.lllabel = QtWidgets.QLabel(self)
        self.lllabel.setText('Left Lung Mask: ')
        self.lllabel.move(400, 120)
        self.lllabel.setFont(font_1)
        self.lllabel.adjustSize()

        self.rllabel = QtWidgets.QLabel(self)
        self.rllabel.setText('Right Lung Mask: ')
        self.rllabel.move(720, 120)
        self.rllabel.setFont(font_1)
        self.rllabel.adjustSize()

        self.dislabel = QtWidgets.QLabel(self)
        self.dislabel.setText('Disease Mask: ')
        self.dislabel.move(400, 490)
        self.dislabel.setFont(font_1)
        self.dislabel.adjustSize()

        self.bglabel = QtWidgets.QLabel(self)
        self.bglabel.setText('Background Mask: ')
        self.bglabel.move(720, 490)
        self.bglabel.setFont(font_1)
        self.bglabel.adjustSize()

        self.hlllabel = QtWidgets.QLabel(self)
        self.hlllabel.setText('Hausdorff: ')
        self.hlllabel.setFont(font_1)
        self.hlllabel.move(400, 420)
        self.hlllabel.adjustSize()
        self.hlledit = QtWidgets.QLineEdit(self)
        self.hlledit.move(480, 420)
        self.hlledit.setText('0')
        self.dlllabel = QtWidgets.QLabel(self)
        self.dlllabel.setText('Dice Coef: ')
        self.dlllabel.setFont(font_1)
        self.dlllabel.move(400, 445)
        self.dlllabel.adjustSize()
        self.dlledit = QtWidgets.QLineEdit(self)
        self.dlledit.move(480, 445)
        self.dlledit.setText('0')

        self.hrllabel = QtWidgets.QLabel(self)
        self.hrllabel.setText('Hausdorff: ')
        self.hrllabel.setFont(font_1)
        self.hrllabel.move(720, 420)
        self.hrllabel.adjustSize()
        self.hrledit = QtWidgets.QLineEdit(self)
        self.hrledit.move(800, 420)
        self.hrledit.setText('0')
        self.drllabel = QtWidgets.QLabel(self)
        self.drllabel.setText('Dice Coef: ')
        self.drllabel.setFont(font_1)
        self.drllabel.move(720, 445)
        self.drllabel.adjustSize()
        self.drledit = QtWidgets.QLineEdit(self)
        self.drledit.move(800, 445)
        self.drledit.setText('0')

        self.hdislabel = QtWidgets.QLabel(self)
        self.hdislabel.setText('Hausdorff: ')
        self.hdislabel.setFont(font_1)
        self.hdislabel.move(400, 790)
        self.hdislabel.adjustSize()
        self.hdisedit = QtWidgets.QLineEdit(self)
        self.hdisedit.move(480, 790)
        self.hdisedit.setText('0')
        self.ddislabel = QtWidgets.QLabel(self)
        self.ddislabel.setText('Dice Coef: ')
        self.ddislabel.setFont(font_1)
        self.ddislabel.move(400, 815)
        self.ddislabel.adjustSize()
        self.ddisedit = QtWidgets.QLineEdit(self)
        self.ddisedit.move(480, 815)
        self.ddisedit.setText('0')

        self.hbglabel = QtWidgets.QLabel(self)
        self.hbglabel.setText('Hausdorff: ')
        self.hbglabel.setFont(font_1)
        self.hbglabel.move(720, 790)
        self.hbglabel.adjustSize()
        self.hbgedit = QtWidgets.QLineEdit(self)
        self.hbgedit.move(800, 790)
        self.hbgedit.setText('0')
        self.dbglabel = QtWidgets.QLabel(self)
        self.dbglabel.setText('Dice Coef: ')
        self.dbglabel.setFont(font_1)
        self.dbglabel.move(720, 815)
        self.dbglabel.adjustSize()
        self.dbgedit = QtWidgets.QLineEdit(self)
        self.dbgedit.move(800, 815)
        self.dbgedit.setText('0')

        self.filename = None
        self.filepath = None
        self.rawdata = None
        self.qPix_img = None
        self.qPix_bg = None
        self.qPix_dis = None
        self.qPix_ll = None
        self.qPix_rl = None
        self.img_bg = None
        self.img_rl = None
        self.img_ll = None
        self.img_dis = None
        self.modelfilename = None
        self.model = None
        self.GT = None

    def LoadModel(self):
        print("Model Loading......")
        self.ResultDisplay.setText("Model Loading......")
        self.ResultDisplay.adjustSize()

        modelfilename, _ = QFileDialog.getOpenFileName(self, "Open Model", "./", "All Files (*);;NIFTI (*.gz)")
        self.modelfilename = modelfilename

        def Tversky_Loss(y_true, y_pred, smooth=1, alpha=0.3, beta=0.7):
            # if flatten:
            y_true = K.flatten(y_true)
            y_pred = K.flatten(y_pred)

            TP = K.sum(y_true * y_pred)
            FP = K.sum((1 - y_true) * y_pred)
            FN = K.sum(y_true * (1 - y_pred))

            tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

            return 1 - tversky_coef

        def Focal_Loss(y_true, y_pred, alpha=0.8, gamma=2.0):
            # if flatten:
            y_true = K.flatten(y_true)
            y_pred = K.flatten(y_pred)

            bce = keras.losses.binary_crossentropy(y_true, y_pred)
            bce_exp = K.exp(-bce)

            loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)
            return loss

        def weighted_bce(weight=0.6):
            def convert_2_logits(y_pred):
                y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
                return tf.math.log(y_pred / (1 - y_pred))

            def weighted_binary_crossentropy(y_true, y_pred):
                y_pred = convert_2_logits(y_pred)
                loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weight)
                return loss

            return weighted_binary_crossentropy

        def Combo_Loss(y_true, y_pred, a=0.4, b=0.2, c=0.4):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)

            return a * weighted_bce()(y_true, y_pred) + b * Focal_Loss(y_true_f, y_pred_f) + c * Tversky_Loss(y_true_f,
                                                                                                              y_pred_f)

        def Dice_coef(y_true, y_pred, smooth=1):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)

            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        self.model = load_model(modelfilename, custom_objects={'Combo_Loss': Combo_Loss, 'Dice_coef': Dice_coef})
        self.ResultDisplay.setText("Model Loaded!")
        self.ResultDisplay.adjustSize()
        print("Model Loaded!")

    def LoadFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Image", "./", "All Files (*);;png (*.png)")
        self.filename = filename
        self.rawdata = cv2.imread(filename)
        img_pil = Image.fromarray(np.uint8(self.rawdata))
        # size = [img_pil.size[0], img_pil.size[1]]
        # size = [x * 2 for x in size]
        # img_pil = img_pil.resize((size))
        self.qPix_img = img_pil.toqpixmap()
        self.ShowMap.setPixmap(self.qPix_img)
        self.ShowMap.adjustSize()
        self.ResultDisplay.setText("Image Loaded!")
        self.ResultDisplay.adjustSize()
        print("Image Loaded!")

    def LoadGT(self):
        filename_GT, _ = QFileDialog.getOpenFileName(self, "Load Ground Truth", "./", "All Files (*);;png (*.png)")
        self.GT = cv2.imread(filename_GT, 0)
        self.ResultDisplay.setText("Ground Truth Loaded")
        self.ResultDisplay.adjustSize()
        print("Ground Truth Loaded")

    def Segment(self):
        if self.rawdata is None:
            return
        if self.model is None:
            return

        print("Segment Processing......")
        self.ResultDisplay.setText("Segment Processing......")
        self.ResultDisplay.adjustSize()
        model = self.model
        imgg = self.rawdata
        image = np.expand_dims(imgg, axis=0)
        pr_mask = model.predict(image)
        pr_mask = pr_mask.squeeze()
        pr_mask_max = np.argmax(pr_mask, axis=-1)
        self.img_bg = (pr_mask_max == 0).astype(np.int8) * 255
        self.img_rl = (pr_mask_max == 1).astype(np.int8) * 255
        self.img_ll = (pr_mask_max == 2).astype(np.int8) * 255
        self.img_dis = (pr_mask_max == 3).astype(np.int8) * 255

        img_pil_ll = Image.fromarray(np.uint8(self.img_ll))
        self.qPix_ll = img_pil_ll.toqpixmap()
        self.llMap.setPixmap(self.qPix_ll)
        self.llMap.adjustSize()

        img_pil_rl = Image.fromarray(np.uint8(self.img_rl))
        self.qPix_rl = img_pil_rl.toqpixmap()
        self.rlMap.setPixmap(self.qPix_rl)
        self.rlMap.adjustSize()

        img_pil_dis = Image.fromarray(np.uint8(self.img_dis))
        self.qPix_dis = img_pil_dis.toqpixmap()
        self.disMap.setPixmap(self.qPix_dis)
        self.disMap.adjustSize()

        img_pil_bg = Image.fromarray(np.uint8(self.img_bg))
        self.qPix_bg = img_pil_bg.toqpixmap()
        self.bgMap.setPixmap(self.qPix_bg)
        self.bgMap.adjustSize()

        print("Segment Done!")
        self.ResultDisplay.setText("Segment Done!")
        self.ResultDisplay.adjustSize()

        if self.GT is None:
            return

        def hausdorff(pred_mask, tru_mask):
            pred_mask_max = np.argmax(pred_mask, axis=-1)
            pr_bg = (pred_mask_max == 0).astype(np.int8)
            pr_rl = (pred_mask_max == 1).astype(np.int8)
            pr_ll = (pred_mask_max == 2).astype(np.int8)
            pr_dis = (pred_mask_max == 3).astype(np.int8)
            truth_bg = (tru_mask == 0).astype(np.int8)
            truth_rl = (tru_mask == 1).astype(np.int8)
            truth_ll = (tru_mask == 2).astype(np.int8)
            truth_dis = (tru_mask == 3).astype(np.int8)
            d_bg = directed_hausdorff(pr_bg, truth_bg)[0]
            d_rl = directed_hausdorff(pr_rl, truth_rl)[0]
            d_ll = directed_hausdorff(pr_ll, truth_ll)[0]
            d_dis = directed_hausdorff(pr_dis, truth_dis)[0]
            return d_bg, d_rl, d_ll, d_dis

        def dice_coef(y_true, y_pred, smooth=1):
            y_true_f = y_true.flatten()
            y_pred_f = y_pred.flatten()
            intersection = y_true_f * y_pred_f
            intersection = intersection.sum()
            return (2 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

        def Dice(pred_mask, tru_mask):
            pred_mask_max = np.argmax(pred_mask, axis=-1)
            pr_bg = (pred_mask_max == 0).astype(np.int8)
            pr_rl = (pred_mask_max == 1).astype(np.int8)
            pr_ll = (pred_mask_max == 2).astype(np.int8)
            pr_dis = (pred_mask_max == 3).astype(np.int8)
            truth_bg = (tru_mask == 0).astype(np.int8)
            truth_rl = (tru_mask == 1).astype(np.int8)
            truth_ll = (tru_mask == 2).astype(np.int8)
            truth_dis = (tru_mask == 3).astype(np.int8)
            dice_bg = dice_coef(truth_bg, pr_bg)
            dice_rl = dice_coef(truth_rl, pr_rl)
            dice_ll = dice_coef(truth_ll, pr_ll)
            dice_dis = dice_coef(truth_dis, pr_dis)
            return dice_bg, dice_rl, dice_ll, dice_dis

        truth = self.GT
        prediction = pr_mask
        Hd_bg, Hd_rl, Hd_ll, Hd_dis = hausdorff(prediction, truth)
        Di_bg, Di_rl, Di_ll, Di_dis = Dice(prediction, truth)
        self.hbgedit.setText(str(round(Hd_bg, 4)))
        self.hrledit.setText(str(round(Hd_rl, 4)))
        self.hlledit.setText(str(round(Hd_ll, 4)))
        self.hdisedit.setText(str(round(Hd_dis, 4)))
        self.dbgedit.setText(str(round(Di_bg, 4)))
        self.drledit.setText(str(round(Di_rl, 4)))
        self.dlledit.setText(str(round(Di_ll, 4)))
        self.ddisedit.setText(str(round(Di_dis, 4)))

        def Dice_coef(y_true, y_pred, smooth=1):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)

            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
