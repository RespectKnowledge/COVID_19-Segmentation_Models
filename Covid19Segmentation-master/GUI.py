
#...........................Imported libraries..................
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import qimage2ndarray
import  cv2
import  os
from PyQt5.QtCore import Qt
#####################################################################
import loadnif as nif
#import  main as mn 
#import  numpy as np



class Ui_Dialog(object):
    
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1500, 540)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 20, 71, 16))
        self.label.setObjectName("label")
        self.segmetLabel = QtWidgets.QLabel(Dialog)
        self.segmetLabel.setGeometry(QtCore.QRect(479, 40, 384, 308))
        self.segmetLabel.setStyleSheet("background-color: white; border: 1px inset grey; min-height: 200px;")

        self.segmetLabel.setObjectName("segmetLabel")
        self.Upload_bott = QtWidgets.QPushButton(Dialog)
        self.Upload_bott.setGeometry(QtCore.QRect(120, 15, 150, 23))
        self.Upload_bott.setObjectName("Upload_bott")
        self.Segment_bott = QtWidgets.QPushButton(Dialog)
        self.Segment_bott.setGeometry(QtCore.QRect(395, 80, 80, 23))
        self.Segment_bott.setObjectName("Segment_bott")
        
        self.train_bott = QtWidgets.QPushButton(Dialog)
        self.train_bott.setGeometry(QtCore.QRect(395, 140, 80, 23))
        self.train_bott.setObjectName("train_bott")
        
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(500, 20, 151, 16))
        self.label_2.setObjectName("label_2")
      
        self.imageTest = QtWidgets.QLabel(Dialog)
        self.imageTest.setStyleSheet("background-color: white; border: 1px inset grey; min-height: 200px;")
        self.imageTest.setGeometry(QtCore.QRect(5, 40, 384, 308))
        self.imageTest.setText("")
        self.imageTest.setObjectName("imageTest")
        
        self.slicer = QtWidgets.QSlider(Qt.Horizontal)
        self.slicer.setGeometry(280, 300, 20, 50)
      
        self.infoLabel = QtWidgets.QLabel(Dialog)
        self.infoLabel.setGeometry(QtCore.QRect(920, 20, 121, 16))
        self.infoLabel.setObjectName("infoLabel")
        #####################################################
        self.cnvJPG = QtWidgets.QPushButton(Dialog)
        self.cnvJPG.setGeometry(QtCore.QRect(250, 380, 80, 23))
        self.cnvJPG.setObjectName("cnvJPG")
        # self.plt3D = QtWidgets.QPushButton(Dialog)
        # self.plt3D.setGeometry(QtCore.QRect(160, 380, 80, 23))
        # self.plt3D.setObjectName("plt3D")

        #############################################
        self.jpgSaveLabel = QtWidgets.QLabel(Dialog)
        self.jpgSaveLabel.setGeometry(QtCore.QRect(250, 360, 71, 23))
        self.jpgSaveLabel.setObjectName("jpgSaveLabel")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(185, 360, 91, 23))
        self.label_3.setObjectName("label_3")


        


        # Set background color 
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Background,QtGui.QColor(174, 173, 172));
        Dialog.autoFillBackground()
        Dialog.setPalette(pal)
        

        self.Upload_bott.clicked.connect(self.loadNifti)
        self.cnvJPG.clicked.connect(self.savePNG)
        self.Segment_bott.clicked.connect(self.ActiveShapeModel)
        self.train_bott.clicked.connect(self.Train)

        # self.plt3D.clicked.connect(self.draw3D)



        
        self.msg = QtWidgets.QMessageBox()
        
        
        # disable segment button until the user manually segment all the manual contours required 
        self.Segment_bott.setEnabled(False)
        
        # disable save jpg button until the two images are loaded(test image and segmented image)
        self.cnvJPG.setEnabled(False)
        

        # disable show 3D button
        # self.plt3D.setEnabled(False)

     
        
        # initalise the x,y coordinates 
        self.last_x, self.last_y = None, None

        # initalise variable to hold the image of nifti file 
        self.imginit=0
        
        #self.flagLabelnotempty = 0
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
         
        
    
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "MRI Image"))
        self.Upload_bott.setText(_translate("Dialog", "Upload Nifti volume "))
        self.Segment_bott.setText(_translate("Dialog", "Segment"))
        self.train_bott.setText(_translate("Dialog", "Train"))

        self.label_2.setText(_translate("Dialog", "Segmented Image "))
   
        self.cnvJPG.setText(_translate("Dialog", "JPG"))
        # self.plt3D.setText(_translate("Dialog", "plot 3D"))
        self.jpgSaveLabel.setText(_translate("Dialog", "save to JPG"))
        # self.label_3.setText(_translate("Dialog", "Plot"))





    # this function is used  to load Nifti data   
    def loadNifti(self):

        # Enable Segment button
        self.Segment_bott.setEnabled(True)

        # disable save jpg button until the two images are loaded(test image and segmented image)
        self.cnvJPG.setEnabled(False)

        # disable show 3d button
        # self.plt3D.setEnabled(False)

        # initalise the x,y coordinates 
        self.last_x, self.last_y = None, None

        #self.flagLabelnotempty = 1
        
        
        
        # open browse dialog 
        fileName = QFileDialog.getOpenFileName()
       
        # load nifti from path 
        self.loadedNifti = nif.loadSimpleITK(fileName[0])
        self.loadNiftinibabel = nif.loadNifti(fileName[0])
        
        # clear the label each time we load an image 
        self.segmetLabel.clear()
        self.imageTest.clear()

        # normalize the image to darken the nifti slice image
        #self.threshold = 500 # Adjust as needed
        #self.image_2d_scaled = (np.maximum(self.imginit, 0) / (np.amax(self.imginit) + self.threshold)) * 255.0 
        #self.img=qimage2ndarray.array2qimage(self.image_2d_scaled)
    
        # call specific slice
        self.sliceNum, ok = QInputDialog.getInt(QtWidgets.QWidget() , "Enter slice number","Slice:", 0, 1, self.loadedNifti.GetSize()[2]-1, 1, )
        if ok:
            self.imageslice = nif.getSliceITK(self.loadedNifti, self.sliceNum)
        else: 
             self.msg.setWindowTitle("Warning")
             self.msg.setInformativeText('You must enter slice number')
             self.msg.exec()
            
        self.img=qimage2ndarray.array2qimage(self.imageslice)


        # add the image to label     
        self.pixmap = QtGui.QPixmap(self.img)    
        self.imageTest.setPixmap(self.pixmap)
        self.imageTest.setGeometry(QtCore.QRect(5, 40, self.loadedNifti.GetSize()[0], self.loadedNifti.GetSize()[1])) #(x, y, width, height)
        self.imageTest.mousePressEvent = self.drawMove
        #information Message
   
    def Train(self):
        self.msg.setWindowTitle("Warning")
        self.msg.setInformativeText('Wait few minutes to finish training..')
        self.msg.exec()
        
        mn.Train()
        
        self.msg.setWindowTitle("Warning")
        self.msg.setInformativeText('Training is done. Upload the nifti file and check the result')
        self.msg.exec()
    # Activates when the user start press and moving the mouse to draw the contours
    def drawMove(self, e):
        
    
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()
        print('press', self.last_x, self.last_y)
        
        self.imageSliceITK = nif.getNotNumpySliceITK(self.loadedNifti, 1)
        spacing = self.imageSliceITK.GetSpacing()[0]
        
        print('auto', roi.extract_roi(self.loadNiftinibabel, spacing))
        
        # Refresh    
        self.imageTest.update()
     

    
         
    # this function activates when the user press JPG button, which saves the two images in folder Nifti_JPG  
    def savePNG(self):     
        self.path = './Nifti_JPG'
        # Save Nifti MRI image to file 
        cv2.imwrite(os.path.join(self.path,'testImage{0}.jpg'.format(self.sliceNum)), self.maskImg) 
        # same Nifti mri image with segmentation above it
        cv2.imwrite(os.path.join(self.path,'segmentedImg{0}.jpg'.format(self.sliceNum)), self.imageslice) 
 
        # pop up message to check image is saved
        self.msg.setWindowTitle("Save To JPG ")
        self.msg.setInformativeText('Images has been Saved.')
        self.msg.exec()
               

    # Activate when the user press 'segment' button    
    def ActiveShapeModel(self):
        
        # enable saving the image button 
        self.cnvJPG.setEnabled(True)
        
        # enable plt 3D button 
        # self.plt3D.setEnabled(True)
        
        # clear the label each time the user press segment, to not store alot above each other 
        self.segmetLabel.clear()
  
        self.imageSliceITK = nif.getNotNumpySliceITK(self.loadedNifti, self.sliceNum)
        spacing = self.imageSliceITK.GetSpacing()[0]
        
        centroid = roi.extract_roi(self.loadNiftinibabel, spacing)
        
        path = '../training/patient050/patient050_frame01.nii.gz'
        simpitkImgSys = nif.loadSimpleITK(path)
        
        # find centroid of the LV in the image slice

        res = mn.run(self.imageslice, centroid)
        
        print(res.shape)
                
        
       
        # draw the Segmentation results the snakes and edit on the image 
        # cv2.fillConvexPoly(self.img,np.array(res,'int32'),(180, 0, 0))
        # cv2.fillConvexPoly(self.img,np.array(snake2,'int32'),(100, 0, 50))
        # cv2.fillConvexPoly(self.img,np.array(snake3,'int32'),(250, 15, 0))
        # cv2.fillConvexPoly(self.img,np.array(snake4,'int32'),(250, 15, 0))

        self.maskImg = np.zeros(self.imageslice.shape)
        
        cv2.fillConvexPoly(self.maskImg,np.array(res.T,'int32'),(180, 0, 0))
        # cv2.fillConvexPoly(self.maskImg,np.array(snake2,'int32'),(100, 0, 50))
        # cv2.fillConvexPoly(self.maskImg,np.array(snake3,'int32'),(250, 15, 0))
        # cv2.fillConvexPoly(self.maskImg,np.array(snake4,'int32'),(250, 15, 0))

        
        self.segmentedImg = qimage2ndarray.array2qimage(self.maskImg)
        self.pixmap2 = QtGui.QPixmap(self.segmentedImg)    
        self.segmetLabel.setGeometry(QtCore.QRect(479, 40, self.loadedNifti.GetSize()[0], self.loadedNifti.GetSize()[1])) #(x, y, width, height)

        self.segmetLabel.setPixmap(self.pixmap2.scaled(self.loadedNifti.GetSize()[0],self.loadedNifti.GetSize()[1]))
    
    # Activate when press on plot 3D button to draw the 3D shapes 
    def draw3D(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_trisurf(self.x1, self.y1, self.z1)
        #ax.plot_trisurf(self.x1, self.y1, self.z, cmap = cm.cool)
        surf2 = ax.plot_trisurf(self.x2, self.y2, self.z2)
        surf3 = ax.plot_trisurf(self.x3, self.y3, self.z3)
        surf3 = ax.plot_trisurf(self.x4, self.y4, self.z4)
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)
            plt.show()
        # rotate the axes and update

        fig.savefig('./3D_Plots/Plot3D{0}.png'.format(self.dataset.InstanceNumber))
        
        # pop up message to check image is saved
        self.msg.setWindowTitle("3D plot")
        self.msg.setInformativeText('3D figure is saved.')
        self.msg.exec()


    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    Dialog.update()
    sys.exit(app.exec_())
    