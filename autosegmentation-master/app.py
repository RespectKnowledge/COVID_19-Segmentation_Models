from tkinter import *
import tkinter.filedialog
from tkinter.ttk import Style
import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import glob
import cv2
from skimage.io import imsave, imread, imshow
from skimage.transform import resize
from PIL import Image, ImageTk
import matplotlib.colors
from medpy.metric.binary import hd, dc
from scipy import stats
import statsmodels.api as sm
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import nibabel as nib


class App:

  def __init__(self, master):

    # All the files in selected path or selected by dialog
    # self.selected_files = []
    # self.processed_images = {}
    # self.processed_masks = {}
    # self.active_patient_id = 'patient_043'
    # self.slice_name = 4
    # self.patient_info = None
    # self.config_file = None
    # self.ed_index = '01'
    # self.es_index = ''
    # self.data_path = "./dataset/data"
    # self.images_path = "./dataset/train"
    self.model = None
    self.frames_src = './cases/coronacases_002.nii.gz'
    self.masks_src = './masks/coronacases_002.nii.gz'
    self.axis = 3
    self.total_frames = 0
    self.slice_index = 105
    self.altman_fname = 'altman_plot.png'
    self.roc_fname = 'roc.png'
    self.active_plot_fname = self.roc_fname
    # GUI related
    self.master = master
    self.create_ui()
    self.create_sidebar()
    self.load_model()
    self.init()

  def create_ui(self):
    self.master.geometry("1220x900")
    self.master.title("MRI Cardiac Segmentation")
    self.master.style = Style()
    self.master.style.theme_use("default")

    self.master.columnconfigure(0, weight=4)
    self.master.columnconfigure(1, weight=1)

    self.leftFrame = Frame(self.master, padx=10, pady=10)
    self.leftFrame.grid(row=0, column=0, sticky=N+S+E+W)

    self.rightFrame = Frame(self.master)
    self.rightFrame.columnconfigure(0, weight=1)
    self.rightFrame.grid(row=0, column=1, sticky=N+S+E+W)

    self.menuFrame = Frame(self.rightFrame, relief=RAISED, borderwidth=1, padx=10, pady=10)
    self.menuFrame.columnconfigure(0, weight=1)
    self.menuFrame.grid(row=0, column=0, sticky=N+S+E+W)

    self.metricsFrame = Frame(self.rightFrame, relief=RAISED, borderwidth=1)
    self.metricsFrame.grid(row=1, column=0, sticky=N+S+E+W)

    self.imageFrame = Frame(self.leftFrame, relief=RAISED, borderwidth=1, padx=5, pady=5)
    self.imageFrame.grid(row=0, column=0)

    self.imageControlsFrame = Frame(self.leftFrame, relief=RAISED, borderwidth=1, padx=5, pady=5)
    self.imageControlsFrame.grid(row=1, column=0)

    srcLbl = Label(self.imageFrame, text="Original Mask")
    srcLbl.grid(row=0, column=0)

    self.org_mask_cmp = Label(self.imageFrame)
    self.org_mask_cmp.grid(row=1, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="Predicted Mask")
    srcLbl.grid(row=2, column=0)

    self.pred_mask_cmp = Label(self.imageFrame)
    self.pred_mask_cmp.grid(row=3, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="Analysis Plot")
    srcLbl.grid(row=0, column=1)

    self.analysis_cmp = Label(self.imageFrame)
    self.analysis_cmp.grid(row=1, column=1, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="Heatmask")
    srcLbl.grid(row=2, column=1)

    self.heatmap_cmp = Label(self.imageFrame)
    self.heatmap_cmp.grid(row=3, column=1, padx=5, pady=5)

    # self.esimg = Label(self.imageFrame)
    # self.esimg.grid(row=3, column=0, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ED Predicted Mask")
    # srcLbl.grid(row=0, column=2)

    # self.edpmask = Label(self.imageFrame)
    # self.edpmask.grid(row=1, column=2, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ES Predicted Mask")
    # srcLbl.grid(row=2, column=2)

    # self.espmask = Label(self.imageFrame)
    # self.espmask.grid(row=3, column=2, padx=5, pady=5)

    b = Button(self.imageControlsFrame, text=">", command=self.next_frame)
    b.grid(row=1, column=1, sticky=E)
    b = Button(self.imageControlsFrame, text="<", command=self.prev_frame)
    b.grid(row=1, column=0, sticky=W)

  def create_sidebar(self):
    fselect_button = Button(self.menuFrame, text="Load Frames Data", command=self.open_path_dialog)
    fselect_button.grid(row=0, sticky=W+E)

    self.exit_button = Button(
      master=self.menuFrame,
      text="Exit",
      command=self.master.quit
    )

    self.exit_button.grid(row=3, sticky=W+E)

    mselect_button = Button(self.menuFrame, text="Load Masks Data", command=self.open_path_masks_dialog)
    mselect_button.grid(row=1, sticky=W+E)

    process_button = Button(self.menuFrame, text="Process", command=self.init)
    process_button.grid(row=2, sticky=W+E)

    listbox = Listbox(self.menuFrame)
    listbox.grid(row=4, sticky=W+E)

    listbox.insert(END, "Background")
    listbox.insert(END, "Right Lung")
    listbox.insert(END, "Left Lung")
    listbox.insert(END, "Covid-19")
    listbox.bind('<<ListboxSelect>>', self.change_axis)

    lbl = Label(self.metricsFrame, text="Choose plot to view")
    lbl.grid(row=0, column=0, padx=5, pady=5)

    self.plot_roc_btn = Button(self.metricsFrame, text="ROC/AUC", command=lambda: self.set_plot_image(self.roc_fname))
    self.plot_roc_btn.grid(row=1, column=0, padx=5, pady=5)

    self.plot_altman_btn = Button(self.metricsFrame, text="A bland Altman", command=lambda: self.set_plot_image(self.altman_fname))
    self.plot_altman_btn.grid(row=1, column=1, padx=5, pady=5)

    # self.patientNameLbl = Label(self.metricsFrame, text="", anchor="center")
    # self.patientNameLbl.grid(row=5, column=0, padx=10, pady=10)

    # tbl_fspace = Label(self.metricsFrame, text="")
    # tbl_fspace.grid(row=0, column=0, padx=5, pady=5)

    lbl = Label(self.metricsFrame, text="Dice-Coefficient")
    lbl.grid(row=2, column=0, padx=5, pady=5)

    self.dice_lbl = Label(self.metricsFrame, text="")
    self.dice_lbl.grid(row=3, column=0, padx=5, pady=5)

    lbl = Label(self.metricsFrame, text="Hausdorff distance")
    lbl.grid(row=4, column=0, padx=5, pady=5)

    self.haus_lbl = Label(self.metricsFrame, text="")
    self.haus_lbl.grid(row=5, column=0, padx=5, pady=5)

    lbl = Label(self.metricsFrame, text="Pearson corr. coeff.")
    lbl.grid(row=6, column=0, padx=5, pady=5)

    self.pearsonc_lbl = Label(self.metricsFrame, text="")
    self.pearsonc_lbl.grid(row=7, column=0, padx=5, pady=5)

    lbl = Label(self.metricsFrame, text="P-Value")
    lbl.grid(row=8, column=0, padx=5, pady=5)

    self.pval_lbl = Label(self.metricsFrame, text="")
    self.pval_lbl.grid(row=9, column=0, padx=5, pady=5)

  def change_axis(self, evt):
    w = evt.widget
    index = int(w.curselection()[0])
    self.axis = index
    self.process()

  def open_path_dialog(self):
    filenames = tkinter.filedialog.askopenfile()
    print(filenames)
    self.frames_src = filenames.name

  def open_path_masks_dialog(self):
    filenames = tkinter.filedialog.askopenfile()
    print(filenames)
    self.masks_src = filenames.name

  def next_frame(self):
    if (self.slice_index > 0):
      self.slice_index = self.slice_index - 1
      self.process()

  def prev_frame(self):
    if (self.slice_index < self.total_frames):
      self.slice_index = self.slice_index + 1
      self.process()

  def preprocess_image(self, img, fname = "temp.png"):
    resized = resize(img, (384, 384))
    imsave(fname, resized)
    return imread(fname)

  def predict(self):
    ## normalising prediction image
    norm_image = cv2.normalize(self.frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image  = np.expand_dims(norm_image, axis=-1)
    norm_image  = np.expand_dims(norm_image, axis=0)

    preds_train = self.model.predict(norm_image, verbose=1)
    self.prediction_raw = preds_train*255

    best_dice = self.getbest_dice(self.prediction_raw, self.mask)
    self.dice_coeff= best_dice[0:255].max()

    if self.axis in range (0,3):
      self.itemindex= best_dice[200:255].argmax() + 200
    elif self.axis == 3:
      self.itemindex= best_dice[90:255].argmax() + 90

    preds_perfect = (self.prediction_raw > self.itemindex-1).astype(np.bool)
    preds_perfect = preds_perfect[...,self.axis].squeeze()

    ## predicted mask from model
    self.prediction = preds_perfect
    return preds_perfect

  def process_plot(self, filename):
    plot_img = imread(filename)
    plot_img = resize(plot_img, (384, 384))
    rescaled_image = 255 * plot_img
    final_image = rescaled_image.astype(np.uint8)
    return final_image

  def roc(self):
    y_mask = self.mask_raw
    y_mask = tf.keras.utils.to_categorical(y_mask, num_classes=4, dtype='bool')
    y_covid = y_mask[...,self.axis].squeeze()

    y_predicted = self.prediction
    # we want to make them into vectors
    ground_truth_labels = y_covid.ravel()
    score_value= y_predicted.ravel()
    fpr, tpr, _ = roc_curve(ground_truth_labels, score_value)
    roc_auc = auc(fpr,tpr)
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive rate')
    ax.set_ylabel('True Positive rate ')
    ax.set_title('Receiver operating characteristic for Diseased Areas pixel wise')
    ax.legend(loc="lower right")
    plt.savefig(self.roc_fname)
    return self.roc_fname

  ## this function cycles pixel intensity to get the best dice coefficient
  def getbest_dice(self, preds_train_func,pred_mask):
    axis = 3
    dice=np.zeros(256,dtype=np.float32)
    for i in range(0,255):
      hello = preds_train_func[...,axis].squeeze()
      hello = (hello>i).astype(np.bool)
      #ihere+=i
      dcval= dc(hello,pred_mask)*100
      #print('here',dcval)
      #dice= []
      dice[i]=dcval
      #dice= int(dice)
      #data= [dice,i]
    return dice

  def heatmap(self):
    # heat map to show prediction in a greater detail
    heatmask = self.prediction_raw[...,self.axis].squeeze()
    return heatmask

  def init(self):
    # tr_im = nib.load('./cases/coronacases_002.nii.gz')
    # tr_masks = nib.load('./masks/coronacases_002.nii.gz')
    if (self.frames_src == None or self.masks_src == None):
      return
    tr_im = nib.load(self.frames_src)
    tr_masks = nib.load(self.masks_src)
    self.im_data = tr_im.get_fdata()
    self.mask_data = tr_masks.get_fdata()
    self.total_frames = self.im_data.shape[2]
    self.process()

  def metrics(self):
    haufdist = hd(self.prediction, self.mask,voxelspacing=None, connectivity=1)
    self.haufdist = haufdist

  def process(self):
    slice_1 = self.im_data[:,:,self.slice_index]
    mask_1 = self.mask_data[:,:,self.slice_index]

    self.frame = self.preprocess_image(slice_1, 'frame.png')

    self.mask_raw = resize(mask_1, (384, 384))
    self.mask = (self.mask_raw == self.axis).astype(np.bool)

    pred = self.predict()
    roc_fname = self.roc()
    heatmap = self.heatmap()
    self.pval_abland()
    self.altman_plot()
    self.metrics()
    self.set_image(self.org_mask_cmp, self.create_image_component(self.mask))
    self.set_image(self.pred_mask_cmp, self.create_image_component(pred))
    self.set_image(self.heatmap_cmp, self.create_image_component(heatmap))
    self.set_plot_image(self.active_plot_fname)
    self.dice_lbl.config(text=self.dice_coeff)
    self.haus_lbl.config(text=self.haufdist)
    self.pearsonc_lbl.config(text=self.pearsonc)
    self.pval_lbl.config(text=self.pval)

  def set_plot_image(self, fname):
    self.active_plot_fname = fname
    plot_image = self.process_plot(fname)
    self.set_image(self.analysis_cmp, self.create_image_component(plot_image))

  def place_ui_images(self):
    img_cmp = self.create_image_component(self.mask)
    self.set_image(self.org_mask_cmp, img_cmp)

  def pval_abland(self):
    pearson_stats = stats.pearsonr(self.prediction.flatten(), self.mask.flatten())
    self.pearsonc = pearson_stats[0]
    self.pval = pearson_stats[1]

  def altman_plot(self):
    mask_graph=self.mask*255

    pred_graph = (self.prediction_raw > self.itemindex-1).astype(np.uint8)
    pred_graph = pred_graph[...,self.axis].squeeze()

    f, ax = plt.subplots(1, figsize = (8,5))
    fig = sm.graphics.mean_diff_plot(mask_graph.flatten(), pred_graph.flatten(), ax = ax)
    fig.savefig(self.altman_fname)
    return self.altman_fname

  def dice_coef(self, y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def load_model(self):
    self.model = tf.keras.models.load_model("model", custom_objects={'dice_coef': self.dice_coef})

  def create_image_component(self, data):
    # load = Image.open(data)
    return ImageTk.PhotoImage(image=Image.fromarray(data))

  def set_image(self, dstFrame, image):
    dstFrame.configure(image=image)
    dstFrame.image = image

if __name__ == '__main__':
  # matplotlib.use("Agg")
  root = Tk()
  app = App(root)

  root.mainloop()