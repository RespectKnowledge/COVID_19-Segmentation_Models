from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.stats import pearsonr
import numpy as np
import  matplotlib.pyplot as plt 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


# Dice Cof. 
def dice_coef(result,reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

    #..........................................................

def jc(result, reference):
    # Jaccard coefficient
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def Hausdorff(result, reference, voxelspacing=None, connectivity=1):
    # Hausdorff Distance.
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd   


def pearsonr_Corr(results, references):
    """
    Volume correlation.
    
    Computes the linear correlation in binary object volume between the
    contents of the successive binary images supplied. Measured through
    the Pearson product-moment correlation coefficient. 
    
    Parameters
    ----------
    results : sequence of array_like
        Ordered list of input data containing objects. Each array_like will be
        converted into binary: background where 0, object everywhere else.
    references : sequence of array_like
        Ordered list of input data containing objects. Each array_like will be
        converted into binary: background where 0, object everywhere else.
        The order must be the same as for ``results``.
    
    Returns
    -------
    r : float
        The correlation coefficient between -1 and 1.
    p : float
        The two-side p value.
        
    """
    results = np.atleast_2d(np.array(results).astype(np.bool))
    references = np.atleast_2d(np.array(references).astype(np.bool))
    
    results_volumes = [np.count_nonzero(r) for r in results]
    references_volumes = [np.count_nonzero(r) for r in references]
    
    return pearsonr(results_volumes, references_volumes)    

def showLossAccu(history):
    plt.figure(figsize=(30, 5), num = 'Metrics')
    plt.subplot(121)
    plt.plot(history['f1-score'])
    plt.plot(history['val_f1-score'])
    plt.title('Model accuracy')
    plt.ylabel('precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')

    #print(history['f1-score'])


# Reciever Operator Charachtarisitics 
def ROC(results, reference):

    flattenGT=results.ravel()
    flattenPD=reference.ravel()

    fpr_f1, tpr_f1, thresholds_fcn = roc_curve(results, reference)
    auc_f= roc_auc_score(results,reference)

    # Plot, axis [-1, 1] 
    axisRange = [-0.01, 0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    # plot perfromance for binary segemnatation
    lw = 1
    plt.grid(True)
    plt.plot(fpr_f1, tpr_f1,'-b',label='Overall, AUC = '+str(round(auc_f,3)))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xticks(axisRange)
    plt.yticks(axisRange)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')