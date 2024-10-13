import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def visualize(viz, xtitle, ytitle, title, show = False, dir = None, legend = True):
    '''
    Visualize multiple plots on a same fig

    Input:
    -----
    viz {List of ([List], str)} : plots and their corresponding legends
    show : Whether to show the plot (Default: True)
    dir  : PATH for saving resulting plot

    Output:
    -----
    '''
    fig, axes = plt.subplots()
    for v in viz:
        axes.plot(np.arange(1, v[0].shape[0] + 1), v[0], label = v[1])
    if legend:
        axes.legend()
    axes.set_xlabel(xtitle)
    axes.set_ylabel(ytitle)
    axes.set_title(title)
    if dir is not None:
        plt.savefig(dir)
    if show:
        plt.show()

def cleanseNA(fea_vec):
    '''
    Cleanse a feature vector of a dataset by removing all missing instances.

    Input:
    -----
    fea_vec {numpy.ndarray} : Feature vector
    label {numpy.ndarray}   : Label vector

    Output:
    -----
    fea_vec {numpy.ndarray} : Cleansed feature
    '''
    avail_index = np.argwhere(~np.isnan(fea_vec)).flatten().tolist()
    return fea_vec[avail_index]

def makeVal(df, feature):
    '''
    Obtaining list of distinct value regarding specified feature (0-based)
    '''
    df[feature] = df[feature].astype(float)
    val = df[feature].unique()
    val = np.sort(val)
    val = cleanseNA(val)
    return val

def makePrebins(df, feature, label, num_classes = 2):
    '''
    Pre-discretizing feature into pre-bins

    Input:
    -----
    df (pandas.DataFrame)   : List of instances to pre-discretize.
    feature                 : interested feature for evaluation.
    label                   : label column in dataset. MUST be in the form of 0 ... (num_class-1) without NaN values
    num_class               : number of class

    Output:
    -----
    val (numpy.ndarray)    : List of distinct value regarding specified feature (0-based)
    freq (numpy.ndarray)   : Frequency of each aforementioned value (1-based)
    valdict (dictionary)   : val - freq mapping
    '''
    # Prepare discretized values
    df[feature] = df[feature].astype(float)
    val = df[feature].unique()
    val = np.sort(val)
    val = cleanseNA(val)

    # print(df[label])

    catcode = pd.Series(pd.Categorical(df[label], categories= df[label].unique())).cat.codes
    # print(catcode)
    # Get number of class
    num_classes = max(catcode) + 1

    valdict = []
    for i in range(num_classes):
        valdict.append(dict.fromkeys(val, 0))

    for i in range(len(df)):
        if np.isnan(df[feature][i]) or catcode[i] == -1:
            continue
        valdict[catcode[i]][df[feature][i]] += 1

    # Build feature frequency with freq[0] being dummy value
    freq = []
    for i in range(num_classes):
        freq.append([0] + list(valdict[i].values()))
    
    freq = np.array(freq)

    return val, freq, valdict

def initMode(mode, val= 1e9):
    '''
    Initial value for corresponding mode ('min' or 'max').
    Default absolute value: 1e9
    '''
    if mode == 'min':
        return val
    else:
        return -val

def binSearch(arr, val):
    '''
    Binary search for an index idx of arr such that arr[idx] = val
    arr is 0-based
    return -1 if there is no such index
    '''
    L = 0
    R = len(arr)-1
    while(L <= R):
        mid = int((L + R) / 2)
        if arr[mid] == val:
            return mid
        if arr[mid] < val:
            L = mid + 1
        else:
            R = mid - 1
    return -1

def discretizeFea(df, fea, split):
    '''
    Discretize df[fea] w.r.t split 

    Output:
    -----
    A ndarray of discretized feature, with values ranging from 0 to n_bin-1
    '''
    # print(df[fea])
    full_split = [df[fea].min()-1] + list(split) + [df[fea].max()+1] # Adding the smallest and largest value to the split-list
    # if fea == 'Y':
    #     print(df[fea])
    #     print(pd.cut(df[fea], bins= full_split, labels= False))
    return pd.cut(df[fea], bins= full_split, labels= False)

def mask2Split(mask, val):
    '''
    Remove zeros from mask * val, forming a corresponding split scheme w.r.t mask

    Input:
    -----
    mask {np.ndarray}   : Value mask of the split
    val {np.ndarray}    : Value list of the feature

    Both arrays are 0-based.

    Ouput:
    -----
    split {np.ndarray}  : The split over feature induced by mask.
    '''
    splt = mask * val
    return splt[splt != 0]

def split2Mask(split, val):
    '''
    Convert from split to mask, regarding val

    Input:
    -----
    split {List}        : Split values
    val {np.ndarray}    : Value list of the feature

    Both arrays are 0-based.

    Ouput:
    -----
    mask {np.ndarray}   : The corresponding mask
    '''
    splt_ptr = 0
    mask = np.zeros_like(val).astype(int)
    
    for i in range(len(val)):
        if splt_ptr < len(split) and math.fabs(val[i] - split[splt_ptr]) < 1e-3:
            mask[i] = 1
            splt_ptr += 1
        else:
            mask[i] = 0

    return mask