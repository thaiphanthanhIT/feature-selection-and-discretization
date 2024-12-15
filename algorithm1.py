import math
from collections import Counter
import numpy as np
import pandas as pd
from numpy import *
import utils
from binalgo import scoreDP

def mutual_information(X, Y):
    """
        Calculate Mutual Information between two discrete variables X and Y

        Input:
        -----
        X {pandas.Series}          : Input variable X
        Y {pandas.Series}          : Input variable Y

        Output:
        ------
        MI {float}                 : Mutual Information value between X and Y
    """
    counter_X = Counter(X)
    counter_Y = Counter(Y)

    N = len(X)

    P_X = {x: count / N for x, count in counter_X.items()}
    P_Y = {y: count / N for y, count in counter_Y.items()}

    joint_counts = Counter(zip(X, Y))
    P_XY = {key: count / N for key, count in joint_counts.items()}

    MI = 0
    for (x, y), p_xy in P_XY.items():
        if p_xy > 0:
            MI += p_xy * math.log2(p_xy / (P_X[x] * P_Y[y]))

    return MI

def algorithm1(F, C):
    """
        Select and discretize features based on mutual information and dynamic programming

        Input:
        -----
        F {DataFrame}          : Feature matrix
        C {DataFrame}          : Class labels

        Output:
        ------
        Fc {list}               : Discretized features
        Dc {list}               : Number of splits for each feature
        Jc {list}               : Scores for the selected features
        split_val {list}        : List of cut points for each feature
    """
    result = pd.concat([F, C], axis=1)
    Fc = np.zeros((F.shape[0], 1))
    Dc = []
    split_val = []
    Jc = []
    cnt = 0

    for i in F.columns:
        val, freq, _ = utils.makePrebins(result, i, C.columns[0])
        opt_score, spl_val, j = scoreDP(val, freq)
        Jrel = mutual_information(pd.cut(F[:][i], bins=[float('-inf')] + spl_val, labels=False), C['class'].T)
        if Jrel > 0:
            Fi = F[:][i].values
            Fi = Fi.reshape(-1, 1)
            Fc = insert(Fc, [cnt + 1], Fi, axis=1)
            Dc.append(j)
            Jc.append(Jrel)
            split_val.append(spl_val)
            cnt = cnt + 1
    Fc = np.delete(Fc, 0, axis=1)

    return Fc, Dc, Jc, split_val

