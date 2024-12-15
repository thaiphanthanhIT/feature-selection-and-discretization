import math
from collections import Counter
import numpy as np
import pandas as pd

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

def compute_prob(X, Y, Z):
    """
        Compute conditional probabilities from three variables X, Y, and Z

        Input:
        -----
        X {pandas.Series}       : Input variable X
        Y {pandas.Series}       : Input variable Y
        Z {pandas.Series}       : Conditioning variable Z

        Output:
        ------
        P_z {float}             : Probability of Z
        P_xz {float}            : Conditional probability of X given Z
        P_yz {float}            : Conditional probability of Y given Z
        P_xyz {float}           : Conditional probability of X and Y given Z
    """
    n = len(Z)
    P_z = Counter(Z)
    for k in P_z:
        P_z[k] /= n

    P_xz = {}
    P_yz = {}
    P_xyz = {}

    for x, y, z in zip(X, Y, Z):
        if (x, z) not in P_xz:
            P_xz[(x, z)] = 0
        P_xz[(x, z)] += 1

        if (y, z) not in P_yz:
            P_yz[(y, z)] = 0
        P_yz[(y, z)] += 1

        if (x, y, z) not in P_xyz:
            P_xyz[(x, y, z)] = 0
        P_xyz[(x, y, z)] += 1

    for (x, z) in P_xz:
        P_xz[(x, z)] /= Counter(Z)[z]
    for (y, z) in P_yz:
        P_yz[(y, z)] /= Counter(Z)[z]
    for (x, y, z) in P_xyz:
        P_xyz[(x, y, z)] /= Counter(Z)[z]

    return P_z, P_xz, P_yz, P_xyz

def CMI(X, Y, Z):
    """
        Compute Conditional Mutual Information between X and Y conditioned on Z

        Input:
        -----
        X {pandas.Series}          : Input variable X
        Y {pandas.Series}          : Input variable Y
        Z {pandas.Series}          : Conditioning variable Z

        Output:
        ------
        I_xyz {float}              : Conditional Mutual Information (CMI) value
    """
    P_z, P_xz, P_yz, P_xyz = compute_prob(X, Y, Z)
    I_xyz = 0

    for (x, y, z) in P_xyz:
        p_xyz = P_xyz[(x, y, z)]
        p_xz = P_xz[(x, z)]
        p_yz = P_yz[(y, z)]
        p_z = P_z[z]

        if p_xyz > 0 and p_xz > 0 and p_yz > 0:
            I_xyz += p_z * p_xyz * np.log(p_xyz / (p_xz * p_yz))

    return I_xyz

def algorithm2(fi, C, di, S, split_val, Jjmi_pre, e):
    """
        Evaluate and refine the inclusion of a feature in the selected set

        Input:
        -----
        fi {array}             : Feature values
        C {DataFrame}          : Class labels
        di {int}               : Number of splits
        S {list}               : Selected features
        split_val {list}       : Split values for discretization
        Jjmi_pre {float}       : Previous score
        e {float}              : Threshold parameter

        Output:
        ------
        Jjmi {float}           : New score
        check {bool}           : Inclusion status of the feature
    """
    check = 0
    Jjmi = 0

    fi_discretized = pd.cut(fi, bins=[float('-inf')] + split_val, labels=False)
    Jjmi_ori = mutual_information(fi_discretized, C['class'].T)
    for i in range(0, len(S)):
        Jjmi_ori = Jjmi_ori + (CMI(fi_discretized, S[i], C['class'].T) - mutual_information(fi_discretized, S[i])) / len(S)
    if Jjmi_ori > (Jjmi_pre - e / len(S)):
        check = 1
    print(Jjmi_ori, Jjmi)

    return Jjmi_ori, check
