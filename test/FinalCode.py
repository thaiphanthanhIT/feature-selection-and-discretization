import math
from collections import Counter

import numpy as np
from catboost import CatBoostClassifier
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo
import pandas as pd
from numpy import *
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import chi2
from sklearn.svm import SVC

import utils
from binalgo import scoreDP

def shan_entropy(c):
    """
        Calculate Shannon Entropy from a histogram

        Input:
        -----
        c {pandas.Series}       : Histogram counting the number of elements

        Output:
        ------
        H {float}               : The Shannon Entropy value
    """
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H

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

def remove_elements(arr, p, q):
    """
        Remove two elements at indices p and q from a list

        Input:
        -----
        arr {list}             : Input list
        p {int}                : Index of the first element to remove
        q {int}                : Index of the second element to remove

        Output:
        ------
        result {list}          : List after removing the elements
    """
    return [arr[i] for i in range(len(arr)) if i != p and i != q]

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

def mDSM(F, C, e):
    """
    Proposed score-wise Dynamic programming algorithm

    Input:
    -----
    F {DataFrame}          : Feature matrix
    C {DataFrame}          : Class labels
    e {float}              : Threshold parameter

    Output:
    ------
    S {list}               : Selected discretized features
    D {list}               : Number of splits for each selected feature
    """
    Fc, Dc, Jc, split_val = algorithm1(F, C)
    Fc = Fc.T

    # Sort features based on their mutual information scores (Jc)
    sorted_indices = np.argsort(Jc)[::-1]
    Fc = [Fc[i] for i in sorted_indices]
    Dc = [Dc[i] for i in sorted_indices]
    Jc = [Jc[i] for i in sorted_indices]
    split_val = [split_val[i] for i in sorted_indices]

    S = []
    D = []
    S.append(pd.cut(Fc[0], bins=[float('-inf')] + split_val[0], labels=False))
    D.append(Dc[0])

    Fc.pop(0)
    Dc.pop(0)
    split_val.pop(0)

    T = -np.inf
    for i in range(len(Fc)):
        fi = Fc[i]
        di = Dc[i]
        Jjmi, check = algorithm2(fi, C, di, S, split_val[i], T, e)
        if check:
            S.append(pd.cut(Fc[i], bins=[float('-inf')] + split_val[i], labels=False))
            D.append(Dc[i])
            T = Jjmi

    S_ori = S[:]
    tmp = -np.inf
    min_position_tmp = -1
    print(len(S))
    while True:
        ep = np.zeros(len(S))
        for p in range(0, len(S)):
            ep[p] = mutual_information(S[p], C['class'].T)
            for i in range(0, len(S)):
                if i == p:
                    continue
                ep[p] = ep[p] + (CMI(S[p], S[i], C['class'].T) - mutual_information(S[p], S[i])) / (len(S) - 1)
        min_position = np.argmin(ep)
        if min_position_tmp == -1:
            min_position_tmp = min_position
            tmp = ep[min_position]
            S_ori = S
            S = S[:min_position_tmp] + S[min_position_tmp + 1:]
            continue
        print(ep[min_position], tmp + e / len(S), np.median(ep))
        if (np.median(ep)) > (tmp + e / len(S)):
            min_position_tmp = min_position
            tmp = ep[min_position]
            S_ori = S
            S = S[:min_position_tmp] + S[min_position_tmp + 1:]
        else:
            break
    S = S_ori
    print(len(S))
    return S, D

# dataset = fetch_ucirepo(id=94)
#
# X = dataset.data.features
# y = dataset.data.targets
# y_ = dataset.data.targets
# weights = [0, 1, 2, 3, 4, 5, 6]
# y__ = y_.dot(weights)
# y = pd.DataFrame(y__, columns=['class'])

# X['Attribute38'].fillna(X['Attribute38'].mean(), inplace=True)
# X['Attribute4'].fillna(X['Attribute4'].mean(), inplace=True)
# y = y.drop('NSP', axis=1)
# X = X.drop('age', axis=1)
# X = X.drop('MDVP:Shimmer', axis=1)
# X = X.values
# y.columns = ['class']
# le = LabelEncoder()
# X = X.drop('conformation_name', axis=1)
# X = X.drop('molecule_name', axis=1)
# y['class'] = le.fit_transform(y['class'])
# X['conformation_name'] = le.fit_transform(X['conformation_name'])
# X['molecule_name'] = le.fit_transform(X['molecule_name'])
# X['Attribute1'] = le.fit_transform(X['Attribute1'])
# X['Attribute3'] = le.fit_transform(X['Attribute3'])
# X['Attribute4'] = le.fit_transform(X['Attribute4'])
# X['Attribute6'] = le.fit_transform(X['Attribute6'])
# X['Attribute7'] = le.fit_transform(X['Attribute7'])
# X['Attribute9'] = le.fit_transform(X['Attribute9'])
# X['Attribute10'] = le.fit_transform(X['Attribute10'])
# X['Attribute12'] = le.fit_transform(X['Attribute12'])
# X['Attribute14'] = le.fit_transform(X['Attribute14'])
# X['Attribute15'] = le.fit_transform(X['Attribute15'])
# X['Attribute17'] = le.fit_transform(X['Attribute17'])
# X['Attribute19'] = le.fit_transform(X['Attribute19'])
# X['Attribute20'] = le.fit_transform(X['Attribute20'])
# # y = y.to_numpy()

# Arrhythmia
# with open('arrhythmia.data', 'r') as file:
#     lines = file.readlines()
#     features = []
#     targets = []
#     for line in lines:
#         row = [float(i) if i != '?' else np.nan for i in line.strip().split(",")]
#         features.append(row[:-1])
#         targets.append(row[-1])
#     X = pd.DataFrame(features)
#     y = pd.DataFrame(targets, columns=["class"])
#     X.fillna(X.mean(), inplace=True)
#     y.fillna(y.mean(), inplace=True)

#semeion
# data = pd.read_csv("semeion.data", delimiter=" ", header=None)
# X = []
# y = []
# for _, row in data.iterrows():
#     row_list = row.tolist()
#     feature_values = row_list[:-11]
#     target_values = row_list[-11:-1]
#     target_sum = sum(value * index for index, value in enumerate(target_values))
#     y.append(target_sum)
#     X.append(feature_values)
# X = pd.DataFrame(X, columns=[f'feature_{i + 1}' for i in range(len(X[0]))])
# y = pd.DataFrame(y, columns=['class'])

#madelon
# data = pd.read_csv("F:\Download\mdlon.csv")
# last_column = data.columns[-1]
# X = data.drop(columns=[data.columns[-1]])
# y = data[[last_column]]
# y.columns = ['class']
# print(X, y)
# file_path1 = "madelon_train.data"
# X1 = pd.read_csv(file_path1, delim_whitespace=True, header=None)
# file_path2 = "madelon_valid.data"
# X2 = pd.read_csv(file_path2, delim_whitespace=True, header=None)
# X = pd.concat([X1, X2], ignore_index=True)
# file_path_label1 = "madelon_train.labels"
# y1 = pd.read_csv(file_path_label1, delim_whitespace=True, header=None)
# file_path_label2 = "madelon_valid.labels"
# y2 = pd.read_csv(file_path_label2, delim_whitespace=True, header=None)
# y = pd.concat([y1, y2], ignore_index=True)
# y.columns = ['class']
# print(X,y)

# lung, colon (colon e = -0.028)
# file_path = 'lung.mat'
# data = loadmat(file_path)
# X = pd.DataFrame(data['X'])
# y = pd.DataFrame(data['Y'], columns=['class'])

# # dbword
# file_path = 'dbworld_bodies.mat'
# data = loadmat(file_path)
# X = pd.DataFrame(data['inputs'])
# y = pd.DataFrame(data['labels'], columns=['class'])

# heart, pima
df = pd.read_csv("F:\Download\dataset_heart.csv")
y = df[[df.columns[-1]]]
X = df[df.columns[:-1]]
y.columns = ['class']

# parkinsons
# with open("F:\Download\parkinsons.data.txt", "r") as file:
#     data = [line.strip().split(",") for line in file]
# df = pd.DataFrame(data)
# df = df.apply(pd.to_numeric, errors='coerce')
# target_column = len(df.columns) - 7
# y = df[[target_column]]
# X = df.drop(columns=target_column)
# X = X.drop(0, axis=1)
# y.columns = ['class']

# BreastTissue
# file_path = 'F:\Download\BreastTissue.xls'
# df = pd.read_excel(file_path, sheet_name='Data')
# print(df)
# y = df[[df.columns[1]]]
# X = df[df.columns[2:]]
# print(X, y)
# y.columns = ['class']
# le = LabelEncoder()
# y['class'] = le.fit_transform(y['class'])

# Libras
# with open("movement_libras.data", "r") as file:
#     data = [line.strip().split(",") for line in file]
# df = pd.DataFrame(data)
# df = df.apply(pd.to_numeric, errors='coerce')
# target_column = len(df.columns) - 1
# y = df[[target_column]]
# X = df.drop(columns=target_column)
# y.columns = ['class']
# print(X, y)

e = 0.5
S, D = mDSM(X, y, e)

y = y.values
X = np.array(S).T
# X_test_selected = np.array([X_test[:, i] for i in range(len(S))]).T

# big
clf = SVC(kernel='linear')
# clf = CatBoostClassifier(verbose=0)
# clf = KNeighborsClassifier(n_neighbors=len(unique(y)))
scores_selected = cross_val_score(clf, X, y, cv=10)
print(e)
print(len(S))
print(f"Độ chính xác trung bình với các đặc trưng đã chọn (10-CV): {scores_selected.mean():.4f}")

# # clf = SVC(kernel='linear')
# clf = KNeighborsClassifier(n_neighbors=2)
# scores_selected = cross_val_score(clf, X, y, cv=10)
# print(e)
# print(len(S))
# print(f"Độ chính xác trung bình với các đặc trưng đã chọn (10-CV): {scores_selected.mean():.4f}")

# small
# model = SVC(kernel='linear')
# loo = LeaveOneOut()
# y_true, y_pred = [], []
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     model.fit(X_train, y_train)
#     y_pred.append(model.predict(X_test)[0])
#     y_true.append(y_test[0])
# accuracy = accuracy_score(y_true, y_pred)
# print(f'Độ chính xác trung bình với các đặc trưng đã chọn (LOO): {accuracy:.4f}')

# clf_all = SVC(kernel='linear')
#
# scores_all = cross_val_score(clf_all, X, y, cv=10)
# print(f"Độ chính xác trung bình với tất cả đặc trưng (10-CV): {scores_all.mean():.4f}")