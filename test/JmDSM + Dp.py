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


def calc_MI(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H

def mutual_information(X, Y):

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
    return [arr[i] for i in range(len(arr)) if i != p and i != q]

def algorithm1_change(F, C):
    number_features = F.shape[1]
    number_samples = F.shape[0]
    result = pd.concat([F, C], axis=1)
    Fc = np.zeros((number_samples, 1))
    Dc = []
    split_val = []
    Jc = []
    cnt = 0
    for i in F.columns:
        val, freq, _ = utils.makePrebins(result, i, C.columns[0])
        opt_score, spl_val, j = scoreDP(val, freq)
        print(i)
        # fi = F[:][i].values.T
        # I_fi_C = mutual_information(fi, C['class'].T)
        # chi2_value = 2 * number_samples * np.log(2) * I_fi_C
        # degree_of_freedom = (len(np.unique(fi)) - 1) * (len(np.unique(C)) - 1)
        fci_discretized = pd.cut(F[:][i], bins=[float('-inf')] + spl_val, labels=False)
        Jrel = mutual_information(fci_discretized, C['class'].T)
        # chi2_R = chi2.sf(chi2_value, degree_of_freedom)
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
def algorithm2(fi, C, di, S, split_val, JmDSM_pre, e):
    j = di
    check = 0
    JmDSM_ori = 0
    JmDSM = 0
    chi2_Rrc = 0
    d_new = 0

    # discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='uniform')
    # fi_discretized = discretizer.fit_transform(fi.reshape(-1, 1)).flatten()
    fi_discretized = pd.cut(fi, bins=[float('-inf')] + split_val, labels=False)
    # chi2_Rrc = chi2.sf((2*len(fi_discretized)*np.log(2)*mutual_information(fi, C['class'].T)), (len(fi_discretized) - 1)*(len(np.unique(C)) - 1))
    # JmDSM = mutual_information(fi_discretized, C['class'].T) - ((len(np.unique(C)) - 1)*(j - 1))/(2*len(fi_discretized)*np.log(2))
    JmDSM_ori = mutual_information(fi_discretized, C['class'].T)
    c = 0
    c_test = 0
    r = 0
    r_test = 0
    for i in range(0, len(S)):
        # c = c + CMI(fi_discretized, S[i], C['class'].T)# - (len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2))
        # c_test = c_test + chi2.sf((2*len(fi_discretized)*np.log(2)*CMI(fi_discretized, S[i], C['class'].T)), len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1))
        # r = r + mutual_information(fi_discretized, S[i]) - ((j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2))
        # r_test = r_test + chi2.sf(2*len(fi_discretized)*np.log(2)*mutual_information(fi, S[i]), (j - 1)*(len(np.unique(S[i])) - 1))
        # JmDSM = JmDSM + (CMI(fi_discretized, S[i], C['class'].T) - (len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2)) - mutual_information(fi_discretized, S[i]) + ((j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2)))/len(S)
        # chi2_Rrc = chi2_Rrc + (chi2.sf((2*len(fi_discretized)*np.log(2)*CMI(fi_discretized, S[i], C['class'].T)), len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1)) - chi2.sf(2*len(fi_discretized)*np.log(2)*mutual_information(fi, S[i]), (j - 1)*(len(np.unique(S[i])) - 1)))/len(S)
        JmDSM_ori = JmDSM_ori + (CMI(fi_discretized, S[i], C['class'].T) - mutual_information(fi_discretized, S[i]))/len(S)
    if JmDSM_ori > (JmDSM_pre - e/len(S)):
        check = 1
    print(JmDSM_ori, JmDSM)

    return JmDSM_ori, check


def mDSM(F, C, e):
    Fc, Dc, Jc, split_val = algorithm1_change(F, C)
    Fc = Fc.T

    sorted_indices = np.argsort(Jc)[::-1]
    Fc = [Fc[i] for i in sorted_indices]
    Dc = [Dc[i] for i in sorted_indices]
    Jc = [Jc[i] for i in sorted_indices]
    split_val = [split_val[i] for i in sorted_indices]

    S = []
    # discretizer = KBinsDiscretizer(n_bins=Dc[0], encode='ordinal', strategy='uniform')
    # fci_discretized = discretizer.fit_transform(Fc[0].reshape(-1, 1)).flatten()
    fci_discretized = pd.cut(Fc[0], bins=[float('-inf')] + split_val[0], labels=False)
    S.append(fci_discretized)
    D = [Dc[0]]
    Fc.pop(0)
    Dc.pop(0)
    split_val.pop(0)
    T = -inf
    for i in range(0, len(Fc)):
        fi = Fc[i]
        di = Dc[i]
        JmDSM, check = algorithm2(fi, C, di, S, split_val[i], T, e)
        if check:
            # discretizer = KBinsDiscretizer(n_bins=di, encode='ordinal', strategy='uniform')
            # fci_discretized = discretizer.fit_transform(fi.reshape(-1, 1)).flatten()
            fci_discretized = pd.cut(Fc[i], bins=[float('-inf')] + split_val[i], labels=False)
            S.append(fci_discretized)
            D.append(Dc[i])
            T = JmDSM
    tmp_100 = -inf
    min_position_tmp_100th = -1
    S_ori = []
    tmp = -inf
    min_position_tmp = -1
    print(len(S))
    while 1:
        ep = np.zeros(len(S))
        for p in range(0, len(S)):
            ep[p] = mutual_information(S[p], C['class'].T)
            for i in range(0, len(S)):
                if i == p:
                    continue
                ep[p] = ep[p] + (CMI(S[p], S[i], C['class'].T) - mutual_information(S[p], S[i])) / (len(S) - 1)
        if min_position_tmp_100th == -1:
            # min_position_tmp = min_position
            # tmp = ep[min_position]
            k = 100
            sorted_ep = np.partition(ep, k - 1)  # Lấy mảng sao cho phần tử nhỏ thứ k nằm ở vị trí k-1
            tmp_100 = sorted_ep[k - 1]
            min_position_tmp_100th = np.where(ep == tmp_100)[0][0]  # Tìm chỉ số đầu tiên
            indices_to_remove = np.argsort(ep)[:k]
            S_ori = S
            S = [s for i, s in enumerate(S) if i not in indices_to_remove]
            continue
        print(tmp_100 + e/len(S), np.median(ep))
        if (np.median(ep)) > (tmp_100 + e/len(S)):
            # min_position_tmp = min_position
            # tmp = ep[min_position]
            sorted_ep = np.partition(ep, k - 1)  # Lấy mảng sao cho phần tử nhỏ thứ k nằm ở vị trí k-1
            tmp_100 = sorted_ep[k - 1]
            min_position_tmp_100th = np.where(ep == tmp_100)[0][0]  # Tìm chỉ số đầu tiên
            indices_to_remove = np.argsort(ep)[:k]
            S_ori = S
            S = [s for i, s in enumerate(S) if i not in indices_to_remove]
        else:
            break
    while 1:
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
        print(ep[min_position], tmp + e/len(S), np.median(ep))
        if (np.median(ep)) > (tmp + e/len(S)):
            min_position_tmp = min_position
            tmp = ep[min_position]
            S_ori = S
            S = S[:min_position_tmp] + S[min_position_tmp+1:]
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
file_path = 'lung.mat'
data = loadmat(file_path)
X = pd.DataFrame(data['X'])
y = pd.DataFrame(data['Y'], columns=['class'])

# # dbword
# file_path = 'dbworld_bodies.mat'
# data = loadmat(file_path)
# X = pd.DataFrame(data['inputs'])
# y = pd.DataFrame(data['labels'], columns=['class'])

# heart, pima
# df = pd.read_csv("F:\Download\diabetes.csv")
# y = df[[df.columns[-1]]]
# X = df[df.columns[:-1]]
# y.columns = ['class']

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

e = 20
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
model = SVC(kernel='linear')
loo = LeaveOneOut()
y_true, y_pred = [], []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred.append(model.predict(X_test)[0])
    y_true.append(y_test[0])
accuracy = accuracy_score(y_true, y_pred)
print(f'Độ chính xác trung bình với các đặc trưng đã chọn (LOO): {accuracy:.4f}')

# clf_all = SVC(kernel='linear')
#
# scores_all = cross_val_score(clf_all, X, y, cv=10)
# print(f"Độ chính xác trung bình với tất cả đặc trưng (10-CV): {scores_all.mean():.4f}")