import math
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo
import pandas as pd
from numpy import *
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
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


def algorithm1_change(F, C, max_d):
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
        fi = F[:][i].values.T
        I_fi_C = mutual_information(fi, C.T)
        chi2_value = 2 * number_samples * np.log(2) * I_fi_C
        degree_of_freedom = (number_samples - 1) * (len(np.unique(C)) - 1)
        fci_discretized = pd.cut(F[:][i], bins=[float('-inf')] + spl_val, labels=False)
        Jrel = mutual_information(fci_discretized, C.T)
        chi2_R = chi2.sf(chi2_value, degree_of_freedom)
        if Jrel > chi2_R:
            Fi = F[:][i].values
            Fi = Fi.reshape(-1, 1)
            Fc = insert(Fc, [cnt + 1], Fi, axis=1)
            Dc.append(j)
            Jc.append(Jrel)
            split_val.append(spl_val)
            cnt = cnt + 1
    Fc = np.delete(Fc, 0, axis=1)
    return Fc, Dc, Jc, split_val

def algorithm1(F, C, max_d):
    number_features = F.shape[0]
    number_samples = F.shape[1]

    Fc = np.zeros((number_samples, 1))
    Dc = []
    Jc = []
    cnt = 0
    for i in range(0, number_features):
        I_fi_C = mutual_information(F[i], C) # (len(np.unique(F[i])) - 1)*(len(np.unique(C)) - 1)/(2 * number_samples * np.log(2))
        chi2_value = 2 * number_samples * np.log(2) * I_fi_C

        for j in range(2, max_d + 1):
            bin_edges = np.percentile(F[i], np.linspace(0, 100, j))
            fi_discretized = np.digitize(F[i], bin_edges, right=True)
            # discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='uniform')
            # fi_discretized = discretizer.fit_transform(F[i].reshape(-1, 1)).flatten()

            degree_of_freedom = (number_samples - 1)*(len(np.unique(C)) - 1)
            Jrel = mutual_information(fi_discretized, C)
            chi2_R = chi2.sf(chi2_value, degree_of_freedom)

            if Jrel > chi2_R:
                Fi = F[i].reshape(-1, 1)
                Fc = insert(Fc, [cnt + 1], Fi, axis=1)
                Dc.append(j)
                Jc.append(Jrel)
                cnt = cnt+1
                break
    Fc = np.delete(Fc, 0, axis=1)
    return Fc, Dc, Jc


def algorithm2(fi, C, di, delta, S):
    Ji = None
    T = None
    best_di = di

    for j in range(di, di + delta + 1):
        if j < 2:
            continue

        discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='uniform')
        fi_discretized = discretizer.fit_transform(fi.reshape(-1, 1)).flatten()
        chi2_Rrc = chi2.sf((2*len(fi_discretized)*np.log(2)*mutual_information(fi, C)), (len(fi_discretized) - 1)*(len(np.unique(C)) - 1))
        JmDSM = mutual_information(fi_discretized, C) - ((len(np.unique(C)) - 1)*(j - 1))/(2*len(fi_discretized)*np.log(2))
        for i in range(0, len(S)):
            k = mutual_information(fi_discretized, S[i]) - ((len(np.unique(C)) - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2))
            n = chi2.sf(2*len(fi_discretized)*np.log(2)*(mutual_information(fi, S[i])-((len(np.unique(C)) - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2))), (len(np.unique(C)) - 1)*(len(np.unique(S[i])) - 1))
            JmDSM = JmDSM + (CMI(fi_discretized, S[i], C) - (len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2)) - mutual_information(fi_discretized, S[i]) + ((len(np.unique(C)) - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2)))/len(S)
            chi2_Rrc = chi2_Rrc + (chi2.sf((2*len(fi_discretized)*np.log(2)*CMI(fi_discretized, S[i], C)), len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1)) - chi2.sf(2*len(fi_discretized)*np.log(2)*mutual_information(fi, S[i]), (len(np.unique(C)) - 1)*(len(np.unique(S[i])) - 1)))/len(S)
        # if JmDSM < 0:
        #     continue
        if JmDSM < chi2_Rrc:
            best_di = j
            Ji = JmDSM
            T = chi2_Rrc

    return Ji, T, best_di


def mDSM(F, C, maxd, delta):
    Fc, Dc, Jc, split_val = algorithm1_change(F, C, maxd)
    Fc = Fc.T

    sorted_indices = np.argsort(Jc)[::-1]
    Fc = [Fc[i] for i in sorted_indices]
    Dc = [Dc[i] for i in sorted_indices]
    Jc = [Jc[i] for i in sorted_indices]
    split_val = [split_val[i] for i in sorted_indices]

    S = []
    # discretizer = KBinsDiscretizer(n_bins=Dc[0], encode='ordinal', strategy='uniform')
    # fci_discretized = discretizer.fit_transform(Fc[0].reshape(-1, 1)).flatten()
    fci_discretized = pd.cut(Fc[0], bins=[float('-inf')]+split_val[0], labels=False)
    S.append(fci_discretized)
    D = [Dc[0]]
    Fc.pop(0)
    Dc.pop(0)
    split_val.pop(0)
    for i in range(0, len(Fc)):
        # fi = Fc[i]
        # di = Dc[i]
        # JmDSM, T, dnew = algorithm2(fi, C, di, delta, S)
        # discretizer = KBinsDiscretizer(n_bins=dnew, encode='ordinal', strategy='uniform')
        # fi_discretized = discretizer.fit_transform(fi.reshape(-1, 1)).flatten()
        # if JmDSM is not None and JmDSM < T:
        #     S.append(fi_discretized)
        #     D.append(dnew)

        fci_discretized = pd.cut(Fc[i], bins=[float('-inf')]+split_val[i], labels=False)
        S.append(fci_discretized)
        D.append(Dc[i])

    return S, D


dataset = fetch_ucirepo(id=50)

X = dataset.data.features
y = dataset.data.targets
# X = X.drop('ID', axis=1)
# X = X.values


max_d = int(len(y)/10)
delta = 2
S, D = mDSM(X, y, max_d, delta)
y = y.values
X = np.array(S).T
# X_test_selected = np.array([X_test[:, i] for i in range(len(S))]).T

# big
# clf = SVC(kernel='linear')
clf = KNeighborsClassifier(n_neighbors=len(unique(y)))
scores_selected = cross_val_score(clf, X, y, cv=10)
print(len(S))
print(f"Độ chính xác trung bình với các đặc trưng đã chọn (10-CV): {scores_selected.mean():.4f}")

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