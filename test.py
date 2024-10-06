from collections import Counter

import numpy as np
from numpy import *
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import chi2

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

def algorithm1(F, C, max_d):
    number_features = F.shape[0]
    number_samples = F.shape[1]

    Fc = np.zeros((number_samples, 1))
    Dc = []
    Jc = []
    cnt = 0
    for i in range(0, number_features):
        I_fi_C = mutual_info_score(F[i], C)
        chi2_value = 2 * number_samples * np.log(2) * I_fi_C

        for j in range(2, max_d + 1):
            discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='uniform')
            fi_discretized = discretizer.fit_transform(F[i].reshape(-1, 1)).flatten()

            degree_of_freedom = (len(np.unique(F[i])) - 1)*(len(np.unique(C)) - 1)
            Jrel = mutual_info_score(fi_discretized, C) + ((j - 1)*(len(np.unique(C)) - 1))/(2 * number_samples * np.log(2))
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

    for j in range(di - delta, di + delta + 1):
        if j < 2:
            continue

        discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='uniform')
        fi_discretized = discretizer.fit_transform(fi.reshape(-1, 1)).flatten()
        tmp = len(S)
        chi2_Rrc = 2 * len(fi_discretized) * np.log(2) * mutual_info_score(fi, C)
        JmDSM = mutual_info_score(fi_discretized, C) - ((len(np.unique(C)) - 1)*(j - 1))/(2*len(fi_discretized)*np.log(2))
        for i in range(0, len(S)):
            f = S[i]
            k = (len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2))
            q = CMI(fi_discretized, S[i], C)
            JmDSM = JmDSM + (CMI(fi_discretized, S[i], C) - (len(np.unique(C))*(j - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2)) - mutual_info_score(fi_discretized, S[i]) + ((len(np.unique(C)) - 1)*(len(np.unique(S[i])) - 1))/(2*len(fi_discretized)*np.log(2)))/len(S)
            chi2_Rrc = chi2_Rrc + (2 * len(fi_discretized) * np.log(2) * CMI(fi_discretized, S[i], C) - 2 * len(fi_discretized) * np.log(2) * mutual_info_score(fi, S[i]))/len(S)
        if JmDSM > chi2_Rrc:
            best_di = j
            Ji = JmDSM
            T = chi2_Rrc

    return Ji, T, best_di


def mDSM(F, C, maxd, delta):
    Fc, Dc, Jc = algorithm1(F, C, maxd)
    Fc = Fc.T

    sorted_indices = np.argsort(Jc)[::-1]
    Fc = [Fc[i] for i in sorted_indices]
    Dc = [Dc[i] for i in sorted_indices]
    Jc = [Jc[i] for i in sorted_indices]

    S = []
    discretizer = KBinsDiscretizer(n_bins=Dc[0], encode='ordinal', strategy='uniform')
    fci_discretized = discretizer.fit_transform(Fc[0].reshape(-1, 1)).flatten()
    S.append(Fc[0])
    D = [Dc[0]]
    Fc.pop(0)
    Dc.pop(0)
    for i in range(0, len(Fc)):
        fi = Fc[i]
        di = Dc[i]

        JmDSM, T, dnew = algorithm2(fi, C, di, delta, S)
        discretizer = KBinsDiscretizer(n_bins=dnew, encode='ordinal', strategy='uniform')
        fi_discretized = discretizer.fit_transform(fi.reshape(-1, 1)).flatten()
        if JmDSM is not None and JmDSM > T:
            S.append(fi_discretized)
            D.append(dnew)

    return S, D

X = np.array([[1, 17, 18, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
     [23, 19, 10, 3, 50, 24, 10, 2, 30, 20, 18, 25, 38, 7, 20, 29, 4, 3, 27, 3],
     [2, 4, 6, 8, 28, 30, 14, 16, 18, 20, 12, 14, 26, 28, 30, 32, 34, 36, 38, 40],
     [19, 15, 20, 17, 14, 5, 1, 6, 14, 18, 20, 19, 14, 18, 3, 8, 19, 5, 4, 7],
     [8, 18, 19, 16, 30, 21, 24, 23, 22, 20, 19, 12, 13, 16, 19, 20, 10, 22, 22, 25]])
y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
X = X.T
# iris = load_iris()
# X = iris.data
# y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_d = 10
delta = 2
S, D = mDSM(X_train.T, y_train, max_d, delta)

clf = RandomForestClassifier()

X_train_selected = np.array(S).T
X_test_selected = np.array([X_test[:, i] for i in range(len(S))]).T

clf.fit(X_train_selected, y_train)

y_pred = clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print(f"Độ chính xác của mô hình khi sử dụng các đặc trưng đã chọn: {accuracy:.4f}")

clf_all = RandomForestClassifier()
clf_all.fit(X_train, y_train)

y_pred_all = clf_all.predict(X_test)
accuracy_all = accuracy_score(y_test, y_pred_all)

print(f"Độ chính xác của mô hình khi sử dụng tất cả các đặc trưng: {accuracy_all:.4f}")

