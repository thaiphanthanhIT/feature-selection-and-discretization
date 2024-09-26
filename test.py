import numpy as np
from numpy import *
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import chi2


def algorithm1(F, C, max_d):
    number_features = F.shape[0]
    number_samples = F.shape[1]

    Fc = np.zeros((number_samples, 1))
    Dc = []
    Jc = []

    for i in range(0, number_features):
        I_fi_C = mutual_info_classif(F[i].reshape(1, -1), C)
        chi2_value = 2 * number_samples * np.log(2) * I_fi_C

        for j in range(2, max_d + 1):
            discretizer = KBinsDiscretizer(n_bins=j, encode='ordinal', strategy='uniform')
            fi_discretized = discretizer.fit_transform(F[i].reshape(-1, 1)).flatten()

            degree_of_freedom = (len(np.unique(F[i])) - 1)*(len(np.unique(C)) - 1)
            MI = mutual_info_score(fi_discretized, C)
            Jrel = chi2.ppf(I_fi_C, degree_of_freedom)

            if Jrel > chi2_value:
                Fi = F[i].reshape(-1, 1)
                Fc = insert(Fc, [i + 1], Fi, axis=1)
                Dc.append(j)
                Jc.append(Jrel)
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

        chi2_Rrc = 1
        JmDSM = mutual_info_score(fi_discretized, C) - ((len(np.unique(C)) - 1)*(j - 1))/(2*len(fi_discretized)*np.log(2))
        for i in range(0, len(S)):
            JmDSM = JmDSM + (mutual_info_score(np.column_stack([fi_discretized, S[i].flatten()]), C))/len(S)
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
    S.append(fci_discretized)
    D = [Dc[0]]
    Fc.pop(0)
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


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_d = 30
delta = 0
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
