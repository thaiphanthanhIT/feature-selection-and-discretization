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
from finalAlgorithm import mDSM

dataset = fetch_ucirepo(id=50)

X = dataset.data.features
y = dataset.data.targets
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
y.columns = ['class']
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
# df = pd.read_csv("F:\Download\dataset_heart.csv")
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

e = 2
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