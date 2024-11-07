import pandas as pd
import numpy as np

# Đọc dữ liệu từ tệp và tách thành các dòng
# with open('arrhythmia.data', 'r') as file:
#     lines = file.readlines()

#     # Tạo danh sách cho các feature và target
#     features = []
#     targets = []

#     for line in lines:
#         row = [float(i) if i != '?' else np.nan for i in line.strip().split(",")]
#         features.append(row[:-1])  # Lấy tất cả các giá trị trừ giá trị cuối làm feature
#         targets.append(row[-1])  # Lấy giá trị cuối cùng làm target

#     # Tạo DataFrame cho features và target
#     X = pd.DataFrame(features)
#     y = pd.DataFrame(targets, columns=["class"])

#     # Điền giá trị trung bình cho các giá trị NaN trong cả X và y
#     X.fillna(X.mean(), inplace=True)
#     y.fillna(y.mean(), inplace=True)
features = []
with open('train_data/madelon_train.data', 'r') as f:
    for line in f:
        row = [float(x) for x in line.split()]
        features.append(row)

X = pd.DataFrame(features)

targets = []
with open('train_data/madelon_train.labels', 'r') as f:
    for line in f:
        targets.append(int(line.strip()))  # Chuyển mỗi nhãn thành kiểu số nguyên (int)

# Chuyển nhãn thành một DataFrame hoặc Series nếu muốn
y = pd.DataFrame(targets, columns=["class"])
    # Hiển thị DataFrames X và y
print("Features (X):")
print(X)
print("\nTarget (y):")
print(y)
num_features = X.shape[1]

print("Số lượng feature:", num_features)