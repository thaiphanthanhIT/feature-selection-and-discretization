import openml
import pandas as pd

# # Tải dataset Spambase từ OpenML với ID tương ứng (ID = 44 cho Spambase)
spambase = openml.datasets.get_dataset(44)

# Lấy dữ liệu thành DataFrame
X, y, _, _ = spambase.get_data(target=spambase.default_target_attribute)

# Kiểm tra kích thước của X và y
print(X.shape, y.shape)

# Hiển thị 5 dòng đầu tiên của dataset
print(X.head())