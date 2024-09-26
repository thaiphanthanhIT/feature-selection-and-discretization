import numpy as np

# Mảng cột
column_array = np.array([[1], [2], [3], [1], [2], [4]])

# Tìm các phần tử khác nhau
unique_elements = np.unique(column_array)

# Đếm số lượng phần tử khác nhau
num_unique = len(unique_elements)

print("Số phần tử khác nhau:", num_unique)
print("Các phần tử khác nhau:", unique_elements)
