import numpy as np

# Ví dụ mảng ep
ep = np.array([1,2,3,4,5,6,7,8,9])  # Mảng có 500 phần tử ngẫu nhiên từ 0 đến 999

# Tìm phần tử nhỏ thứ 100
k = 9
sorted_ep = np.partition(ep, k-1)  # Lấy mảng sao cho phần tử nhỏ thứ k nằm ở vị trí k-1
smallest_100th_value = sorted_ep[k-1]

print(f"Phần tử nhỏ thứ {k}: {smallest_100th_value}")

# Nếu cần vị trí của phần tử đó
min_position_100th = np.where(ep == smallest_100th_value)[0][0]  # Tìm chỉ số đầu tiên
print(f"Vị trí phần tử nhỏ thứ {k}: {min_position_100th}")
