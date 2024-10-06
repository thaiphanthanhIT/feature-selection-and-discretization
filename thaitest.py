import numpy as np
from collections import Counter

def compute_probabilities(X, Y, Z):
    """Tính xác suất có điều kiện từ dữ liệu X, Y và Z."""
    n = len(Z)  # Tổng số mẫu

    # Tính P(Z)
    P_z = Counter(Z)
    for k in P_z:
        P_z[k] /= n

    # Tính P(X|Z), P(Y|Z), và P(X, Y|Z)
    P_xz = {}
    P_yz = {}
    P_xyz = {}

    for x, y, z in zip(X, Y, Z):
        # P(x|z)
        if (x, z) not in P_xz:
            P_xz[(x, z)] = 0
        P_xz[(x, z)] += 1

        # P(y|z)
        if (y, z) not in P_yz:
            P_yz[(y, z)] = 0
        P_yz[(y, z)] += 1

        # P(x, y|z)
        if (x, y, z) not in P_xyz:
            P_xyz[(x, y, z)] = 0
        P_xyz[(x, y, z)] += 1

    # Chuẩn hóa các xác suất
    for (x, z) in P_xz:
        P_xz[(x, z)] /= Counter(Z)[z]
    for (y, z) in P_yz:
        P_yz[(y, z)] /= Counter(Z)[z]
    for (x, y, z) in P_xyz:
        P_xyz[(x, y, z)] /= Counter(Z)[z]

    return P_z, P_xz, P_yz, P_xyz

def conditional_mutual_information(X, Y, Z):
    """Tính thông tin tương hỗ có điều kiện I(X; Y | Z)."""
    P_z, P_xz, P_yz, P_xyz = compute_probabilities(X, Y, Z)
    I_xyz = 0

    # Tính toán thông tin tương hỗ có điều kiện
    for (x, y, z) in P_xyz:
        p_xyz = P_xyz[(x, y, z)]
        p_xz = P_xz[(x, z)]
        p_yz = P_yz[(y, z)]
        p_z = P_z[z]

        if p_xyz > 0 and p_xz > 0 and p_yz > 0:
            I_xyz += p_z * p_xyz * np.log(p_xyz / (p_xz * p_yz))

    return I_xyz

# Dữ liệu X, Y, Z
X = [1, 17, 18, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
Y = [23, 19, 10, 3, 50, 24, 10, 2, 30, 20, 18, 25, 38, 7, 20, 29, 4, 3, 27, 3]
Z = [1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

# Tính thông tin tương hỗ có điều kiện
I_xyz = conditional_mutual_information(X, Y, Z)
print("Conditional Mutual Information I(X; Y | Z):", I_xyz)
