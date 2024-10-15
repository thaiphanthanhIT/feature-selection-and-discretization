from enum import unique

import numpy as np

def convert_array_class(A):
    sorted_A = sorted(set(A))
    rank_map = {value: rank for rank, value in enumerate(sorted_A)}

    result = [rank_map[value] for value in A]
    return result

def convert_array_feature(A):
    sorted_A = sorted(set(A))
    rank_map = {value: rank + 1 for rank, value in enumerate(sorted_A)}

    result = [rank_map[value] for value in A]
    return result
def calculate_frequency(bin, cls):
    num_bins = len(np.unique(bin)) + 1
    num_classes = len(np.unique(cls))
    freq = np.zeros((num_classes, num_bins), dtype=int)

    for b, c in zip(bin, cls):
        freq[c, b] += 1

    return freq

# A = [5, 10, 15, 10]
# B = [0, 0, 1, 1]
# val = convert_array(A)
# freq = calculate_frequency(val, B)
# print(val)
# print(freq)

