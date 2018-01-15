import numpy as np
from collections import Counter


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def calculate_entropy(hidden_array):
    hidden_string_array = []
    for row in hidden_array:
        string = ''
        for digit in row:
            string += str(digit)
        hidden_string_array.append(string)

    # Calculate status entropy
    counter_for_h = Counter(hidden_string_array)
    p_x = np.array(list(counter_for_h.values())) / len(hidden_array)
    entropy_h = np.sum(-p_x * np.log(p_x))

    # Calculate frequency entropy
    counter_for_k = Counter(counter_for_h.values())
    p_x2 = np.array([k * v / len(hidden_array) for k, v in counter_for_k.items()])
    entropy_k = np.sum(-p_x2 * np.log(p_x2))

    return entropy_h, entropy_k

