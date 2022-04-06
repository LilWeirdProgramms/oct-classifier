import math
import re
import numpy as np
import tensorflow.keras as k

def extract_position(string_to_search: str):
    b_pos = re.findall(r"b[0-9][0-9]?", string_to_search)
    c_pos = re.findall(r"c[0-9][0-9]?", string_to_search)
    if b_pos and c_pos:
        b_pos = int(b_pos[0][1:])
        c_pos = int(c_pos[0][1:])
    else:
        raise ValueError("Please Name input files b[number] c[number] to show its position in bag")
    return b_pos, c_pos


def reorder_by_name(files_to_reorder: list, model_output: np.array, dim=(20, 20)) -> np.array:
    reshaped_files = np.zeros(dim)
    for file, output in zip(files_to_reorder, model_output):
        b_dim, c_dim = extract_position(file)
        reshaped_files[b_dim, c_dim] = output
    return reshaped_files


def calc_receptive_field3D(model: k.Model):
    last_receptive_field_size = (1, 1, 1)
    for layer in model.layers:
        if layer.__class__.__name__ == "Conv3D":
            last_receptive_field_size = np.array(layer.strides) * np.array(last_receptive_field_size) + \
                                        np.array(layer.kernel_size) - np.array(layer.strides)
        elif layer.__class__.__name__ == "MaxPooling3D":
            last_receptive_field_size = np.array(layer.pool_size) * np.array(last_receptive_field_size) + \
                                        np.array(layer.pool_size) - np.array(layer.pool_size)
    return last_receptive_field_size
    #stride * last_receptive_field_size * (kernel_size - stride)


def fourier_coefficients(k, N):
    return [np.exp(-1j * 2 * np.pi * k * n / N) for n in range(N)]


def fourier_coefficients_of_series(N):
    return [fourier_coefficients(k, N) for k in range(N)]


import matplotlib.pyplot as plt
if __name__ == "__main__":
    fc = fourier_coefficients_of_series(16)
    # plt.imshow(np.angle(fc))
    # plt.show()
    plt.imshow(np.real(fc))
    plt.show()
    plt.imshow(np.imag(fc))
    plt.show()
    print(extract_position("b1, c20"))
    print(extract_position("a1_b2_c3"))
    print(reorder_by_name(["b1_c1", "b2_c3"], [5, 4]))
