import numpy as np
import matplotlib.pyplot as plt
from BinaryReader import BinaryReader, InstanceDim
from InputList import diabetic_training_files
import os

def create_testbinary():
    fake_input_size = (6,4,3)
    fake_input_data = []
    for i in range(fake_input_size[2]):
        for j in range(fake_input_size[1]):
            for k in range(fake_input_size[0]):
                fake_input_data.append(i+j)
    print(fake_input_data)
    fake_input_data_bytes = [num.to_bytes(2, byteorder='little') for num in fake_input_data]
    with open("testbinary.bin", "wb") as f:
        for element in fake_input_data_bytes:
            f.write(element)

def test_binary_to_instance():
    br = BinaryReader()
    generator = br.instance_from_binaries_generator(diabetic_training_files, 0)
    i, j = 0, 0
    for elem in generator:
        if i == np.random.randint(30, 90):
            plt.imsave("instance_test/b-scan_test_nr" + str(j) + ".png", elem[0][:,:,-1])
            plt.imsave("instance_test/c-scan_test_nr" + str(j) + ".png", elem[0][:,3,:])
            i = 0
            j += 1
        i += 1

def test_instance_generator():
    create_testbinary()
    br = BinaryReader()
    br.ascan_length = 6
    br.bscan_length = 4
    br.cscan_length = 3
    b_size = 2
    with open("testbinary.bin", "rb") as f:
        generator1 = br._create_instance(f, InstanceDim(2,2,2,1))
        expected_instance1 = np.array([
           [[0, 1],
            [1, 2]],
           [[0, 1],
            [1, 2]],
           [[0, 1],
            [1, 2]],
           [[0, 1],
            [1, 2]],
           [[0, 1],
            [1, 2]],
           [[0, 1],
            [1, 2]],
        ])
        assert np.array_equal(np.array(generator1.tolist()), expected_instance1)
        f.seek(2*b_size*br.ascan_length, os.SEEK_SET)
        generator2 = br._create_instance(f, InstanceDim(b_size,2,2,1))
        expected_instance2 = np.array([
                [[2, 3],
                 [3, 4]],
                [[2, 3],
                 [3, 4]],
                [[2, 3],
                 [3, 4]],
                [[2, 3],
                 [3, 4]],
                [[2, 3],
                 [3, 4]],
                [[2, 3],
                 [3, 4]],
            ])
        assert np.array_equal(np.array(generator2.tolist()), expected_instance2)




