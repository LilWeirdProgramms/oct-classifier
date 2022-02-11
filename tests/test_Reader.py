import numpy as np
import matplotlib.pyplot as plt
from BinaryReader import BinaryReader, InstanceDim
from InputList import diabetic_training_files
import os
import InputList

alen = 6
blen = 4
clen = 3


def test_80_percent():
    br = BinaryReader()
    split = br._one_or_80_percent([1, 2, 3, 4])
    assert split == 3
    split = br._one_or_80_percent([1, 2])
    assert split == 1
    split = br._one_or_80_percent([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert split == 8


def create_testbinary():
    fake_input_size = (alen, blen, clen)
    fake_input_data = []
    for i in range(fake_input_size[2]):
        for j in range(fake_input_size[1]):
            for k in range(fake_input_size[0]):
                fake_input_data.append(i+j)
    create_binary(fake_input_data)


def create_testbinary2():
    fake_input_size = (1, 8, 8)
    fake_input_data = []
    for i in range(fake_input_size[2]):
        for j in range(fake_input_size[1]):
            for k in range(fake_input_size[0]):
                fake_input_data.append(i * fake_input_size[1] + j)
    create_binary(fake_input_data)


def create_binary(data):
    fake_input_data_bytes = [num.to_bytes(2, byteorder='little') for num in data]
    with open("testbinary.bin", "wb") as f:
        for element in fake_input_data_bytes:
            f.write(element)


def _test_binary_to_instance():
    br = BinaryReader()
    generator = br.instance_from_binaries_generator(diabetic_training_files)
    i, j = 0, 0
    for elem in generator:
        if i == np.random.randint(30, 90):
            plt.imsave("instance_test/b-scan_test_nr" + str(j) + ".png", elem[0][:,:,-1])
            plt.imsave("instance_test/c-scan_test_nr" + str(j) + ".png", elem[0][:,3,:])
            i = 0
            j += 1
        i += 1

def test_diabetic_healthy_diff():
    a = plt.figure(figsize=(16, 10))
    br = BinaryReader()
    generator = br.instance_from_binaries_generator(
        [InputList.diabetic_training_files[4]]
    )
    for elem in generator:
        a = elem[0][:, 0, 0, 0]
        b = elem[0][:, 1, 0, 0]
        plt.plot(a, "--", alpha=0.7)
        plt.plot(b, "--", alpha=0.7)
        break
    plt.savefig('test.png')

def test_info_map():
    br = BinaryReader()
    br._create_info_map("D87/rechts/raw_1536x2048x2045x2_30515.bin", [1, 2])
    assert br.info_map[0][0] == "30515"


def test_instance_generator2():
    create_testbinary2()
    br = BinaryReader()
    br.ascan_length = 1
    br.bscan_length = 8
    br.cscan_length = 8
    b_size = 2
    c_size = 2
    with open("testbinary.bin", "rb") as f:
        generator1 = br._create_instance(f, InstanceDim(2, 2, 4, 4))
        expected_instance1 = np.array([
           [[[0], [8]],
            [[1], [9]]],
        ])
        assert np.array_equal(generator1, expected_instance1)
        f.seek(2*b_size*br.ascan_length, os.SEEK_SET)
        generator2 = br._create_instance(f, InstanceDim(2, 2, 4, 4))
        expected_instance2 = np.array([
           [[[2], [10]],
            [[3], [11]]],
        ])
        assert np.array_equal(generator2, expected_instance2)
        temp = 2*b_size + 0*c_size*b_size
        f.seek(2*temp*br.ascan_length, os.SEEK_SET)
        generator3 = br._create_instance(f, InstanceDim(2, 2, 4, 4))
        expected_instance3 = np.array([
           [[[4], [12]],
            [[5], [13]]],
        ])
        assert np.array_equal(generator3, expected_instance3)
        temp = 0*b_size + 1*c_size*br.bscan_length
        f.seek(2*temp*br.ascan_length, os.SEEK_SET)
        generator4 = br._create_instance(f, InstanceDim(2, 2, 4, 4))
        expected_instance4 = np.array([
           [[[16], [24]],
            [[17], [25]]],
        ])
        assert np.array_equal(generator4, expected_instance4)
        temp = 1*b_size + 2*c_size*br.bscan_length
        f.seek(2*temp*br.ascan_length, os.SEEK_SET)
        generator5 = br._create_instance(f, InstanceDim(2, 2, 4, 4))
        expected_instance5 = np.array([
           [[[34], [42]],
            [[35], [43]]],
        ])
        assert np.array_equal(generator5, expected_instance5)


def _test_instance_generator():
    create_testbinary()
    br = BinaryReader()
    br.ascan_length = alen
    br.bscan_length = blen
    br.cscan_length = clen
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




