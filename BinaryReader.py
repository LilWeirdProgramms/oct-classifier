import InputList
from collections import namedtuple
import numpy as np
import os

class BinaryReader:


    def __init__(self):
        self.ascan_length = 1536
        self.bscan_length = 2047
        self.cscan_length = 2045
        self.files = pathlib

    # Mix diabetic and healthy files and store the label somewhere TODO
    def get_slices_from_binary(self):
        bscans = np.empty((n_bscans, Ascan_length, b_dim), dtype=dt)

        with open(fullpath, "rb") as f:
            for i in range(n_bscans):
                for j in range(b_dim):
                    bscans[i,:,j] = np.fromfile(f, dtype=dt, count=a_dim)


    def decide_slice_size(self):
        # for dim in [self.Bscan_length, self.Ascan_length]:
        #     for i in range(1,dim):
        #         if dim%i < 2:
        #             print(i, dim//i, dim%i) -> Slice 89, 73 mal
        Slice = namedtuple("Slice", "bsize csize")
        self.cscan_length = 2044 # -> One has to go
        return Slice(89, 73)

    """ Randomly select file and File Batches->Should be created earlier """ #TODO
    def batches(self):
        yield 0

    def get_test_samples(self):
        pass


# Batches from one File??
# -> Can I Make test hdf5 Dataset?

hallo = BinaryReader()
