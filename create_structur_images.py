import numpy as np
import matplotlib.pyplot as plt
import skimage.io as sk_io
import InputListUtils
import os
from multiprocessing import Pool
import shutil

def check_if_struc_already_calced(name):
    calculate = True
    existing_names = os.listdir(f"data/healthy_structure")
    if name in existing_names:
        calculate = False
    existing_names = os.listdir(f"data/diabetic_structure")
    if name in existing_names:
        calculate = False
    return calculate


def process_structure(struc_path):
    struc_name = file_name_generator(struc_path, type=type)
    if check_if_struc_already_calced(os.path.basename(struc_name)):
        try:
            struc_image = np.zeros((csize, bsize))
            with open(struc_path, "rb") as f:
                for i in range(csize):
                    read_from_file = np.fromfile(f, dtype="uint8", count=asize * bsize)
                    read_from_file = read_from_file.reshape((bsize, asize))
                    struc_image[i] = np.mean(read_from_file, axis=1)
            sk_io.imsave(struc_name, struc_image)
            print(f"Saved to {struc_name}")
        except:
            print(f"Failed to calc {struc_name}")


def file_name_generator(old_path, type="healthy"):
    basename = os.path.basename(old_path).replace(".bin", ".png").replace("struc_", "enf_")
    folder = f"data/{type}_structure"
    return os.path.join(folder, basename)

if False:
    type = "healthy"
    h_list = [path for path, label in InputListUtils.__fill_healthy_input_list(type="struc")]
    with Pool(2) as p:
        p.map(process_structure, h_list)

    type = "diabetic"
    d_list = [path for path, label in InputListUtils.__fill_diabetic_input_list(type="struc")]
    with Pool(4) as p:
        p.map(process_structure, d_list)

#struc_path = "/media/julius/My Passport/MOON1e/H1/rechts/struc_1536x2048x2044x2_2666.bin"
asize = 1536
bsize = 2048
csize = 2044

def get_id(name):
    return name.split("_")[-1].split(".")[0]

def sort_files(type="healthy"):
    test_files = [get_id(name) for name in os.listdir(f"data/{type}_images/test_files")]
    train_files = [get_id(name) for name in os.listdir(f"data/{type}_images/train_files")]
    structure_files = os.listdir(f"data/{type}_structure")
    for structure_image in structure_files:
        id = get_id(structure_image)
        if id in test_files:
            shutil.move(os.path.join(f"data/{type}_structure", structure_image),
                        os.path.join(f"data/{type}_structure/test_files", structure_image))
        if id in train_files:
            shutil.move(os.path.join(f"data/{type}_structure", structure_image),
                        os.path.join(f"data/{type}_structure/train_files", structure_image))

sort_files("diabetic")
# #d_list = InputListUtils.__fill_diabetic_input_list()
# new_im = sk_io.imread("test.png")
# print(new_im)

