import pandas as pd
import InputList
import re
import os

server_location = "/mnt/p_Zeiss_Clin/Projects/UWF OCTA/Clinical data/MOON1"
healthy_retina_location = "data/healthy_retina_file_list.txt"
diabetic_retina_location = "data/diabetic_retina_file_list.txt"

def find_binaries(query, label, location=InputList._server_location, type="raw"):
    all_folders = os.listdir(location)
    my_diabetic_folder = []
    for folder in all_folders:
        if re.search(query, folder):
            my_diabetic_folder.append(folder)
    my_diabetic_files = []
    for path in [os.path.join(location, folder) for folder in my_diabetic_folder]:
        for path, subdirs, files in os.walk(path):
            for file in files:
                if re.search(rf"{type}_1536x2048x204.x2.+.bin$", file):
                    my_diabetic_files.append((os.path.join(path, file), label))
    return my_diabetic_files

list_of_test_folders = ["D99", "D79", "D107", "D105", "D116", "D23", "D88", "D57", "H1", "H20", "H35"]
def find_binaries_test_train(query, label, location=InputList._server_location, type="raw"):
    all_folders = os.listdir(location)
    my_diabetic_folder = []
    for folder in all_folders:
        if re.search(query, folder):
            my_diabetic_folder.append(folder)
    my_test_files = []
    my_train_files = []
    for folder in my_diabetic_folder:
        path = os.path.join(location, folder)
        if folder in list_of_test_folders:
            for path, subdirs, files in os.walk(path):
                for file in files:
                    if re.search(rf"{type}_1536x2048x204.x2.+.bin$", file):
                        my_test_files.append((os.path.join(path, file), label))
        else:
            for path, subdirs, files in os.walk(path):
                for file in files:
                    if re.search(rf"{type}_1536x2048x204.x2.+.bin$", file):
                        my_train_files.append((os.path.join(path, file), label))
    return my_train_files, my_test_files


def get_test_train_file_lists(type="raw"):
    # train_bin_healthy, test_bin_healthy = find_binaries_test_train(r"^H([0-9]|[0-9][0-9])", 0,
    #                                                        location=server_location, type=type)
    train_bin_diabetic, test_bin_diabetic = find_binaries_test_train(r"^D([8-9][0-9]|[0-9][0-9][0-9])$", 1,
                                                            location=server_location, type=type)
    # train_list = train_bin_healthy + train_bin_diabetic
    # test_list = test_bin_healthy + test_bin_diabetic
    train_list = train_bin_diabetic
    test_list = test_bin_diabetic
    return train_list, test_list



def __fill_healthy_input_list(type="raw"):
    file_list = []
    file_list.extend(find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=server_location, type=type))
    if os.path.exists(hdd1 := "/media/julius/My Passport/MOON1e"):
        file_list.extend(
            find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=hdd1, type=type))
    if os.path.exists(hdd2 := "/media/julius/My Passport1/MOON1e"):
        file_list.extend(
            find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=hdd2, type=type))
    InputList.training_files = file_list
    return file_list


def __fill_diabetic_input_list(fill_from=r"^D([0-9]|[0-9][0-9]|[0-9][0-9][0-9])$", type="raw"):
    file_list = []
    file_list.extend(find_binaries(fill_from, 1, location=server_location, type=type))
    InputList.training_files = file_list
    return file_list


def read_from_file(path, label):
    file_list = []
    with open(path, 'r') as f:
        for item in f.read().splitlines():
            file_list.append((item, label))
    return file_list


def write_to_file(file_list, path):
    with open(path, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item[0])


def create_retina_file_lists():
    id = image2d.ImageDataset()
    __fill_healthy_input_list()
    healthy_image_list = id.get_training_files()
    write_to_file(healthy_image_list, healthy_retina_location)
    __fill_diabetic_input_list()
    diabetic_image_list = id.get_training_files()
    write_to_file(diabetic_image_list, diabetic_retina_location)


def create_raw_file_lists():
    __fill_healthy_input_list()
    write_to_file(InputList.training_files, "data/healthy_raw_file_list.txt")
    __fill_diabetic_input_list()
    write_to_file(InputList.training_files, "data/diabetic_raw_file_list.txt")


import shutil
def copy_images_to_data():
    file_list = read_from_file(healthy_retina_location, 0)
    healthy_image_list = []
    for file, label in file_list:
        file_name = os.path.basename(file)
        destination = os.path.join("data/healthy_images", file_name)
        if os.path.exists(destination):
            raise FileExistsError(f"File {file_name} would be overwritten")
        shutil.copyfile(file, destination)
    healthy_image_list.append(destination)
    file_list = read_from_file(diabetic_retina_location, 1)
    for file, label in file_list:
        file_name = os.path.basename(file)
        destination = os.path.join("data/diabetic_images", file_name)
        if os.path.exists(destination):
            raise FileExistsError(f"File {file_name} would be overwritten")
        shutil.copyfile(file, destination)


def copy_images_to_data2(image_class="diabetic", image_type="angio", copy_from=r"^D([0-9]|[0-9][0-9]|[0-9][0-9][0-9])$",
                         copy_to="new_files", remove_wrong_dim=True):
    __fill_diabetic_input_list(copy_from)

    id = image2d.ImageDataset(image_type, image=remove_wrong_dim)
    diabetic_image_list = id.get_training_files()

    for file, label in diabetic_image_list:
        file_name = os.path.basename(file)
        destination = os.path.join(f"data/{image_class}_images/{copy_to}", file_name)
        if os.path.exists(destination):
            print(f"File {file_name} would be overwritten by {file}")
        else:
            shutil.copyfile(file, destination)


def copy_images_to_data3(type, image_type="angio"):
    if type == "healthy":
        __fill_healthy_input_list()
    if type == "diabetic":
        __fill_diabetic_input_list()
    id = image2d.ImageDataset(image_type)
    found = id.get_training_files()
    present = get_file_list_from_folder(f"data/{type}_images", -1, only_training=False)
    search_dict = dict.fromkeys([os.path.basename(file) for file, label in present])
    for found_file, label in found:
        try:
            search_dict[os.path.basename(found_file)]
        except KeyError:
            destination = os.path.join(f"data/{type}_images/new_files", os.path.basename(found_file))
            shutil.copyfile(found_file, destination)


import pandas
from collections import defaultdict
def copy_enf_files_to_retinas():
    merge_data_in_dict = defaultdict(lambda: [])
    for path, subdirs, files in os.walk(f"data"):
        # if 'trash_files' in subdirs:
        #     subdirs.remove('trash_files')  # don't visit CVS directories
        for file_path in files:
            file_name = os.path.basename(file_path)
            if "1536x2048x204" in file_name:
                data_name = file_name.split(".")[0].split("_")[-1]
                merge_data_in_dict[data_name].append(file_name)
    ret_cntr, struc_cntr, onh_cntr = 0, 0, 0
    for key in merge_data_in_dict:
        for name in merge_data_in_dict[key]:
            if "retina" in name:
                ret_cntr += 1
            if "enf" in name:
                struc_cntr += 1
            if "onh" in name:
                onh_cntr += 1
    print(f"Number of Samples: {len(merge_data_in_dict)}")
    print(f"Number of Retina Images: {ret_cntr}")
    print(f"Number of Struct Images: {struc_cntr}")
    print(f"Number of ONH Overlays: {onh_cntr}")


import typing
import pathlib
def get_file_list_from_folder(file_list_folder: typing.Union[str, pathlib.Path], label, only_training=True):
    path_list = []
    for dirpath, subdirs, filenames in os.walk(file_list_folder):
        if 'trash_files' in subdirs and only_training:
            subdirs.remove('trash_files')
        if 'test_files' in subdirs and only_training:
            subdirs.remove('test_files')
        for f in filenames:
            path_list.append((os.path.abspath(os.path.join(dirpath, f)), label))
    return path_list


if __name__ == "__main__":
    #create_retina_file_lists()
    #create_raw_file_lists()
    #copy_images_to_data3("diabetic", "angio")
    #copy_enf_files_to_retinas("diabetic")
    copy_images_to_data3("healthy")
    #copy_images_to_data2(image_type="overlay", remove_wrong_dim=False)
    #copy_enf_files_to_retinas()