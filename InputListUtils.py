import InputList
import re
import os
import image2d

healthy_retina_location = "data/healthy_retina_file_list.txt"
diabetic_retina_location = "data/diabetic_retina_file_list.txt"

def find_binaries(query, label, location=InputList._server_location):
    all_folders = os.listdir(location)
    my_diabetic_folder = []
    for folder in all_folders:
        if re.search(query, folder):
            my_diabetic_folder.append(folder)

    my_diabetic_files = []
    for path in [os.path.join(location, folder) for folder in my_diabetic_folder]:
        for path, subdirs, files in os.walk(path):
            for file in files:
                if re.search(r"raw_1536x2048x204.x2.+.bin$", file):
                    my_diabetic_files.append((os.path.join(path, file), label))
    return my_diabetic_files


def __fill_healthy_input_list():
    file_list = []
    file_list.extend(find_binaries(r"^H([0-9]|[0-9][0-9])", 0))
    if os.path.exists(hdd1 := "/media/julius/My Passport/MOON1e"):
        file_list.extend(
            find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=hdd1))
    if os.path.exists(hdd2 := "/media/julius/My Passport1/MOON1e"):
        file_list.extend(
            find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=hdd2))
    InputList.training_files = file_list


def __fill_diabetic_input_list():
    file_list = []
    file_list.extend(find_binaries(r"^D([0-9]|[0-9][0-9]|[0-9][0-9][0-9])$", 0))
    InputList.training_files = file_list


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


def copy_images_to_data2():
    __fill_diabetic_input_list()
    id = image2d.ImageDataset()
    diabetic_image_list = id.get_training_files()
    for file, label in diabetic_image_list:
        file_name = os.path.basename(file)
        destination = os.path.join("data/diabetic_images/test_files", file_name)
        if os.path.exists(destination):
            raise FileExistsError(f"File {file_name} would be overwritten")
        shutil.copyfile(file, destination)


def copy_images_to_data3(type, image_type="angio"):
    if type == "healthy":
        __fill_healthy_input_list()
    if type == "diabetic":
        __fill_diabetic_input_list()

    id = image2d.ImageDataset(image_type)
    found = id.get_training_files()
    present = get_file_list_from_folder(f"data/{type}_images", -1)
    search_dict = dict.fromkeys([os.path.basename(file) for file, label in present])
    for found_file, label in found:
        try:
            search_dict[os.path.basename(found_file)]
        except KeyError:
            destination = os.path.join(f"data/{type}_images/new_files", os.path.basename(found_file))
            shutil.copyfile(found_file, destination)


def copy_enf_files_to_retinas(type):
    for path, subdirs, files in os.walk(f"data/{type}_images"):
        if 'trash_files' in subdirs:
            subdirs.remove('trash_files')  # don't visit CVS directories
        for name in files:
            print(os.path.join(path, name))

import typing
import pathlib
def get_file_list_from_folder(file_list_folder: typing.Union[str, pathlib.Path], label):
    path_list = []
    for dirpath, subdirs, filenames in os.walk(file_list_folder):
        if 'trash_files' in subdirs:
            subdirs.remove('trash_files')  # don't visit CVS directories
        if 'test_files' in subdirs:
            subdirs.remove('test_files')  # don't visit CVS directories
        for f in filenames:
            path_list.append((os.path.abspath(os.path.join(dirpath, f)), label))
    return path_list


if __name__ == "__main__":
    #create_retina_file_lists()
    #create_raw_file_lists()
    #copy_images_to_data3("diabetic", "angio")
    copy_images_to_data3("healthy")
