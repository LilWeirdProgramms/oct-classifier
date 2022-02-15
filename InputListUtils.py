import InputList
import re
import os


def generate_diabetic_files_list():
    all_folders = os.listdir(InputList._server_location)
    my_diabetic_folder = []
    for folder in all_folders:
        if re.search(r"^D([0-9][0-9][0-9]|[4-9][0-9])", folder):
            my_diabetic_folder.append(folder)

    my_diabetic_files = []
    for path in [os.path.join(InputList._server_location, folder) for folder in my_diabetic_folder]:
        for path, subdirs, files in os.walk(path):
            for file in files:
                if re.search(r"raw_1536x2048x2045x2.+.bin$", file):
                    my_diabetic_files.append((os.path.join(path, file), 0))


    InputList.diabetic_training_files = my_diabetic_files[:-10]
    InputList.diabetic_testing_files = my_diabetic_files[-10:]

#print(os.listdir(InputList._server_location))
