import InputList
import re
import os

def find_binaries(query, label):
    all_folders = os.listdir(InputList._server_location)
    my_diabetic_folder = []
    for folder in all_folders:
        if re.search(query, folder):
            my_diabetic_folder.append(folder)

    my_diabetic_files = []
    for path in [os.path.join(InputList._server_location, folder) for folder in my_diabetic_folder]:
        for path, subdirs, files in os.walk(path):
            for file in files:
                if re.search(r"raw_1536x2048x204.x2.+.bin$", file):
                    my_diabetic_files.append((os.path.join(path, file), label))
    return my_diabetic_files

def write_to_file():
    found_diabetic_files = find_binaries(r"^D([0-9][0-9][0-9]|[4-9][0-9])", 1)
    found_healthy_files = find_binaries(r"^H([0-9]|[0-9][0-9])", 0)

    with open('diabetic_training_files.txt', 'w') as f:
        for item in found_diabetic_files[240:260]:
            f.write("%s\n" % item[0])
    with open('healthy_training_files.txt', 'w') as f:
        for item in found_healthy_files[2:]:
            f.write("%s\n" % item[0])
