import shutil
from InputListUtils import find_binaries
import os


def choose_zip_name(folder):
    parent_name = os.path.split(folder)[-1]
    if parent_name == "rechts":
        zip_name = "right_eye"
    elif parent_name == "links":
        zip_name = "left_eye"
    else:
        zip_name = "parsing_error"
    return zip_name

def choose_parent_name(folder):
    parent_name = os.path.split(os.path.split(folder)[0])[-1]
    return parent_name


file_list = []
file_list.extend(find_binaries(r"^H([3-9][0-9])", 0))
# if os.path.exists(hdd1 := "/media/julius/My Passport/MOON1e"):
#     file_list.extend(
#         find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=hdd1)
#     )
# if os.path.exists(hdd2 := "/media/julius/My Passport1/MOON1e"):
#     file_list.extend(
#         find_binaries(r"^H([0-9]|[0-9][0-9])", 0, location=hdd2)
#     )

store_folder = "/home/julius/Desktop/send_pics"
if not os.path.exists(store_folder):
    os.mkdir(store_folder)

for file in file_list[:]:
    file = file[0]
    folder = os.path.dirname(file)

    parent_name = choose_parent_name(folder)
    copy_to_folder = os.path.join(store_folder, parent_name)
    if not os.path.exists(copy_to_folder):
        os.mkdir(copy_to_folder)

    zip_name = choose_zip_name(folder)
    zip_folder = os.path.join(copy_to_folder, zip_name)
    if not os.path.exists(zip_folder):
        os.mkdir(zip_folder)

    retina = file.replace("raw", "retina").replace("bin", "png")
    struct = file.replace("raw", "enf").replace("bin", "png")
    choroid = file.replace("raw", "choroid").replace("bin", "png")
    retina_path = os.path.join(folder, retina)
    choroid_path = os.path.join(folder, choroid)
    struct_path = os.path.join(folder, struct)
    if os.path.exists(retina_path):
        shutil.copy(retina_path, os.path.join(zip_folder, "retina_angiography.png"))
    else:
        print(f"{retina_path} does not exist")
    if os.path.exists(struct_path):
        shutil.copy(struct_path, os.path.join(zip_folder, "retina_structure.png"))
    else:
        print(f"{struct_path} does not exist")
    if os.path.exists(choroid_path):
        shutil.copy(choroid_path, os.path.join(zip_folder, "choroid_angiography.png"))
    else:
        print(f"{choroid_path} does not exist")
    shutil.make_archive(zip_folder, 'zip', zip_folder)
