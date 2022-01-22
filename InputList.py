import os

set_of_healthy_training_files = {
    "a"
}
set_of_diabetic_training_files = {
    "D87/rechts/raw_1536x2048x2045x2_30515.bin"
}
set_of_healthy_testing_files = {
    "b"
}
set_of_diabetic_testing_files = {
    "c"
}

server_location = "/mnt/server/Projects/UWF OCTA/Clinical data/MOON1/"
diabetic_location = server_location
healthy_location = server_location

healthy_training_files = [os.path.join(healthy_location, healthy_file) for healthy_file in set_of_healthy_training_files]
diabetic_training_files = [os.path.join(diabetic_location, diabetic_file) for diabetic_file in set_of_diabetic_training_files]
healthy_testing_files = [os.path.join(healthy_location, healthy_file) for healthy_file in set_of_healthy_testing_files]
diabetic_testing_files = [os.path.join(diabetic_location, diabetic_file) for diabetic_file in set_of_diabetic_testing_files]

assert len(set_of_diabetic_training_files.intersection(set_of_diabetic_testing_files, set_of_healthy_training_files,
                                                       set_of_healthy_testing_files)) == 0, \
    "Duplicated Input Files found, Check that each File is only used in one Dataset"
