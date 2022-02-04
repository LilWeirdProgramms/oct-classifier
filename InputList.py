import os

_set_of_healthy_training_files = {
    "H3/raw_1536x2048x2045x2_9201.bin",  # Bir
    "H4/raw_1536x2048x2045x2_12417.bin",  # Jon
    "H5/raw_1536x2048x2045x2_17600.bin"  # Anj, sar, kath
}
_set_of_diabetic_training_files = {
    "D87/rechts/raw_1536x2048x2045x2_30515.bin",
    "D93/rechts besser/raw_1536x2048x2045x2_7592.bin",
    "D108/rechts/raw_1536x2048x2045x2_20962.bin"
}
# Currently all right eyes
_set_of_healthy_testing_files = {
}
_set_of_diabetic_testing_files = {
}

_server_location = "/mnt/server/Projects/UWF OCTA/Clinical data/MOON1/"
_diabetic_location = _server_location
_healthy_location = _server_location

healthy_training_files = [(os.path.join(_healthy_location, healthy_file), 0)
                          for healthy_file in _set_of_healthy_training_files]
diabetic_training_files = [(os.path.join(_diabetic_location, diabetic_file), 1)
                           for diabetic_file in _set_of_diabetic_training_files]
healthy_testing_files = [(os.path.join(_healthy_location, healthy_file), 0)
                         for healthy_file in _set_of_healthy_testing_files]
diabetic_testing_files = [(os.path.join(_diabetic_location, diabetic_file), 1)
                          for diabetic_file in _set_of_diabetic_testing_files]

training_files = healthy_training_files + diabetic_training_files
testing_files = healthy_testing_files + diabetic_testing_files

assert len(_set_of_diabetic_training_files.intersection(_set_of_diabetic_testing_files,
                                                        _set_of_healthy_training_files,
                                                        _set_of_healthy_testing_files)) == 0, \
    "Duplicated Input Files found, Check that each File is only used in one Dataset"
