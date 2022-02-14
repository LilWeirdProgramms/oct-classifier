import os

_set_of_healthy_training_files = {
    "H3/raw_1536x2048x2045x2_9201.bin",  # Bir
    "H3/raw_1536x2048x2045x2_9579.bin",
    "H4/raw_1536x2048x2045x2_12417.bin",  # Jon
    "H4/raw_1536x2048x2045x2_13025.bin",
    "H5/raw_1536x2048x2045x2_17600.bin",  # Anj
    "H5/raw_1536x2048x2045x2_18044.bin",
    "H6/rechts/raw_1536x2048x2045x2_8042.bin",  # sar
    "H6/links/raw_1536x2048x2045x2_8920.bin",
    "H7/rechts/raw_1536x2048x2045x2_9772.bin",  # kath
    "H7/links/raw_1536x2048x2045x2_10419.bin"
}
_set_of_diabetic_training_files = {
    "D78/rechts/raw_1536x2048x2045x2_18038.bin",
    "D78/links/raw_1536x2048x2045x2_18749.bin",
    "D85/rechts/raw_1536x2048x2045x2_19200.bin",
    "D87/rechts/raw_1536x2048x2045x2_30515.bin",
    "D88/links/raw_1536x2048x2045x2_19064.bin",
    "D89/links/raw_1536x2048x2045x2_25535.bin",
    "D93/rechts besser/raw_1536x2048x2045x2_7592.bin",
    "D93/links/raw_1536x2048x2045x2_8245.bin",
    "D94/links/raw_1536x2048x2045x2_2563.bin",
    "D108/rechts/raw_1536x2048x2045x2_20962.bin"
}
# Currently all right eyes
_set_of_healthy_testing_files = {
}
_set_of_diabetic_testing_files = {
    "D71/rechts/raw_1536x2048x2045x2_2591.bin"
}

_server_location = "/mnt/p_Zeiss/Projects/UWF OCTA/Clinical data/MOON1/"
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
