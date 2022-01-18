set_of_testing_files = {
    "D87/rechts/raw_1536x2048x2045x2_30515.bin"
}
set_of_diabetic_files = {
    "D87/rechts/raw_1536x2048x2045x2_30515.bin"
}
set_of_healthy_files = {
    "D87/rechts/raw_1536x2048x2045x2_30515.bin"
}
server_location = "/mnt/server/Projects/UWF OCTA/Clinical data/MOON1/"
diabetic_location = ""
healthy_location = ""

assert len(set_of_diabetic_files.intersection(set_of_testing_files, set_of_healthy_files)) == 0, \
    "Duplicated Input Files found, Check that each File is only used in one Dataset"
