from PreprocessMultiChannelMILCombination import PreprocessMultiChannelMILCombination
from PreprocessMultiChannelMILImageData import PreprocessMultiChannelMILImageData
from PreprocessMILImageData import PreprocessMILImageData
from PreprocessImageData import PreprocessImageData
from PreprocessData import PreprocessData
from PreprocessRawData import PreprocessRawData
from RawModelBuilder import RawModel


def get_data_class(data_type="images", strategy="mil"):
    if data_type == "images" or data_type == "structure":
        return PreprocessMILImageData
    if data_type == "combined" and strategy == "mil":
        return PreprocessMultiChannelMILImageData
    if data_type == "weighted" and strategy == "mil":
        return PreprocessMultiChannelMILCombination
    if data_type == "imsupervised" or data_type == "strucsupervised":
        return PreprocessImageData
    if data_type == "images" and strategy == "supervised":
        return PreprocessImageData
    if data_type == "images" and strategy == "supervised":
        return PreprocessImageData
