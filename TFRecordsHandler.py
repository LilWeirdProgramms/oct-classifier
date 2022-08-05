import tensorflow as tf
import BinaryReader
import numpy as np
import logging

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_image(image, label):
  # define the dictionary -- the structure -- of our single example
  data = {
    'ascan': _int64_feature(image.shape[0]),
    'bscan': _int64_feature(image.shape[1]),
    'cscan': _int64_feature(image.shape[2]),
    'channels': _int64_feature(image.shape[3]),
    'raw_image': _bytes_feature(serialize_array(image)),
    'label': _int64_feature(label)
  }
  # create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))
  return out

def write_images_to_tfr_short(image, label, filename:str="images"):  # TODO: Write batch size elements
  filename= filename + ".tfrecords"
  #options = tf.io.TFRecordOptions(compression_type="GZIP")
  with tf.io.TFRecordWriter(filename, #options=options
                            ) as writer:
    out = parse_single_image(image=image, label=label)
    writer.write(out.SerializeToString())

def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
    'ascan': tf.io.FixedLenFeature([], tf.int64),
    'bscan': tf.io.FixedLenFeature([], tf.int64),
    'cscan': tf.io.FixedLenFeature([], tf.int64),
    'channels': tf.io.FixedLenFeature([], tf.int64),
    'raw_image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)
    logging.info(content)
    ascan = content['ascan']
    bscan = content['bscan']
    cscan = content['cscan']
    channels = content['channels']
    label = content['label']
    raw_image = content['raw_image']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.float32)
    feature = tf.reshape(feature, shape=[int(ascan), int(bscan), int(cscan), int(channels)])
    feature = tf.cast(feature, "float16")
    label = tf.reshape(label, shape=[1])
    return (feature, label)


def get_dataset_small(filenames):
  # create the dataset
  dataset = tf.data.TFRecordDataset(filenames)

  # pass every single feature through our mapping function
  dataset = dataset.map(
    parse_tfr_element
  )
  return dataset

import BinaryReader
import InputList
import os

def lala():
  br = BinaryReader.BinaryReader()
  bag = br.create_test_dataset([InputList.diabetic_testing_files[0]])
  for instance in bag.take(1):
      write_images_to_tfr_short(*instance)


data_location = "/mnt/NewHDD"


def tfrecords_real_writer(file_list):
    br = BinaryReader.BinaryReader()
    for file in file_list:
        i = 0
        folder = extract_D_or_H_number(file)
        dirname = os.path.join(data_location, f"tfrecords_norm/{file[1]}_{os.path.basename(file[0][-9:-4])}_{folder}")
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for instance, label, b_position, c_position in br.instance_from_binaries_generator([file]):
            filename = os.path.join(dirname, f"b{b_position}_c{c_position}")
            write_images_to_tfr_short(instance, int(label), filename=filename)

def tfrecords_normalize_writer(file_list):
    br = BinaryReader.BinaryReader()
    for file in file_list:
        i = 0
        folder = extract_D_or_H_number(file)
        dirname = os.path.join(data_location, f"tfrecords_norm/{file[1]}_{os.path.basename(file[0][-9:-4])}_{folder}")
        background_mean, background_var = get_background(file)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for instance, label, b_position, c_position in br.instance_from_binaries_generator([file]):
            normalized_instance = (instance.reshape((1536, 102*102)) - background_mean.reshape((1536, 1))) / \
                       background_var.reshape((1536, 1))**0.5
            normalized_instance_reshape = normalized_instance.reshape((1536, 102, 102, 1)).astype("float32")
            filename = os.path.join(dirname, f"b{b_position}_c{c_position}")
            write_images_to_tfr_short(normalized_instance_reshape, int(label), filename=filename)

def get_background(file):
    dirname = os.path.join(data_location, f"tfrecords/{file[1]}_{os.path.basename(file[0][-9:-4])}")
    file = os.path.join(dirname, f"b0_c0.tfrecords")
    dataset = get_dataset_small([file])
    for elem in dataset:
        mean_scan = np.mean(elem[0].numpy().reshape((1536, 102 * 102)), axis=1)
        var_scan = np.var(elem[0].numpy().reshape((1536, 102 * 102)), axis=1)
    return mean_scan, var_scan


def is_boundary_instance(file_name, instance_label, bag_size):
    position_string = file_name.split(".")[0]
    b_pos_str, c_pos_str = position_string.split("_")
    b_pos = int(b_pos_str[1:])
    c_pos = int(c_pos_str[1:])
    instance_label *= 2
    if b_pos == 0 and c_pos == 0:
        is_corner = True
    elif b_pos == bag_size[0] - 1 and c_pos == bag_size[1] - 1:
        is_corner = True
        instance_label += 1
    else:
        is_corner = False
    return is_corner, instance_label


def get_boundary_files(self, data_location):
    data_location = "/mnt/NewHDD/tfrecords"
    boundary_files = []
    for instance in os.listdir(data_location):
        label = int(instance[0])
        instance = os.path.join(data_location, instance)
        for file in os.listdir(instance):
            is_corner_instance, corner_label = is_boundary_instance(file, label, 20)
            if is_corner_instance:
                boundary_files.append(os.path.join(instance, file))
    return boundary_files

def extract_D_or_H_number(file):
    full_path = file[0].split("/")
    for i, value in enumerate(full_path):
        if value == "MOON1":
            D_or_H_index = i + 1
    return full_path[D_or_H_index]


def to_file():
    file_list = []
    with open('healthy_training_files.txt', 'r') as f:
        for item in f.read().splitlines():
            file_list.append((item, 0))
    with open('diabetic_training_files.txt', 'r') as f:
        for item in f.read().splitlines():
            file_list.append((item, 1))
    tfrecords_normalize_writer(file_list)

if __name__ == '__main__':
    to_file()