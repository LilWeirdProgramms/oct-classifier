import tensorflow as tf
import BinaryReader
import numpy as np


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
  with tf.io.TFRecordWriter(filename,
                            #options=options
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

  ascan = content['ascan']
  bscan = content['bscan']
  cscan = content['cscan']
  channels = content['channels']
  label = content['label']
  raw_image = content['raw_image']

  # get our 'feature'-- our image -- and reshape it appropriately
  feature = tf.io.parse_tensor(raw_image, out_type=tf.uint16)
  feature = tf.reshape(feature, shape=[ascan, bscan, cscan, channels])
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

def tfrecords_real_writer():
    br = BinaryReader.BinaryReader()
    for file in file_list:
        i = 0
        dirname = os.path.join(data_location, f"tfrecords/{file[1]}_{os.path.basename(file[0][-9:-4])}")
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for instance, label, b_position, c_position in br.instance_from_binaries_generator([file]):
            filename = os.path.join(dirname, f"b{b_position}_c{c_position}")
            write_images_to_tfr_short(instance, int(label), filename=filename)

def to_file():
    data_location = "/mnt/NewHDD"
    file_list = []
    with open('healthy_training_files.txt', 'r') as f:
        for item in f.read().splitlines():
            file_list.append((item, 0))
    with open('diabetic_training_files.txt', 'r') as f:
        for item in f.read().splitlines():
            file_list.append((item, 1))
    tfrecords_real_writer()

