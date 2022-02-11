import os

num_volumes = 10
num_samples_per_volume = 73 * 89
num_samples = num_samples_per_volume * num_volumes
num_samples_per_tfrecords = 50
num_tfrecords_files = num_samples // num_samples_per_tfrecords
tfrecords_dir = "/mnt/NewHDD/tfrecords"

if num_samples % num_samples_per_tfrecords:
    num_tfrecords_files += 1

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(image, path, example):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "area": float_feature(example["area"]),
        "bbox": float_feature_list(example["bbox"]),
        "category_id": int64_feature(example["category_id"]),
        "id": int64_feature(example["id"]),
        "image_id": int64_feature(example["image_id"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
