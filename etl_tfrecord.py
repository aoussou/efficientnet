"""
Title: Creating TFRecords
Author: [Dimitre Oliveira](https://www.linkedin.com/in/dimitre-oliveira-7a1a0113a/)
Date created: 2021/02/27
Last modified: 2021/02/27
Description: Converting data to the TFRecord format.
"""

import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

home_dir = Path.home()
home_path = os.fspath(home_dir)
images_dir = os.path.join(home_path, "dev/data/ocr_JP/etlcdb-image-extractor/etl_data/images/ETL9G")

tfrecords_dir = "tfrecords"
path_pattern = os.path.join(images_dir, '*', '*.jpg')
all_files = glob.glob(path_pattern)

char_list = os.listdir(images_dir)

num_images = len(all_files)
num_samples = 4096
num_tfrecords = num_images // num_samples
if num_images % num_samples:
    num_tfrecords += 1  # add one record if there are any remaining samples

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder

"""
## Define TFRecords helper functions
"""


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, example):
    feature = {
        "image": image_feature(image),
        "category_id": int64_feature(example["category_id"]),
        "id": int64_feature(example["id"]),
        "image_id": int64_feature(example["image_id"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example


"""
## Generate data in the TFRecord format
Let's generate the COCO2017 data in the TFRecord format. The format will be
`file_{number}.tfrec` (this is optional, but including the number sequences in the file
names can make counting easier).
"""

for tfrec_num in range(num_tfrecords):
    samples = annotations[(tfrec_num * num_samples): ((tfrec_num + 1) * num_samples)]

    with tf.io.TFRecordWriter(
            tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
    ) as writer:
        for sample in samples:
            image_path = f"{images_dir}/{sample['image_id']:012d}.jpg"
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            example = create_example(image, image_path, sample)
            writer.write(example.SerializeToString())
