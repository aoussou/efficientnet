import glob

import tensorflow as tf
from pathlib import Path
import os


def get_all_image_paths(dir_path, ext="jpg"):
    home_dir = Path.home()
    home_path = os.fspath(home_dir)

    full_dir_path = os.path.join(home_path, dir_path)
    image_path_pattern = os.path.join(full_dir_path, '*', '*.' + ext)
    all_files = glob.glob(image_path_pattern)

    return all_files


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


def create_example(image, char_id, image_id):
    feature = {
        "image": image_feature(image),
        "char_id": int64_feature(char_id),
        "image_id": int64_feature(image_id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "char_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=1)

    return example


def prepare_sample(features):
    image = tf.image.resize(features["image"], size=(224, 224))
    return image, features["char_id"]
