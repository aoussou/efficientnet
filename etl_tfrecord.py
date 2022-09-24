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

label_dictionary = dict(zip(char_list, list(range(len(char_list)))))

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


# for tfrec_num in range(num_tfrecords):
#     samples = all_files[(tfrec_num * num_samples): ((tfrec_num + 1) * num_samples)]
#
#     with tf.io.TFRecordWriter(
#             tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
#     ) as writer:
#         for image_path in samples:
#             # print(image_path)
#             image = tf.io.decode_jpeg(tf.io.read_file(image_path))
#
#             split_path = os.path.normpath(image_path).split(os.path.sep)
#             char = split_path[-2]
#             file_name = split_path[-1]
#             image_id = int(Path(file_name).with_suffix('').stem)
#
#             char_id = label_dictionary[char]
#             example = create_example(image, char_id, image_id)
#             writer.write(example.SerializeToString())


raw_dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/file_00-{num_samples}.tfrec")

parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != "image":
            print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(features["image"].numpy())
    plt.show()

###############################
def prepare_sample(features):
    image = tf.image.resize(features["image"], size=(224, 224))
    return image, features["char_id"]


def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
batch_size = 32
epochs = 1
steps_per_epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

input_tensor = tf.keras.layers.Input(shape=(224, 224, 1), name="image")
input_conc = tf.keras.layers.Concatenate()([input_tensor, input_tensor, input_tensor])
model = tf.keras.applications.EfficientNetB0(
    input_tensor=input_conc, weights=None, classes=len(label_dictionary)
)


print("compiling model")
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

print("fitting model")
model.fit(
    x=get_dataset(train_filenames, batch_size),
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)
