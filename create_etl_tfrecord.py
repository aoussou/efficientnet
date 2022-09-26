import tensorflow as tf
import json
import glob
import os
from pathlib import Path
from utils import create_example, parse_tfrecord_fn, get_all_image_paths
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def write_tfrecords(file_list, save_dir, label_dictionary, num_samples):
    # print(file_list)

    num_images = len(file_list)
    num_tfrecords = num_images // num_samples
    if num_images % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # creating TFRecords output folder

    for tfrec_num in tqdm(range(num_tfrecords)):

        samples = file_list[(tfrec_num * num_samples): ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
                save_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
        ) as writer:
            for image_path in samples:
                # print(image_path)
                image = tf.io.decode_jpeg(tf.io.read_file(image_path))

                split_path = os.path.normpath(image_path).split(os.path.sep)
                char = split_path[-2]
                file_name = split_path[-1]
                image_id = int(Path(file_name).with_suffix('').stem)

                char_id = label_dictionary[char]
                example = create_example(image, char_id, image_id)
                writer.write(example.SerializeToString())


train_dir_etl = "dvpt/etlcdb-image-extractor/etl_data/images/train"
val_dir_etl = "dvpt/etlcdb-image-extractor/etl_data/images/val"

train_dir_traditional_chinese = "dvpt/handwritting_data_all/traditional_chinese/train"
val_dir_traditional_chinese = "dvpt/handwritting_data_all/traditional_chinese/val"

# val_dir = "dev/data/ocr_JP/etlcdb-image-extractor/etl_data/images/val"

train_images_etl = get_all_image_paths(train_dir_etl)
val_images_etl = get_all_image_paths(val_dir_etl)

train_images_traditional_chinese = get_all_image_paths(train_dir_traditional_chinese,ext='png')
val_images_traditional_chinese = get_all_image_paths(val_dir_traditional_chinese,ext='png')

all_train_images = train_images_etl + train_images_traditional_chinese
all_val_images = val_images_etl + val_images_traditional_chinese

random.shuffle(all_train_images)
# random.shuffle(val_images)

home_dir = Path.home()
home_path = os.fspath(home_dir)
full_train_dir_path_etl = os.path.join(home_path, train_dir_etl)

full_train_dir_path_traditional_chinese = os.path.join(home_path, train_dir_traditional_chinese)

list_japanese_characters = os.listdir(full_train_dir_path_etl)
list_traditional_chinese_characters = os.listdir(full_train_dir_path_traditional_chinese)

all_characters_list = list_japanese_characters + list_traditional_chinese_characters
list_unique_characters = list(set(all_characters_list))

label_dictionary = dict(zip(list_unique_characters, list(range(len(list_unique_characters)))))

# SAVE A DICTIONARY
dict_ = dict()
with open(os.path.join('label_dictionary.json'), 'w') as fp:
    json.dump(label_dictionary, fp)
fp.close()

train_tf_record_dir = os.path.join('tfrecords', 'train')
val_tf_record_dir = os.path.join('tfrecords', 'val')
num_samples = 1024

write_tfrecords(all_train_images, train_tf_record_dir, label_dictionary, num_samples)
write_tfrecords(all_val_images, val_tf_record_dir, label_dictionary, num_samples)

# CHECK RECORDS SUCCESSFULLY CREATED

raw_dataset = tf.data.TFRecordDataset(f"{train_tf_record_dir}/file_00-{num_samples}.tfrec")

parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != "image":
            print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(features["image"].numpy())
    plt.show()
