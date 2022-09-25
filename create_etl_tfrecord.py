import tensorflow as tf
import json
import glob
import os
from pathlib import Path
from utils import create_example, parse_tfrecord_fn, get_all_image_paths
import matplotlib.pyplot as plt
from tqdm import tqdm

def write_tfrecords(file_list, save_dir, label_dictionary, num_samples):
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


# CHECK RECORDS SUCCESSFULLY CREATED

# home_dir = Path.home()
# home_path = os.fspath(home_dir)
#
# train_images_dir = os.path.join(home_path, )
# train_path_pattern = os.path.join(train_images_dir, '*', '*.jpg')
# train_files = glob.glob(train_path_pattern)
#
# val_images_dir = os.path.join(home_path, )
# val_path_pattern = os.path.join(val_images_dir, '*', '*.jpg')
# val_files = glob.glob(val_path_pattern)

train_dir = "dvpt/etlcdb-image-extractor/etl_data/images/train"
train_images_dir = get_all_image_paths(train_dir)

val_dir = "dvpt/etlcdb-image-extractor/etl_data/images/val"
val_images_dir = get_all_image_paths(val_dir)

char_list = os.listdir(train_images_dir)

label_dictionary = dict(zip(char_list, list(range(len(char_list)))))

# SAVE A DICTIONARY
dict_ = dict()
with open(os.path.join('label_dictionary.json'), 'w') as fp:
    json.dump(label_dictionary, fp)
fp.close()

train_tf_record_dir = os.path.join('tfrecords', 'train')
val_tf_record_dir = os.path.join('tfrecords', 'val')
num_samples = 1024

write_tfrecords(train_files, train_tf_record_dir, label_dictionary, num_samples)
write_tfrecords(val_files, val_tf_record_dir, label_dictionary, num_samples)


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
