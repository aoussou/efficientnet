import json
import os.path
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from utils import parse_tfrecord_fn, prepare_sample, get_all_image_paths

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print(tf.__version__)
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(
            filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset


train_tf_record_dir = os.path.join("tfrecords", "train")
val_tf_record_dir = os.path.join("tfrecords", "val")

train_filenames = tf.io.gfile.glob(f"{train_tf_record_dir}/*.tfrec")
batch_size = 128
epochs = 40

val_filenames = tf.io.gfile.glob(f"{val_tf_record_dir}/*.tfrec")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = (tf.data.TFRecordDataset(train_filenames,num_parallel_reads=AUTOTUNE)
                 .map(parse_tfrecord_fn,num_parallel_calls=AUTOTUNE)
                 .map(prepare_sample, num_parallel_calls=AUTOTUNE)
                 .batch(batch_size)
                 .shuffle(batch_size*5)
                 .prefetch(AUTOTUNE)
                 )

val_dataset = (tf.data.TFRecordDataset(
    val_filenames)
               .map(parse_tfrecord_fn)
               .map(prepare_sample)
               .batch(batch_size)
               )


# AUTOTUNE = tf.data.AUTOTUNE
#
# train_dataset = (tf.data.TFRecordDataset(
#     train_filenames, num_parallel_reads=AUTOTUNE)
#                  .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
#                  .map(prepare_sample, num_parallel_calls=AUTOTUNE)
#                  .shuffle(batch_size)
#                  .batch(batch_size)
#                  .prefetch(AUTOTUNE)
#                  )
#
# val_filenames = tf.io.gfile.glob(f"{val_tf_record_dir}/*.tfrec")
# val_dataset = (tf.data.TFRecordDataset(
#     val_filenames)
#                .map(parse_tfrecord_fn)
#                .map(prepare_sample)
#                .batch(batch_size)
#                .prefetch(AUTOTUNE)
#                )


filepath = 'best_model.hdf5'

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

callbacks = [checkpoint]

label_dictionary = json.load(open("label_dictionary.json"))

img_augmentation = Sequential(
    [
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

input_tensor = tf.keras.layers.Input(shape=(224, 224, 1), name="image")
concatenated_input = tf.keras.layers.Concatenate()([input_tensor, input_tensor, input_tensor])
x = img_augmentation(concatenated_input)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.applications.EfficientNetB0(
        input_tensor=x, weights=None, classes=len(label_dictionary)
    )

    print("compiling model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    print("fitting model")
    model.fit(
        x=train_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_dataset,
        # validation_steps=val_steps_per_epoch
    )

    print("done")
