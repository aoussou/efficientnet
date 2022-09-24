import json
import os.path
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from utils import parse_tfrecord_fn, prepare_sample


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
batch_size = 32
epochs = 2
steps_per_epoch = 50
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (tf.data.TFRecordDataset(
    train_filenames, num_parallel_reads=AUTOTUNE)
                 .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
                 .map(prepare_sample, num_parallel_calls=AUTOTUNE)
                 .shuffle(batch_size * 10)
                 .batch(batch_size)
                 .prefetch(AUTOTUNE)
                 )

val_filenames = tf.io.gfile.glob(f"{val_tf_record_dir}/*.tfrec")
val_dataset = (tf.data.TFRecordDataset(
    val_filenames)
               .map(parse_tfrecord_fn)
               .map(prepare_sample)
               .batch(batch_size)
               .prefetch(AUTOTUNE)
               )

filepath = 'best_model.hdf5'


checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

callbacks = [checkpoint]

label_dictionary = json.load(open("label_dictionary.json"))

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
    x=train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    validation_data=val_dataset,
    callbacks=callbacks
)

print("done")
