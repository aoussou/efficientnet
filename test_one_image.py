import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('models/best_model.hdf5', compile=False)

image_path = "data/screenshot.png"

image = tf.io.decode_jpeg(tf.io.read_file(image_path))
image_slice = image[:, :, 0:-1]
gray_scale_image = tf.image.rgb_to_grayscale(image_slice)
resized_image = tf.image.resize(gray_scale_image, size=(224, 224))

resized_image = tf.expand_dims(resized_image, 0)
pred = model.predict(resized_image)
argmax = np.argmax(pred)
