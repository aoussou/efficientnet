
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
model = load_model('models/best_model_elt_only_87.hdf5', compile=False)

image_path = "data/000617.jpg"
image = tf.io.decode_jpeg(tf.io.read_file(image_path))


resized_image = tf.image.resize(image, size=(224, 224))


resized_image = tf.expand_dims(resized_image,0)
pred = model.predict(resized_image)
argmax = np.argmax(pred)