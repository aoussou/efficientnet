import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('models/best_model.hdf5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# Save the model.
with open('models/model.tflite', 'wb') as f:
  f.write(tflite_model)
