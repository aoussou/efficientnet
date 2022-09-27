import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('models/best_model.hdf5', compile=False)


class NewModel(model):

    def __int__(self):
        model.__init__(self)

    def predict(self, x):
        return model.predict(tf.expand_dims(x, 0))


newModel = NewModel(model)

converter = tf.lite.TFLiteConverter.from_keras_model(newModel)


tflite_model = converter.convert()

TFLITE_FILE_PATH = 'models/model.tflite'

# Save the model.
# with open('models/model.tflite', 'wb') as f:
#     f.write(tflite_model)
# f.close()

interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
interpreter.allocate_tensors()
STOP

run_model = tf.function(lambda x: model(x))
# to get the concrete function from callable graph
concrete_funct = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
# convert concrete function into TF Lite model using TFLiteConverter

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_funct])
tflite_model = converter.convert()

TFLITE_FILE_PATH = 'models/model.tflite'

with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)
f.close()

STOP

interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_path = "data/screenshot.png"
image = tf.io.decode_jpeg(tf.io.read_file(image_path))
image_slice = image[:, :, 0:-1]
gray_scale_image = tf.image.rgb_to_grayscale(image_slice)
resized_image = tf.image.resize(gray_scale_image, size=(224, 224))
# input_data = tf.expand_dims(resized_image, 0)
input_data = resized_image
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(np.argmax(output_data))
