
import tensorflow as tf
import IPython.display as display
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


img_path0 = './data/001418.jpg'
img_path1 = '/home/ceo/dev/data/ocr_JP/etlcdb-image-extractor/test_dataset/train/0x4e00/000078.png'

im = cv2.imread(img_path0,cv2.IMREAD_GRAYSCALE)
im_mplt = imread(img_path0,0)
plt.imshow(im.astype("uint8"),cmap="gray")
plt.show()

# This is an example, just using the cat image.
image_string = open(img_path0, 'rb').read()

label = 0

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):

  feature = {
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')

image_path_list = [img_path0,img_path1]

record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for img_path in image_path_list:
    image_string = open(img_path, 'rb').read()

    tf_example = image_example(image_string,0)
    writer.write(tf_example.SerializeToString())


# Create a dictionary describing the features.
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}



raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')



def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))

  print(type(image_raw))
  plt.imshow(image_raw.astype("uint8"))