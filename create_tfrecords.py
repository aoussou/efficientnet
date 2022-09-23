
import tensorflow as tf
import IPython.display as display
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread
img_path = './data/001418.jpg'

# cat_in_snow  = tf.keras.utils.get_file(
#     '/home/ceo/dev/data/ocr_JP/etlcdb-image-extractor/test_dataset/train/0x4e00/000078.png'
# )

# cat_in_snow: object  = tf.keras.utils.get_file(
#     '320px-Felis_catus-cat_on_snow.jpg',
#     'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')

# for thing in cat_in_snow:
#     print(thing)

im = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
im_mplt = imread(img_path,0)
# cv2.imshow("piu",im)
# display.display(display.Image(filename=img_path))
plt.imshow(im.astype("uint8"),cmap="gray")
plt.show()
# plt.imshow(im.astype("uint8"))
# plt.show()

# print(cat_in_snow)