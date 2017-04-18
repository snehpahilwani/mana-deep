from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import load_model

import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import  cv2
import scipy.misc
from scipy import spatial
from PIL import Image

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("modelnew1.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adagrad')
img = plt.imread('/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test/op_U3850_B.jpg.tif')
img = cv2.imread('/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test/op_U3850_B.jpg.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = img/255.
img[img<100] = 20
img[img>=100] = 0
plt.imshow(img)
plt.show()
#img = np.invert(img.astype('int32'))
X = np.array([img])
X = X.astype('float32')# / np.max(X)
X = np.reshape(X, (len(X),  224, 224, 1))
#X_ = np.array([np.invert(img)])
#X_ = X.astype('float32')#/ float(np.max(X_))
#plt.imshow(img)
#X_ = np.reshape(X_, (len(X_), 224, 224, 1))
#score = loaded_model.evaluate(X_, X_, verbose=0)
pred = loaded_model.predict(X, verbose=0)[0]
plt.imshow(pred.reshape((224,224)))
#pred = pred * 255
#plt.imshow(img)
plt.show()
