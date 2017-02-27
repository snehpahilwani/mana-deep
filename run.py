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



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adagrad')
img = plt.imread('/home/arvind/Desktop/Manatee_dataset/cleaned_data/test/op_U1150_B.tif.tif')

X = np.array([img])
X = X.astype('float32') / float(np.max(X))
X = np.reshape(X, (len(X),  224, 224, 1))
X_ = np.array([np.invert(img)])
X_ = X_.astype('float32') / float(np.max(X_))
X_ = np.reshape(X_, (len(X_), 224, 224, 1))
score = loaded_model.evaluate(X_, X_, verbose=0)



input_img = Input(shape=(224, 224,1))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(224,224,1), weights=loaded_model.layers[1].get_weights())(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', weights=loaded_model.layers[3].get_weights())(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', weights=loaded_model.layers[5].get_weights())(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)
model = Model(input_img, encoded)
#model.compile(loss='binary_crossentropy', optimizer='adagrad')
print score
print model.predict(X_, verbose=1).shape
