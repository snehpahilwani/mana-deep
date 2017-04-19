from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import regularizers
import tensorflow as tf
tf.python.control_flow_ops = tf

from PIL import Image


import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import  cv2
import scipy.misc

input_img = Input(shape=(224, 224,1))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(224,224,1))(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', activity_regularizer=regularizers.activity_l1(10e-5))(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adagrad', loss='mse')

mypath = 'cleaned_data/train/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = []
masks = np.zeros((224,224))
for filen in files:
#	img = cv2.imread(mypath+filen)
	img = Image.open(mypath+filen)
	img = np.asarray(img, dtype=np.uint8)
	img = np.invert(img)
	masks += img
	images.append(np.array([img]))
images_train = np.array(images[:-100])
images_test = np.array(images[-100:])
images_train = images_train.astype('float32')#/ float(np.max(images_train))
images_test = images_test.astype('float32') #/ float(np.max(images_test))
images_train_denoised = []
images_test_denoised = []

for img in images_test:
	_img = np.reshape(np.copy(img),(224,224))
	_img[masks>1000] = 0.
	_img[_img>0.0] = 50.
	images_test_denoised.append(_img)


for img in images_train:
	_img = np.reshape(np.copy(img),(224,224))
	_img[masks>1000] = 0.
        _img[_img>0.0] = 50.
	images_train_denoised.append(_img)
#print np.max(images_train_op[0])
#plt.imshow(np.reshape(images_train_op[50],(224,224)))
#plt.show()

images_train_denoised = np.array(images_train_denoised)
images_test_denoised = np.array(images_test_denoised)

images_train = np.reshape(images_train, (len(images_train),  224, 224, 1))
images_test = np.reshape(images_test, (len(images_test), 224, 224, 1))

images_train_denoised = np.reshape(images_train_denoised, (len(images_train_denoised),  224, 224, 1))
images_test_denoised = np.reshape(images_test_denoised, (len(images_test_denoised), 224, 224, 1))

#print images_test_op.shape
#print images_train_op.shape

print autoencoder.summary()
autoencoder.load_weights('model1.h5')
autoencoder.fit(images_train, images_train_denoised,
                nb_epoch=200,
                batch_size=128,
                shuffle=True,
                validation_data=(images_test, images_test_denoised),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])



# serialize model to JSON
model_json = autoencoder.to_json()
with open("modelold.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model12.h5")
print("Saved model to disk")
