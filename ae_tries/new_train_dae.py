from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers

import tensorflow as tf

tf.python.control_flow_ops = tf

import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import  cv2
import scipy.misc
import random

random.seed(9)


input_img = Input(shape=(224, 224,1))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(224,224,1))(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', activity_regularizer=regularizers.activity_l1(10e-5))(x)
encoded = MaxPooling2D((1, 1), border_mode='same')(x)

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((1, 1))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

mypath = '/home/arvind/tt/cleaned_data/train/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = []
mask = np.zeros((224,224))
for filen in files:
	img = cv2.imread(mypath+filen)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img[img<30] = 101
        img[img<100] = 1
        img[img>=100] = 0
        mask = mask + img
	img[img>0] = 1
	images.append(np.array([img]))
images_train = np.array(images[:-100])
images_test = np.array(images[-100:])

images_train = images_train.astype('float32')#/ 255. #float(np.max(images_train))
images_test = images_test.astype('float32')#/ 255. #float(np.max(images_test))

images_train_masked = []
images_test_masked = []

av = np.average(mask)

for img in images_test:
	_img = np.reshape(np.copy(img),(224,224))
	_img[mask>av] = 0
	images_test_masked.append(np.array([_img]))


for img in images_train:
	_img = np.reshape(np.copy(img),(224,224))
	_img[mask>av] = 0
	images_train_masked.append(np.array([_img]))
#print np.max(images_train_op[0])
#plt.imshow(np.reshape(images_train_op[50],(224,224)))
#plt.show()
# images_train_op = np.array(images_train_op)
# images_test_op = np.array(images_test_op)

images_train = np.reshape(images_train, (len(images_train),  224, 224, 1))
images_test = np.reshape(images_test, (len(images_test), 224, 224, 1))

images_train_masked = np.reshape(images_train_masked, (len(images_train_masked),  224, 224, 1))
images_test_masked = np.reshape(images_test_masked, (len(images_test_masked), 224, 224, 1))

noise_factor = 0.01
x_train_noisy = images_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images_train.shape)
x_test_noisy = images_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)



#print images_test_op.shape
#print images_train_op.shape

print autoencoder.summary()
filepath="best_model_dae.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#autoencoder.load_weights('model_right1.h5')


history = autoencoder.fit(images_train, images_train_masked,
                nb_epoch=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(images_test, images_test_masked),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), checkpoint])


# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model_best_dae.h5")
print("Saved model to disk")
