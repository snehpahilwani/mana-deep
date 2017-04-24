from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping ,LearningRateScheduler
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

from keras.optimizers import Adam

mypath = '/home/arvind/arvind/cleaned_data/train/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = []
for filen in files:
        img = cv2.imread(mypath+filen)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.invert(img)
	img = img/np.float32(np.max(img))
	img[img>0.50] = 1
	img[img!=1] = 0
	img = cv2.resize(img, (224,224))
        images.append(np.array([img]))
	images.append(np.array([np.fliplr(img)]))
        images.append(np.array([np.flipud(img)]))
        images.append(np.array([np.fliplr(np.flipud(img))]))
	
print 'Training with ', len(images), ' samples'
autoencoder = None
with tf.device('/gpu:0'):
	input_img = Input(shape=(224, 224, 1))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(224,224, 1))(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)

	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3,activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1 , 3, 3, activation='hard_sigmoid', border_mode='same')(x)

	autoencoder = Model(input_img, decoded)

	autoencoder.compile(optimizer="adamax", loss='mse')


images_train = np.array(images[:-400])
images_test = np.array(images[-400:])

images_train = images_train.astype('float32')#/255. # / float(np.max(images_train))
images_test = images_test.astype('float32') #/255.# / float(np.max(images_test))



#print np.max(images_train_op[0])
#plt.imshow(np.reshape(images_train_op[50],(224,224)))
#plt.show()
# images_train_op = np.array(images_train_op)
# images_test_op = np.array(images_test_op)

images_train = np.reshape(images_train, (len(images_train),224, 224, 1))
images_test = np.reshape(images_test, (len(images_test), 224, 224, 1))

#images_train_op = np.reshape(images_train_op, (len(images_train_op),  224, 224, 1))

#images_test_op = np.reshape(images_test_op, (len(images_test_op), 224, 224, 1))

#print images_test_op.shape
#print images_train_op.shape

def step_decay(epoch):
	initial_lrate = 0.1
	lrate = 0.1 * 0.999 * epoch
	if lrate < 0.0001:
		return 0.0001
	return lrate 

def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


print autoencoder.summary()

autoencoder.load_weights("model_best_4.h5")
saver = ModelCheckpoint("model_iter.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
stopper = EarlyStopping(monitor='loss', patience=50, verbose=1, mode='auto')

lrdecay = LearningRateScheduler(step_decay)

autoencoder.fit(images_train, images_train,
                nb_epoch=10000,
                batch_size=256,
                shuffle=True,
                validation_data=(images_test,images_test), verbose=1, callbacks=[saver, stopper])
# serialize model to JSON
model_json = autoencoder.to_json()
with open("model4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model_best_4.h5")
print("Saved model to disk")
