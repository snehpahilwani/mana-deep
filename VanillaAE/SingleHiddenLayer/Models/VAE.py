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

mypath = 'C:/Python3_5_2/ManateeData/cleaned_data/train/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = []
for filen in files:
    img = cv2.imread(mypath+filen,0)
    img = np.invert(img)
    img = img/np.float32(np.max(img))
    img[img>0.50] = 1
    img[img!=1] = 0
    img = cv2.resize(img, (224,224))
    images.append(np.array([img]))
    images.append(np.array([np.fliplr(img)]))
    images.append(np.array([np.flipud(img)]))
    images.append(np.array([np.fliplr(np.flipud(img))]))

print ('Training with ', len(images), ' samples')

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 1638.78125, assuming the input is 50176 floats

# this is our input placeholder
input_img = Input(shape=(50176,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.activity_l1(10e-1))(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(50176, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]


# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adamax', loss='binary_crossentropy')

images_train = np.array(images[:-400])
images_test = np.array(images[-400:])

images_train = images_train.astype('float32')
images_test = images_test.astype('float32')

images_train = images_train.reshape((len(images_train), np.prod(images_train.shape[1:])))
images_test = images_test.reshape((len(images_test), np.prod(images_test.shape[1:])))

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

autoencoder.load_weights("model_iter.h5")
	
saver = ModelCheckpoint("model_iter.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
stopper = EarlyStopping(monitor='loss', patience=50, verbose=1, mode='auto')

lrdecay = LearningRateScheduler(step_decay)

autoencoder.fit(images_train, images_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(images_test,images_test), verbose=1, callbacks=[saver, stopper])

# serialize model to JSON
model_json = autoencoder.to_json()
with open("model4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model_best.h5")
print("Saved model to disk")

encoded_imgs = encoder.predict(images_test)
decoded_imgs = decoder.predict(encoded_imgs)
n=2
for i in range(n):
    # display original
    plt.imshow(images_test[i].reshape(224, 224))
    plt.gray()
    plt.savefig('Origmanatee'+str(i)+'.png')

    # display reconstruction
    plt.imshow(decoded_imgs[i].reshape(224, 224))
    plt.gray()
    plt.savefig('manatee'+str(i)+'.png')
plt.show()
