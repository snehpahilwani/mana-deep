from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras import regularizers
from keras import regularizers
from keras.callbacks import TensorBoard
from PIL import Image
import os
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 1638.78125, assuming the input is 50176 floats

# this is our input placeholder
input_img = Input(shape=(50176,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.activity_l1(10e-5))(input_img)

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

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#Load images

# Append all the absolute image paths in a list image_paths  
image_paths_train = "..\Input\train" 
image_paths_test = "..\Input\test"
# images list will contain face image data. i.e. pixel intensities   
images_train = []  
images_test = [] 

print("Train Preprocess") 
for image_path in os.listdir(image_paths_train):  
     # Read the image and convert to grayscale  
     image_pil = Image.open(image_paths_train+"\\" +image_path)
     # Convert the image format into numpy array  
     image = np.array(np.invert(image_pil))
     images_train.append(image)  
     
print("Test Preprocess")
for image_path in os.listdir(image_paths_test):  
     # Read the image and convert to grayscale  
     image_pil = Image.open(image_paths_test+"\\"+image_path)
     # Convert the image format into numpy array  
     image = np.array(np.invert(image_pil))
     images_test.append(image)  

     
trainData = np.asarray(images_train) 
testData = np.asarray(images_test)
x_train = trainData.astype('float32') /255.0
x_test = testData.astype('float32')/255.0
print (x_train)
print (x_test)

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
				
# encode and decode some manatees
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# For visualizing reconstructed image
import matplotlib.pyplot as plt
n = 5  # how many manatees we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(224, 224))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(224, 224))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




