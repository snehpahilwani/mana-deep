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


# this is our input placeholder
input_img = Input(shape=(50176,))

# "encoded" is the encoded representation of the input
#encoded = Dense(784, activation = 'relu')(input_img)
encoded = Dense(128, activation='relu',activity_regularizer=regularizers.activity_l1(10e-4))(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
#decoded = Dense(784, activation='relu')(decoded)
decoded = Dense(50176, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(32,))

# retrieve the last layer of the autoencoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

#compile
autoencoder.compile(optimizer='sgd', loss='binary_crossentropy')

#Load images
# Append all the absolute image paths in a list image_paths  
image_paths_train = "C:\\Python3_5_2\\ManateeData\\cleaned_data\\train" 
image_paths_test = "C:\\Python3_5_2\\ManateeData\\cleaned_data\\test"
# images list will contain face image data. i.e. pixel intensities   
images_train = []  
images_test = [] 

print("train Preprocess") 
for image_path in os.listdir(image_paths_train):  
     # Read the image and convert to grayscale  
     image_pil = Image.open("C:\\Python3_5_2\\ManateeData\\cleaned_data\\train\\" +image_path)
     #image_pil = image_pil.resize((224,224),PIL.Image.NEAREST)
     # Convert the image format into numpy array  
     image = np.array(np.invert(image_pil))
     '''
     one_image = image
    
     plt.axis('off')
     plt.imshow(one_image, cmap=cm.binary)
     plt.show()
     '''
     images_train.append(image)  
     
print("Test Preprocess")
for image_path in os.listdir(image_paths_test):  
     # Read the image and convert to grayscale  
     image_pil = Image.open("C:\\Python3_5_2\\ManateeData\\cleaned_data\\test\\"+image_path)
     #image_pil = image_pil.resize((224,224),PIL.Image.NEAREST)
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

print (autoencoder.summary())

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
n = 15  # how many manatees we will display
#plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    #ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(224, 224))
    plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    plt.savefig('Origmanatee'+str(i)+'.png')


    # display reconstruction
    #ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(224, 224))
    plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    plt.savefig('manatee'+str(i)+'.png')
plt.show()
				

# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")