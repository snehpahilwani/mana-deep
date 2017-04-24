
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers


from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import  cv2
import scipy.misc
from scipy import spatial
from PIL import Image
import heapq
import sys
import cStringIO
import base64
from PIL import Image
import cv2
from StringIO import StringIO
import numpy as np
import json

global model_load_status
model_load_status = False
th = 70
v = 20
model_file = '/home/arvind/MyStuff/Desktop/Manatee_dataset/allmods/new_train/model_iter.h5'

mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
X_test = []
masks = np.zeros((224,224))
for filen1 in files1:
    img1 = cv2.imread(mypath1+filen1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<th] = v
    img1[img1>=th] = 0
    masks = masks + img1
masks = masks / v

if not model_load_status:
    input_img = Input(shape=(224, 224,1))
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', input_shape=(224,224,1))(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x) 
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', activity_regularizer=regularizers.activity_l1(10e-5))(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)
    model = Model(input_img, encoded)
    model.compile(loss='binary_crossentropy', optimizer='adagrad', verbose=0)
    # In[4]:
    model.load_weights(model_file, by_name=True)
    model_load_status = True

def push_pqueue(queue, priority, value):
    if len(queue)>20:
       heapq.heappushpop(queue, (priority, value))
    else:
        heapq.heappush(queue, (priority, value))


mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
X_test = []
for filen1 in files1:
    img1 = cv2.imread(mypath1+filen1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<=th] = v
    img1[masks>60] = 0
    img1[img1>th] = 0
    X_test.append(np.array([img1]))
X_test = np.array(X_test).astype('float32')#/ float(np.max(X))
X_test = np.reshape(X_test, (len(X_test),  224, 224, 1))
X_test_pred = model.predict(X_test, verbose=0)


mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
X_train = []
for filen1 in files1:
    img1 = cv2.imread(mypath1+filen1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<=th] = v
    img1[masks>60] = 0
    img1[img1>th] = 0
    X_train.append(np.array([img1]))
X_train = np.array(X_train).astype('float32')#/ float(np.max(X))
X_train = np.reshape(X_train, (len(X_train),  224, 224, 1))
X_train_pred = model.predict(X_train, verbose=0)



mypath = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print 'model loaded ready to serve'


def predict(img_png_b64):
    pqueue = []
    tempimg = cStringIO.StringIO(img_png_b64.decode('base64'))
    img1 = Image.open(tempimg)
    img1 = np.array(img1)
    img1 = cv2.resize(img1, (224,224))
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<=th] = v
    img1[masks>60] = 0
    img1[img1>th] = 0
    X_train = []
    X_train.append(np.array([img1]))
    X_train = np.array(X_train).astype('float32')#/ float(np.max(X))
    X_train = np.reshape(X_train, (len(X_train),  224, 224, 1))
    pred = model.predict(X_train, verbose=0)[0]
    msk = cv2.resize(img1, (28, 28))
    msk =  np.repeat(msk[:, :, np.newaxis], 8, axis=2)
    msk = msk.flatten()
    pred = pred.flatten()
    pred[msk!=0] = 5
    for j in np.arange(0, len(files)):
            filen = files[j]
            tpred = X_train_pred[j].flatten()
            tpred[msk!=0] = tpred[msk!=0] * 5
            score = 1 - spatial.distance.cosine(tpred, pred)
            push_pqueue(pqueue, score, filen)
    return json.dumps(heapq.nlargest(len(pqueue), pqueue))