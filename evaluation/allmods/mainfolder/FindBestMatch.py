
# coding: utf-8

# In[1]:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
# In[2]:

th = int(sys.argv[1])
v = int(sys.argv[2])
mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train_new/'
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
#img1[masks>20] = 0
#print np.average(masks)
#plt.imshow(img1)


# In[3]:

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

model.load_weights(sys.argv[3], by_name=True)


# In[5]:

def push_pqueue(queue, priority, value):
    if len(queue)>10:
       heapq.heappushpop(queue, (priority, value))
    else:
        heapq.heappush(queue, (priority, value))


mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test_new/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
X_test = []
for filen1 in files1:
    img1 = cv2.imread(mypath1+filen1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<th] = v
    img1[masks>60] = 0
    img1[img1>=th] = 0
    X_test.append(np.array([img1]))
X_test = np.array(X_test).astype('float32')#/ float(np.max(X))
X_test = np.reshape(X_test, (len(X_test),  224, 224, 1))
X_test_pred = model.predict(X_test, verbose=0)


# In[8]:

mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train_new/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
X_train = []
for filen1 in files1:
    img1 = cv2.imread(mypath1+filen1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<th] = v
    img1[masks>60] = 0
    img1[img1>=th] = 0
    X_train.append(np.array([img1]))
X_train = np.array(X_train).astype('float32')#/ float(np.max(X))
X_train = np.reshape(X_train, (len(X_train),  224, 224, 1))
X_train_pred = model.predict(X_train, verbose=0)


# In[9]:

mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test_new/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
top10_correct = 0
top5_correct = 0
top1_correct = 0
for i in np.arange(0, len(files1)):
    filen1 = files1[i]
    pred = X_test_pred[i]
    mypath = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train_new/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    masks = np.zeros((224,224))
    max_confidence = 0.0
    max_file = None
    pqueue = []
    for j in np.arange(0, len(files)):
            filen = files[j]
            tpred = X_train_pred[j]
            score = 1 - spatial.distance.cosine(tpred.sum(axis=2).flatten(), pred.sum(axis=2).flatten())
            push_pqueue(pqueue, score, filen)
            if max_confidence < score:
                max_confidence = score
                max_file = filen
   
    i = 0
    for top20 in heapq.nlargest(len(pqueue), pqueue):
	i += 1
        if top20[1].split('_')[1].split('.')[0] == filen1.split('_')[1].split('.')[0]:
            if i>5:
                top10_correct+=1
            elif i>=1:
                top10_correct+=1
                top5_correct+=1
            elif i>=0:
                top10_correct+=1
                top5_correct+=1
                top1_correct+=1
            break


print "\n!@#$", top10_correct/float(len(files1)), top5_correct/float(len(files1)), top1_correct,"\n"
