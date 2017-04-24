
# coding: utf-8

# In[1]:
import os

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
#masks = np.zeros(masks.shape)
#img1[masks>20] = 0
#print np.average(masks)
#plt.imshow(img1)

#masks = cv2.cvtColor(cv2.imread('/home/arvind/Desktop/mask.jpg'), cv2.COLOR_BGR2GRAY)
#masks = cv2.resize(masks, (224,224))
#masks[masks<100] = 71
#masks[masks!=71] = 0

# In[3]:



# In[5]:

def push_pqueue(queue, priority, value):
    if len(queue)>20:
       heapq.heappushpop(queue, (priority, value))
    else:
        heapq.heappush(queue, (priority, value))


mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test_roc/'
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
X_test_pred = np.reshape(X_test, (len(X_test),  224, 224, 1))
#X_test_pred = model.predict(X_test, verbose=0)


# In[8]:

mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train_roc/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
X_train = []
files1 = files1[:100 * int(sys.argv[4])]
for filen1 in files1:
    img1 = cv2.imread(mypath1+filen1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<=th] = v
    img1[masks>60] = 0
    img1[img1>th] = 0
    X_train.append(np.array([img1]))
X_train = np.array(X_train).astype('float32')#/ float(np.max(X))
X_train_pred = np.reshape(X_train, (len(X_train),  224, 224, 1))
#X_train_pred = model.predict(X_train, verbose=0)


# In[9]:
import time

mypath = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/train_roc/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files = files[:100 * int(sys.argv[4])]

start_time = time.time()
mypath1 = '/home/arvind/MyStuff/Desktop/Manatee_dataset/cleaned_data/test_roc/'
files1 = [f for f in listdir(mypath1) if isfile(join(mypath1, f))]
cutoff = float(sys.argv[5])
true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0
for i in np.arange(0, len(files1)):
    filen1 = files1[i]
    pred = X_test_pred[i]
    max_confidence = 0.0
    max_file = None
    pqueue = []
    for j in np.arange(0, len(files)):
            filen = files[j]
            tpred = X_train_pred[j]
            score = 1- spatial.distance.cosine(tpred.flatten(), pred.flatten())
            if score > cutoff:
            	if filen.split('_')[1].split('.')[0] == filen1.split('_')[1].split('.')[0]:
			true_positive+=1
		else:
			false_positive+=1
	    else:
		if filen.split('_')[1].split('.')[0] == filen1.split('_')[1].split('.')[0]:
                        false_negative+=1
                else:
                        true_negative+=1

print "\n!@#$", float(len(files1)) , float(len(files)), true_positive, false_negative, false_positive, true_negative,"\n"
