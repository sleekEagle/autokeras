#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:22:03 2020

@author: sleek_eagle
"""

import autokeras as ak
from os import listdir
from os.path import isfile, join
import autokeras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model,Model,model_from_json
from keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau

#path to save spectrograms as np arrays

path='/scratch/lnw8px/emodb_audio/spec/spec/'

emotions=np.array(['W','L','E','A','F','T','N'])
def get_emo_onehot(file_name):
    emotion=file_name[5]
    emo_num=to_onehot(np.where(emotions==emotion)[0][0])
    return emo_num

def get_emo_num(file_name):
    emotion=file_name[5]
    emo_num=np.where(emotions==emotion)[0][0]
    return emo_num

def to_onehot(num):
    num_classes=emotions.shape[0]
    out = np.empty([0,num_classes])
    for x in np.nditer(num):
        onehot = np.zeros(num_classes)
        onehot[int(x)] = 1
        out = np.append(out,[onehot],axis = 0)
    return out


def get_file_list(file_path,ext):
    with open(file_path) as f:
        lines = f.readlines()
        files=[]
        for line in lines:
            file_name=line[1:-2]+"."+ext
            files.append(file_name)
    return files

def get_data_list(files):
    path='/scratch/lnw8px/emodb_audio/spec/spec/'
    data=[]
    labels=[]
    for file in files:
        if(len(file)<11):
            continue
        ar=np.load(path+file)
        data.append(ar)
        label=get_emo_num(file)
        labels.append(label)
    return data,labels

#***************prepare data
#***************************
#***************************
train_files=get_file_list('/scratch/lnw8px/emodb_audio/file_names/train.txt','npy')
vali_files=get_file_list('/scratch/lnw8px/emodb_audio/file_names/vali.txt','npy')
test_files=get_file_list('/scratch/lnw8px/emodb_audio/file_names/test.txt','npy')

train_data,train_labels=get_data_list(train_files)
vali_data,vali_labels=get_data_list(vali_files)
test_data,test_labels=get_data_list(test_files)

train_data=np.array(train_data)
vali_data=np.array(vali_data)
test_data=np.array(test_data)

train_labels=np.array(train_labels)
vali_labels=np.array(vali_labels)
test_labels=np.array(test_labels)
#*******************************************
#*******************************************

#***************use autokeras for auto ML
#***************************
#***************************
clf = ak.ImageClassifier(max_trials=100,seed=9)
clf.fit(x=train_data,y=train_labels,epochs=20)

#export model to tensorflow keras
model=clf.export_model()


#***************prepare data again
#***************************
#***************************
def get_data_list(files):
    path='/scratch/lnw8px/emodb_audio/spec/spec/'
    data=[]
    labels=[]
    for file in files:
        if(len(file)<11):
            continue
        ar=np.load(path+file)
        data.append(ar)
        label=get_emo_onehot(file)
        labels.append(label)
    return data,labels

train_data,train_labels=get_data_list(train_files)
vali_data,vali_labels=get_data_list(vali_files)
test_data,test_labels=get_data_list(test_files)

train_data=np.array(train_data)
vali_data=np.array(vali_data)
test_data=np.array(test_data)

train_labels=np.array(train_labels)
vali_labels=np.array(vali_labels)
test_labels=np.array(test_labels)

train_data=np.expand_dims(train_data,axis=-1)
vali_data=np.expand_dims(vali_data,axis=-1)
test_data=np.expand_dims(test_data,axis=-1)

train_labels=np.array(train_labels)
train_labels=np.squeeze(train_labels)
vali_labels=np.array(vali_labels)
vali_labels=np.squeeze(vali_labels)
test_labels=np.array(test_labels)
test_labels=np.squeeze(test_labels)
#*******************************************
#*******************************************


rms=optimizers.RMSprop(lr=0.001, rho=0.9)
lrs = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50,verbose=1)
model.compile(optimizer=rms, loss='categorical_crossentropy',metrics=['accuracy'])

#train with whole data because autokeras just use a fraction
history=model.fit(x=train_data,y=train_labels,batch_size=128,epochs=200,validation_data=(vali_data,vali_labels),callbacks=[lrs])
#plot results
fig=plt.figure()
plt.plot(history.history['val_loss'])
plt.xlabel("number of epochs")
plt.ylabel('validation loss')
fig.savefig('/home/lnw8px/models/audio_CNN/noise_analyze/results/autokeras_vali_loss.jpg')




