# -*- coding: utf-8 -*-
#%%
from keras.datasets import mnist #import the dataset
from keras.models import Sequential #import the type of modeel
from keras.layers.core import Dense,Dropout,Activation,Flatten #import layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D #import convolution layers
from keras.utils import np_utils


#to plot,import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%%
#batch size to train
batch_size=128
#Number of output classes
nb_classes=10
#Number of epochs to train
nb_epoch=3
#input image dimensios
img_rows,img_cols=28,28
#Number of convolutional filters to use
nb_filters=32
#size of pooling area for max pooling
nb_pool=2
#convolution kernal size
nb_conv=3

#%%
#The data suffled and split between train and test sets
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#reshape the data
X_train=X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
X_test=X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train /=255
X_test /=255
print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
#define class vectors to binary class matrices
Y_train=np_utils.to_categorical(y_train,nb_classes)
Y_test=np_utils.to_categorical(y_test,nb_classes)
i=4600
plt.imshow(X_train[i,0],interpolation='nearest')
print('label:',Y_train[i,:])

#%%
model = Sequential()
 
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                         border_mode='valid',
                         input_shape=(1, img_rows, img_cols)))
convout1=Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2=Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
 
#%% 
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,show_accuracy=True,
           verbose=1, validation_data=(X_test, Y_test))
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,show_accuracy=True,
           verbose=1, validation_split=0.2)

#%%

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])

#%%
# saving weights
fname = "weights-mnist-cnn.hdf5"
model.save_weights(fname,overwrite=True)
# saving bias
fname = "biases-mnist-cnn.hdf5"
model.save_biases(fname,overwrite=True)
#%%
# Loading weights
fname = "weights-mnist-cnn.hdf5"
model.load_weights(fname)
# Loading bias
fname = "bias-mnist-cnn.hdf5"
model.load_bias(fname)
#%%
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:10])
#%%
#confusion matrix
from sklearn.metrics import classification_report,confusion_matrix

Y_pred = model.predict(X_test)
print(Y_pred)
#y_pred = np.argmax(Y_pred, axis=1)
y_pred = model.predict_classes(X_test)
print(y_pred)
p=model.predict_proba(X_test) # to predict probability
target_names = ['class 0(0)', 'class 1(1)', 'class 2(2)', 'class 3(3)', 'class 4(4)', 'class 5(5)', 'class 6(6)', 'class 7(7)', 'class 8(8)', 'class 9(9)']
print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_test,axis=1), y_pred))

#%%
model.count_params()
  