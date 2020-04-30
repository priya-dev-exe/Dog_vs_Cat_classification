import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
import pickle

pickle_in = open(r'D:\Projects\Dogs vs Cats\X.pickle','rb')
X = pickle.load(pickle_in)

pickle_in = open(r'D:\Projects\Dogs vs Cats\y.pickle','rb')
y = pickle.load(pickle_in)

#normalising the data
X = X/255.0

print(X)

#Let's make model

model = Sequential()

model.add(Conv2D(256,(3,3),input_shape = X.shape[1:])) #we need to give input shape in first layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))


#This memorising the image whether it is dog or it is cat
#This is the main brain 
model.add(Dense(1))
model.add(Activation('sigmoid')) #we have two features only that's why we are using sigmoid

model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

#batch size is the size of images you want to give it to neural network to tain once at a time
#validation_split = 0.3 means that use 70% image to check whether it is dog or cat and rest 30% use it for crosscheck
model.fit(X,y,batch_size = 16, epochs = 10,validation_split = 0.3)

#saving 
model.save(r'D:\Projects\Dogs vs Cats\Dogs_vs_Cats_CNN.model')





