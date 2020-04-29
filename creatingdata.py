import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os

Datadir = r'D:\Projects\Dogs vs Cats\PetImages/'
CATEGORIES = ['Dog', 'Cat']

for i in CATEGORIES:
    path = os.path.join(Datadir,i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = 'gray')
        plt.show()
        break
    break

img_size = 120

new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap='gray')
plt.show()

training_data = []

def create_training_data():
    for i in CATEGORIES:

        path = os.path.join(Datadir,i)
        class_num = CATEGORIES.index(i)
        #removing corrupt images from dataset and making taining dataset
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)
#printing first 10 data of training set
for sample in training_data[:10]:
    print(sample)

x = []
y = []

for features,label in training_data:
    x.append(features)
    y.append(label)

print(x[0].reshape(-1,img_size,img_size,1))

#converting it into numpy array for faster processing
X = np.array(x).reshape(-1,img_size,img_size,1)

print(X)

import pickle

#making file to store converted data

pickle_out = open(r'D:\Projects\Dogs vs Cats\X.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open(r'D:\Projects\Dogs vs Cats\Y.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()








            
                


