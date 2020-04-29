import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

CATEGORIES = ['Dog','Cat']
image = r'D:\Projects\Dogs vs Cats\Test_images\437.jpg'

def prepare(filepath):
    img_size = 120
    img_array = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)



model = tf.keras.models.load_model(r'D:\Projects\Dogs vs Cats\Dogs_vs_Cats_CNN.model')
prediction = model.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])

img=mpimg.imread(image)
imgplot=plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()
