import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Load the saved model
model = tf.keras.models.load_model('./InceptionV3.h5')

dir = './test/'
dir_fire = './fire/'
dir_normal = './normal/'

for filename in os.listdir(dir):
    if filename.endswith(".jpg"):
        im = image.load_img(dir + filename)
        # Resizing into 224x224 because we trained the model with this image size.
        im = im.resize((224, 224))

        img_array = image.img_to_array(im)
        img_array = np.expand_dims(img_array, axis=0) / 255

        probabilities = model.predict(img_array)[0]
        print("probabilities = ", probabilities)
        prediction = np.argmax(probabilities)
        print("prediction = ", prediction)
        if prediction == 0:
            print("There is a FIRE!!!")
            image.save_img(dir_fire + filename, im)
        else:
            print("There is a NORMAL....")
            image.save_img(dir_normal + filename, im)
        continue

