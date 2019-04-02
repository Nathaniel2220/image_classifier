# -*- coding: utf-8 -*-
"""
@author: Nathaniel Livingston
Adam Smith AI Spring 2018
Convolutional Nueral Network
"""
import sys
import random
import numpy as np

from os import listdir
from PIL import Image as PImage
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Dropout


directory = sys.argv[1] # Read command line arguments
saveName = sys.argv[2]+".dnn"

if (not directory): # If there's nothing there, like an empty string
    print ("Goodbye.")

imageList = listdir(directory)

training_data = []
labels = []

print("Loading Images...")

for image in imageList:
        
    if(image[-4:] == ".jpg"): # Make sure it's a JPG
        img = PImage.open(directory +"\\"+ image)
        if(np.array(img).shape != (100,100,3)): # Make sure it's in color
            #print("Black and white photo detected.")
            continue
                 
        else:    
            if (image[0:1] == "c"): # Check for identities here
                training_data.append(np.array(img)) # Then load everything up
                labels.append(np.array([0,1]))
            else:
                training_data.append(np.array(img))
                labels.append(np.array([1,0]))
            img.load() # this seems to "close" the image
        
print("Done! Now time to train!")

c = list(zip(training_data,labels)) # shuffle up the data
random.shuffle(c)
training_data,labels = zip(*c)

# Time to create our model!
model = Sequential()

# Five convolutional layers with pooling layers
model.add(Conv2D(32, kernel_size=(3, 3),strides = (1,1), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
 
model.add(Conv2D(32, kernel_size=(3, 3),strides = (1,1), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
 
model.add(Conv2D(32, kernel_size=(3, 3),strides = (1,1), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(32, kernel_size=(3, 3),strides = (1,1), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(32, kernel_size=(3, 3),strides = (1,1), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.5))
 
model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(2, activation='softmax')) # add a softmax layer as the last step

# compile it
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# And let's start training away!
model.fit(np.array(training_data), np.array(labels),
          batch_size=100, 
          epochs=70, 
          validation_split=0.2,
          verbose = 1
          )

# Saves the wieghts as a dnn file
model.save(saveName, include_optimizer = False)