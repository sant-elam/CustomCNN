# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:49:31 2021

@author: ****
"""

import  os
import glob
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from keras.utils import np_utils





import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_sequential(no_of_classes, target_size=28):

    model = Sequential()
    
    # FIRST LAYER
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(target_size, target_size, 1)))
    model.add(MaxPooling2D())  
    
    # SECOND LAYER
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D())  
    
    # THIRD LAYER
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D())  
    
    
    model.add(Flatten())  
    
    model.add(Dense(512, activation='relu'))
              
    model.add(Dense(no_of_classes, activation='softmax'))
    
    model.compile( optimizer='adam',
                   loss = 'categorical_crossentropy',
                   metrics = 'accuracy')
    
    return model


def read_image_resize(dataset_folder):
    
    images =[]
    labels =[]
    for directory in os.listdir(dataset_folder):
        #print(directory)
        
        path_dir = os.path.join(dataset_folder, directory)
        
       # print(path_dir)
        
        all_bmp_images = glob.glob(path_dir + "**/*.bmp")
        
        #print(all_bmp_images)
        for bmp_image in all_bmp_images:
            #print(bmp_image)
            image = cv.imread(bmp_image, cv.IMREAD_GRAYSCALE)
            
            
            #2.2 Resize the images to target size.. 28x28
            height, width = image.shape
            
            max_value = (float) (np.amax([height, width]))
            
            scale_factor = (float) (TARGET_SIZE/max_value)
            
            if scale_factor < 1.0:         
                
                width_new = (int) ( width * scale_factor)
                height_new = (int) ( height * scale_factor)
                
                image = cv.resize(image, (width_new, height_new))
                
            else:
                width_new = width
                height_new = height
            
            #2.3 Pad the images..
            
            width = TARGET_SIZE
            height= TARGET_SIZE
            
            diff_width = width - width_new
            diff_height = height - height_new
            
            
            top = diff_height // 2
            bottom = diff_height - top
            
            left = diff_width // 2
            right = diff_width - left
            
            print(top, bottom, left, right)
            
            color_fill = 0
            image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, color_fill)
       
            images.append(image)
            
            labels.append(directory)
            
    return images, labels


def split_train_test(images, labels, testsize= 0.33, randomstate=1):
    
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size= testsize, random_state = randomstate)
           
     
    #2.4 Convert the images/dataset to numpy array
    x_train = np.array(x_train, dtype=object)    
    x_test = np.array(x_test, dtype=object)
        
         #2.5 Reshape the array
    no_of_train = x_train.shape[0]     
    no_of_test  = x_test.shape[0]
    
    x_train = x_train.reshape(no_of_train, TARGET_SIZE, TARGET_SIZE, 1)
    x_test  = x_test.reshape(no_of_test, TARGET_SIZE, TARGET_SIZE, 1)
    
    #print(x_train[0])
         
         #2.6 Convert array to float
    x_train = x_train.astype('float32')     
    x_test = x_test.astype('float32')
    
    
    #print(x_train[0])
         
         #2.7 Noramlized or scale - 0:1
         
    x_train /= 255
    x_test /= 255
    
    return x_train, x_test, y_train, y_test

def one_hot_encode(y_train, y_test, label_array, no_of_classes):
    y_train_keys=[]
    for y_label in y_train:
        values =    label_array[y_label]  
        y_train_keys.append(values)

    y_test_keys=[]
    for y_label in y_test:
        values =    label_array[y_label]  
        y_test_keys.append(values)
        
    '''    
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11      
    0  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  ]
    0  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  ] 
    
    -  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0  ] 
    R  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1  ]
    
    '''
    y_train = np_utils.to_categorical(y_train_keys, NO_OF_CLASSES)
    y_test = np_utils.to_categorical(y_test_keys, NO_OF_CLASSES)
        
    return y_train, y_test


# 1. PREPARE THE DATASET


# 2. TRAIN THE DATASET


     #2.1 Load the images/dataset
TARGET_SIZE = 28
     
dataset_folder = "C:/Users/Santosh/OneDrive/Desktop/CHAR_ DETECTION/DATA_SET"


images, labels = read_image_resize(dataset_folder)

plt.imshow(images[20], cmap='gray')

    
#2.4 Split the images into Train set and Test set
x_train, x_test, y_train, y_test = split_train_test(images, labels, 0.33, 2)
           
print(y_test)   
print(y_train)  

print(x_train[0])
print(x_test[0])
    
# 3. Use ONE HOT ENCODE for the classes-lables.
     # Encode the Train and Test label/classes
     
NO_OF_CLASSES = 12     
label_array = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '-':10, 'R':11 }
     
print(y_test) 
y_train, y_test = one_hot_encode(y_train, y_test, label_array, NO_OF_CLASSES)

print(y_test)

# 4. SIMPLE MODEL TO TRAIN..
     # 3 layes models... 

model = create_sequential(NO_OF_CLASSES, TARGET_SIZE)

model.summary()


Epochs_to_train = 50

# 5. TRAIN the dataset with this MODEL..
model.fit(x_train, y_train,
          validation_data = ( x_test, y_test),
          epochs = Epochs_to_train)

# 6. PLOT and see how the model performs on this data
print(model.history.history)

history = model.history.history

plt.subplot(2, 1, 1)
plt.title("MODEL LOSS")
plt.ylabel("Loss")
plt.xlabel("No of Epochs")

plt.plot(history['loss'])
plt.plot(history['val_loss'])

plt.legend(['train', 'test'], loc='upper right')


plt.subplot(2, 1, 2)
plt.title("MODEL Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("No of Epochs")

plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])

plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()


# 7. Save this model..



# 8. Predict/Classify new images..
