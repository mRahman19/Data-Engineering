# Dataset - In this dataset we have 10000 images, 8000 for training set and 2000 for test set. 
# On training set 4000 images for dog and 4000 for cat of total 8000 images. 
# And on test set 1000 images for dog and 1000 images for cat of total 2000 images

# Part - 1 
# Build CNN architecture

# Import Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize CNN
classifier = Sequential()

# Step - 1 - Convolution, Create 32 feature map by 3 by 3 feature detector and use 64 by 64 pixel to avoid high computation time
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))

# Step - 2 - Pooling, use 2 by 2 pool size 
classifier.add(MaxPooling2D(2,2))

# Adding second convolution and pooling layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(2,2))

# Step - 3 - Flattening
classifier.add(Flatten())

# Step - 4 - Full connection
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compile CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part - 2
# Flatting images to the CNN
from keras.preprocessing.image import ImageDataGenerator

# Use data augmentation function to reduce overfitting model and also useful to get good result with small amount of images
# Train datagen and test datagen both are augmented function
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Use 8000 image for traning, 4000 for dog and 4000 for cat
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         validation_set=2000)

## We get .86 accuracy on training set and .80 on test set, we can improve accuracy by parameter tuning, increase pixel size, increase feature detector and pool size. 

#---------------------------------------------------------------------------------------------------#

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'





