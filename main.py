#This project proposal based on the tutorial from tensorflow at https://www.tensorflow.org/tutorials/keras/classification
# and the keras tutorial at https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

# TensorFlow and tf.keras
import tensorflow as tf

import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt



#import fashion data
#fashion_mnist = tf.keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


datagen = ImageDataGenerator(rescale = 1/255)
train_iterator = datagen.flow_from_directory('data/train/', class_mode='binary', target_size=(128, 72), color_mode='grayscale')
validation_iterator = datagen.flow_from_directory('data/validation/', class_mode='binary', target_size=(128, 72), color_mode='grayscale')
test_iterator = datagen.flow_from_directory('data/test/', class_mode='binary', target_size=(128, 72), color_mode='grayscale')

#set up the layers
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(128, 72)),
#     tf.keras.layers.Dense(9216, activation='relu'),
#     tf.keras.layers.Dense(2)
# ])

model = Sequential()
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(128,72,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(250, activation='relu'))
#model.add(tf.keras.layers.Dense(2,activation='softmax'))

# print(model.layers[0].input_shape)
# print(model.layers[0].output_shape)
# print(model.layers[1].input_shape)
# print(model.layers[1].output_shape)
# print(model.layers[2].input_shape)
# print(model.layers[2].output_shape)





#add loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train the model
history = model.fit(train_iterator, epochs=15, validation_data=validation_iterator)

for i in range(len(history.history["val_accuracy"])):
    plt.scatter(x=i, y=history.history["val_accuracy"][i])

plt.show()

print(history.history)
print(history.history['val_accuracy'])
#loss = model.evaluate_generator(test_iterator, steps=24)
#print('\nloss:', loss)
