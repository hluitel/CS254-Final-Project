#This project proposal based on the tutorial from tensorflow at https://www.tensorflow.org/tutorials/keras/classification
# and the keras tutorial at https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator

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
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 72)),
    tf.keras.layers.Dense(9216, activation='relu'),
    tf.keras.layers.Dense(2)
])

print(model.layers[0].input_shape)
print(model.layers[0].output_shape)
print(model.layers[1].input_shape)
print(model.layers[1].output_shape)
print(model.layers[2].input_shape)
print(model.layers[2].output_shape)





#add loss function
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train the model
model.fit(train_iterator, epochs=10, validation_data=validation_iterator)

#loss = model.evaluate_generator(test_iterator, steps=24)
#print('\nloss:', loss)
