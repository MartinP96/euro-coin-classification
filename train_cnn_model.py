'''
- script name: train_cnn_model.py
- author: Martin P.
- description: This script is used for training a CNN model.

'''

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Define hyperparameters
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 100
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 900//BATCH_SIZE
EPOCHS = 500

# Image generators for training/test set
training_data_generator = ImageDataGenerator(rescale = 1./255,
                                             brightness_range=[0.5, 1],
                                             rotation_range=90)

testing_data_generator = ImageDataGenerator(rescale = 1./255,
                                            brightness_range=[0.5, 1],
                                            rotation_range=90)


# Dataset folder
src = "data/"

training_set = training_data_generator.flow_from_directory(f"{src}+train/",
                                                           target_size = (INPUT_SIZE, INPUT_SIZE),
                                                           batch_size = BATCH_SIZE,
                                                           class_mode = 'categorical')

test_set = testing_data_generator.flow_from_directory(f"{src}+test/"+'test/',
                                                      target_size = (INPUT_SIZE, INPUT_SIZE),
                                                      batch_size = BATCH_SIZE,
                                                      class_mode = 'categorical')

# Build CNN model
model = Sequential()
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 8, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# Train model
history = model.fit(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=1, validation_data=test_set)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Save model
# model.save('model.h5')

