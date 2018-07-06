from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Initialize the CNN
model = Sequential()
BATCH_SIZE=32
CLASSES = 2
EPOCHS = 5
MODEL_SAVE_PATH='./models/model_food_101.h5'
MODEL_SAVE_PATH_WEIGHTS='./models/weights_food_101.h5'
TEST_SET_PATH='./data_set/training_set'
TRAIN_SET_PATH='./data_set/test_set'
VALIDATION_SET_PATH='./data_set/validation_set'

# Convolution and Max pooling
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten
model.add(Flatten())
# Full connection
model.add(Dense(128, activation='relu'))
model.add(Dense(CLASSES, activation='softmax'))
# Compile classifier
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
training_set = train_datagen.flow_from_directory(TEST_SET_PATH, target_size=(128, 128), batch_size=BATCH_SIZE,class_mode='categorical')
test_set = test_datagen.flow_from_directory(TRAIN_SET_PATH, target_size=(128, 128), batch_size=BATCH_SIZE,class_mode='categorical')
validation_set = test_datagen.flow_from_directory(VALIDATION_SET_PATH, target_size=(128, 128), batch_size=BATCH_SIZE,class_mode='categorical')

history= model.fit_generator(training_set, steps_per_epoch=800 / 32, epochs=EPOCHS, validation_data=validation_set,validation_steps=200 / 32)

# results
score=model.evaluate_generator(test_set)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# save model
target_dir = './models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save(MODEL_SAVE_PATH)
model.save_weights(MODEL_SAVE_PATH_WEIGHTS)