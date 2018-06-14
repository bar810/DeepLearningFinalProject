# taken from: https://github.com/Kemanth/Food-Classification-from-Images-Using-Convolutional-Neural-Networks-in-Keras-using-Tensorflow

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
#load model

img_width, img_height = 128, 128
model_path = '/Users/barbrownshtein/PycharmProjects/FinalProject/DeepLearningFinalProject/MyProject/NeuralNetworks/Food_101/models/model.h5'
model_weights_path = '/Users/barbrownshtein/PycharmProjects/FinalProject/DeepLearningFinalProject/MyProject/NeuralNetworks/Food_101/models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

#Prediction on a new picture
from keras.preprocessing import image as image_utils

from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import Tk,Label,Canvas,NW,Entry,Button

def predict(inputUrl):
    global url
    url = (inputUrl)
    print(url)
    response = requests.get(url)
    test_image = Image.open(BytesIO(response.content))
    put_image = test_image.resize((400, 400))
    test_image = test_image.resize((128, 128))
    img = ImageTk.PhotoImage(put_image)
    pic = Label(image=img)
    pic.pack()
    pic.image = img
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict_on_batch(test_image)
    print(result)
    return result