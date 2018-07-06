# taken from: https://github.com/Kemanth/Food-Classification-from-Images-Using-Convolutional-Neural-Networks-in-Keras-using-Tensorflow
#Prediction on a new picture
from keras.preprocessing import image as image_utils

from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import Tk,Label,Canvas,NW,Entry,Button
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import  tensorflow as tf
#load model

global graph
img_width, img_height = 128, 128
model_path = '../../MyProject/NeuralNetworks/Food_101/models/model_food_non_food.h5'
model_weights_path = '../../MyProject/NeuralNetworks/Food_101/models/weights_food_non_food.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)
graph=tf.get_default_graph()

def predict(inputUrl):
    global url

    url = (inputUrl)
    response = requests.get(url)
    test_image = Image.open(BytesIO(response.content))
    put_image = test_image.resize((400, 400))
    test_image = test_image.resize((128, 128))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result=''
    with graph.as_default():
        result = model.predict_on_batch(test_image)

    if result[0][0] == 1:
        return  'NonFood'
    elif result[0][1] == 1:
        return 'Food'

