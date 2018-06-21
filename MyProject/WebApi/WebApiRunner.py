# THIS WEB-API BASED ON THIS TUTORIAL: https://codeburst.io/this-is-how-easy-it-is-to-create-a-rest-api-8a25122ab1f3
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
import MyProject.NeuralNetworks.Food_101.food_label_predict as foodClassifier
import MyProject.NeuralNetworks.Food_101.food_non_food_predict as foodNonFoodClassifier

from tkinter import Tk,Label,Canvas,NW,Entry,Button

import urllib
app = Flask(__name__)
api = Api(app)


class Query(Resource):
    def get(self,arg):
        arg=(request.args.get("url"))
        if not (arg.startswith("http")):
            print("invalid input")
            res="invalid"
            return res,200
        print("the input is: "+arg)
        # CHECK IF THE PICTURE IS A FOOD OR NOT
        res=foodNonFoodClassifier.predict(arg)
        if(res!='Food'):
            print("Non food input")
            return res,200
        # CHECK WHAT KIND OF FOOD
        res=foodClassifier.predict(arg)
        print("result: "+ res)
        return res,200


# QUERY SHOULD SEND TO 127.0.0.1:5000/bar?url=PATH_TO_INPUT
api.add_resource(Query, "/<string:arg>")
app.run(debug=True)


