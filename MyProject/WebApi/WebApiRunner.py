# THIS WEB-API BASED ON THIS TUTORIAL: https://codeburst.io/this-is-how-easy-it-is-to-create-a-rest-api-8a25122ab1f3
from flask import Flask
from flask_restful import Api, Resource, reqparse
import MyProject.NeuralNetworks.RunPrediction as prediction

app = Flask(__name__)
api = Api(app)

class Query(Resource):
    def get(self,arg):
        print("Input: " + arg)
        result=prediction.runPrediction(arg)
        print("Result: "+result)
        return result,200

# QUERY SHOULD SEND TO 127.0.0.1:5000/PATH_TO_INPUT
api.add_resource(Query, "/<string:arg>")
app.run(debug=True)