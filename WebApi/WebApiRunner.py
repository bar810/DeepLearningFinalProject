# THIS WEB-API BASED ON THIS TUTORIAL: https://codeburst.io/this-is-how-easy-it-is-to-create-a-rest-api-8a25122ab1f3
from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

class Query(Resource):
    def get(self):
        result="APPLY HERE SOME FUNCTION"
        print("function return result: "+result)
        return result,200

# QUERY SHOULD SEND TO 127.0.0.1:5000/
#TODO :: add param
api.add_resource(Query, "/")
app.run(debug=True)