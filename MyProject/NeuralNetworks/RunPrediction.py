import MyProject.NeuralNetworks.Classifier.food_cnn_predict_by_query as classfier


def runPrediction(arg):
    # CHECK FOOD - NO - FOOD
    # TODO

    # CLASSIFY THE INPUT
    retval=classfier.predict(arg)
    return retval