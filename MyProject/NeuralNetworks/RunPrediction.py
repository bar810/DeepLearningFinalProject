import MyProject.NeuralNetworks.CIFAR10.CIFAR10_CNN as cifar10
import MyProject.NeuralNetworks.CNN_1.CNN1 as cnn1


def runPrediction(arg):
    # CHECK FOOD - NO - FOOD
    retval= cnn1.predict(arg)
    if retval=="NO_FOOD":
        return "NO_FOOD"

    # CLASSIFY THE INPUT
    retval=cifar10.predict(arg)
    return retval