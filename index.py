from clases.AlgorithmsML.DecisionTree import DT
from clases.AlgorithmsML.GaussianNaiveBayes import GNB
from clases.AlgorithmsML.LogisticRegresion import LR
from clases.AlgorithmsML.MultiLayerPerceptron import MLP
from clases.AlgorithmsML.RandomForest import RF
from clases.AlgorithmsML.XGBoost import XGB
from clases.AlgorithmsML.KNearestNeighbor import KNN
from clases.AlgorithmsDL.ConvolutionNeuralNetworkClass import CNN
from clases.AlgorithmsDL.AutoencoderClass import AE
from clases.ImbalancedPerformance import ImbalancedPerformanceClass

ip = ImbalancedPerformanceClass()
ip.solve_imbalanced("creditcard.csv")

def ML_test(ip):
    #Crear los modelos a utilizar
    ip.performanceML(DT(ip))
    ip.performanceML(GNB(ip))
    ip.performanceML(KNN(ip))
    ip.performanceML(LR(ip))
    ip.performanceML(MLP(ip))
    ip.performanceML(RF(ip))
    ip.performanceML(XGB(ip))
    return ip


def ConvolutionNeuralNetwork(ip):
    ip.performanceCNN(CNN(ip))
    return ip


def AutoEncoder(ip):
    ip.performanceAE(AE(ip))
    return ip


# ip = ML_test(ip)
# ip = ConvolutionNeuralNetwork(ip)
ip = AutoEncoder(ip)
# print(ip.show_comparison())
