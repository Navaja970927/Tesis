from clases.AutoencoderClass import AutoencoderClass
from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from clases.AlgorithmsML.DecisionTree import DT
from clases.AlgorithmsML.GaussianNaiveBayes import GNB
from clases.AlgorithmsML.LogisticRegresion import LR
from clases.AlgorithmsML.MultiLayerPerceptron import MLP
from clases.AlgorithmsML.RandomForest import RF
from clases.AlgorithmsML.XGBoost import XGB
from clases.AlgorithmsML.KNearestNeighbor import KNN


def ML_test():
    #Cargar los datos y aplicar los algoritmos para el problema desbalanceado
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")

    #Crear los modelos a utilizar
    ip.performance(DT(ip))
    ip.performance(GNB(ip))
    ip.performance(KNN(ip))
    ip.performance(LR(ip))
    ip.performance(MLP(ip))
    ip.performance(RF(ip))
    ip.performance(XGB(ip))

    #Obtener tabla comparativa de los algoritmos
    print(ip.show_comparison())

