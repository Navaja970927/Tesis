from clases.AlgorithmsML.DecisionTree import DT
from clases.AlgorithmsML.GaussianNaiveBayes import GNB
from clases.AlgorithmsML.LogisticRegresion import LR
from clases.AlgorithmsML.MultiLayerPerceptron import MLP
from clases.AlgorithmsML.RandomForest import RF
from clases.AlgorithmsML.XGBoost import XGB
from clases.AlgorithmsML.KNearestNeighbor import KNN
from clases.AlgorithmsDL.ConvolutionNeuralNetworkClass import CNN
from clases.AlgorithmsDL.AutoencoderClass import AE
from clases.AlgorithmsDL.DenoisingAutoencoderClass import DAE
from clases.AlgorithmsDL.RestrictedBoltzmanMachineClass import RBM
from clases.AlgorithmsDL.RecurrentNeuralNetworkClass import RNN
from clases.AlgorithmsDL.LongShortTermMemoryClass import LSTM
from clases.AlgorithmsDL.BackpropagationNeuralNetworkClass import BPNN
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


def DL_test(ip):
    # Crear los modelos a utilizar
    ip.performanceCNN(CNN(ip))
    ip.performanceAE(AE(ip))
    ip.performanceDAE(DAE(ip))
    ip.performanceRBM(RBM(ip))
    ip.performanceRNN(RNN(ip))
    ip.performanceLSTM(LSTM(ip))
    ip.performanceBPNN(BPNN(ip))
    return ip


def ConvolutionNeuralNetwork(ip):
    ip.performanceCNN(CNN(ip))
    return ip


def AutoEncoder(ip):
    ip.performanceAE(AE(ip))
    return ip


def DenoisingAutoencoder(ip):
    ip.performanceDAE(DAE(ip))
    return ip


def RestrictedBoltzmanMachine(ip):
    ip.performanceRBM(RBM(ip))
    return ip


def RecurrentNeuralNetwork(ip):
    ip.performanceRNN(RNN(ip))
    return ip


def LongShortTermMemory(ip):
    ip.performanceLSTM(LSTM(ip))
    return ip


def BackpropagationNeuralNetwork(ip):
    ip.performanceBPNN(BPNN(ip))
    return ip


ip = ML_test(ip)
# ip = ConvolutionNeuralNetwork(ip)
# ip = AutoEncoder(ip)
# ip = DenoisingAutoencoder(ip)
# ip = RecurrentNeuralNetwork(ip)
# ip = LongShortTermMemory(ip)
# ip = RestrictedBoltzmanMachine(ip)
# ip = BackpropagationNeuralNetwork(ip)
ip = DL_test(ip)
print(ip.show_comparison())
