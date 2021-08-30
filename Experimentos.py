from clases.AlgorithmsML.DecisionTree import *
from clases.AlgorithmsML.GaussianNaiveBayes import *
from clases.AlgorithmsML.LogisticRegresion import *
from clases.AlgorithmsML.MultiLayerPerceptron import *
from clases.AlgorithmsML.RandomForest import *
from clases.AlgorithmsML.XGBoost import *
from clases.AlgorithmsML.KNearestNeighbor import *
from clases.AlgorithmsDL.ConvolutionNeuralNetworkClass import *
from clases.AlgorithmsDL.AutoencoderClass import *
from clases.AlgorithmsDL.DenoisingAutoencoderClass import *
from clases.AlgorithmsDL.RestrictedBoltzmanMachineClass import *
from clases.AlgorithmsDL.RecurrentNeuralNetworkClass import *
from clases.AlgorithmsDL.LongShortTermMemoryClass import *
from clases.AlgorithmsDL.BackpropagationNeuralNetworkClass import *
from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from openpyxl import load_workbook
import pandas as pd


def experimento1():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv", 0.1)
    ip.performanceML(KNNImbalanced(ip))
    ip.solve_imbalanced("creditcard.csv", 0.2)
    ip.performanceML(KNNImbalanced(ip))
    ip.solve_imbalanced("creditcard.csv", 0.3)
    ip.performanceML(KNNImbalanced(ip))
    ip.solve_imbalanced("creditcard.csv", 0.4)
    ip.performanceML(KNNImbalanced(ip))
    ip.solve_imbalanced("creditcard.csv", 0.5)
    ip.performanceML(KNNImbalanced(ip))
    ip.show_comparison_test_size()


def experimento2():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.performanceML(DTImbalanced(ip))
    ip.performanceML(GNBImbalanced(ip))
    ip.performanceML(KNNImbalanced(ip))
    ip.performanceML(LRImbalanced(ip))
    ip.performanceML(MLPImbalanced(ip))
    ip.performanceML(RFImbalanced(ip))
    ip.performanceML(XGBImbalanced(ip))
    comparision = ip.show_comparison()
    book = load_workbook('Experimentos.xlsx')
    comparision = comparision.sort_values('F1 Score', ascending=False)
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    comparision.to_excel(writer, sheet_name='Experimento2', index=False)
    writer.save()
    writer.close()


def experimento3():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.epochs = 1
    ip.performanceCNN(CNNImbalanced(ip), [0.5, 0.4, 0.3])
    ip.performanceRNN(RNNImbalanced(ip), [0.5, 0.4, 0.3])
    ip.performanceCNN(CNNImbalanced(ip), [0.3, 0.4, 0.5])
    ip.performanceRNN(RNNImbalanced(ip), [0.3, 0.4, 0.5])
    ip.performanceCNN(CNNImbalanced(ip), [0.5, 0.5, 0.5])
    ip.performanceRNN(RNNImbalanced(ip), [0.5, 0.5, 0.5])
    ip.performanceCNN(CNNImbalanced(ip), [0.4, 0.4, 0.4])
    ip.performanceRNN(RNNImbalanced(ip), [0.4, 0.4, 0.4])
    ip.performanceCNN(CNNImbalanced(ip), [0.3, 0.3, 0.3])
    ip.performanceRNN(RNNImbalanced(ip), [0.3, 0.3, 0.3])
    ip.performanceCNN(CNNImbalanced(ip), [0.2, 0.2, 0.2])
    ip.performanceRNN(RNNImbalanced(ip), [0.2, 0.2, 0.2])
    ip.performanceCNN(CNNImbalanced(ip), [0.2, 0.2, 0.5])
    ip.performanceRNN(RNNImbalanced(ip), [0.2, 0.2, 0.5])
    ip.performanceCNN(CNNImbalanced(ip), [0.5, 0.2, 0.2])
    ip.performanceRNN(RNNImbalanced(ip), [0.5, 0.2, 0.2])
    comparision = ip.show_comparison_dropout()
    values = comparision.values
    df1 = pd.DataFrame(values[:2], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                            'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df2 = pd.DataFrame(values[2:4], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                             'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df3 = pd.DataFrame(values[4:6], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                             'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df4 = pd.DataFrame(values[6:8], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                             'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df5 = pd.DataFrame(values[8:10], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                              'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df6 = pd.DataFrame(values[10:12], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df7 = pd.DataFrame(values[12:14], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df8 = pd.DataFrame(values[14:16], columns=['Dropout', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    comparision = comparision.sort_values('F1 Score', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace', mode='a')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df1.to_excel(writer, sheet_name='Experimento3', index=False, startrow=0)
    df2.to_excel(writer, sheet_name='Experimento3', index=False, startrow=4)
    df3.to_excel(writer, sheet_name='Experimento3', index=False, startrow=8)
    df4.to_excel(writer, sheet_name='Experimento3', index=False, startrow=12)
    df5.to_excel(writer, sheet_name='Experimento3', index=False, startrow=16)
    df6.to_excel(writer, sheet_name='Experimento3', index=False, startrow=20)
    df7.to_excel(writer, sheet_name='Experimento3', index=False, startrow=24)
    df8.to_excel(writer, sheet_name='Experimento3', index=False, startrow=28)
    comparision.to_excel(writer, sheet_name='Experimento3', index=False, startcol=8)
    writer.save()
    writer.close()


def experimento4():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.epochs = 1
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRBM(RBMImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    ip.epochs = 20
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRBM(RBMImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    ip.epochs = 50
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRBM(RBMImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    ip.epochs = 100
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRBM(RBMImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    comparision = ip.show_comparison_eposh()
    values = comparision.values
    df1 = pd.DataFrame(values[:8], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                            'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df2 = pd.DataFrame(values[8:16], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                              'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df3 = pd.DataFrame(values[16:24], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df4 = pd.DataFrame(values[24:32], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    comparision = comparision.sort_values('F1 Score', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df1.to_excel(writer, sheet_name='Experimento4', index=False, startrow=0)
    df2.to_excel(writer, sheet_name='Experimento4', index=False, startrow=9)
    df3.to_excel(writer, sheet_name='Experimento4', index=False, startrow=18)
    df4.to_excel(writer, sheet_name='Experimento4', index=False, startrow=27)
    comparision.to_excel(writer, sheet_name='Experimento4', index=False, startcol=8)
    writer.save()
    writer.close()


def experimento5():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.epochs = 1
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    ip.epochs = 20
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    ip.epochs = 50
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    ip.epochs = 100
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceAE(AEImbalanced(ip))
    ip.performanceDAE(DAEImbalanced(ip))
    ip.performanceRNN(RNNImbalanced(ip))
    ip.performanceLSTM(LSTMImbalanced(ip))
    ip.performanceBPNN(BPNNImbalanced(ip))
    comparision = ip.show_comparison_train()
    values = comparision.values
    df1 = pd.DataFrame(values[:6], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                            'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train', ascending=False)
    df2 = pd.DataFrame(values[6:12], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                              'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train', ascending=False)
    df3 = pd.DataFrame(values[12:18], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                               'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train', ascending=False)
    df4 = pd.DataFrame(values[18:24], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                               'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train', ascending=False)
    comparision = comparision.sort_values('F1 Score Train', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df1.to_excel(writer, sheet_name='Experimento5', index=False, startrow=0)
    df2.to_excel(writer, sheet_name='Experimento5', index=False, startrow=8)
    df3.to_excel(writer, sheet_name='Experimento5', index=False, startrow=16)
    df4.to_excel(writer, sheet_name='Experimento5', index=False, startrow=24)
    comparision.to_excel(writer, sheet_name='Experimento5', index=False, startcol=8)
    writer.save()
    writer.close()

def experimento6():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.epochs = 1
    ip.performanceCNN(CNN(ip))
    ip.epochs = 20
    ip.performanceCNN(CNN(ip))
    ip.epochs = 50
    ip.performanceCNN(CNN(ip))
    ip.epochs = 100
    ip.performanceCNN(CNN(ip))
    comparision = ip.show_comparison_eposh()
    values = comparision.values
    df1 = pd.DataFrame(values[:5], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                            'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df2 = pd.DataFrame(values[5:10], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                              'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df3 = pd.DataFrame(values[10:15], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    df4 = pd.DataFrame(values[15:20], columns=['Eposh', 'Model', 'Accuracy', 'AUC', 'Precision Score',
                                               'Recall Score', 'F1 Score']).sort_values('F1 Score', ascending=False)
    comparision = comparision.sort_values('F1 Score', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df1.to_excel(writer, sheet_name='Experimento6', index=False, startrow=0)
    df2.to_excel(writer, sheet_name='Experimento6', index=False, startrow=7)
    df3.to_excel(writer, sheet_name='Experimento6', index=False, startrow=14)
    df4.to_excel(writer, sheet_name='Experimento6', index=False, startrow=21)
    comparision.to_excel(writer, sheet_name='Experimento6', index=False, startcol=8)
    writer.save()
    writer.close()


def experimento7():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.epochs = 1
    ip.performanceCNN(CNN(ip))
    ip.epochs = 20
    ip.performanceCNN(CNN(ip))
    ip.epochs = 50
    ip.performanceCNN(CNN(ip))
    ip.epochs = 100
    ip.performanceCNN(CNN(ip))
    comparision = ip.show_comparison_train()
    values = comparision.values
    df1 = pd.DataFrame(values[:5], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                            'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train',
                                                                                                 ascending=False)
    df2 = pd.DataFrame(values[5:10], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                              'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train',
                                                                                                   ascending=False)
    df3 = pd.DataFrame(values[10:15], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                               'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train',
                                                                                                    ascending=False)
    df4 = pd.DataFrame(values[15:20], columns=['Eposh', 'Model', 'Accuracy Train', 'Precision Score Train',
                                               'Recall Score Train', 'F1 Score Train']).sort_values('F1 Score Train',
                                                                                                    ascending=False)
    comparision = comparision.sort_values('F1 Score Train', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    df1.to_excel(writer, sheet_name='Experimento7', index=False, startrow=0)
    df2.to_excel(writer, sheet_name='Experimento7', index=False, startrow=7)
    df3.to_excel(writer, sheet_name='Experimento7', index=False, startrow=14)
    df4.to_excel(writer, sheet_name='Experimento7', index=False, startrow=21)
    comparision.to_excel(writer, sheet_name='Experimento7', index=False, startcol=8)
    writer.save()
    writer.close()


def experimento8():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.performanceML(DT(ip))
    ip.performanceML(GNB(ip))
    ip.performanceML(KNN(ip))
    ip.performanceML(LR(ip))
    ip.performanceML(MLP(ip))
    ip.performanceML(RF(ip))
    ip.performanceML(XGB(ip))
    comparision = ip.show_comparison()
    comparision = comparision.sort_values('F1 Score', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    comparision.to_excel(writer, sheet_name='Experimento8', index=False)
    writer.save()
    writer.close()


def experimento9():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.performanceML(RFSMOTE(ip))
    ip.performanceML(RFADASYN(ip))
    ip.performanceML(XGBOverSample(ip))
    ip.performanceML(XGBSMOTE(ip))
    ip.epochs = 1
    ip.performanceCNN(CNNSMOTE(ip))
    ip.epochs = 20
    ip.performanceCNN(CNNOverSample(ip))
    ip.epochs = 100
    ip.performanceCNN(CNNImbalanced(ip))
    ip.performanceCNN(CNNADASYN(ip))
    comparision = ip.show_comparison_eposh()
    comparision = comparision.sort_values('F1 Score', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    comparision.to_excel(writer, sheet_name='Experimento9', index=False)
    writer.save()
    writer.close()


def experimento10():
    ip = ImbalancedPerformanceClass()
    ip.solve_imbalanced("creditcard.csv")
    ip.performanceML(DT(ip))
    ip.performanceML(GNB(ip))
    ip.performanceML(KNN(ip))
    ip.performanceML(LR(ip))
    ip.performanceML(MLP(ip))
    ip.performanceML(RF(ip))
    ip.performanceML(XGB(ip))
    ip.epochs = 1
    ip.performanceCNN(CNN(ip))
    ip.performanceAE(AE(ip))
    ip.performanceDAE(DAE(ip))
    ip.performanceRBM(RBM(ip))
    ip.performanceRNN(RNN(ip))
    ip.performanceLSTM(LSTM(ip))
    ip.performanceBPNN(BPNN(ip))
    ip.epochs = 20
    ip.performanceCNN(CNN(ip))
    ip.performanceAE(AE(ip))
    ip.performanceDAE(DAE(ip))
    ip.performanceRBM(RBM(ip))
    ip.performanceRNN(RNN(ip))
    ip.performanceLSTM(LSTM(ip))
    ip.performanceBPNN(BPNN(ip))
    ip.epochs = 50
    ip.performanceCNN(CNN(ip))
    ip.performanceAE(AE(ip))
    ip.performanceDAE(DAE(ip))
    ip.performanceRBM(RBM(ip))
    ip.performanceRNN(RNN(ip))
    ip.performanceLSTM(LSTM(ip))
    ip.performanceBPNN(BPNN(ip))
    ip.epochs = 100
    ip.performanceCNN(CNN(ip))
    ip.performanceAE(AE(ip))
    ip.performanceDAE(DAE(ip))
    ip.performanceRBM(RBM(ip))
    ip.performanceRNN(RNN(ip))
    ip.performanceLSTM(LSTM(ip))
    ip.performanceBPNN(BPNN(ip))
    comparision = ip.show_comparison_general()
    comparision = comparision.sort_values('F1 Score', ascending=False)
    book = load_workbook('Experimentos.xlsx')
    writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    comparision.to_excel(writer, sheet_name='Experimento10', index=False)
    writer.save()
    writer.close()
