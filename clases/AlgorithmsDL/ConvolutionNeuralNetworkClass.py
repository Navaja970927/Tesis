import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPool1D
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from clases.ImbalancedPerformance import ImbalancedPerformanceClass


def CNN(ip):
    CNNModel = []

    CNNModel.append(('CNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    CNNModel.append(('CNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    CNNModel.append(('CNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    CNNModel.append(('CNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    CNNModel.append(('CNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return CNNModel


def CNNImbalanced(ip):
    CNNModel = []
    CNNModel.append(('CNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return CNNModel


def CNNUnderSample(ip):
    CNNModel = []
    CNNModel.append(('CNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return CNNModel


def CNNOverSample(ip):
    CNNModel = []
    CNNModel.append(('CNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return CNNModel


def CNNSMOTE(ip):
    CNNModel = []
    CNNModel.append(('CNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return CNNModel


def CNNADASYN(ip):
    CNNModel = []
    CNNModel.append(('CNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return CNNModel


def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

