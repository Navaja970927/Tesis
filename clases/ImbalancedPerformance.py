from sklearn.linear_model import LogisticRegression
from clases.ImbalancedSolution.RandomUnderSampleClass import RamdomUnderSampleClass
from clases.ImbalancedSolution.RandomOverSampleClass import RamdomOverSampleClass
from clases.ImbalancedSolution.SMOTEClass import SMOTEClass
from clases.ImbalancedSolution.ADASYNClass import ADASYNClass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.neural_network import BernoulliRBM
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization, GaussianNoise, SimpleRNN, LSTM, RNN
from keras.layers import Conv1D, ReLU, RepeatVector, TimeDistributed, MaxPooling1D, UpSampling1D, Conv2D, Reshape
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
from openpyxl import load_workbook


class ImbalancedPerformanceClass:
    epochs = 100
    test_size_current = 0.2
    test_size = []
    eps = []
    dropouts = []
    names = []
    aucs_tests = []
    accuracy_tests = []
    precision_tests = []
    recall_tests = []
    f1_score_tests = []
    histories = []

    accuracy_train = []
    recall_train = []
    precision_train = []
    f1_score_train = []

    def solve_imbalanced(self, csv, test_size=0.2):
        # save test_size_current
        self.test_size_current = test_size

        #Use Ramdom Under Sample Dataset
        self.rus = RamdomUnderSampleClass()
        self.rus.read_data(csv, test_size)
        self.rus.rus()
        self.df = self.rus.df
        self.X = self.rus.X
        self.y = self.rus.y

        #Use Ramdom Over Sample Dataset
        self.ros = RamdomOverSampleClass()
        self.ros.read_data(csv, test_size)
        self.ros.ros()

        #Use SMOTE Dataset
        self.smote = SMOTEClass()
        self.smote.read_data(csv, test_size)
        self.smote.smote()

        #Use ADASYN Dataset
        self.adasyn = ADASYNClass()
        self.adasyn.read_data(csv, test_size)
        self.adasyn.adasyn()

        #Saving principal data
        self.X_train = self.rus.X_train
        self.y_train = self.rus.y_train
        self.X_test = self.rus.X_test
        self.y_test = self.rus.y_test

        #Saving Ramdom Under Sample data
        self.X_train_under = self.rus.X_train_under
        self.y_train_under = self.rus.y_train_under
        self.X_test_under = self.rus.X_test_under
        self.y_test_under = self.rus.y_test_under

        #Saving Random Over Sample data
        self.X_train_over = self.ros.X_train_over
        self.y_train_over = self.ros.y_train_over
        self.X_test_over = self.ros.X_test_over
        self.y_test_over = self.ros.y_test_over

        #Saving SMOTE data
        self.X_train_smote = self.smote.X_train_smote
        self.y_train_smote = self.smote.y_train_smote
        self.X_test_smote = self.smote.X_test_smote
        self.y_test_smote = self.smote.y_test_smote

        #Saving ADASYN data
        self.X_train_adasyn = self.adasyn.X_train_adasyn
        self.y_train_adasyn = self.adasyn.y_train_adasyn
        self.X_test_adasyn = self.adasyn.X_test_adasyn
        self.y_test_adasyn = self.adasyn.y_test_adasyn

    def performanceML(self, model):
        for name, model, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(0)

            # appending dropout
            self.dropouts.append('0')

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            self.history = model.fit(X_train, y_train)

            # predictions test
            self.y_test_pred = model.predict(X_test)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred)
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred)
            self.recall_tests.append(Recall_score_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred)
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            self.accuracy_train.append(0)

            # calculate precision train
            self.precision_train.append(0)

            # calculate recall train
            self.recall_train.append(0)

            # calculate F1 train
            self.f1_score_train.append(0)

            # draw confusion matrix
            cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy :{0:0.5f}'.format(Accuracy_test))
            print('Test AUC : {0:0.5f}'.format(Aucs_test))
            print('Test Precision : {0:0.5f}'.format(Precision_score_test))
            print('Test Recall : {0:0.5f}'.format(Recall_score_test))
            print('Test F1 : {0:0.5f}'.format(F1Score_test))
            print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceCNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))
            batch_size = 112
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            plot_model(model, to_file='PNG/modelos/CNNmodel.png', show_shapes=True)
            self.history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, verbose=1,
                                     validation_data=(X_test, y_test))
            self.histories.append(self.history)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = (np.array(self.y_test_pred)).round()
            self.y_test_pred = np.nan_to_num(self.y_test_pred)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceAE(self, model, dropout=[0.5, 0.4, 0.3]):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dropout(dropout[0])(encoder)
            encoder = Dense(1, activation="relu")(encoder)
            decoder = Dropout(dropout[1])(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
            decoder = Dropout(dropout[2])(decoder)
            decoder = Dense(input_dim, activation='relu')(decoder)
            decoder = Dense(1, activation='sigmoid')(decoder)

            autoencoder = Model(inputs=input_layer, outputs=decoder)
            batch_size = 32
            autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                                metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            plot_model(autoencoder, to_file='PNG/modelos/AEmodel.png', show_shapes=True)
            self.history = autoencoder.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, shuffle=True,
                                           validation_data=(X_test, y_test))
            self.histories.append(self.history)

            # predictions
            self.y_test_pred = autoencoder.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceDAE(self, model):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            self.dropouts.append('0')

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)

            # build model
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = GaussianNoise(0.2)(input_layer)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dense(1, activation='relu')(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
            decoder = Dense(input_dim, activation='relu')(decoder)
            decoder = Dense(1, activation='sigmoid')(decoder)

            denoising = Model(inputs=input_layer, outputs=decoder)
            denoising.compile(loss='binary_crossentropy', optimizer=SGD(),
                              metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            plot_model(denoising, to_file='PNG/modelos/DAEmodel.png', show_shapes=True)
            self.history = denoising.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test))
            self.histories.append(self.history)

            # predictions
            self.y_test_pred = denoising.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            if np.isnan(Accuracy_train):
                Accuracy_train = np.nan_to_num(Accuracy_train)
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            if np.isnan(Precision_score_train):
                Precision_score_train = np.nan_to_num(Precision_score_train)
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            if np.isnan(Recall_score_train):
                Recall_score_train = np.nan_to_num(Recall_score_train)
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            if np.isnan(F1Score_train):
                F1Score_train = np.nan_to_num(F1Score_train)
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceRNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(SimpleRNN(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(14))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            plot_model(model, to_file='PNG/modelos/RNNmodel.png', show_shapes=True)
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)
            self.histories.append(self.history)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : '+ str(AUC_score))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceLSTM(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(LSTM(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(14))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            plot_model(model, to_file='PNG/modelos/LSTMmodel.png', show_shapes=True)
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)
            self.histories.append(self.history)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceBPNN(self, model):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending eposhs
            self.dropouts.append('0')

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

            # build model
            model = Sequential()
            model.add(Dense(14, input_shape=(29,), activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss="binary_crossentropy",
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            plot_model(model, to_file='PNG/modelos/BPNNmodel.png', show_shapes=True)
            self.history = model.fit(X_train, y_train, epochs=self.epochs, verbose=1, validation_data=(X_test, y_test))
            self.histories.append(self.history)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceCNN_AE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))

            encoding_dim = 14
            input_layer = Input(shape=(1,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dropout(dropout[0])(encoder)
            encoder = Dense(1, activation="relu")(encoder)
            decoder = Dropout(dropout[1])(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
            decoder = Dropout(dropout[2])(decoder)
            decoder = Dense(1, activation='sigmoid')(decoder)

            autoencoder = Model(inputs=input_layer, outputs=decoder)

            model.add(autoencoder)

            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = (np.array(self.y_test_pred)).round()
            self.y_test_pred = np.nan_to_num(self.y_test_pred)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceCNN_BPNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(Dense(14, input_shape=(29,), activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            model.add(Dense(1, activation='sigmoid'))
            batch_size = 112
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = (np.array(self.y_test_pred)).round()
            self.y_test_pred = np.nan_to_num(self.y_test_pred)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceCNN_DAE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))

            encoding_dim = 14
            input_layer = Input(shape=(1,))
            encoder = GaussianNoise(0.2)(input_layer)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dense(1, activation='relu')(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
            decoder = Dense(1, activation='sigmoid')(decoder)

            denoising = Model(inputs=input_layer, outputs=decoder)

            model.add(denoising)

            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = (np.array(self.y_test_pred)).round()
            self.y_test_pred = np.nan_to_num(self.y_test_pred)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceCNN_LSTM(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(LSTM(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(14))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))
            batch_size = 112
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = (np.array(self.y_test_pred)).round()
            self.y_test_pred = np.nan_to_num(self.y_test_pred)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceCNN_RNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Dense(64, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(SimpleRNN(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(14))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))
            batch_size = 112
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=batch_size, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = (np.array(self.y_test_pred)).round()
            self.y_test_pred = np.nan_to_num(self.y_test_pred)

            # calculate accuracy test
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc test
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # calculating F1 test
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate precision test
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall test
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceBPNN_CNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending eposhs
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # build model
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = Sequential()
            model.add(Dense(29, activation='relu'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            model.add(Conv1D(32, 1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[0]))

            model.add(Conv1D(1, 1, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout[1]))

            model.add(Dense(10, activation='relu'))
            model.add(Dropout(dropout[2]))

            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer='adam', loss="binary_crossentropy",
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceBPNN_AE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending eposhs
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)

            # build model
            model = Sequential()
            model.add(Dense(14, input_shape=(29,), activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dropout(dropout[0])(encoder)
            encoder = Dense(1, activation="relu")(encoder)
            decoder = Dropout(dropout[1])(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
            decoder = Dropout(dropout[2])(decoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            autoencoder = Model(inputs=input_layer, outputs=decoder)

            model.add(autoencoder)

            model.compile(optimizer='adam', loss="binary_crossentropy",
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceBPNN_DAE(self, model):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending eposhs
            self.dropouts.append('0')

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)

            # build model
            model = Sequential()
            model.add(Dense(14, input_shape=(29,), activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = GaussianNoise(0.2)(input_layer)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dense(1, activation='relu')(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            denoising = Model(inputs=input_layer, outputs=decoder)
            model.add(denoising)

            model.compile(optimizer='adam', loss="binary_crossentropy",
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted', zero_division=1)
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceAE_DAE(self, model, dropout=[0.5, 0.4, 0.3]):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dropout(dropout[0])(encoder)
            encoder = Dense(1, activation="relu")(encoder)
            decoder = Dropout(dropout[1])(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
            decoder = Dropout(dropout[2])(decoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            autoencoder = Model(inputs=input_layer, outputs=decoder)

            encoder = GaussianNoise(0.2)(input_layer)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dense(1, activation='relu')(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            denoising = Model(inputs=input_layer, outputs=decoder)

            model = Sequential()
            model.add(autoencoder)
            model.add(denoising)

            batch_size = 32
            model.compile(optimizer='adam', loss='binary_crossentropy',
                                metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, batch_size=batch_size, shuffle=True)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceAE_BPNN(self, model, dropout=[0.5, 0.4, 0.3]):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)

            # Build model
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dropout(dropout[0])(encoder)
            encoder = Dense(1, activation="relu")(encoder)
            decoder = Dropout(dropout[1])(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
            decoder = Dropout(dropout[2])(decoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            autoencoder = Model(inputs=input_layer, outputs=decoder)

            model = Sequential()
            model.add(autoencoder)
            model.add(Dense(14, activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            batch_size = 32
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, batch_size=batch_size, shuffle=True,
                                     validation_data=(X_test, X_test))

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceDAE_BPNN(self, model):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            self.dropouts.append('0')

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)

            # build model
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = GaussianNoise(0.2)(input_layer)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dense(1, activation='relu')(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            denoising = Model(inputs=input_layer, outputs=decoder)

            model = Sequential()
            model.add(denoising)
            model.add(Dense(14, activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer=SGD(),
                              metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, validation_data=(X_test, X_test))

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            if np.isnan(Accuracy_train):
                Accuracy_train = np.nan_to_num(Accuracy_train)
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            if np.isnan(Precision_score_train):
                Precision_score_train = np.nan_to_num(Precision_score_train)
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            if np.isnan(Recall_score_train):
                Recall_score_train = np.nan_to_num(Recall_score_train)
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            if np.isnan(F1Score_train):
                F1Score_train = np.nan_to_num(F1Score_train)
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceDAE_AE(self, model, dropout=[0.5, 0.4, 0.3]):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)

            # build model
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = GaussianNoise(0.2)(input_layer)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dense(1, activation='relu')(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            denoising = Model(inputs=input_layer, outputs=decoder)

            encoder = Dense(encoding_dim, activation="relu")(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            encoder = Dropout(dropout[0])(encoder)
            encoder = Dense(1, activation="relu")(encoder)
            decoder = Dropout(dropout[1])(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='relu')(decoder)
            decoder = Dropout(dropout[2])(decoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)

            autoencoder = Model(inputs=input_layer, outputs=decoder)

            model = Sequential()
            model.add(denoising)
            model.add(autoencoder)

            model.compile(loss='binary_crossentropy', optimizer=SGD(),
                              metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, X_train, epochs=self.epochs, validation_data=(X_test, X_test))

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            if np.isnan(Accuracy_train):
                Accuracy_train = np.nan_to_num(Accuracy_train)
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            if np.isnan(Precision_score_train):
                Precision_score_train = np.nan_to_num(Precision_score_train)
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            if np.isnan(Recall_score_train):
                Recall_score_train = np.nan_to_num(Recall_score_train)
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            if np.isnan(F1Score_train):
                F1Score_train = np.nan_to_num(F1Score_train)
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Accuracy_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceRNN_AE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(SimpleRNN(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(29))
            model.add(Dropout(dropout[2]))

            model.add(Dense(14, activation="relu"))
            model.add(Dense(7, activation="relu"))
            model.add(Dropout(dropout[0]))
            model.add(Dense(1, activation="relu"))
            model.add(Dropout(dropout[1]))
            model.add(Dense(7, activation='relu'))
            model.add(Dropout(dropout[2]))
            model.add(Dense(14, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : '+ str(AUC_score))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceRNN_DAE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(SimpleRNN(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(14))
            model.add(Dropout(dropout[2]))

            model.add(GaussianNoise(0.2))
            model.add(Dense(14, activation="relu"))
            model.add(Dense(7, activation="relu"))
            model.add(Dense(1, activation='relu'))
            model.add(Dense(7, activation='relu'))
            model.add(Dense(14, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : '+ str(AUC_score))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceRNN_BPNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(SimpleRNN(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(29))
            model.add(Dropout(dropout[2]))

            model.add(Dense(14, activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            AUC_score = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(AUC_score)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : '+ str(AUC_score))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceLSTM_AE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(LSTM(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(29))
            model.add(Dropout(dropout[2]))

            model.add(Dense(14, activation="relu"))
            model.add(Dense(7, activation="relu"))
            model.add(Dropout(dropout[0]))
            model.add(Dense(1, activation="relu"))
            model.add(Dropout(dropout[1]))
            model.add(Dense(7, activation='relu'))
            model.add(Dropout(dropout[2]))
            model.add(Dense(14, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceLSTM_DAE(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(LSTM(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(29))
            model.add(Dropout(dropout[2]))

            model.add(GaussianNoise(0.2))
            model.add(Dense(14, activation="relu"))
            model.add(Dense(7, activation="relu"))
            model.add(Dense(1, activation='relu'))
            model.add(Dense(7, activation='relu'))
            model.add(Dense(14, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def performanceLSTM_BPNN(self, model, dropout=[0.5, 0.4, 0.3]):
        for name, X_train, y_train, X_test, y_test in model:
            # appending eposhs
            self.eps.append(self.epochs)

            # appending dropout
            d = str(dropout[0]) + str('-') + str(dropout[1]) + str('-') + str(dropout[2])
            self.dropouts.append(d)

            # appending test_size_current
            self.test_size.append(self.test_size_current)

            # appending name
            self.names.append(name)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # build model
            model = Sequential()
            model.add(LSTM(29))
            model.add(Dropout(dropout[0]))

            model.add(Dense(14))
            model.add(Dropout(dropout[1]))

            model.add(Dense(29))
            model.add(Dropout(dropout[2]))

            model.add(Dense(14, activation='relu'))
            model.add(Dense(2, activation='softmax'))
            model.add(Dense(14, activation='relu'))
            model.add(Dense(29, activation='sigmoid'))

            model.add(Dense(1, activation='sigmoid'))
            opt = Adam()
            model.compile(optimizer=opt, loss='binary_crossentropy',
                          metrics=[self.precision_m, self.recall_m, self.custom_f1, 'accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)
            self.y_test_pred = np.nan_to_num(self.y_test_pred)
            self.y_test_pred = np.argmax(self.y_test_pred, axis=1)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred, average='weighted')
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred, average='weighted')
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred, average='weighted')
            self.f1_score_tests.append(F1Score_test)

            # calculate accuracy train
            Accuracy_train = self.history.history.get('accuracy')[-1]
            self.accuracy_train.append(Accuracy_train)

            # calculate precision train
            Precision_score_train = self.history.history.get('precision_m')[-1]
            self.precision_train.append(Precision_score_train)

            # calculate recall train
            Recall_score_train = self.history.history.get('recall_m')[-1]
            self.recall_train.append(Recall_score_train)

            # calculate F1 train
            F1Score_train = self.history.history.get('custom_f1')[-1]
            self.f1_score_train.append(F1Score_train)

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name : ", name)
            print('Test Accuracy : ' + str(Accuracy_test))
            print('Test AUC : ' + str(Aucs_test))
            print('Test Precision : ' + str(Precision_score_test))
            print('Test Recall : ' + str(Recall_score_test))
            print('Test F1 : ' + str(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

    def recall_m(self, y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        TP = tf.dtypes.cast(TP, tf.float64)
        Positives = tf.dtypes.cast(Positives, tf.float64)
        recall = TP/(Positives+K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        TP = tf.dtypes.cast(TP, tf.float64)
        Pred_Positives = tf.dtypes.cast(Pred_Positives, tf.float64)
        precision = TP/(Pred_Positives+K.epsilon())
        return precision

    def custom_f1(self, y_true, y_pred):
        precision, recall = self.precision_m(y_true, y_pred), self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def show_comparison(self):
        comparision = {
            'Model': self.names,
            'Accuracy': self.accuracy_tests,
            'AUC': self.aucs_tests,
            'Precision Score': self.precision_tests,
            'Recall Score': self.recall_tests,
            'F1 Score': self.f1_score_tests
        }
        print("Comparing performance of various Classifiers: \n \n")
        comparision = pd.DataFrame(data=comparision)
        print(comparision)
        return comparision

    def show_comparison_general(self):
        comparision = {
            'Epoch': self.eps,
            'Model': self.names,
            'Accuracy': self.accuracy_tests,
            'AUC': self.aucs_tests,
            'Precision Score': self.precision_tests,
            'Recall Score': self.recall_tests,
            'F1 Score': self.f1_score_tests,
            'Accuracy Train': self.accuracy_train,
            'Precision Score Train': self.precision_train,
            'Recall Score Train': self.recall_train,
            'F1 Score Train': self.f1_score_train
        }
        print("Comparing performance of various Classifiers: \n \n")
        comparision = pd.DataFrame(data=comparision)
        print(comparision)
        return comparision

    def show_comparison_dropout(self):
        comparision = {
            'Dropout': self.dropouts,
            'Model': self.names,
            'Accuracy': self.accuracy_tests,
            'AUC': self.aucs_tests,
            'Precision Score': self.precision_tests,
            'Recall Score': self.recall_tests,
            'F1 Score': self.f1_score_tests
        }
        print("Comparing performance of various Classifiers: \n \n")
        comparision = pd.DataFrame(data=comparision)
        print(comparision)
        return comparision

    def show_comparison_eposh(self):
        comparision = {
            'Epoch': self.eps,
            'Model': self.names,
            'Accuracy': self.accuracy_tests,
            'AUC': self.aucs_tests,
            'Precision Score': self.precision_tests,
            'Recall Score': self.recall_tests,
            'F1 Score': self.f1_score_tests
        }
        print("Comparing performance of various Classifiers: \n \n")
        comparision = pd.DataFrame(data=comparision)
        print(comparision)
        return comparision

    def show_comparison_test_size(self):
        comparision = {
            'Test Size': self.test_size,
            'Model': self.names,
            'Accuracy': self.accuracy_tests,
            'AUC': self.aucs_tests,
            'Precision Score': self.precision_tests,
            'Recall Score': self.recall_tests,
            'F1 Score': self.f1_score_tests
        }
        print("Comparing performance of various Classifiers: \n \n")
        comparision = pd.DataFrame(data=comparision).sort_values('F1 Score', ascending=False)
        book = load_workbook('Experimentos.xlsx')
        writer = pd.ExcelWriter('Experimentos.xlsx', engine='openpyxl', if_sheet_exists='replace')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        comparision.to_excel(writer, sheet_name='Experimento1', index=False)
        writer.save()
        writer.close()
        print(comparision)
        return comparision

    def show_comparison_train(self):
        comparision = {
            'Epoch': self.eps,
            'Model': self.names,
            'Accuracy Train': self.accuracy_train,
            'Precision Score Train': self.precision_train,
            'Recall Score Train': self.recall_train,
            'F1 Score Train': self.f1_score_train
        }
        print("Comparing performance of various Classifiers: \n \n")
        comparision = pd.DataFrame(data=comparision)
        print(comparision)
        return comparision
