from clases.ImbalancedSolution.RandomUnderSampleClass import RamdomUnderSampleClass
from clases.ImbalancedSolution.RandomOverSampleClass import RamdomOverSampleClass
from clases.ImbalancedSolution.SMOTEClass import SMOTEClass
from clases.ImbalancedSolution.ADASYNClass import ADASYNClass
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Conv1D
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams


class ImbalancedPerformanceClass:
    names = []
    aucs_tests = []
    accuracy_tests = []
    precision_tests = []
    recall_tests = []
    f1_score_tests = []

    def solve_imbalanced(self, csv):
        #Use Ramdom Under Sample Dataset
        self.rus = RamdomUnderSampleClass()
        self.rus.read_data(csv)
        self.rus.rus()
        self.df = self.rus.df
        self.X = self.rus.X
        self.y = self.rus.y

        #Use Ramdom Over Sample Dataset
        self.ros = RamdomOverSampleClass()
        self.ros.read_data(csv)
        self.ros.ros()

        #Use SMOTE Dataset
        self.smote = SMOTEClass()
        self.smote.read_data(csv)
        self.smote.smote()

        #Use ADASYN Dataset
        self.adasyn = ADASYNClass()
        self.adasyn.read_data(csv)
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
            # appending name
            self.names.append(name)

            # Build model
            model.fit(X_train, y_train)

            # predictions
            self.y_test_pred = model.predict(X_test)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = metrics.roc_auc_score(y_test, self.y_test_pred)
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, self.y_test_pred)
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, self.y_test_pred)
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, self.y_test_pred)
            self.f1_score_tests.append(format(F1Score_test))

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

            fpr, tpr, thresholds = metrics.roc_curve(y_test, self.y_test_pred)
            auc = metrics.roc_auc_score(y_test, self.y_test_pred)
            plt.plot(fpr, tpr, linewidth=2, label=name + ", auc=" + str(auc))

        plt.legend(loc=4)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.show()

    def performanceCNN(self, model):
        for name, X_train, y_train, X_test, y_test in model:
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

            self.epochs = 20
            model = Sequential()
            model.add(Conv1D(32, 2, activation='relu', input_shape=X_train[0].shape))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Conv1D(64, 2, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(1, activation='sigmoid'))
            opt = keras.optimizers.Adam(lr=0.0001)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            self.history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_test, y_test), verbose=1)

            # predictions
            self.y_test_pred = model.predict(X_test)

            # calculate accuracy
            Accuracy_test = keras.metrics.binary_accuracy(y_test, self.y_test_pred)[-1]
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = keras.metrics.AUC(num_thresholds=10)
            y_test = y_test.reshape(y_test.shape[0], 1)
            Aucs_test.update_state(y_test, self.y_test_pred)
            Aucs_test = Aucs_test.result()
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = keras.metrics.Precision()
            Precision_score_test.update_state(y_test, self.y_test_pred)
            Precision_score_test = Precision_score_test.result()
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = keras.metrics.Recall()
            Recall_score_test.update_state(y_test, self.y_test_pred)
            Recall_score_test = Recall_score_test.result()
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = 2*(Precision_score_test*Recall_score_test)/(Precision_score_test+Recall_score_test+K.epsilon())
            self.f1_score_tests.append(format(F1Score_test))

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy :{0:0.5f}'.format(Accuracy_test))
            print('Test AUC : {0:0.5f}'.format(Aucs_test))
            print('Test Precision : {0:0.5f}'.format(Precision_score_test))
            print('Test Recall : {0:0.5f}'.format(Recall_score_test))
            print('Test F1 : {0:0.5f}'.format(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

            fpr, tpr, thresholds = metrics.roc_curve(y_test, self.y_test_pred)
            auc = metrics.roc_auc_score(y_test, self.y_test_pred)
            plt.plot(fpr, tpr, linewidth=2, label=name + ", auc=" + str(auc))

        plt.legend(loc=4)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.show()

        epoch_range = range(1, self.epochs + 1)
        plt.plot(epoch_range, self.history.history['accuracy'])
        plt.plot(epoch_range, self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(epoch_range, self.history.history['loss'])
        plt.plot(epoch_range, self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()

    def performanceAE(self, model):
        sns.set(style='whitegrid', palette='muted', font_scale=1.5)
        rcParams['figure.figsize'] = 14, 8
        RANDOM_SEED = 42
        LABELS = ["Normal", "Fraud"]
        for name, X_train, y_train, X_test, y_test in model:
            # appending name
            self.names.append(name)

            # Build model
            input_dim = X_train.shape[1]
            encoding_dim = 14
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation="tanh",
                            activity_regularizer=keras.regularizers.l1(10e-5))(input_layer)
            encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
            decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
            decoder = Dense(input_dim, activation='relu')(decoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            nb_epoch = 1
            batch_size = 32
            autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            checkpointer = ModelCheckpoint(filepath="model.h5", verbose=0, save_best_only=True)
            tensorboard = TensorBoard(log_dir='.\logs', histogram_freq=0, write_graph=True, write_images=True)
            history = autoencoder.fit(X_train, X_train, epochs= nb_epoch, batch_size= batch_size, shuffle=True,
                                      validation_data=(X_test, X_test), verbose=1,
                                      callbacks=[checkpointer, tensorboard]).history

            # predictions
            self.y_test_pred = autoencoder.predict(X_test)

            # calculate accuracy
            Accuracy_test = keras.metrics.binary_accuracy(y_test, self.y_test_pred)
            self.accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = keras.metrics.AUC(num_thresholds=10)
            y_test = y_test.reshape(y_test.shape[0], 1)
            Aucs_test.update_state(y_test, self.y_test_pred)
            Aucs_test = Aucs_test.result()
            self.aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = keras.metrics.Precision()
            Precision_score_test.update_state(y_test, self.y_test_pred)
            Precision_score_test = Precision_score_test.result()
            self.precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = keras.metrics.Recall()
            Recall_score_test.update_state(y_test, self.y_test_pred)
            Recall_score_test = Recall_score_test.result()
            self.recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = 2*(Precision_score_test*Recall_score_test)/(Precision_score_test+Recall_score_test+K.epsilon())
            self.f1_score_tests.append(format(F1Score_test))

            # draw confusion matrix (have troubles)
            # cnf_matrix = metrics.confusion_matrix(y_test, self.y_test_pred)

            print("Model Name :", name)
            print('Test Accuracy :{0:0.5f}'.format(Accuracy_test))
            print('Test AUC : {0:0.5f}'.format(Aucs_test))
            print('Test Precision : {0:0.5f}'.format(Precision_score_test))
            print('Test Recall : {0:0.5f}'.format(Recall_score_test))
            print('Test F1 : {0:0.5f}'.format(F1Score_test))
            # print('Confusion Matrix : \n', cnf_matrix)
            print("\n")

            fpr, tpr, thresholds = metrics.roc_curve(y_test, self.y_test_pred)
            auc = metrics.roc_auc_score(y_test, self.y_test_pred)
            plt.plot(fpr, tpr, linewidth=2, label=name + ", auc=" + str(auc))

        plt.legend(loc=4)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.show()

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
        return comparision.sort_values('F1 Score', ascending=False)
