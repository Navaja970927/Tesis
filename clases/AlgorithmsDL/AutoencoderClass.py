import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn import metrics
import matplotlib.gridspec as gridspec
from clases.ImbalancedPerformance import ImbalancedPerformanceClass


def AE(ip):

    AEModel = []

    AEModel.append(('AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    AEModel.append(('AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    AEModel.append(('AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    AEModel.append(('AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    AEModel.append(('AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return AEModel

class AutoencoderClass:
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    rcParams['figure.figsize'] = 14, 8
    RANDOM_SEED = 42
    LABELS = ["Normal", "Fraud"]

    def read_data(self, csv):
        self.df = pd.read_csv(csv)
        self.frauds = self.df[self.df.Class == 1]
        self.normal = self.df[self.df.Class == 0]

    def is_read_data(self):
        print(self.df.isnull().values.any())

    def prepare_data(self):
        self.data = self.df.drop(['Time'], axis=1)
        self.data['Amount'] = StandardScaler().fit_transform(self.data['Amount'].values.reshape(-1, 1))
        self.X_train, self.X_test = train_test_split(self.data, test_size=0.2, random_state=self.RANDOM_SEED)
        self.X_train = self.X_train[self.X_train.Class == 0]
        self.X_train = self.X_train.drop(['Class'], axis=1)
        self.y_test = self.X_test['Class']
        self.X_test = self.X_test.drop(['Class'], axis=1)
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values

    def is_prepared_data(self):
        print(self.X_train.shape)

    def building_model(self):
        self.input_dim = self.X_train.shape[1]
        self.encoding_dim = 14
        self.input_layer = Input(shape=(self.input_dim,))
        self.encoder = Dense(self.encoding_dim, activation="tanh",
                             activity_regularizer=regularizers.l1(10e-5))(self.input_layer)
        self.encoder = Dense(int(self.encoding_dim / 2), activation="relu")(self.encoder)
        self.decoder = Dense(int(self.encoding_dim / 2), activation='tanh')(self.encoder)
        self.decoder = Dense(self.input_dim, activation='relu')(self.decoder)
        self.autoencoder = Model(inputs=self.input_layer, outputs=self.decoder)
        self.nb_epoch = 100
        self.batch_size = 32
        self.autoencoder.compile(optimizer='adam',
                                 loss='mean_squared_error',
                                 metrics=['accuracy'])
        self.checkpointer = ModelCheckpoint(filepath="model.h5",
                                            verbose=0,
                                            save_best_only=True)
        self.tensorboard = TensorBoard(log_dir='.\logs',
                                       histogram_freq=0,
                                       write_graph=True,
                                       write_images=True)
        self.history = self.autoencoder.fit(self.X_train, self.X_train,
                                            epochs=self.nb_epoch,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            validation_data=(self.X_test, self.X_test),
                                            verbose=1,
                                            callbacks=[self.checkpointer, self.tensorboard]).history

    def show_evaluation(self):
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

    def show_distribution_error(self):
        self.predictions = self.autoencoder.predict(self.X_test)
        self.mse = np.mean(np.power(self.X_test - self.predictions, 2), axis=1)
        self.error_df = pd.DataFrame({'reconstruction_error': self.mse, 'true_class': self.y_test})
        print(self.error_df.describe())

    # Reconstruction error without fraud
    def show_error_no_fraud(self):
        self.fig1 = plt.figure()
        self.ax = self.fig1.add_subplot(111)
        self.normal_error_df = self.error_df[(self.error_df['true_class'] == 0) &
                                             (self.error_df['reconstruction_error'] < 10)]
        self._ = self.ax.hist(self.normal_error_df.reconstruction_error.values, bins=10)
        plt.show()

    # Reconstruction error with fraud
    def show_error_fraud(self):
        self.fig2 = plt.figure()
        self.ax = self.fig2.add_subplot(111)
        self.fraud_error_df = self.error_df[self.error_df['true_class'] == 1]
        self._ = self.ax.hist(self.fraud_error_df.reconstruction_error.values, bins=10)
        plt.show()

    # ROC curves
    def show_roc_curves(self):
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.error_df.true_class,
                                                                self.error_df.reconstruction_error)
        self.roc_auc = metrics.auc(self.fpr, self.tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(self.fpr, self.tpr, label='AUC = %0.4f' % self.roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.001, 1])
        plt.ylim([0, 1.001])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    # Precision vs Recall
    def show_precision_vs_recall(self):
        self.precision, self.recall, self.th = metrics.precision_recall_curve(self.error_df.true_class,
                                                                              self.error_df.reconstruction_error)
        plt.plot(self.recall, self.precision, 'b', label='Precision-Recall curve')
        plt.title('Recall vs Precision')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    # Precision
    def show_precision(self):
        self.precision, self.recall, self.th = metrics.precision_recall_curve(self.error_df.true_class,
                                                                              self.error_df.reconstruction_error)
        plt.plot(self.th, self.precision[1:], 'b', label='Threshold-Precision curve')
        plt.title('Precision for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.show()

    # Recall
    def show_recall(self):
        self.precision, self.recall, self.th = metrics.precision_recall_curve(self.error_df.true_class,
                                                                              self.error_df.reconstruction_error)
        plt.plot(self.th, self.recall[1:], 'b', label='Threshold-Recall curve')
        plt.title('Recall for different threshold values')
        plt.xlabel('Reconstruction error')
        plt.ylabel('Recall')
        plt.show()

    # Prediction
    def show_prediction(self):
        self.threshold = 2.9
        self.groups = self.error_df.groupby('true_class')
        self.fig3, self.ax = plt.subplots()
        for name, group in self.groups:
            self.ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                         label="Fraud" if name == 1 else "Normal")
        self.ax.hlines(self.threshold, self.ax.get_xlim()[0], self.ax.get_xlim()[1],
                       colors="r", zorder=100, label='Threshold')
        self.ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.show()

    # Confusion Matrix
    def show_confussion_matrix(self):
        self.y_pred = [1 if e > self.threshold else 0 for e in self.error_df.reconstruction_error.values]
        self.conf_matrix = metrics.confusion_matrix(self.error_df.true_class, self.y_pred)
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.conf_matrix, xticklabels=self.LABELS, yticklabels=self.LABELS, annot=True, fmt="d",
                    cmap=plt.cm.BuGn)
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()

    # Amount of transaction for calss (fraud/normal)
    def show_transaction_by_class(self):
        self.fig4, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        self.fig4.suptitle('Amount per transaction by class(Fraud/normal)')
        bins = 50
        ax1.hist(self.frauds.Amount, bins=bins)
        ax1.set_title('Fraud')
        ax2.hist(self.normal.Amount, bins=bins)
        ax2.set_title('Normal')
        plt.xlabel('Amount ($)')
        plt.ylabel('Number of Transactions')
        plt.xlim((0, 20000))
        plt.yscale('log')
        plt.show()

    # Time based Analysis to understand occurrence of fraud transaction
    def show_occurrence_fraud(self):
        self.fig5, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        self.fig5.suptitle('Time of transaction vs Amount by class')
        ax1.scatter(self.frauds.Time, self.frauds.Amount)
        ax1.set_title('Fraud')
        ax2.scatter(self.normal.Time, self.normal.Amount)
        ax2.set_title('Normal')
        plt.xlabel('Time (in Seconds)')
        plt.ylabel('Amount')
        plt.show()

    # histograms
    def show_histograms(self):
        self.features = self.df.iloc[:, 1:29].columns
        plt.figure(figsize=(12, 28 * 4))
        self.gs = gridspec.GridSpec(28, 1)
        for i, cn in enumerate(self.df[self.features]):
            ax = plt.subplot(self.gs[i])
            sns.distplot(self.df[cn][self.df.Class == 1], bins=50)
            sns.distplot(self.df[cn][self.df.Class == 0], bins=50)
            ax.set_xlabel('')
            ax.set_title('histogram of feature: ' + str(cn))
        plt.show()
