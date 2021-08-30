from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd


class SMOTEClass:

    def read_data(self, csv, test_size=0.2):
        self.df = pd.read_csv(csv)
        self.X = self.df.drop(['Class', 'Amount'], axis=1)
        self.y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                shuffle=True)

    def smote(self):
        self.smote = SMOTE(sampling_strategy='minority')
        self.X_train_smote, self.y_train_smote = self.smote.fit_resample(self.X_train, self.y_train)
        self.X_test_smote, self.y_test_smote = self.X_test, self.y_test

    def show_test(self):
        print("X_train: ", self.X_train.shape)
        print("y_train: ", self.y_train.shape)
        print("X_test: ", self.X_test.shape)
        print("y_test: ", self.y_test.shape)
        print('............')
        print("X_train_smote: ", self.X_train_smote.shape)
        print("y_train_smote: ", self.y_train_smote.shape)
        print("X_test_smote: ", self.X_test_smote.shape)
        print("y_test_smote: ", self.y_test_smote.shape)
