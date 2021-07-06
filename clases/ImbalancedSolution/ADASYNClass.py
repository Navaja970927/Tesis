from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd


class ADASYNClass:

    def read_data(self, csv):
        self.df = pd.read_csv(csv)
        self.X = self.df.drop(['Class', 'Amount'], axis=1)
        self.y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                shuffle=True)

    def adasyn(self):
        self.adasyn = ADASYN(sampling_strategy='minority')
        self.X_train_adasyn, self.y_train_adasyn = self.adasyn.fit_resample(self.X_train, self.y_train)
        self.X_test_adasyn, self.y_test_adasyn = self.X_test, self.y_test

    def show_test(self):
        print("X_train: ", self.X_train.shape)
        print("y_train: ", self.y_train.shape)
        print("X_test: ", self.X_test.shape)
        print("y_test: ", self.y_test.shape)
        print('............')
        print("X_train_adasyn: ", self.X_train_adasyn.shape)
        print("y_train_adasyn: ", self.y_train_adasyn.shape)
        print("X_test_adasyn: ", self.X_test_adasyn.shape)
        print("y_test_adasyn: ", self.y_test_adasyn.shape)
