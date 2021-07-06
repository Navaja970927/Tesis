from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pandas as pd


class RamdomOverSampleClass:

    def read_data(self, csv):
        self.df = pd.read_csv(csv)
        self.X = self.df.drop(['Class', 'Amount'], axis=1)
        self.y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                shuffle=True)

    def ros(self):
        self.ros = RandomOverSampler(sampling_strategy='minority')
        self.X_train_over, self.y_train_over = self.ros.fit_resample(self.X_train, self.y_train)
        self.X_test_over, self.y_test_over = self.X_test, self.y_test

    def show_test(self):
        print("X_train: ", self.X_train.shape)
        print("y_train: ", self.y_train.shape)
        print("X_test: ", self.X_test.shape)
        print("y_test: ", self.y_test.shape)
        print('............')
        print("X_train_over: ", self.X_train_over.shape)
        print("y_train_over: ", self.y_train_over.shape)
        print("X_test_over: ", self.X_test_over.shape)
        print("y_test_over: ", self.y_test_over.shape)
