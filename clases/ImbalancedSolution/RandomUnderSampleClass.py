from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


class RamdomUnderSampleClass:

    def read_data(self, csv):
        self.df = pd.read_csv(csv)
        self.X = self.df.drop(['Class', 'Amount'], axis=1)
        self.y = self.df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                shuffle=True)

    def rus(self):
        self.rus = RandomUnderSampler(sampling_strategy='majority')
        self.X_train_under, self.y_train_under = self.rus.fit_resample(self.X_train, self.y_train)
        self.X_test_under, self.y_test_under = self.X_test, self.y_test

    def show_test(self):
        print("X_train: ", self.X_train.shape)
        print("y_train: ", self.y_train.shape)
        print("X_test: ", self.X_test.shape)
        print("y_test: ", self.y_test.shape)
        print('............')
        print("X_train_under: ", self.X_train_under.shape)
        print("y_train_under: ", self.y_train_under.shape)
        print("X_test_under: ", self.X_test_under.shape)
        print("y_test_under: ", self.y_test_under.shape)

    def performance(model):
        for name, model, X_train, y_train, X_test, y_test in model:
            # appending name
            names.append(name)

            # Build model
            model.fit(X_train, y_train)

            # predictions
            y_test_pred = model.predict(X_test)

            # calculate accuracy
            Accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
            accuracy_tests.append(Accuracy_test)

            # calculate auc
            Aucs_test = metrics.roc_auc_score(y_test, y_test_pred)
            aucs_tests.append(Aucs_test)

            # precision_calculation
            Precision_score_test = metrics.precision_score(y_test, y_test_pred)
            precision_tests.append(Precision_score_test)

            # calculate recall
            Recall_score_test = metrics.recall_score(y_test, y_test_pred)
            recall_tests.append(Recall_score_test)

            # calculating F1
            F1Score_test = metrics.f1_score(y_test, y_test_pred)
            f1_score_tests.append(F1Score_test)

            # draw confusion matrix
            cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
