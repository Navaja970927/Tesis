from clases.ImbalancedSolution.RandomUnderSampleClass import RamdomUnderSampleClass
from clases.ImbalancedSolution.RandomOverSampleClass import RamdomOverSampleClass
from clases.ImbalancedSolution.SMOTEClass import SMOTEClass
from clases.ImbalancedSolution.ADASYNClass import ADASYNClass
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd


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

    def performance(self, model):
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
            self.f1_score_tests.append(F1Score_test)

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
