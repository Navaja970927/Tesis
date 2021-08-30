from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from sklearn.linear_model import LogisticRegression


def LR(ip):

    LRmodel = []

    LRmodel.append(
        ('LR IMBALANCED', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train, ip.y_train,
         ip.X_test, ip.y_test))
    LRmodel.append(('LR UNDERSAMPLE', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_under,
                    ip.y_train_under, ip.X_test_under, ip.y_test_under))
    LRmodel.append(('LR OVERSAMPLE', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_over,
                    ip.y_train_over, ip.X_test_over, ip.y_test_over))
    LRmodel.append(('LR SMOTE', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_smote,
                    ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    LRmodel.append(('LR ADASYN ', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_adasyn,
                    ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return LRmodel


def LRImbalanced(ip):
    LRmodel = []
    LRmodel.append(
        ('LR IMBALANCED', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train, ip.y_train,
         ip.X_test, ip.y_test))
    return LRmodel


def LRUnderSample(ip):
    LRmodel = []
    LRmodel.append(('LR UNDERSAMPLE', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_under,
                    ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return LRmodel


def LROverSample(ip):
    LRmodel = []
    LRmodel.append(('LR OVERSAMPLE', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_over,
                    ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return LRmodel


def LRSMOTE(ip):
    LRmodel = []
    LRmodel.append(('LR SMOTE', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_smote,
                    ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return LRmodel


def LRADASYN(ip):
    LRmodel = []
    LRmodel.append(('LR ADASYN ', LogisticRegression(solver='saga', multi_class='multinomial'), ip.X_train_adasyn,
                    ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return LRmodel
