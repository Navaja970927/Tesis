from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from sklearn.neighbors import KNeighborsClassifier


def KNN(ip):

    KNNmodel = []

    KNNmodel.append(('KNN IMBALANCE', KNeighborsClassifier(), ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    KNNmodel.append(('KNN UNDERSAMPLE', KNeighborsClassifier(), ip.X_train_under, ip.y_train_under, ip.X_test_under,
                     ip.y_test_under))
    KNNmodel.append(('KNN OVERSAMPLE', KNeighborsClassifier(), ip.X_train_over, ip.y_train_over, ip.X_test_over,
                     ip.y_test_over))
    KNNmodel.append(('KNN SMOTE', KNeighborsClassifier(), ip.X_train_smote, ip.y_train_smote, ip.X_test_smote,
                     ip.y_test_smote))
    KNNmodel.append(('KNN ADASYN', KNeighborsClassifier(), ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn,
                     ip.y_test_adasyn))
    return KNNmodel


def KNNImbalanced(ip):
    KNNmodel = []
    KNNmodel.append(('KNN IMBALANCE', KNeighborsClassifier(), ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return KNNmodel


def KNNUnderSample(ip):
    KNNmodel = []
    KNNmodel.append(('KNN UNDERSAMPLE', KNeighborsClassifier(), ip.X_train_under, ip.y_train_under, ip.X_test_under,
                     ip.y_test_under))
    return KNNmodel


def KNNOverSample(ip):
    KNNmodel = []
    KNNmodel.append(('KNN OVERSAMPLE', KNeighborsClassifier(), ip.X_train_over, ip.y_train_over, ip.X_test_over,
                     ip.y_test_over))
    return KNNmodel


def KNNSMOTE(ip):
    KNNmodel = []
    KNNmodel.append(('KNN SMOTE', KNeighborsClassifier(), ip.X_train_smote, ip.y_train_smote, ip.X_test_smote,
                     ip.y_test_smote))
    return KNNmodel


def KNNADASYN(ip):
    KNNmodel = []
    KNNmodel.append(('KNN ADASYN', KNeighborsClassifier(), ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn,
                     ip.y_test_adasyn))
    return KNNmodel
