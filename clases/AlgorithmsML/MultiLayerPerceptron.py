from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from sklearn.neural_network import MLPClassifier


def MLP(ip):

    MLPclassifier = []

    MLPclassifier.append(('MLPClassifier IMBALANCE', MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),
                          ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    MLPclassifier.append(('MLPClassifier UNDERSAMPLE', MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),
                          ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    MLPclassifier.append(('MLPClassifier OVERSAMPLE', MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),
                          ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    MLPclassifier.append((
                         'MLPClassifier SMOTE', MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),
                         ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    MLPclassifier.append(('MLPClassifier  ADASYN', MLPClassifier(hidden_layer_sizes=(200,), max_iter=10000),
                          ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return MLPclassifier
