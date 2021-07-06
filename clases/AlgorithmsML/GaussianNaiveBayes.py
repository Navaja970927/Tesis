from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from sklearn.naive_bayes import GaussianNB


def GNB(ip):

    NBmodel = []

    NBmodel.append(('NB IMBALANCED', GaussianNB(), ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    NBmodel.append(('NB UNDERSAMPLE', GaussianNB(), ip.X_train_under, ip.y_train_under, ip.X_test_under,
                    ip.y_test_under))
    NBmodel.append(('NB OVERSAMPLE', GaussianNB(), ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    NBmodel.append(('NB SMOTE', GaussianNB(), ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    NBmodel.append(('NB ADASYN', GaussianNB(), ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn,
                    ip.y_test_adasyn))

    return NBmodel
