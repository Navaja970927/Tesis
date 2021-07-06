from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from sklearn.ensemble import RandomForestClassifier


def RF(ip):

    RFmodel = []

    RFmodel.append(('RF IMABALANCED', RandomForestClassifier(), ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    RFmodel.append(
        ('RF UNDERSAMPLE', RandomForestClassifier(), ip.X_train_under, ip.y_train_under, ip.X_test_under,
         ip.y_test_under))
    RFmodel.append(('RF OVERSAMPLE', RandomForestClassifier(), ip.X_train_over, ip.y_train_over, ip.X_test_over,
                    ip.y_test_over))
    RFmodel.append(('RF SMOTE', RandomForestClassifier(), ip.X_train_smote, ip.y_train_smote, ip.X_test_smote,
                    ip.y_test_smote))
    RFmodel.append(
        ('RF ADASYN', RandomForestClassifier(), ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return RFmodel
