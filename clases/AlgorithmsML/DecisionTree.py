from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from sklearn.tree import DecisionTreeClassifier


def DT(ip):

    DTmodel = []

    DTmodel.append(('DT IMBALANCED', DecisionTreeClassifier(), ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    DTmodel.append(
        ('DT UNDERSAMPLE', DecisionTreeClassifier(), ip.X_train_under, ip.y_train_under, ip.X_test_under,
         ip.y_test_under))
    DTmodel.append(('DT OVERSAMPLE', DecisionTreeClassifier(), ip.X_train_over, ip.y_train_over, ip.X_test_over,
                    ip.y_test_over))
    DTmodel.append(('DT SMOTE', DecisionTreeClassifier(), ip.X_train_smote, ip.y_train_smote, ip.X_test_smote,
                    ip.y_test_smote))
    DTmodel.append(
        ('DT ADASYN', DecisionTreeClassifier(), ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn,
         ip.y_test_adasyn))

    return DTmodel
