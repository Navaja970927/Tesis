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


def RFImbalanced(ip):
    RFmodel = []
    RFmodel.append(('RF IMABALANCED', RandomForestClassifier(), ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return RFmodel


def RFUnderSample(ip):
    RFmodel = []
    RFmodel.append(
        ('RF UNDERSAMPLE', RandomForestClassifier(), ip.X_train_under, ip.y_train_under, ip.X_test_under,
         ip.y_test_under))
    return RFmodel


def RFOverSample(ip):
    RFmodel = []
    RFmodel.append(('RF OVERSAMPLE', RandomForestClassifier(), ip.X_train_over, ip.y_train_over, ip.X_test_over,
                    ip.y_test_over))
    return RFmodel


def RFSMOTE(ip):
    RFmodel = []
    RFmodel.append(('RF SMOTE', RandomForestClassifier(), ip.X_train_smote, ip.y_train_smote, ip.X_test_smote,
                    ip.y_test_smote))
    return RFmodel


def RFADASYN(ip):
    RFmodel = []
    RFmodel.append(
        ('RF ADASYN', RandomForestClassifier(), ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return RFmodel
