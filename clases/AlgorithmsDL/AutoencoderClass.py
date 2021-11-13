def AE(ip):

    AEModel = []

    AEModel.append(('AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    AEModel.append(('AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    AEModel.append(('AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    AEModel.append(('AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    AEModel.append(('AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return AEModel


def AEImbalanced(ip):
    AEModel = []
    AEModel.append(('AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return AEModel


def AEUnderSample(ip):
    AEModel = []
    AEModel.append(('AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return AEModel


def AEOverSample(ip):
    AEModel = []
    AEModel.append(('AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return AEModel


def AESMOTE(ip):
    AEModel = []
    AEModel.append(('AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return AEModel


def AEADASYN(ip):
    AEModel = []
    AEModel.append(('AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return AEModel
