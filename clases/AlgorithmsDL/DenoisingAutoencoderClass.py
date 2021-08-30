

def DAE(ip):

    DAEModel = []

    DAEModel.append(('DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    DAEModel.append(('DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    DAEModel.append(('DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    DAEModel.append(('DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    DAEModel.append(('DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return DAEModel


def DAEImbalanced(ip):
    DAEModel = []
    DAEModel.append(('DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return DAEModel


def DAEUnderSample(ip):
    DAEModel = []
    DAEModel.append(('DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return DAEModel


def DAEOverSample(ip):
    DAEModel = []
    DAEModel.append(('DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return DAEModel


def DAESMOTE(ip):
    DAEModel = []
    DAEModel.append(('DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return DAEModel


def DAEADASYN(ip):
    DAEModel = []
    DAEModel.append(('DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return DAEModel
