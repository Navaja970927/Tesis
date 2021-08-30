def BPNN_CNN(ip):
    BPNN_CNNModel = []

    BPNN_CNNModel.append(('BPNN_CNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    BPNN_CNNModel.append(('BPNN_CNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    BPNN_CNNModel.append(('BPNN_CNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    BPNN_CNNModel.append(('BPNN_CNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    BPNN_CNNModel.append(('BPNN_CNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return BPNN_CNNModel


def BPNN_CNNImbalanced(ip):
    BPNN_CNNModel = []
    BPNN_CNNModel.append(('BPNN_CNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return BPNN_CNNModel


def BPNN_CNNUnderSample(ip):
    BPNN_CNNModel = []
    BPNN_CNNModel.append(('BPNN_CNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return BPNN_CNNModel


def BPNN_CNNOverSample(ip):
    BPNN_CNNModel = []
    BPNN_CNNModel.append(('BPNN_CNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return BPNN_CNNModel


def BPNN_CNNSMOTE(ip):
    BPNN_CNNModel = []
    BPNN_CNNModel.append(('BPNN_CNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return BPNN_CNNModel


def BPNN_CNNADASYN(ip):
    BPNN_CNNModel = []
    BPNN_CNNModel.append(('BPNN_CNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return BPNN_CNNModel


def BPNN_AE(ip):
    BPNN_AEModel = []

    BPNN_AEModel.append(('BPNN_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    BPNN_AEModel.append(('BPNN_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    BPNN_AEModel.append(('BPNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    BPNN_AEModel.append(('BPNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    BPNN_AEModel.append(('BPNN_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return BPNN_AEModel


def BPNN_AEImbalanced(ip):
    BPNN_AEModel = []
    BPNN_AEModel.append(('BPNN_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return BPNN_AEModel


def BPNN_AEUnderSample(ip):
    BPNN_AEModel = []
    BPNN_AEModel.append(('BPNN_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return BPNN_AEModel


def BPNN_AEOverSample(ip):
    BPNN_AEModel = []
    BPNN_AEModel.append(('BPNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return BPNN_AEModel


def BPNN_AESMOTE(ip):
    BPNN_AEModel = []
    BPNN_AEModel.append(('BPNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return BPNN_AEModel


def BPNN_AEADASYN(ip):
    BPNN_AEModel = []
    BPNN_AEModel.append(('BPNN_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return BPNN_AEModel


def BPNN_DAE(ip):
    BPNN_DAEModel = []

    BPNN_DAEModel.append(('BPNN_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    BPNN_DAEModel.append(('BPNN_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    BPNN_DAEModel.append(('BPNN_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    BPNN_DAEModel.append(('BPNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    BPNN_DAEModel.append(('BPNN_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return BPNN_DAEModel


def BPNN_DAEImbalanced(ip):
    BPNN_DAEModel = []
    BPNN_DAEModel.append(('BPNN_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return BPNN_DAEModel


def BPNN_DAEUnderSample(ip):
    BPNN_DAEModel = []
    BPNN_DAEModel.append(('BPNN_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return BPNN_DAEModel


def BPNN_DAEOverSample(ip):
    BPNN_DAEModel = []
    BPNN_DAEModel.append(('BPNN_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return BPNN_DAEModel


def BPNN_DAESMOTE(ip):
    BPNN_DAEModel = []
    BPNN_DAEModel.append(('BPNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return BPNN_DAEModel


def BPNN_DAEADASYN(ip):
    BPNN_DAEModel = []
    BPNN_DAEModel.append(('BPNN_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return BPNN_DAEModel
