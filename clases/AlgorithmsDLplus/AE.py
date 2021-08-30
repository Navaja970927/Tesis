def AE_BPNN(ip):
    AE_BPNNModel = []

    AE_BPNNModel.append(('AE_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    AE_BPNNModel.append(('AE_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    AE_BPNNModel.append(('AE_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    AE_BPNNModel.append(('AE_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    AE_BPNNModel.append(('AE_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return AE_BPNNModel


def AE_BPNNImbalanced(ip):
    AE_BPNNModel = []
    AE_BPNNModel.append(('AE_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return AE_BPNNModel


def AE_BPNNUnderSample(ip):
    AE_BPNNModel = []
    AE_BPNNModel.append(('AE_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return AE_BPNNModel


def AE_BPNNOverSample(ip):
    AE_BPNNModel = []
    AE_BPNNModel.append(('AE_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return AE_BPNNModel


def AE_BPNNSMOTE(ip):
    AE_BPNNModel = []
    AE_BPNNModel.append(('AE_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return AE_BPNNModel


def AE_BPNNADASYN(ip):
    AE_BPNNModel = []
    AE_BPNNModel.append(('AE_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return AE_BPNNModel


def AE_DAE(ip):
    AE_DAEModel = []

    AE_DAEModel.append(('AE_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    AE_DAEModel.append(('AE_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    AE_DAEModel.append(('AE_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    AE_DAEModel.append(('AE_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    AE_DAEModel.append(('AE_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return AE_DAEModel


def AE_DAEImbalanced(ip):
    AE_DAEModel = []
    AE_DAEModel.append(('AE_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return AE_DAEModel


def AE_DAEUnderSample(ip):
    AE_DAEModel = []
    AE_DAEModel.append(('AE_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return AE_DAEModel


def AE_DAEOverSample(ip):
    AE_DAEModel = []
    AE_DAEModel.append(('AE_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return AE_DAEModel


def AE_DAESMOTE(ip):
    AE_DAEModel = []
    AE_DAEModel.append(('AE_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return AE_DAEModel


def AE_DAEADASYN(ip):
    AE_DAEModel = []
    AE_DAEModel.append(('AE_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return AE_DAEModel
