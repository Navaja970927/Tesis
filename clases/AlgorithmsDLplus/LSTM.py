def LSTM_AE(ip):
    LSTM_AEModel = []

    LSTM_AEModel.append(('LSTM_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    LSTM_AEModel.append(('LSTM_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    LSTM_AEModel.append(('LSTM_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    LSTM_AEModel.append(('LSTM_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    LSTM_AEModel.append(('LSTM_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return LSTM_AEModel


def LSTM_AEImbalanced(ip):
    LSTM_AEModel = []
    LSTM_AEModel.append(('LSTM_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return LSTM_AEModel


def LSTM_AEUnderSample(ip):
    LSTM_AEModel = []
    LSTM_AEModel.append(('LSTM_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return LSTM_AEModel


def LSTM_AEOverSample(ip):
    LSTM_AEModel = []
    LSTM_AEModel.append(('LSTM_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return LSTM_AEModel


def LSTM_AESMOTE(ip):
    LSTM_AEModel = []
    LSTM_AEModel.append(('LSTM_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return LSTM_AEModel


def LSTM_AEADASYN(ip):
    LSTM_AEModel = []
    LSTM_AEModel.append(('LSTM_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return LSTM_AEModel


def LSTM_DAE(ip):
    LSTM_DAEModel = []

    LSTM_DAEModel.append(('LSTM_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    LSTM_DAEModel.append(('LSTM_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    LSTM_DAEModel.append(('LSTM_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    LSTM_DAEModel.append(('LSTM_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    LSTM_DAEModel.append(('LSTM_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return LSTM_DAEModel


def LSTM_DAEImbalanced(ip):
    LSTM_DAEModel = []
    LSTM_DAEModel.append(('LSTM_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return LSTM_DAEModel


def LSTM_DAEUnderSample(ip):
    LSTM_DAEModel = []
    LSTM_DAEModel.append(('LSTM_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return LSTM_DAEModel


def LSTM_DAEOverSample(ip):
    LSTM_DAEModel = []
    LSTM_DAEModel.append(('LSTM_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return LSTM_DAEModel


def LSTM_DAESMOTE(ip):
    LSTM_DAEModel = []
    LSTM_DAEModel.append(('LSTM_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return LSTM_DAEModel


def LSTM_DAEADASYN(ip):
    LSTM_DAEModel = []
    LSTM_DAEModel.append(('LSTM_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return LSTM_DAEModel


def LSTM_BPNN(ip):
    LSTM_BPNNModel = []

    LSTM_BPNNModel.append(('LSTM_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    LSTM_BPNNModel.append(('LSTM_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    LSTM_BPNNModel.append(('LSTM_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    LSTM_BPNNModel.append(('LSTM_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    LSTM_BPNNModel.append(('LSTM_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return LSTM_BPNNModel


def LSTM_BPNNImbalanced(ip):
    LSTM_BPNNModel = []
    LSTM_BPNNModel.append(('LSTM_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return LSTM_BPNNModel


def LSTM_BPNNUnderSample(ip):
    LSTM_BPNNModel = []
    LSTM_BPNNModel.append(('LSTM_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return LSTM_BPNNModel


def LSTM_BPNNOverSample(ip):
    LSTM_BPNNModel = []
    LSTM_BPNNModel.append(('LSTM_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return LSTM_BPNNModel


def LSTM_BPNNSMOTE(ip):
    LSTM_BPNNModel = []
    LSTM_BPNNModel.append(('LSTM_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return LSTM_BPNNModel


def LSTM_BPNNADASYN(ip):
    LSTM_BPNNModel = []
    LSTM_BPNNModel.append(('LSTM_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return LSTM_BPNNModel
