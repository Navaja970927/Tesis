def RNN_AE(ip):
    RNN_AEModel = []

    RNN_AEModel.append(('RNN_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    RNN_AEModel.append(('RNN_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    RNN_AEModel.append(('RNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    RNN_AEModel.append(('RNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    RNN_AEModel.append(('RNN_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return RNN_AEModel


def RNN_AEImbalanced(ip):
    RNN_AEModel = []
    RNN_AEModel.append(('RNN_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return RNN_AEModel


def RNN_AEUnderSample(ip):
    RNN_AEModel = []
    RNN_AEModel.append(('RNN_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return RNN_AEModel


def RNN_AEOverSample(ip):
    RNN_AEModel = []
    RNN_AEModel.append(('RNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return RNN_AEModel


def RNN_AESMOTE(ip):
    RNN_AEModel = []
    RNN_AEModel.append(('RNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return RNN_AEModel


def RNN_AEADASYN(ip):
    RNN_AEModel = []
    RNN_AEModel.append(('RNN_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return RNN_AEModel


def RNN_DAE(ip):
    RNN_DAEModel = []

    RNN_DAEModel.append(('RNN_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    RNN_DAEModel.append(('RNN_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    RNN_DAEModel.append(('RNN_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    RNN_DAEModel.append(('RNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    RNN_DAEModel.append(('RNN_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return RNN_DAEModel


def RNN_DAEImbalanced(ip):
    RNN_DAEModel = []
    RNN_DAEModel.append(('RNN_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return RNN_DAEModel


def RNN_DAEUnderSample(ip):
    RNN_DAEModel = []
    RNN_DAEModel.append(('RNN_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return RNN_DAEModel


def RNN_DAEOverSample(ip):
    RNN_DAEModel = []
    RNN_DAEModel.append(('RNN_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return RNN_DAEModel


def RNN_DAESMOTE(ip):
    RNN_DAEModel = []
    RNN_DAEModel.append(('RNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return RNN_DAEModel


def RNN_DAEADASYN(ip):
    RNN_DAEModel = []
    RNN_DAEModel.append(('RNN_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return RNN_DAEModel


def RNN_BPNN(ip):
    RNN_BPNNModel = []

    RNN_BPNNModel.append(('RNN_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    RNN_BPNNModel.append(('RNN_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    RNN_BPNNModel.append(('RNN_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    RNN_BPNNModel.append(('RNN_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    RNN_BPNNModel.append(('RNN_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return RNN_BPNNModel


def RNN_BPNNImbalanced(ip):
    RNN_BPNNModel = []
    RNN_BPNNModel.append(('RNN_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return RNN_BPNNModel


def RNN_BPNNUnderSample(ip):
    RNN_BPNNModel = []
    RNN_BPNNModel.append(('RNN_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return RNN_BPNNModel


def RNN_BPNNOverSample(ip):
    RNN_BPNNModel = []
    RNN_BPNNModel.append(('RNN_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return RNN_BPNNModel


def RNN_BPNNSMOTE(ip):
    RNN_BPNNModel = []
    RNN_BPNNModel.append(('RNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return RNN_BPNNModel


def RNN_BPNNADASYN(ip):
    RNN_BPNNModel = []
    RNN_BPNNModel.append(('RNN_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return RNN_BPNNModel
