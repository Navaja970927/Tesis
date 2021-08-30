def DAE_BPNN(ip):
    DAE_BPNNModel = []

    DAE_BPNNModel.append(('DAE_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    DAE_BPNNModel.append(('DAE_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    DAE_BPNNModel.append(('DAE_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    DAE_BPNNModel.append(('DAE_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    DAE_BPNNModel.append(('DAE_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return DAE_BPNNModel


def DAE_BPNNImbalanced(ip):
    DAE_BPNNModel = []
    DAE_BPNNModel.append(('DAE_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return DAE_BPNNModel


def DAE_BPNNUnderSample(ip):
    DAE_BPNNModel = []
    DAE_BPNNModel.append(('DAE_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return DAE_BPNNModel


def DAE_BPNNOverSample(ip):
    DAE_BPNNModel = []
    DAE_BPNNModel.append(('DAE_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return DAE_BPNNModel


def DAE_BPNNSMOTE(ip):
    DAE_BPNNModel = []
    DAE_BPNNModel.append(('DAE_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return DAE_BPNNModel


def DAE_BPNNADASYN(ip):
    DAE_BPNNModel = []
    DAE_BPNNModel.append(('DAE_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return DAE_BPNNModel


def DAE_AE(ip):
    DAE_AEModel = []

    DAE_AEModel.append(('DAE_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    DAE_AEModel.append(('DAE_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    DAE_AEModel.append(('DAE_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    DAE_AEModel.append(('DAE_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    DAE_AEModel.append(('DAE_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return DAE_AEModel


def DAE_AEImbalanced(ip):
    DAE_AEModel = []
    DAE_AEModel.append(('DAE_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return DAE_AEModel


def DAE_AEUnderSample(ip):
    DAE_AEModel = []
    DAE_AEModel.append(('DAE_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return DAE_AEModel


def DAE_AEOverSample(ip):
    DAE_AEModel = []
    DAE_AEModel.append(('DAE_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return DAE_AEModel


def DAE_AESMOTE(ip):
    DAE_AEModel = []
    DAE_AEModel.append(('DAE_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return DAE_AEModel


def DAE_AEADASYN(ip):
    DAE_AEModel = []
    DAE_AEModel.append(('DAE_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return DAE_AEModel
