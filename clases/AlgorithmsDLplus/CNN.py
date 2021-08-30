
def CNN_AE(ip):
    CNN_AEModel = []

    CNN_AEModel.append(('CNN_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    CNN_AEModel.append(('CNN_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    CNN_AEModel.append(('CNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    CNN_AEModel.append(('CNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    CNN_AEModel.append(('CNN_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return CNN_AEModel


def CNN_AEImbalanced(ip):
    CNN_AEModel = []
    CNN_AEModel.append(('CNN_AE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return CNN_AEModel


def CNN_AEUnderSample(ip):
    CNN_AEModel = []
    CNN_AEModel.append(('CNN_AE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return CNN_AEModel


def CNN_AEOverSample(ip):
    CNN_AEModel = []
    CNN_AEModel.append(('CNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return CNN_AEModel


def CNN_AESMOTE(ip):
    CNN_AEModel = []
    CNN_AEModel.append(('CNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return CNN_AEModel


def CNN_AEADASYN(ip):
    CNN_AEModel = []
    CNN_AEModel.append(('CNN_AE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return CNN_AEModel


def CNN_BPNN(ip):
    CNN_BPNNModel = []

    CNN_BPNNModel.append(('CNN_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    CNN_BPNNModel.append(('CNN_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    CNN_BPNNModel.append(('CNN_BPNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    CNN_BPNNModel.append(('CNN_BPNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    CNN_BPNNModel.append(('CNN_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return CNN_BPNNModel


def CNN_BPNNImbalanced(ip):
    CNN_BPNNModel = []
    CNN_BPNNModel.append(('CNN_BPNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return CNN_BPNNModel


def CNN_BPNNUnderSample(ip):
    CNN_BPNNModel = []
    CNN_BPNNModel.append(('CNN_BPNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return CNN_BPNNModel


def CNN_BPNNOverSample(ip):
    CNN_BPNNModel = []
    CNN_BPNNModel.append(('CNN_AE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return CNN_BPNNModel


def CNN_BPNNSMOTE(ip):
    CNN_BPNNModel = []
    CNN_BPNNModel.append(('CNN_AE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return CNN_BPNNModel


def CNN_BPNNADASYN(ip):
    CNN_BPNNModel = []
    CNN_BPNNModel.append(('CNN_BPNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return CNN_BPNNModel


def CNN_DAE(ip):
    CNN_DAEModel = []

    CNN_DAEModel.append(('CNN_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    CNN_DAEModel.append(('CNN_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    CNN_DAEModel.append(('CNN_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    CNN_DAEModel.append(('CNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    CNN_DAEModel.append(('CNN_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return CNN_DAEModel


def CNN_DAEImbalanced(ip):
    CNN_DAEModel = []
    CNN_DAEModel.append(('CNN_DAE IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return CNN_DAEModel


def CNN_DAEUnderSample(ip):
    CNN_DAEModel = []
    CNN_DAEModel.append(('CNN_DAE UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return CNN_DAEModel


def CNN_DAEOverSample(ip):
    CNN_DAEModel = []
    CNN_DAEModel.append(('CNN_DAE OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return CNN_DAEModel


def CNN_DAESMOTE(ip):
    CNN_DAEModel = []
    CNN_DAEModel.append(('CNN_DAE SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return CNN_DAEModel


def CNN_DAEADASYN(ip):
    CNN_DAEModel = []
    CNN_DAEModel.append(('CNN_DAE ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return CNN_DAEModel


def CNN_LSTM(ip):
    CNN_LSTMModel = []

    CNN_LSTMModel.append(('CNN_LSTM IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    CNN_LSTMModel.append(('CNN_LSTM UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    CNN_LSTMModel.append(('CNN_LSTM OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    CNN_LSTMModel.append(('CNN_LSTM SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    CNN_LSTMModel.append(('CNN_LSTM ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return CNN_LSTMModel


def CNN_LSTMImbalanced(ip):
    CNN_LSTMModel = []
    CNN_LSTMModel.append(('CNN_LSTM IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return CNN_LSTMModel


def CNN_LSTMUnderSample(ip):
    CNN_LSTMModel = []
    CNN_LSTMModel.append(('CNN_LSTM UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return CNN_LSTMModel


def CNN_LSTMOverSample(ip):
    CNN_LSTMModel = []
    CNN_LSTMModel.append(('CNN_LSTM OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return CNN_LSTMModel


def CNN_LSTMSMOTE(ip):
    CNN_LSTMModel = []
    CNN_LSTMModel.append(('CNN_LSTM SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return CNN_LSTMModel


def CNN_LSTMADASYN(ip):
    CNN_LSTMModel = []
    CNN_LSTMModel.append(('CNN_LSTM ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return CNN_LSTMModel


def CNN_RNN(ip):
    CNN_RNNModel = []

    CNN_RNNModel.append(('CNN_RNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    CNN_RNNModel.append(('CNN_RNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    CNN_RNNModel.append(('CNN_RNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    CNN_RNNModel.append(('CNN_RNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    CNN_RNNModel.append(('CNN_RNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return CNN_RNNModel


def CNN_RNNImbalanced(ip):
    CNN_RNNModel = []
    CNN_RNNModel.append(('CNN_RNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return CNN_RNNModel


def CNN_RNNUnderSample(ip):
    CNN_RNNModel = []
    CNN_RNNModel.append(('CNN_RNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return CNN_RNNModel


def CNN_RNNOverSample(ip):
    CNN_RNNModel = []
    CNN_RNNModel.append(('CNN_RNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return CNN_RNNModel


def CNN_RNNSMOTE(ip):
    CNN_RNNModel = []
    CNN_RNNModel.append(('CNN_RNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return CNN_RNNModel


def CNN_RNNADASYN(ip):
    CNN_RNNModel = []
    CNN_RNNModel.append(('CNN_RNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return CNN_RNNModel