

def LSTM(ip):
    LSTMModel = []

    LSTMModel.append(('LSTM IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    LSTMModel.append(('LSTM UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    LSTMModel.append(('LSTM OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    LSTMModel.append(('LSTM SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    LSTMModel.append(('LSTM ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return LSTMModel


def LSTMImbalanced(ip):
    LSTMModel = []
    LSTMModel.append(('LSTM IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return LSTMModel


def LSTMUnderSample(ip):
    LSTMModel = []
    LSTMModel.append(('LSTM UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return LSTMModel


def LSTMOverSample(ip):
    LSTMModel = []
    LSTMModel.append(('LSTM OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return LSTMModel


def LSTMSMOTE(ip):
    LSTMModel = []
    LSTMModel.append(('LSTM SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return LSTMModel


def LSTMADASYN(ip):
    LSTMModel = []
    LSTMModel.append(('LSTM ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return LSTMModel