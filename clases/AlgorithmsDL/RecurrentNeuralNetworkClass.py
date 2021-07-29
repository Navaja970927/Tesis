


def RNN(ip):
    RNNModel = []

    RNNModel.append(('RNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    RNNModel.append(('RNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    RNNModel.append(('RNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    RNNModel.append(('RNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    RNNModel.append(('RNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return RNNModel