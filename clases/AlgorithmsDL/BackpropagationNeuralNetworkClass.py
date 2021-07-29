

def BNN(ip):
    BNNModel = []

    BNNModel.append(('BNN IMBALANCE', ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    BNNModel.append(('BNN UNDERSAMPLE', ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    BNNModel.append(('BNN OVERSAMPLE', ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    BNNModel.append(('BNN SMOTE', ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    BNNModel.append(('BNN ADASYN', ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return BNNModel

