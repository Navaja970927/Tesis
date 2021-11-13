from xgboost import XGBClassifier


def XGB(ip):

    xgBOOST = []
    xgBOOST.append(('XGBOOST IMBALANCED', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    xgBOOST.append(('XGBOOST UNDERSAMPLE', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    xgBOOST.append(('XGBOOST OVERSAMPLE', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    xgBOOST.append(('XGBOOST SMOTE', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    xgBOOST.append(('XGBOOST ADASYN ', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))

    return xgBOOST


def XGBImbalanced(ip):
    xgBOOST = []
    xgBOOST.append(('XGBOOST IMBALANCED', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train, ip.y_train, ip.X_test, ip.y_test))
    return xgBOOST


def XGBUnderSample(ip):
    xgBOOST = []
    xgBOOST.append(('XGBOOST UNDERSAMPLE', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_under, ip.y_train_under, ip.X_test_under, ip.y_test_under))
    return xgBOOST


def XGBOverSample(ip):
    xgBOOST = []
    xgBOOST.append(('XGBOOST OVERSAMPLE', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_over, ip.y_train_over, ip.X_test_over, ip.y_test_over))
    return xgBOOST


def XGBSMOTE(ip):
    xgBOOST = []
    xgBOOST.append(('XGBOOST SMOTE', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_smote, ip.y_train_smote, ip.X_test_smote, ip.y_test_smote))
    return xgBOOST


def XGBADASYN(ip):
    xgBOOST = []
    xgBOOST.append(('XGBOOST ADASYN ', XGBClassifier(n_estimators=1000, verbosity=1, scale_pos_weight=580),
                    ip.X_train_adasyn, ip.y_train_adasyn, ip.X_test_adasyn, ip.y_test_adasyn))
    return xgBOOST
