"""
基线模型--lightgbm
"""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
import gc


def lgb_model(params=None,
              cv=None,
              X=None,
              y=None,
              columns=None,
              early_stopping_rounds=200,
              categorical_feature=None):

    folds = cv
    if not columns:
        columns = list(X.columns)
    splits = folds.split(X[columns], y)
    y_oof = np.zeros(X.shape[0])
    score = 0

    clfs = []
    for fold_n, (train_index, valid_index) in enumerate(splits):
        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[
            valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='auc',
                verbose=100,
                early_stopping_rounds=early_stopping_rounds,
                categorical_feature=categorical_feature)

        y_pred_valid = clf.predict_proba(X_valid)[:, 1]
        y_oof[valid_index] = y_pred_valid
        print(
            f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

        score += roc_auc_score(y_valid, y_pred_valid) / 5

        del X_train, X_valid, y_train, y_valid
        gc.collect()
        clfs.append(clf)

    print(f"\nMean AUC = {score}")
    print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
    return clfs