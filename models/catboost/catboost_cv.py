import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from catboost import CatBoostRegressor


def main():

    X_train = pd.read_pickle('pd_splits/x_train_msk_merged_2.pkl')
    X_test = pd.read_pickle('pd_splits/x_test_msk_merged_2.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged_2.pkl').to_numpy()
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged_2.pkl').to_numpy()

    X_full = pd.concat([X_train, X_test], axis=0)
    #X_full['level_0'] = X_full['level_0'].astype(str)
    y_full = np.concatenate([y_train, y_test], axis=0)
    assert len(X_full) == len(y_full)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)
    scoring = {
        'neg_mse': 'neg_mean_squared_error',
        'r2': 'r2',
    }

    model = CatBoostRegressor(ignored_features=['index', 'level_0'], verbose=500)
    results = {}
    for metric_name, metric in scoring.items():
        scores = cross_val_score(
            model,
            X_full,
            y_full,
            cv=cv,
            scoring=metric,
            n_jobs=4
        )
        results[metric_name] = scores
        print(f"{metric_name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")


if __name__ == '__main__':
    main()
