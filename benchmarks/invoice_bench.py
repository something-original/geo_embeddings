import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from catboost import CatBoostRegressor
from scipy import stats


def main():
    # all flats columns
    # emb_0, emb_1, ... , emb_191
    # rast_0, rast_1, ... , rast_2559
    # new invoice cols:
    # district_id', 'KktCount', 'AverageBill', 'TruncatedAverageBill',
    #        'MedianBill', 'CachePayPercent', 'IntensityOfNumberBills',
    #        'RevenueIntensity', 'ReceiptTotalCount', 'apartment_count'],
    # redundant 'geometry', 'coordinates', 'polygon', 'Unnamed: 0', 'district_id'
    ds_full = pd.read_csv('pd_splits/flats_checks_raster.csv')
    print(len(ds_full))
    invoice_cols = ['districd_id', 'KktCount', 'AverageBill', 'TruncatedAverageBill', 'MedianBill', 'CachePayPercent',
                    'IntensityOfNumberBills', 'RevenueIntensity', 'ReceiptTotalCount', 'apartment_count']
    emb_cols_1 = [col for col in ds_full.columns if 'emb' in col]
    emb_cols_2 = [col for col in ds_full.columns if 'rast' in col]
    flat_cols = [col for col in ds_full.columns if col not in (invoice_cols+emb_cols_2+emb_cols_1)]
    ds_full.drop(['TruncatedAverageBill', 'MedianBill'], axis=1, inplace=True)
    X_full, y_full = ds_full.drop(['AverageBill'], axis=1), ds_full['AverageBill']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=59)
    all_errors_before = []
    all_errors_after = []

    model_before = CatBoostRegressor(verbose=1000)
    model_after = CatBoostRegressor(verbose=1000)

    for train_index, test_index in cv.split(X_full, y_full):

        # Wout Embeddings
        X_full_invoice_only = X_full.drop(emb_cols_1 + emb_cols_2 + flat_cols, axis=1)
        X_train, X_test = X_full_invoice_only.iloc[train_index], X_full_invoice_only.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

        model_before.fit(X_train, y_train)
        pred_before = model_before.predict(X_test)
        errors_before = np.abs(y_test - pred_before)


        # With Embeddings
        X_full_invoice_rasters = X_full.drop(emb_cols_2 + flat_cols, axis=1)
        X_train, X_test = X_full_invoice_rasters.iloc[train_index], X_full_invoice_rasters.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

        model_after.fit(X_train, y_train)
        pred_after = model_after.predict(X_test)
        errors_after = np.abs(y_test - pred_after)

        all_errors_before.extend(errors_before)
        all_errors_after.extend(errors_after)

    t_stat, p_value = stats.ttest_rel(all_errors_before, all_errors_after)
    print(f"p-value = {p_value:.8f}")

    # В случае с этими данными - ухудшение статистически значимо)
    if p_value < 0.05:
        print("Улучшение статистически значимо (p < 0.05)")
    else:
        print("Улучшение не значимо (p >= 0.05)")

    baseline_error = np.mean(np.abs(y_full - np.median(y_full)))
    print(f"Ошибка бейзлайна: {baseline_error:.3f}")
    print(f"Медианная ошибка до: {np.median(all_errors_before):.3f}")
    print(f"Медианная ошибка после: {np.median(all_errors_after):.3f}")



if __name__ == '__main__':
    main()