#from unittest.mock import inplace

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
import random
import pickle
from tqdm import tqdm


def main():
    ds_full = pd.read_pickle('pd_splits/dist_emb_msc.pkl')
    #print(ds_full.columns)

    X, y = ds_full.drop(['MedianBill'], axis=1), ds_full['MedianBill']
    X.drop(['AverageBill', 'TruncatedAverageBill'], axis=1, inplace=True)
    #print(y.iloc[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=59)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    scaler = StandardScaler()
    numeric_cols = [col for col in X_train.columns if (X_train[col].dtype in ['int64', 'float64', 'int32', 'float32'])]

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # print(np.array([X[col].to_numpy() for col in X.columns if 'emb' in col]).shape)
    # embs_values = np.array([X[col].to_numpy() for col in X.columns if 'emb' in col])
    # print(min(embs_values.min(axis=1)), max(embs_values.max(axis=1)))

    model = CatBoostRegressor()
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost MedianBill WITH aggregated embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    regressor = TabPFNRegressor()
    regressor.fit(X_train, y_train)

    preds = regressor.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print('Tabpfn MedianBill WITH aggregated embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    X, y = ds_full.drop(['MedianBill'], axis=1), ds_full['MedianBill']
    X.drop(['AverageBill', 'TruncatedAverageBill'], axis=1, inplace=True)
    columns_to_drop = [col for col in X.columns if 'emb' in col]
    X.drop(columns_to_drop, axis=1, inplace=True)
    # print(y.iloc[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=59)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    scaler = StandardScaler()
    numeric_cols = [col for col in X_train.columns if (X_train[col].dtype in ['int64', 'float64', 'int32', 'float32'])]

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = CatBoostRegressor()
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost MedianBill WITHOUT aggregated embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    regressor = TabPFNRegressor()
    regressor.fit(X_train, y_train)

    preds = regressor.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print('Tabpfn MedianBill WITHOUT aggregated embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
