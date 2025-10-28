from copyreg import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor


def main():
    to_take_train_rows = 10000
    X_train = pd.read_pickle('pd_splits/x_train_msk_merged.pkl')[:to_take_train_rows]
    X_test = pd.read_pickle('pd_splits/x_test_msk_merged.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()[:to_take_train_rows]
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()

    X_train.drop(['index', 'level_0'], axis=1, inplace=True)
    X_test.drop(['index', 'level_0'], axis=1, inplace=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(len(X_train), len(X_test), len(y_train), len(y_test))

    model = CatBoostRegressor()
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost NO embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    X_train = pd.read_pickle('tabpfn_embeds/X_train_mosc_wembs.pickle')[:to_take_train_rows]
    X_test = pd.read_pickle('tabpfn_embeds/X_test_mosc_wembs.pickle')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()[:to_take_train_rows]
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()

    X_train.drop(['index', 'level_0'], axis=1, inplace=True)
    X_test.drop(['index', 'level_0'], axis=1, inplace=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = CatBoostRegressor()
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost WITH embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    X_train = pd.read_pickle('tabpfn_embeds/X_train_mosc_wembs.pickle')[:to_take_train_rows]
    X_test = pd.read_pickle('tabpfn_embeds/X_test_mosc_wembs.pickle')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()[:to_take_train_rows]
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()

    columns_to_drop = [col for col in X_train.columns if 'emb' not in col]
    X_train.drop(columns_to_drop, axis=1, inplace=True)
    X_test.drop(columns_to_drop, axis=1, inplace=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = CatBoostRegressor()
    model.fit(X_train, y_train, verbose=100)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print('Catboost ONLY embeds')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
