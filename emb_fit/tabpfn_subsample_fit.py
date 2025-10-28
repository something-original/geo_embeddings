import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
import random
import pickle
from tqdm import tqdm
import warnings


def inverse_scale(preds, scaler):
    preds_log = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    return 10**preds_log - 1


def main():

    X_train = pd.read_pickle('pd_splits/x_train_msk_merged.pkl')
    X_test = pd.read_pickle('pd_splits/x_test_msk_merged.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged.pkl').to_numpy()
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged.pkl').to_numpy()
    X_train.drop(['level_0', 'index'], axis=1, inplace=True)
    X_test.drop(['level_0', 'index'], axis=1, inplace=True)

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    y_train_log = np.log10(y_train + 1)
    target_scaler = MinMaxScaler(feature_range=(0, 10))
    y_train_scaled = target_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()

    tabpfn_subsample_reg = TabPFNRegressor(
        ignore_pretraining_limits=True,  # (bool) Enables handling datasets beyond pretraining constraints.
        n_estimators=32,  # (int) Number of estimators in the ensemble for robustness.
        inference_config={
            "SUBSAMPLE_SAMPLES": 10000  # (int) Controls sample subsampling per inference to avoid OOM errors.
        },
    )

    print('RandomForestTabpfn fit started')
    tabpfn_subsample_reg.fit(X_train, y_train_scaled)

    # # Predict on the test set
    # print('Predictions started')
    # preds = tabpfn_subsample_reg.predict(X_test)
    # preds_unscaled = inverse_scale(preds, target_scaler)
    # print('Predictions completed')
    #
    # # Evaluate
    # mse = mean_squared_error(y_test, preds_unscaled)
    # r2 = r2_score(y_test, preds_unscaled)
    #
    # print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    # print("R² Score:", r2)

    def get_embeddings_in_batches(model, data, batch_size=5000):
        embeddings = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i + batch_size]
            batch_embeds = model.get_embeddings(batch)
            embeddings.append(batch_embeds.mean(axis=0).squeeze())
        return np.vstack(embeddings)

    print('Calculating train embeddings...')
    train_embeddings_avg = get_embeddings_in_batches(tabpfn_subsample_reg, X_train)

    print('Calculating test embeddings...')
    test_embeddings_avg = get_embeddings_in_batches(tabpfn_subsample_reg, X_test)

    print(train_embeddings_avg.shape, test_embeddings_avg.shape)

    emb_cols = [f'emb_{i}' for i in range(train_embeddings_avg.shape[1])]
    train_emb_df = pd.DataFrame(train_embeddings_avg, columns=emb_cols, index=X_train.index)
    test_emb_df = pd.DataFrame(test_embeddings_avg, columns=emb_cols, index=X_test.index)

    X_train_with_emb = pd.concat([X_train, train_emb_df], axis=1)
    X_test_with_emb = pd.concat([X_test, test_emb_df], axis=1)

    X_train_with_emb.to_pickle('X_train_mosc_subs_wembs.pickle')
    X_test_with_emb.to_pickle('X_test_mosc_subs_wembs.pickle')








if __name__ == '__main__':
    main()

