import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabpfn import TabPFNRegressor
import random
import pickle
from tqdm import tqdm

def inverse_scale(preds, scaler):
    preds_log = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    return 10**preds_log - 1


def main():
    # Load data, tabpfn can't train (officially) on more than 10k examples
    X_train = pd.read_pickle('pd_splits/x_train_spb_merged.pkl')
    X_test = pd.read_pickle('pd_splits/x_test_spb_merged.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_spb_merged.pkl').to_numpy()
    y_test = pd.read_pickle('pd_splits/y_test_spb_merged.pkl').to_numpy()
    X_train.drop(['level_0', ], axis=1, inplace=True)
    X_test.drop(['level_0', ], axis=1, inplace=True)

    print(X_train.columns)
    print(len(X_train), len(y_train), len(X_test), len(y_test))

    y_train_log = np.log10(y_train+1)
    target_scaler = MinMaxScaler(feature_range=(0, 10))

    y_train_scaled = target_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()

    # Initialize the regressor
    regressor = TabPFNRegressor(ignore_pretraining_limits=True)
    print('Fit started')
    regressor.fit(X_train.iloc[:10000], y_train_scaled[:10000])
    print('Fit completed')

    def get_embeddings_in_batches(model, data, batch_size=10000):
        embeddings = []
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.iloc[i:i + batch_size]
            batch_embeds = model.get_embeddings(batch)
            embeddings.append(batch_embeds.mean(axis=0).squeeze())
        return np.vstack(embeddings)

    # print('Calculating train embeddings...')
    # train_embeddings_avg = get_embeddings_in_batches(regressor, X_train)
    #
    # print('Calculating test embeddings...')
    # test_embeddings_avg = get_embeddings_in_batches(regressor, X_test)
    #
    # emb_cols = [f'emb_{i}' for i in range(train_embeddings_avg.shape[1])]
    # train_emb_df = pd.DataFrame(train_embeddings_avg, columns=emb_cols, index=X_train.index)
    # test_emb_df = pd.DataFrame(test_embeddings_avg, columns=emb_cols, index=X_test.index)
    #
    # X_train_with_emb = pd.concat([X_train, train_emb_df], axis=1)
    # X_test_with_emb = pd.concat([X_test, test_emb_df], axis=1)
    #
    # X_train_with_emb.to_pickle('X_train_mosc_wembs.pickle')
    # X_test_with_emb.to_pickle('X_test_mosc_wembs.pickle')

    # Getting embeddings of input
    # print('Embeddings calculation')
    # embs = regressor.get_embeddings(X_test)
    # print(len(X_test), type(embs), embs.shape)  # 3107 -> (8, 3107, 192)

    # Predict on the test set
    print('Predictions started')
    preds = regressor.predict(X_test)
    preds_unscaled = inverse_scale(preds, target_scaler)
    print('Predictions completed')

    # Evaluate
    mse = mean_squared_error(y_test, preds_unscaled)
    r2 = r2_score(y_test, preds_unscaled)

    print("Root Mean Squared Error (RMSE):", mse**0.5)
    print("R² Score:", r2)


if __name__ == '__main__':
    main()
