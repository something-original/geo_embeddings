import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from catboost import CatBoostRegressor
from scipy import stats


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():

    X_train = pd.read_pickle('pd_splits/x_train_msk_merged_2.pkl')
    X_test = pd.read_pickle('pd_splits/x_test_msk_merged_2.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged_2.pkl').to_numpy()
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged_2.pkl').to_numpy()

    X_full = pd.concat([X_train, X_test], axis=0)
    X_full['level_0'] = X_full['level_0'].astype(str)
    y_full = np.concatenate([y_train, y_test], axis=0)
    assert len(X_full) == len(y_full)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.10, random_state=59)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    model = CatBoostRegressor(ignored_features=['index', 'level_0'], verbose=500)
    model.fit(X_train, y_train)

    predictions_before = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions_before)
    errors_before = np.abs(y_test - predictions_before)
    print(errors_before[:10])
    r2 = r2_score(y_test, predictions_before)

    print('Catboost Msk WITHOUT Raster Embeds:')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    folder_path = 'map_embeds/maps_emb_msc'
    embeddings_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_key = os.path.splitext(file_name)[0]
            embeddings_dict[file_key] = np.load(os.path.join(folder_path, file_name))

    X_train = pd.read_pickle('pd_splits/x_train_msk_merged_2.pkl')
    X_test = pd.read_pickle('pd_splits/x_test_msk_merged_2.pkl')
    y_train = pd.read_pickle('pd_splits/y_train_msk_merged_2.pkl').to_numpy()
    y_test = pd.read_pickle('pd_splits/y_test_msk_merged_2.pkl').to_numpy()

    X_full = pd.concat([X_train, X_test], axis=0)
    X_full['level_0'] = X_full['level_0'].astype(str)
    X_full['embedding'] = X_full['level_0'].apply(lambda x: embeddings_dict.get(str(x), None))
    y_full = np.concatenate([y_train, y_test], axis=0)

    missing_mask = X_full['embedding'].isnull()
    X_full = X_full[~missing_mask].copy()
    y_full = y_full[~missing_mask]

    assert len(X_full) == len(y_full)
    print(len(missing_mask))

    embeddings_matrix = np.vstack(X_full['embedding'].values)
    embeddings_df = pd.DataFrame(embeddings_matrix,
                                 columns=[f'emb_rastr_{i}' for i in range(embeddings_matrix.shape[1])],
                                 index=X_full.index)

    X_full = X_full.drop(columns=['embedding'])
    X_full = pd.concat([X_full, embeddings_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.10, random_state=59)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    model = CatBoostRegressor(ignored_features=['index', 'level_0'], verbose=500)
    model.fit(X_train, y_train)

    predictions_after = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions_after)
    errors_after = np.abs(y_test - predictions_after)
    print(errors_after[:10])
    r2 = r2_score(y_test, predictions_after)

    print('Catboost Msk WITH Raster Embeds:')
    print("Root Mean Squared Error (RMSE):", mse ** 0.5)
    print("R² Score:", r2)

    t_stat, p_value = stats.ttest_rel(errors_before, errors_after)
    print(f"p-value = {p_value:.8f}")

    if p_value < 0.05:
        print("Улучшение статистически значимо (p < 0.05)")
    else:
        print("Улучшение не значимо (p >= 0.05)")

    baseline_error = np.mean(np.abs(y_test - np.median(y_test)))
    print(f"Ошибка бейзлайна: {baseline_error:.3f}")
    print(f"Медианная ошибка до: {np.median(errors_before):.3f}")
    print(f"Медианная ошибка после: {np.median(errors_after):.3f}")
    print(f"MAPE до: {mape(y_test, predictions_before):.2f}%")
    print(f"MAPE после: {mape(y_test, predictions_after):.2f}%")


if __name__ == '__main__':
    main()