import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
from pytorch_widedeep.models import SAINT, WideDeep
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.training import Trainer
from pytorch_widedeep.callbacks import LRHistory, EarlyStopping


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train = pd.read_pickle('pd_splits_0606/x_train_ekb_merged.pkl')
    X_test = pd.read_pickle('pd_splits_0606/x_test_ekb_merged.pkl')
    y_train = pd.read_pickle('pd_splits_0606/y_train_ekb_merged.pkl').to_numpy()
    y_test = pd.read_pickle('pd_splits_0606/y_test_ekb_merged.pkl').to_numpy()

    # data isn't perfect, mb fix later)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_train.mean(), inplace=True)
    target_scaler = MinMaxScaler(feature_range=(0, 10))
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print("Scaled mean/std:", y_train_scaled.mean(), y_train_scaled.std())

    column_idx = {k: v for v, k in enumerate(X_train.columns)}

    preprocessor = TabPreprocessor(
        continuous_cols=X_train.columns.tolist(),
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # print("NaNs in X_train:", np.isnan(X_train_processed).any())
    # print("Infs in X_train:", np.isinf(X_train_processed).any())
    # print("NaNs in y_train:", np.isnan(y_train).any())
    # print("Infs in y_train:", np.isinf(y_train).any())
    #
    # # Check value ranges
    # print("X_train min/max:", X_train_processed.min(), X_train_processed.max())
    # print("y_train min/max:", y_train.min(), y_train.max())

    saint = SAINT(
        column_idx=column_idx,
        continuous_cols=X_train.columns.tolist(),
        input_dim=64,
        n_heads=8,
        n_blocks=4,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_dims=[128, 64, 32, 1],
        mlp_activation="relu",
    )

    model = WideDeep(deeptabular=saint)
    model.to(device)
    print('model initialized')

    n_epochs=50
    callbacks = [
        LRHistory(n_epochs=n_epochs),
        EarlyStopping(patience=5, monitor='train_loss'),
    ]

    trainer = Trainer(
        model=model,
        objective="regression",
        lr=1e-4,
        device=device,
        clip_grad_norm=1.0,
        callbacks=callbacks,
    )

    print('trainer fit started')
    trainer.fit(
        X_train={"X_tab": X_train_processed, "target": y_train_scaled},
        X_test={"X_tab": X_test_processed, "target": y_test_scaled},
        n_epochs=n_epochs,
        batch_size=32
    )
    print('trainer fit ended')

    torch.save(model.state_dict(), 'saint_model_spb.pth')

    model.eval()
    preds_scaled = trainer.predict(X_tab=X_test_processed, batch_size=32)
    y_pred = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")

    r2 = r2_score(y_test, y_pred)
    print(f"Test R²: {r2:.4f}")
