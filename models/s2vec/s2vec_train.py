import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from model import S2VecModel


torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TabularImageDataset(Dataset):
    """
    Датасет для преобразования табличных данных в 'изображения' для S2Vec.
    """
    def __init__(self, csv_path, img_size=32, fill_value=-1.0, sep=';'):
        """
        Args:
            csv_path: Путь к CSV файлу.
            img_size: Размер стороны квадратного 'изображения' (H и W). 
                      Общее кол-во признаков должно быть <= img_size**2.
            fill_value: Значение, которым заполняются NaN и недостающие признаки.
        """
        self.img_size = img_size
        self.fill_value = fill_value
        self.total_pixels = img_size * img_size
        self.sep = sep

        self.df = pd.read_csv(csv_path, sep=self.sep)
        self.features = self.df.values.astype(np.float32)

        n_features = self.features.shape[1]
        if n_features > self.total_pixels:
            raise ValueError(f"Too many features ({n_features}) for image {img_size}x{img_size}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        row = self.features[idx]

        row = np.nan_to_num(
            row,
            nan=self.fill_value,
            posinf=self.fill_value,
            neginf=self.fill_value
        )

        n_features = len(row)
        if n_features < self.total_pixels:
            padding = np.full((self.total_pixels - n_features,), self.fill_value, dtype=np.float32)
            row = np.concatenate([row, padding])

        image = row.reshape(1, self.img_size, self.img_size)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(0)


def get_dataloader(csv_path, img_size=32, batch_size=64, num_workers=4, shuffle=False):
    dataset = TabularImageDataset(csv_path, img_size=img_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def prepare_and_save_dataset(
    dataset_path: str,
    features_to_drop: list[str],
    train_path: str,
    val_path: str,
    csv_sep: str = ';',
    id_col: str | None = None,
    use_scaler: bool = False,
    scaler: StandardScaler | None = None,
    test_size: float = 0.25,
    nan_fill_threshold: float = 0.9,
) -> None:

    df_full = pd.read_csv(dataset_path, sep=csv_sep)
    logger.info(f'Full dataset shape: {df_full.shape}')
    mask_all_nan = df_full.isna().all(axis=1)

    if id_col:
        dropped_ids = df_full.loc[mask_all_nan, id_col].tolist()
        logger.info(f'Dropped ids: {dropped_ids}')

    df_full.drop(columns=features_to_drop, inplace=True, errors='ignore')
    df_full = df_full[~mask_all_nan].reset_index(drop=True)    
    df_full = df_full.select_dtypes(include=[np.number])
    logger.info(f'Shape after dropping: {df_full.shape}')

    df_full = df_full.replace([np.inf, -np.inf], np.nan)

    logger.info(f'Nan threshold: {nan_fill_threshold}')
    nan_ratio = df_full.isnull().mean()

    cols_to_drop = nan_ratio[nan_ratio > nan_fill_threshold].index
    df_full = df_full.drop(columns=cols_to_drop)
    if cols_to_drop.any():
        print(f"Columns dropped with nan ratios: {list(cols_to_drop)}")

    df_train, df_val = train_test_split(df_full, test_size=test_size, random_state=42)
    logger.info(f'Train shape: {df_train.shape}')
    logger.info(f'Val shape: {df_val.shape}')

    if use_scaler and scaler is not None:
        df_train_for_scaler = df_train.fillna(df_train.median())
        df_val_for_scaler = df_val.fillna(df_val.median())

        train_scaled = pd.DataFrame(
            scaler.fit_transform(df_train_for_scaler), columns=df_train.columns
        )
        val_scaled = pd.DataFrame(
            scaler.transform(df_val_for_scaler), columns=df_val.columns
        )

        train_scaled_final = np.where(df_train.isnull(), -1.0, train_scaled.values)
        val_scaled_final = np.where(df_val.isnull(), -1.0, val_scaled.values)

        df_train = pd.DataFrame(train_scaled_final)
        df_val = pd.DataFrame(val_scaled_final)

    df_train.to_csv(train_path, sep=csv_sep, index=False)
    df_val.to_csv(val_path, sep=csv_sep, index=False)


if __name__ == '__main__':

    root_dir = Path(__file__).resolve().parents[2]
    path_parts = [root_dir, 'datasets', 'mun_data']

    dataset_path = os.path.join(*path_parts, 'indicator_values.csv')
    train_path = os.path.join(*path_parts, 'indicator_values_train.csv')
    val_path = os.path.join(*path_parts, 'indicator_values_val.csv')

    features_to_drop = ['municipality_id']
    scaler = StandardScaler()

    prepare_and_save_dataset(
        dataset_path=dataset_path,
        features_to_drop=features_to_drop,
        train_path=train_path,
        val_path=val_path,
        csv_sep=';',
        use_scaler=True,
        scaler=scaler,
        test_size=0.25,
    )

    IMG_SIZE = 128
    PATCH_SIZE = 16
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    IN_CH = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    train_loader = get_dataloader(
        train_path,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = get_dataloader(
        val_path,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    model = S2VecModel(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_ch=IN_CH,
        num_heads=8,
        encoder_layers=6,
        decoder_layers=2,
        embed_dim=256,
        decoder_dim=128,
        mask_ratio=0.75,
        lr=LEARNING_RATE
    )

    # 3. Настройка Trainer (PyTorch Lightning)
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=os.path.join(root_dir, 'models', 's2vec', 'checkpoints'),
        filename='s2vec-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    early_stopping = EarlyStopping(
        monitor='validation_loss',
        patience=5,
        mode='min',
        verbose=True,
        check_finite=True,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        precision='32-true',
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    model = model.to(DEVICE)
    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for batch in train_loader:
            imgs, _ = batch
            imgs = imgs.to(DEVICE)

            latent, _, _ = model.encode(imgs)
            cls_embeddings = latent[:, 0, :] 

            embeddings_list.append(cls_embeddings.cpu().numpy())

    final_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Shape of resulting embeddings: {final_embeddings.shape}")

    np.save(os.path.join(root_dir, 'emb_fit', 's2vec', 's2vec_embs.npy'), final_embeddings)
