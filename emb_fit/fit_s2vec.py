from utils import get_dataloader
from models import S2VecModel

import os
from pathlib import Path

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler


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
