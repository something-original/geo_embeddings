from utils import get_dataloader
from models import S2VecModel

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def train_s2vec(
    train_path: str,
    val_path: str,
    checkpoint_path: str,
    device: str,
) -> S2VecModel:

    IMG_SIZE = 128
    PATCH_SIZE = 16
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    IN_CH = 1

    print(f"Using device: {device}")

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

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=checkpoint_path,
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
        accelerator='gpu' if device == 'cuda' else 'cpu',
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

    return model


def get_s2vec_embeddings(
    model: S2VecModel,
    loader: DataLoader,
    embs_save_path: str,
    device: str,
) -> None:

    model = model.to(device)
    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for batch in loader:
            imgs, _ = batch
            imgs = imgs.to(device)

            latent, _, _ = model.encode(imgs)
            cls_embeddings = latent[:, 0, :]

            embeddings_list.append(cls_embeddings.cpu().numpy())

    final_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Shape of resulting embeddings: {final_embeddings.shape}")

    np.save(embs_save_path, final_embeddings)
