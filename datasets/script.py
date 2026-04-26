import pandas as pd

kwargs = {'sep': ',', 'encoding': 'utf-8'}

msk = pd.read_csv(r'D:\itmo\geo_embeddings\datasets\merged_df_emb_msc.csv', **kwargs)
spb = pd.read_csv(r'D:\itmo\geo_embeddings\datasets\merged_df_emb_spb.csv', **kwargs)
ekb = pd.read_csv(r'D:\itmo\geo_embeddings\datasets\merged_df_emb_ekb.csv', **kwargs)

debug = 1