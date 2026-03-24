import asyncio
import aiohttp
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


async def download_all_files(base_download_url, mun_datasets_df):
    tasks = []
    timeout = aiohttp.ClientTimeout(total=600)
    semaphore = asyncio.Semaphore(10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for section_id, dataset_code in zip(
            mun_datasets_df['Код раздела'],
            mun_datasets_df['Код показателя']
        ):
            task = asyncio.create_task(
                download_one_file(session, semaphore, base_download_url, str(section_id), str(dataset_code))
            )
            tasks.append(task)
        await asyncio.gather(*tasks)


async def download_one_file(session, semaphore, base_download_url, section_id, dataset_code):
    async with semaphore:
        download_url = base_download_url.replace("<section_id>", section_id)
        download_url = download_url.replace("<dataset_code>", dataset_code)
        print(f"URL: {download_url}")

        root_dir = Path(__file__).resolve().parents[2]
        save_folder_path = os.path.join(root_dir, "datasets", "mun_data")
        os.makedirs(save_folder_path, exist_ok=True)
        save_file_path = os.path.join(save_folder_path, f"{dataset_code}.zip")

        async with session.get(download_url) as response:
            with open(save_file_path, 'wb') as f:
                f.write(await response.read())


def main():
    load_dotenv()
    base_download_url = os.getenv("TOCHNO_ST_BASE_LINK")

    root_dir = Path(__file__).resolve().parents[2]
    mun_datasets_df_path = os.path.join(root_dir, "datasets", "mun_datasets_metadata.csv")
    mun_datasets_df = pd.read_csv(mun_datasets_df_path, sep=';')

    asyncio.run(download_all_files(base_download_url, mun_datasets_df))


if __name__ == '__main__':
    main()
