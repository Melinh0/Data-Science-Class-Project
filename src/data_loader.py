
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .config import DATA_RAW, DATE_COLS

ARQUIVOS = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

DATASET_KAGGLE = "olistbr/brazilian-ecommerce"


def _baixar_kaggle() -> Path:
    import kagglehub

    src = Path(kagglehub.dataset_download(DATASET_KAGGLE))
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS.values():
        destino = DATA_RAW / nome
        if not destino.exists():
            shutil.copy(src / nome, destino)
    return DATA_RAW


def _resolver_dir_dados() -> Path:
    if all((DATA_RAW / n).exists() for n in ARQUIVOS.values()):
        return DATA_RAW
    print("[data] data/raw incompleto -> baixando do Kaggle...")
    return _baixar_kaggle()


def carregar_dados() -> dict[str, pd.DataFrame]:
    base = _resolver_dir_dados()
    dfs: dict[str, pd.DataFrame] = {}

    for chave, arquivo in ARQUIVOS.items():
        caminho = base / arquivo
        dtype = None
        if chave in ("customers", "geolocation", "sellers"):
            col_zip = {
                "customers": "customer_zip_code_prefix",
                "geolocation": "geolocation_zip_code_prefix",
                "sellers": "seller_zip_code_prefix",
            }[chave]
            dtype = {col_zip: "string"}
        kwargs: dict = {"dtype": dtype}
        if chave == "category_translation":
            kwargs["encoding"] = "utf-8-sig"

        df = pd.read_csv(caminho, **kwargs)

        if chave == "orders":
            for col in DATE_COLS:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        if chave == "reviews":
            for col in ("review_creation_date", "review_answer_timestamp"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if chave == "items":
            df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")

        dfs[chave] = df
        print(f"[data] {arquivo}: {len(df):,} linhas x {df.shape[1]} colunas")

    return dfs
