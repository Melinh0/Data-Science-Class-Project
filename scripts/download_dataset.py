from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DATASET = "olistbr/brazilian-ecommerce"
URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

ARQUIVOS = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

RAIZ = Path(__file__).resolve().parents[1]


def faltando(destino: Path) -> list[str]:
    return [nome for nome in ARQUIVOS if not (destino / nome).exists()]


def baixar(destino: Path) -> None:
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "O pacote 'kagglehub' não está instalado. Instale com:\n"
            "    pip install kagglehub"
        )

    try:
        origem = Path(kagglehub.dataset_download(DATASET))
    except Exception as erro:
        sys.exit(
            f"Falha ao baixar '{DATASET}' do Kaggle: {erro}\n\n"
            "Se for um problema de credenciais, defina KAGGLE_USERNAME e KAGGLE_KEY ou crie\n"
            "~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).\n"
            f"Alternativa manual: baixe o zip em {URL} e extraia os CSVs em {destino}/"
        )

    ausentes_na_origem = [nome for nome in ARQUIVOS if not (origem / nome).exists()]
    if ausentes_na_origem:
        sys.exit(f"O download não contém os arquivos esperados: {ausentes_na_origem}")

    destino.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS:
        shutil.copy2(origem / nome, destino / nome)
        print(f"  copiado: {nome}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa o dataset Olist para data/raw/.")
    parser.add_argument("--dest", type=Path, default=RAIZ / "data" / "raw",
                        help="pasta de destino (padrão: data/raw/ na raiz do projeto)")
    parser.add_argument("--force", action="store_true",
                        help="baixa e sobrescreve mesmo que os arquivos já existam")
    args = parser.parse_args()

    destino: Path = args.dest
    if not args.force and not faltando(destino):
        print(f"Os {len(ARQUIVOS)} arquivos já estão em {destino}. Nada a fazer (use --force para baixar de novo).")
        return

    ausentes = faltando(destino)
    print(f"Baixando '{DATASET}' ({len(ausentes)} de {len(ARQUIVOS)} arquivos ausentes em {destino})...")
    baixar(destino)

    ainda_faltam = faltando(destino)
    if ainda_faltam:
        sys.exit(f"Ainda faltam arquivos em {destino}: {ainda_faltam}")
    print(f"Pronto: {len(ARQUIVOS)} arquivos em {destino}.")


if __name__ == "__main__":
    main()
