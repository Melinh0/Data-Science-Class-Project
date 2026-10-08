
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANALYSIS_CSV, DATE_COLS, STATUS_CSV, TARGET
from .config import SCORE_COL, NEGATIVE_MAX
from .data_loader import carregar_dados

RAIO_TERRA_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dla, dlo = la2 - la1, lo2 - lo1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * RAIO_TERRA_KM * np.arcsin(np.sqrt(a))


def _agrupar_raras(s: pd.Series, min_n: int = 100, rotulo: str = "Outras categorias") -> pd.Series:
    contagem = s.value_counts(dropna=False)
    validos = set(contagem[contagem >= min_n].index)
    return s.where(s.isin(validos), rotulo)


def construir_bases() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = carregar_dados()

    orders, customers, items = d["orders"], d["customers"], d["items"]
    reviews, payments, products = d["reviews"], d["payments"], d["products"]
    sellers, geo, cat_tr = d["sellers"], d["geolocation"], d["category_translation"]

    reviews = reviews.dropna(subset=["order_id"]).copy()
    reviews["tem_comentario"] = (
        reviews["review_comment_title"].fillna("").str.strip().ne("")
        | reviews["review_comment_message"].fillna("").str.strip().ne("")
    )
    reviews["comentario_chars"] = (
        reviews["review_comment_title"].fillna("").str.len()
        + reviews["review_comment_message"].fillna("").str.len()
    )
    rev = (
        reviews.sort_values("review_creation_date")
        .drop_duplicates("order_id", keep="first")
        .set_index("order_id")[
            [SCORE_COL, "review_creation_date", "tem_comentario", "comentario_chars"]
        ]
    )

    cat_tr = cat_tr.rename(columns={"product_category_name_english": "categoria_en"})
    products = products.merge(
        cat_tr[["product_category_name", "categoria_en"]], on="product_category_name", how="left"
    )
    items = items.merge(
        products[["product_id", "categoria_en", "product_weight_g", "product_photos_qty"]],
        on="product_id", how="left",
    ).merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
        on="seller_id", how="left",
    ).rename(columns={"seller_zip_code_prefix": "seller_zip"})

    agg_itens = items.groupby("order_id").agg(
        n_itens=("order_item_id", "count"),
        preco_total=("price", "sum"),
        frete_total=("freight_value", "sum"),
    )
    principal = (
        items.loc[items.groupby("order_id")["price"].idxmax()]
        .set_index("order_id")[
            ["categoria_en", "product_weight_g", "product_photos_qty", "seller_zip", "seller_state"]
        ]
        .rename(
            columns={
                "categoria_en": "categoria",
                "product_weight_g": "peso_g",
                "product_photos_qty": "n_fotos",
                "seller_state": "uf_vendedor",
            }
        )
    )

    pag_princ = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")[["payment_type"]]
        .rename(columns={"payment_type": "pagamento_tipo"})
    )
    agg_pag = payments.groupby("order_id").agg(
        n_parcelas=("payment_installments", "max"),
        valor_pago=("payment_value", "sum"),
    )

    centroides = geo.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "median"),
        lng=("geolocation_lng", "median"),
    )

    customers = customers.rename(
        columns={
            "customer_state": "uf_cliente",
            "customer_city": "cidade_cliente",
            "customer_zip_code_prefix": "cep_cliente",
        }
    )

    base = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(agg_itens, on="order_id", how="left")
        .merge(principal, on="order_id", how="left")
        .merge(agg_pag, on="order_id", how="left")
        .merge(pag_princ, on="order_id", how="left")
        .merge(rev, on="order_id", how="left")
    )

    cli = centroides.rename(columns={"lat": "lat_cli", "lng": "lng_cli"})
    vend = centroides.rename(columns={"lat": "lat_vend", "lng": "lng_vend"})
    base = base.merge(
        cli, left_on="cep_cliente", right_index=True, how="left"
    ).merge(
        vend, left_on="seller_zip", right_index=True, how="left"
    )
    base["distancia_km"] = _haversine_km(
        base["lat_cli"].to_numpy(), base["lng_cli"].to_numpy(),
        base["lat_vend"].to_numpy(), base["lng_vend"].to_numpy(),
    )
    base = base.drop(columns=["lat_cli", "lng_cli", "lat_vend", "lng_vend"])

    base["atraso_dias"] = (
        (base["order_delivered_customer_date"] - base["order_estimated_delivery_date"])
        .dt.total_seconds() / 86400
    ).round(2)
    base["entrega_dias"] = (
        (base["order_delivered_customer_date"] - base["order_purchase_timestamp"])
        .dt.total_seconds() / 86400
    ).round(2)
    base["manuseio_dias"] = (
        (base["order_delivered_carrier_date"] - base["order_purchase_timestamp"])
        .dt.total_seconds() / 86400
    ).round(2)
    base["aprovacao_h"] = (
        (base["order_approved_at"] - base["order_purchase_timestamp"])
        .dt.total_seconds() / 3600
    ).round(2)

    base["valor_pedido"] = base["preco_total"] + base["frete_total"]
    base["frete_ratio"] = (base["frete_total"] / base["valor_pedido"]).round(4)
    base["preco_unitario_medio"] = (base["preco_total"] / base["n_itens"]).round(2)

    base["prazo_cumprido"] = base["atraso_dias"] <= 0
    base["faixa_atraso"] = pd.cut(
        base["atraso_dias"],
        bins=[-np.inf, -7, 0, 7, np.inf],
        labels=[
            "Adiantado > 7 dias",
            "No prazo (até 7d antes)",
            "Atraso 1-7 dias",
            "Atraso > 7 dias",
        ],
    ).astype("string")

    base["mes_compra"] = base["order_purchase_timestamp"].dt.strftime("%Y-%m")
    base["dia_semana_compra"] = base["order_purchase_timestamp"].dt.day_name()
    base["hora_compra"] = base["order_purchase_timestamp"].dt.hour

    base["categoria"] = base["categoria"].fillna("sem_categoria")
    base["categoria_grupo"] = _agrupar_raras(base["categoria"], min_n=100)

    base[SCORE_COL] = base[SCORE_COL].astype("float")
    base[TARGET] = (base[SCORE_COL] <= NEGATIVE_MAX).astype("float")
    base.loc[base[SCORE_COL].isna(), TARGET] = np.nan

    status_df = base[["order_id", "order_status", SCORE_COL]].copy()

    analise = base[
        (base["order_status"] == "delivered")
        & base["order_delivered_customer_date"].notna()
        & base[SCORE_COL].notna()
    ].copy()

    print(
        f"[features] base de análise: {len(analise):,} pedidos entregues com review "
        f"| insatisfeitos (nota<=2): {analise[TARGET].mean():.1%}"
    )
    return analise, status_df


def salvar_bases() -> None:
    analise, status = construir_bases()
    analise.to_csv(ANALYSIS_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
    status.to_csv(STATUS_CSV, index=False)
    print(f"[features] salvo -> {ANALYSIS_CSV}")
    print(f"[features] salvo -> {STATUS_CSV}")


def carregar_analise() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_CSV, parse_dates=DATE_COLS)
    df["faixa_atraso"] = df["faixa_atraso"].astype("string")
    return df
