from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE,
)
MIN_FREQ_OHE = 30
def pipeline_preprocess(
    escalar_numerico: bool = True,
    numericas: list[str] | None = None,
    categoricas: list[str] | None = None,
    etapa_extra_numerica: tuple[str, object] | None = None,
) -> ColumnTransformer:
    numericas = NUMERIC_FEATURES if numericas is None else numericas
    categoricas = CATEGORICAL_FEATURES if categoricas is None else categoricas
    passos_num = [("imputer", SimpleImputer(strategy="median"))]
    if escalar_numerico:
        passos_num.append(("scaler", StandardScaler()))
    if etapa_extra_numerica is not None:
        passos_num.append(etapa_extra_numerica)
    blocos = []
    if numericas:
        blocos.append(("num", Pipeline(passos_num), numericas))
    if categoricas:
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_FREQ_OHE,
                                  sparse_output=False)),
        ])
        blocos.append(("cat", cat, categoricas))
    return ColumnTransformer(blocos, remainder="drop")
def nomes_saida(pre: ColumnTransformer) -> list[str]:
    return [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
def features_do_modelo() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
def dividir(df: pd.DataFrame):
    X = df[features_do_modelo()].copy()
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
def dividir_temporal(df: pd.DataFrame, coluna: str = "order_purchase_timestamp"):
    ordenado = df.sort_values(coluna)
    corte = int(len(ordenado) * (1 - TEST_SIZE))
    tr, te = ordenado.iloc[:corte], ordenado.iloc[corte:]
    cols = features_do_modelo()
    return tr[cols], te[cols], tr[TARGET].astype(int), te[TARGET].astype(int)
def aplicar_preprocess(df: pd.DataFrame, escalar_numerico: bool = True) -> pd.DataFrame:
    pre = pipeline_preprocess(escalar_numerico)
    Xt = pre.fit_transform(df[features_do_modelo()])
    return pd.DataFrame(Xt, columns=nomes_saida(pre), index=df.index)
