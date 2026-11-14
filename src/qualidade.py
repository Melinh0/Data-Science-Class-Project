from __future__ import annotations
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import (
    ANALYSIS_CSV, CATEGORICAL_FEATURES, DATE_COLS, DISTANCIA_MAX_KM, FIGURES,
    NUMERIC_CANDIDATAS, NUMERIC_FEATURES, PREP_DIR, PREP_JSON, REDUNDANTES, STATUS_CSV,
    TABLES, TARGET, rotulo,
)
from .features import CATEGORIAS_EQUIVALENTES
from .preprocess import dividir, features_do_modelo, nomes_saida, pipeline_preprocess
REGRAS_INVALIDAS = [
    ("manuseio_dias", "postagem anterior à compra",
     lambda d: d["manuseio_dias"] < 0),
    ("manuseio_dias", "postagem posterior à entrega ao cliente",
     lambda d: d["manuseio_dias"] > d["entrega_dias"]),
    ("entrega_dias", "entrega anterior ou igual à compra",
     lambda d: d["entrega_dias"] <= 0),
    ("aprovacao_h", "aprovação anterior à compra",
     lambda d: d["aprovacao_h"] < 0),
    ("n_parcelas", "pagamento com 0 parcelas (mínimo possível = 1)",
     lambda d: d["n_parcelas"] < 1),
    ("peso_g", "peso do produto ≤ 0 g",
     lambda d: d["peso_g"] <= 0),
    ("distancia_km", f"distância > {DISTANCIA_MAX_KM} km (fora do território brasileiro)",
     lambda d: d["distancia_km"] > DISTANCIA_MAX_KM),
    ("preco_total", "preço total ≤ 0",
     lambda d: d["preco_total"] <= 0),
    ("frete_total", "frete negativo",
     lambda d: d["frete_total"] < 0),
    ("n_itens", "pedido sem itens",
     lambda d: d["n_itens"] < 1),
]
VEREDITO_EXTREMOS = {
    "atraso_dias": ("legítimo", "atrasos longos coerentes com entrega_dias e com notas baixas"),
    "entrega_dias": ("legítimo", "entregas > 60 dias existem na operação e quase sempre estão atrasadas"),
    "manuseio_dias": ("legítimo na cauda positiva", "negativos já tratados como inválidos; cauda positiva = vendedor lento"),
    "aprovacao_h": ("legítimo", "boletos aprovados após vários dias"),
    "frete_total": ("legítimo", "itens pesados/volumosos e rotas longas; frete é coerente com peso e distância"),
    "preco_total": ("legítimo", "itens de alto valor existem no marketplace; preço = soma dos itens do pedido"),
    "frete_ratio": ("legítimo", "pedidos baratos com frete alto (razão alta) são esperados"),
    "n_itens": ("legítimo", "pedidos com muitos itens existem (máx. 21)"),
    "distancia_km": ("legítimo até o limite", "acima do limite geográfico = erro de coordenada (tratado)"),
    "n_parcelas": ("legítimo", "parcelamentos longos (até 24x) são possíveis"),
    "peso_g": ("legítimo", "móveis e eletrodomésticos pesados (até ~40 kg)"),
    "n_fotos": ("legítimo", "anúncios com muitas fotos"),
    "hora_compra": ("sem extremos", "variável limitada a 0-23"),
    "valor_pedido": ("legítimo (redundante)", "soma de preço e frete; extremos são pedidos caros e coerentes com preco_total"),
    "preco_unitario_medio": ("legítimo (redundante)", "preço médio dos itens; extremos são itens caros"),
}
def aplicar_regras(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col, _, cond in REGRAS_INVALIDAS:
        if col in d.columns:
            mascara = cond(d).fillna(False)
            d.loc[mascara, col] = np.nan
    return d
def _vif(X: pd.DataFrame) -> pd.Series:
    A = X.fillna(X.median()).to_numpy(dtype=float)
    A = (A - A.mean(axis=0)) / np.where(A.std(axis=0) == 0, 1, A.std(axis=0))
    out = {}
    for j, c in enumerate(X.columns):
        y = A[:, j]
        Xo = np.delete(A, j, axis=1)
        coef, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        r2 = 1 - ((y - Xo @ coef) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        out[c] = 1e6 if r2 >= 1 - 1e-9 else float(min(1 / (1 - r2), 1e6))
    return pd.Series(out, name="vif")
def _tukey_extremos(s: pd.Series) -> tuple[int, int]:
    x = s.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0, 0
    return int((x > q3 + 3 * iqr).sum()), int((x < q1 - 3 * iqr).sum())
def diagnosticar(bruto: pd.DataFrame, limpo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    linhas: list[dict] = []
    n = len(bruto)
    def add(cat, item, n_afetados, decisao, justificativa):
        linhas.append({
            "categoria": cat, "item": item,
            "n": np.nan if n_afetados is None else int(n_afetados),
            "pct": np.nan if n_afetados is None else round(100 * n_afetados / n, 3),
            "decisao": decisao, "justificativa": justificativa,
        })
    for col in NUMERIC_CANDIDATAS + CATEGORICAL_FEATURES:
        k = int(bruto[col].isna().sum())
        if k:
            if col in CATEGORICAL_FEATURES:
                dec, just = "imputar pela moda (treino)", "ausência pontual, categoria dominante"
            elif col == "n_fotos":
                dec, just = "imputar pela mediana (treino)", "produto sem cadastro de fotos; MCAR plausível"
            elif col == "distancia_km":
                dec, just = "imputar pela mediana (treino)", "CEP sem coordenadas na base de geolocalização"
            else:
                dec, just = "imputar pela mediana (treino)", "ausência residual (< 0,1%)"
            add("Ausentes", col, k, dec, just)
    dup_id = int(bruto["order_id"].duplicated().sum())
    add("Duplicatas", "order_id repetido", dup_id,
        "nenhuma ação" if dup_id == 0 else "remover",
        "granularidade = 1 pedido por linha (revisões duplicadas tratadas em features.py)")
    cols_dup = NUMERIC_CANDIDATAS + CATEGORICAL_FEATURES + [TARGET]
    dup_linha = int(bruto.duplicated(subset=cols_dup).sum())
    add("Duplicatas", "linhas idênticas nas features (sem ids)", dup_linha, "mantidas",
        "pedidos distintos (order_id únicos) podem coincidir em todas as features; sem evidência de duplicação")
    for antigo, novo in CATEGORIAS_EQUIVALENTES.items():
        add("Categorias equivalentes", f"{antigo} → {novo}", None,
            "unificadas em features.py",
            "versão '_2' da mesma família (hipótese de domínio)" if antigo.endswith("_2")
            else "erro de grafia no dicionário de categorias")
    for col, descricao, cond in REGRAS_INVALIDAS:
        k = int(cond(bruto).fillna(False).sum())
        add("Valores inválidos", f"{col}: {descricao}", k,
            "→ NaN e imputação pelo treino" if k else "nenhuma ocorrência",
            "incoerente com o domínio; não corrigível com a informação disponível")
    for col in NUMERIC_CANDIDATAS:
        sup, inf = _tukey_extremos(limpo[col])
        if sup + inf == 0:
            continue
        ver, just = VEREDITO_EXTREMOS.get(col, ("a investigar", ""))
        extremos_sup = limpo[col] > limpo[col].quantile(0.99)
        pct_neg = limpo.loc[extremos_sup, TARGET].mean() * 100
        add("Valores extremos", f"{col} (> 3×IQR: {sup} sup. / {inf} inf.; "
            f"{pct_neg:.0f}% de 1-2 no top 1%)", sup + inf,
            f"mantidos ({ver})", just)
    for col in bruto.columns:
        if bruto[col].nunique(dropna=True) <= 1:
            add("Constantes / identificadores", f"{col} (constante)", n, "descartada",
                "sem variância após o recorte analítico")
    for col in ("order_id", "customer_id", "customer_unique_id", "cep_cliente", "seller_zip",
                "cidade_cliente"):
        if col in bruto.columns:
            add("Constantes / identificadores", f"{col} (identificador)", n, "descartada",
                "não generaliza; risco de memorização")
    num = limpo[NUMERIC_CANDIDATAS]
    corr = num.corr()
    pares = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    )
    pares.columns = ["variavel_1", "variavel_2", "r"]
    pares = pares[pares["r"].abs() >= 0.8].sort_values("r", key=np.abs, ascending=False)
    for _, r in pares.iterrows():
        add("Redundância", f"{r['variavel_1']} × {r['variavel_2']} (r = {r['r']:.3f})", None,
            "ver decisão abaixo", "correlação linear muito alta")
    ident = {
        "valor_pedido = preco_total + frete_total":
            float((limpo["valor_pedido"] - limpo["preco_total"] - limpo["frete_total"]).abs().max()),
        "preco_unitario_medio = preco_total / n_itens":
            float((limpo["preco_unitario_medio"] - limpo["preco_total"] / limpo["n_itens"]).abs().max()),
    }
    for expr, desvio in ident.items():
        add("Redundância", f"identidade: {expr} (desvio máx. = {desvio:.2f})", None, "variável derivada removida",
            "contém exatamente a mesma informação das variáveis originais")
    for col, motivo in REDUNDANTES.items():
        add("Redundância", f"remoção de {col}", None, "descartada do modelo", motivo)
    vif_antes = _vif(num)
    vif_depois = _vif(limpo[NUMERIC_FEATURES])
    vif = pd.concat([vif_antes.rename("vif_antes"), vif_depois.rename("vif_depois")], axis=1)
    vif.index.name = "variavel"
    vif.reset_index().to_csv(TABLES / "tab17_vif.csv", index=False)
    info = {
        "vif_max_antes": float(vif_antes.max()),
        "vif_max_depois": float(vif_depois.max()),
        "pares_correlacionados": pares.round(4).to_dict("records"),
        "identidades": ident,
    }
    return pd.DataFrame(linhas), info
def executar_etapa_prep() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes da etapa 'prep'.")
    bruto = pd.read_csv(ANALYSIS_CSV, parse_dates=DATE_COLS)
    limpo = aplicar_regras(bruto)
    print(f"[prep] base: {len(bruto):,} pedidos | {bruto.shape[1]} colunas")
    diag, info_diag = diagnosticar(bruto, limpo)
    diag.to_csv(TABLES / "tab15_diagnostico_qualidade.csv", index=False)
    print(f"[prep] tabela -> tab15_diagnostico_qualidade.csv ({len(diag)} verificações)")
    inval = diag[diag["categoria"] == "Valores inválidos"]
    for _, r in inval[inval["n"] > 0].iterrows():
        print(f"[prep]   inválido: {r['item']} -> {int(r['n']):,}")
    inval_pos = inval[inval["n"] > 0].sort_values("n")
    if not inval_pos.empty:
        rotulos = []
        for item in inval_pos["item"]:
            if ":" in item and item.split(":", 1)[0].strip() in NUMERIC_CANDIDATAS:
                col, desc = item.split(":", 1)
                rotulos.append(f"{rotulo(col.strip())}: {desc.strip()}")
            else:
                rotulos.append(item)
        fig, ax = plt.subplots(figsize=(10, 0.75 * len(inval_pos) + 2.4))
        barras = ax.barh(rotulos, inval_pos["n"], color="#c44e52")
        ax.bar_label(barras, fmt="%d", padding=3, fontsize=9)
        ax.set_xlabel("Ocorrências na base analítica (95.824 pedidos)", fontsize=11)
        fig.suptitle("Valores impossíveis ou incoerentes: tratados como ausentes\n"
                     "antes da divisão treino/teste (imputação aprendida só no treino)",
                     fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(FIGURES / "fig11_valores_impossiveis.png")
        plt.close(fig)
        print("[prep] figura -> fig11_valores_impossiveis.png")
    print(f"[prep] VIF máx.: {info_diag['vif_max_antes']:.1f} (antes) -> "
          f"{info_diag['vif_max_depois']:.1f} (depois da remoção das redundantes)")
    X_train, X_test, y_train, y_test = dividir(limpo)
    pre = pipeline_preprocess(escalar_numerico=True)
    Xt_train = pre.fit_transform(X_train)
    Xt_test = pre.transform(X_test)
    nomes = nomes_saida(pre)
    def salvar(nome, matriz, indice):
        pd.DataFrame(matriz, columns=nomes, index=indice).round(5).to_csv(
            PREP_DIR / nome, compression="gzip"
        )
    salvar("X_train_prep.csv.gz", Xt_train, X_train.index)
    salvar("X_test_prep.csv.gz", Xt_test, X_test.index)
    y_train.to_csv(PREP_DIR / "y_train.csv", index=True)
    y_test.to_csv(PREP_DIR / "y_test.csv", index=True)
    cols_extra = ["order_id", "order_purchase_timestamp", "review_score"]
    for nome, X, y in (("treino_limpo.csv.gz", X_train, y_train), ("teste_limpo.csv.gz", X_test, y_test)):
        pd.concat([limpo.loc[X.index, cols_extra], X, y], axis=1).to_csv(
            PREP_DIR / nome, compression="gzip", index=False
        )
    status = pd.read_csv(STATUS_CSV)
    n_cat_ohe = sum(1 for n in nomes if any(n.startswith(c + "_") for c in CATEGORICAL_FEATURES))
    ausentes_bruto = int(bruto[features_do_modelo()].isna().sum().sum())
    ausentes_limpo = int(limpo[features_do_modelo()].isna().sum().sum())
    info = {
        "n_total": int(len(status)),
        "n_delivered": int((status["order_status"] == "delivered").sum()),
        "n_base": int(len(bruto)),
        "n_colunas_base": int(bruto.shape[1]),
        "n_features_candidatas": len(NUMERIC_CANDIDATAS) + len(CATEGORICAL_FEATURES),
        "n_features_modelo": len(features_do_modelo()),
        "redundantes_removidas": REDUNDANTES,
        "n_invalidos_tratados": {
            f"{r['item']}": int(r["n"]) for _, r in inval.iterrows() if r["n"] > 0
        },
        "n_invalidos_total": int(inval["n"].sum()),
        "ausentes_antes_regras": ausentes_bruto,
        "ausentes_depois_regras": ausentes_limpo,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "taxa_treino": float(y_train.mean()),
        "taxa_teste": float(y_test.mean()),
        "n_dim_saida": int(Xt_train.shape[1]),
        "n_dim_ohe": int(n_cat_ohe),
        "n_dim_num": len(NUMERIC_FEATURES),
        **info_diag,
    }
    PREP_JSON.write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[prep] base final: treino {Xt_train.shape} / teste {Xt_test.shape} -> {PREP_DIR}")
    print("[prep] etapa concluída.")
