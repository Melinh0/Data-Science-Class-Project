
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES, ETAPAS_MD, FIGURES, MODEL_JSON, NUMERIC_FEATURES,
    RANDOM_STATE, STATUS_CSV, TABLES, TARGET, TEST_SIZE,
)
from .data_loader import carregar_dados
from .features import carregar_analise
from .model import _preprocessador
from .report import _fig, _inte, _md, _n, _pct

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MOTIVOS_DESCARTE = {
    "order_id": "Identificador único do pedido (1:1 com a linha)",
    "customer_id": "Identificador de ligação pedido-cliente",
    "customer_unique_id": "Identificador de cliente (alta cardinalidade)",
    "cep_cliente": "Identificador geográfico fino (equivalente à cidade)",
    "cidade_cliente": "Identificador de cidade — redundante com uf_cliente + distancia_km",
    "seller_zip": "Identificador geográfico do vendedor",
    "order_status": "Pós-recorte: a base já contém apenas pedidos entregues",
    "categoria": "Redundante com categoria_grupo (níveis raros agrupados)",
    "valor_pago": "Redundante com valor_pedido/preco_total",
    "prazo_cumprido": "Redundante com o sinal de atraso_dias (contínua mantida)",
    "faixa_atraso": "Versão agrupada de atraso_dias (usada apenas na EDA)",
    "review_score": "VAZAMENTO: deriva diretamente do alvo",
    "review_creation_date": "VAZAMENTO: conhecida somente após a avaliação",
    "tem_comentario": "VAZAMENTO: manifestação pós-avaliação",
    "comentario_chars": "VAZAMENTO: conteúdo da avaliação (pós-resposta)",
    "order_purchase_timestamp": "Data bruta -> hora_compra, mes_compra, dia_semana_compra",
    "order_approved_at": "Data bruta -> aprovacao_h",
    "order_delivered_carrier_date": "Data bruta -> manuseio_dias",
    "order_delivered_customer_date": "Data bruta -> entrega_dias",
    "order_estimated_delivery_date": "Data bruta -> atraso_dias (real - prometido)",
}


def _p_fmt(p: float) -> str:
    if p == 0:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.1e}".replace(".", ",")
    return f"{p:.3f}".replace(".", ",")


def _tipo(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "booleana"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "data"
    if pd.api.types.is_numeric_dtype(s):
        return "numérica"
    return "categórica"


def _papel(col: str) -> str:
    if col == TARGET:
        return "alvo"
    if col in FEATURES:
        return "feature"
    return "descartada"


def _inventario(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for c in df.columns:
        s = df[c]
        linhas.append({
            "variavel": c,
            "papel": _papel(c),
            "tipo": _tipo(s),
            "pct_ausentes": round(float(s.isna().mean()) * 100, 2),
            "n_unicos": int(s.nunique(dropna=True)),
        })
    return pd.DataFrame(linhas)


def _descritivas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = df[NUMERIC_FEATURES]
    tab_num = pd.DataFrame({
        "variavel": NUMERIC_FEATURES,
        "media": [_n(x, 2) for x in num.mean()],
        "desvio_padrao": [_n(x, 2) for x in num.std()],
        "min": [_n(x, 2) for x in num.min()],
        "mediana": [_n(x, 2) for x in num.median()],
        "max": [_n(x, 2) for x in num.max()],
        "ausentes": [int(x) for x in num.isna().sum()],
    })
    cat = df[CATEGORICAL_FEATURES]
    linhas_cat = []
    for c in CATEGORICAL_FEATURES:
        s = cat[c]
        vc = s.value_counts(dropna=True)
        linhas_cat.append({
            "variavel": c,
            "n_niveis": int(s.nunique(dropna=True)),
            "nivel_mais_frequente": str(vc.index[0]),
            "pct_nivel_frequente": _pct(vc.iloc[0] / max(int(vc.sum()), 1)),
            "ausentes": int(s.isna().sum()),
        })
    return tab_num, pd.DataFrame(linhas_cat)


def _evidencia(df: pd.DataFrame, col: str) -> dict:
    dados = df[[col, TARGET]].dropna()
    x = dados.loc[dados[TARGET] == 1, col]
    y = dados.loc[dados[TARGET] == 0, col]
    if col in NUMERIC_FEATURES:
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        efeito = 2 * stat / (len(x) * len(y)) - 1
        return {"teste": "Mann-Whitney U", "medida": "correlação rank-bisserial r",
                "efeito": float(efeito), "p_valor": float(p)}
    tabela = pd.crosstab(dados[col], dados[TARGET])
    chi2, p, _, _ = stats.chi2_contingency(tabela)
    n = float(tabela.to_numpy().sum())
    v = float(np.sqrt(chi2 / (n * (min(tabela.shape) - 1))))
    return {"teste": "Qui-quadrado de independência", "medida": "V de Cramér",
            "efeito": v, "p_valor": float(p)}


def _selecao(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    linhas = []
    n_sig = 0
    for col in FEATURES:
        ev = _evidencia(df, col)
        sig = ev["p_valor"] < 0.05
        n_sig += int(sig)
        linhas.append({
            "variavel": col,
            "decisao": "mantida",
            "criterio": "evidência estatística + relevância operacional",
            "teste": ev["teste"],
            "p_valor": float(f"{ev['p_valor']:.3g}"),
            "significativo_005": "sim" if sig else "não",
            "medida_efeito": ev["medida"],
            "valor_efeito": round(ev["efeito"], 4),
        })
    for col in [c for c in df.columns if _papel(c) == "descartada"]:
        motivo = MOTIVOS_DESCARTE.get(col, "Irrelevante para o modelo")
        criterio = "vazamento" if motivo.startswith("VAZAMENTO") else (
            "identificador" if "Identificador" in motivo else (
                "engenharia de atributos" if "->" in motivo else "redundância"
            )
        )
        linhas.append({
            "variavel": col,
            "decisao": "descartada",
            "criterio": criterio,
            "teste": motivo,
            "p_valor": np.nan,
            "significativo_005": "—",
            "medida_efeito": "—",
            "valor_efeito": np.nan,
        })
    linhas.append({
        "variavel": TARGET,
        "decisao": "alvo",
        "criterio": "resposta (review_score <= 2)",
        "teste": "—",
        "p_valor": np.nan,
        "significativo_005": "—",
        "medida_efeito": "—",
        "valor_efeito": np.nan,
    })
    tab = pd.DataFrame(linhas)
    return tab, n_sig


def _fig_correlacao(corr: pd.DataFrame) -> None:
    rotulos = [c.replace("_", " ") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.4, xticklabels=rotulos, yticklabels=rotulos,
        cbar_kws={"label": "r de Pearson"}, ax=ax,
    )
    ax.set_title("Correlação entre variáveis numéricas e o alvo")
    fig.savefig(FIGURES / "fig16_matriz_correlacao.png")
    plt.close(fig)
    print("[etapas] figura -> fig16_matriz_correlacao.png")


def _preprocessamento_resultados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    status = pd.read_csv(STATUS_CSV)
    n_total = len(status)
    n_delivered = int((status["order_status"] == "delivered").sum())
    n_base = len(df)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    pre = _preprocessador(escalar=False)
    X_treino = pre.fit_transform(X_train)
    nomes = list(pre.get_feature_names_out(FEATURES))
    saidas_cat = [n for n in nomes if n.startswith("cat__")]
    agrupados = [n for n in saidas_cat if "infrequent" in n]
    n_niveis_brutos = int(X[CATEGORICAL_FEATURES].nunique().sum())
    n_missing = int(X.isna().sum().sum())
    pesos = len(y_train) / (2 * np.bincount(y_train))

    linhas = [
        {"etapa": "1. Recorte dos pedidos",
         "entrada": _inte(n_total), "saida": _inte(n_base),
         "detalhe": f"entregues ({_inte(n_delivered)}) e com avaliação "
                    f"(−{_inte(n_delivered - n_base)} sem review)"},
        {"etapa": "2. Seleção de atributos",
         "entrada": f"{len(df.columns)} colunas",
         "saida": f"{len(FEATURES)} features",
         "detalhe": f"{len(MOTIVOS_DESCARTE)} colunas descartadas (Seção 3) + alvo"},
        {"etapa": "3. Imputação de ausências",
         "entrada": f"{_inte(n_missing)} células", "saida": "0 células",
         "detalhe": "numéricas -> mediana; categóricas -> moda"},
        {"etapa": "4. One-Hot Encoding",
         "entrada": f"{len(CATEGORICAL_FEATURES)} categóricas ({n_niveis_brutos} níveis)",
         "saida": f"{len(saidas_cat)} colunas",
         "detalhe": f"min_frequency=30 agrupa níveis raros em {len(agrupados)} colunas 'infrequent'"},
        {"etapa": "5. Padronização (z-score)",
         "entrada": f"{len(NUMERIC_FEATURES)} numéricas", "saida": "idem",
         "detalhe": "StandardScaler aplicado apenas na regressão logística"},
        {"etapa": "6. Divisão treino/teste",
         "entrada": _inte(n_base),
         "saida": f"{_inte(len(X_train))} / {_inte(len(X_test))}",
         "detalhe": f"80/20 estratificada — insatisfação {_pct(y_train.mean())} no treino, "
                    f"{_pct(y_test.mean())} no teste"},
        {"etapa": "7. Balanceamento das classes",
         "entrada": f"{_pct(y_train.mean())} positivos",
         "saida": f"pesos {_n(pesos[1], 2)} / {_n(pesos[0], 2)}",
         "detalhe": "class_weight='balanced' (positivo recebe mais peso no custo)"},
        {"etapa": "8. Matriz final de modelagem",
         "entrada": "—",
         "saida": f"{_inte(X_treino.shape[0])} × {X_treino.shape[1]}",
         "detalhe": f"treino ({_inte(X_test.shape[0])} × {X_treino.shape[1]} no teste) — "
                    f"{len(NUMERIC_FEATURES)} numéricas + {len(saidas_cat)} OHE"},
    ]
    info = {
        "n_niveis_brutos": n_niveis_brutos,
        "n_ohe": len(saidas_cat),
        "n_agrupados": len(agrupados),
        "n_missing": n_missing,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "peso_pos": float(pesos[1]),
        "peso_neg": float(pesos[0]),
        "n_delivered": n_delivered,
        "n_total": n_total,
        "taxa_treino": float(y_train.mean()),
        "taxa_teste": float(y_test.mean()),
    }
    return pd.DataFrame(linhas), info


def gerar_relatorio_etapas() -> None:
    df = carregar_analise()
    brutos = carregar_dados()
    modelo = json.loads(MODEL_JSON.read_text())

    tab_notas = pd.read_csv(TABLES / "tab01_resumo_notas.csv")
    tab_testes = pd.read_csv(TABLES / "tab05_testes_hipotese.csv")
    tab_or_atraso = pd.read_csv(TABLES / "tab06_odds_ratio_atraso.csv").iloc[0]
    tab_coef = pd.read_csv(TABLES / "tab07_odds_ratios_logistica.csv")
    tab_imp = pd.read_csv(TABLES / "tab08_importancia_permutacao.csv")
    tab_met = pd.read_csv(TABLES / "tab09_metricas_modelos.csv")

    inventario = _inventario(df)
    inventario.to_csv(TABLES / "tab10_inventario_variaveis.csv", index=False)
    print(f"[etapas] tabela -> tab10_inventario_variaveis.csv ({len(inventario)} colunas)")
    inv_exib = inventario.copy()
    inv_exib["pct_ausentes"] = inv_exib["pct_ausentes"].map(
        lambda x: f"{x:.2f}".replace(".", ",")
    )

    tab_num, tab_cat = _descritivas(df)
    pd.concat([tab_num, tab_cat], ignore_index=True).to_csv(
        TABLES / "tab11_descritivas_features.csv", index=False
    )
    print("[etapas] tabela -> tab11_descritivas_features.csv")

    numericas = df[NUMERIC_FEATURES + [TARGET]].corr()
    numericas.to_csv(TABLES / "tab12_correlacoes_numericas.csv")
    _fig_correlacao(numericas)
    corr_feat = numericas.loc[NUMERIC_FEATURES, NUMERIC_FEATURES]
    pares = (
        corr_feat.where(np.triu(np.ones(corr_feat.shape), k=1).astype(bool))
        .stack().reset_index()
    )
    pares.columns = ["variavel_1", "variavel_2", "r"]
    pares["abs_r"] = pares["r"].abs()
    pares_fortes = pares[pares["abs_r"] >= 0.80].sort_values("abs_r", ascending=False)

    tab_sel, n_sig = _selecao(df)
    tab_sel.to_csv(TABLES / "tab13_selecao_atributos.csv", index=False)
    print(f"[etapas] tabela -> tab13_selecao_atributos.csv ({len(tab_sel)} linhas)")

    tab_prep, info = _preprocessamento_resultados(df)
    tab_prep.to_csv(TABLES / "tab14_preprocessamento.csv", index=False)
    print("[etapas] tabela -> tab14_preprocessamento.csv")

    met = modelo["metricas"]
    m_lr = met["regressao_logistica"]
    m_hgb = met["gradient_boosting"]
    melhor = modelo["melhor_modelo"]

    pares_md = pares_fortes.head(8).copy()
    pares_md["r"] = pares_md["r"].map(lambda x: _n(x, 3))
    pares_md = pares_md.drop(columns="abs_r")

    sel_exib = tab_sel.copy()
    sel_exib["p_valor"] = sel_exib["p_valor"].map(
        lambda p: "—" if pd.isna(p) else _p_fmt(p)
    )
    pag_p = _p_fmt(
        tab_sel.loc[tab_sel["variavel"] == "pagamento_tipo", "p_valor"].iloc[0]
    )
    hora_p = _p_fmt(
        tab_sel.loc[tab_sel["variavel"] == "hora_compra", "p_valor"].iloc[0]
    )
    sel_exib["valor_efeito"] = sel_exib["valor_efeito"].map(
        lambda x: "—" if pd.isna(x) else f"{x:.4f}".replace(".", ",")
    )

    sel_features = sel_exib[sel_exib["decisao"] == "mantida"]
    sel_descartadas = sel_exib[sel_exib["decisao"] == "descartada"]

    conf_lr = m_lr["matriz_confusao"]
    conf_hgb = m_hgb["matriz_confusao"]

    coef_exib = tab_coef.copy()
    coef_exib["variavel"] = (
        coef_exib["variavel"]
        .str.replace("infrequent_sklearn", "outros (agrupados)", regex=False)
        .str.replace("categoria_grupo_", "categoria: ", regex=False)
        .str.replace("uf_cliente_", "UF cliente: ", regex=False)
        .str.replace("uf_vendedor_", "UF vendedor: ", regex=False)
        .str.replace("pagamento_tipo_", "pagamento: ", regex=False)
    )
    coef_exib["odds_ratio"] = coef_exib["odds_ratio"].map(
        lambda x: f"{x:.2f}".replace(".", ",")
    )
    or_risco = coef_exib[tab_coef["odds_ratio"] > 1].head(8)
    or_prot = coef_exib[tab_coef["odds_ratio"] < 1].tail(8).iloc[::-1]
    imp_hgb = tab_imp[tab_imp["modelo"] == "gradient_boosting"].head(10).copy()
    imp_hgb["importancia_media"] = imp_hgb["importancia_media"].map(
        lambda x: f"{x:.4f}".replace(".", ",")
    )
    imp_hgb["importancia_std"] = imp_hgb["importancia_std"].map(
        lambda x: f"{x:.4f}".replace(".", ",")
    )

    shap_ok = modelo.get("shap_disponivel", False)
    shap_md = ""
    if shap_ok:
        shap_md = f"""
{_fig('fig14_shap_importancia.png', 'Importância média |SHAP| no gradient boosting')}
{_fig('fig15_shap_beeswarm.png', 'SHAP beeswarm: direção e magnitude de cada variável')}
"""

    partes: list[str] = []

    partes.append(f"""# Relatório de etapas — da análise de variáveis ao modelo

**Trilha completa de Machine Learning aplicada ao dataset
[Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**

> **Objetivo:** apresentar, etapa por etapa, os resultados de toda a jornada:
> análise de variáveis → seleção de atributos → pré-processamento →
> implementação e avaliação do modelo.
>
> **Pergunta que orienta o trabalho:** *Quais fatores estão associados à
> insatisfação dos clientes e como essas evidências podem apoiar decisões de
> melhoria operacional?*

**Sumário das etapas**

| Seção | Etapa | Resultado principal |
|---|---|---|
| 1 | Base de entrada | {_inte(len(df))} pedidos, {len(df.columns)} colunas |
| 2 | Análise de variáveis | inventário, descritivas, correlações |
| 3 | Seleção de atributos | {len(df.columns)} → {len(FEATURES)} features + alvo |
| 4 | Pré-processamento | imputação, OHE, escala, divisão, balanceamento |
| 5 | Implementação do modelo | 2 algoritmos comparados por validação cruzada |
| 6 | Avaliação e resultado final | AUC, precisão/recall, matriz de confusão, interpretação |

---

## 1. Base de entrada

O dataset possui **9 arquivos** relacionados por `order_id`:

{_md(pd.DataFrame({
    "tabela": list(brutos.keys()),
    "linhas": [_inte(len(v)) for v in brutos.values()],
    "colunas": [v.shape[1] for v in brutos.values()],
}))}

Após o recorte analítico (pedidos **entregues**, com data de entrega e **com
avaliação** — 1 review por pedido, a mais antiga), a base de análise tem
**{_inte(len(df))} pedidos** e **{len(df.columns)} colunas**
({len(FEATURES)} candidatas a feature, 1 alvo e {len(MOTIVOS_DESCARTE)} colunas
descartadas — detalhe na Seção 3).

{_fig('fig01_status_pedidos.png', 'Distribuição dos status na base bruta')}

**Variável-alvo:** `insatisfeito = 1` quando `review_score <= 2`.
Taxa de insatisfação: **{_pct(df[TARGET].mean())}** — problema de classe
desbalanceada, o que orienta decisões de pré-processamento (Seção 4).

---

## 2. Análise de variáveis

### 2.1. Inventário completo ({len(df.columns)} colunas)

{_md(inv_exib)}

### 2.2. Estatísticas descritivas das features candidatas

**Numéricas:**

{_md(tab_num)}

**Categóricas:**

{_md(tab_cat)}

### 2.3. Distribuição do alvo e das variáveis-chave

Notas: **{_pct(tab_notas.loc[tab_notas['nota'] == 5, 'pct'].iloc[0] / 100)}** são nota 5 e
**{_pct(tab_notas.loc[tab_notas['nota'].isin([1, 2]), 'pct'].sum() / 100)}** são notas 1-2
(base polarizada):

{_fig('fig02_distribuicao_notas.png', 'Distribuição das notas de avaliação')}

{_fig('fig03_evolucao_temporal.png', 'Evolução mensal da nota média e da insatisfação')}

### 2.4. Correlação entre as variáveis numéricas

{_fig('fig16_matriz_correlacao.png', 'Matriz de correlação (Pearson) das variáveis numéricas + alvo')}

Pares com correlação alta (|r| ≥ 0,80) — insumo para a Seção 3:

{_md(pares_md) if len(pares_md) else "*Nenhum par de features atinge |r| ≥ 0,80.*"}

---

## 3. Seleção de atributos

**Resultado: {len(df.columns)} colunas → {len(FEATURES)} features + 1 alvo**
({len(MOTIVOS_DESCARTE)} colunas descartadas).

### 3.1. Critérios aplicados

| # | Critério | Colunas | Lógica |
|---|---|---|---|
| 1 | Vazamento (leakage) | 4 | qualquer informação derivada da própria avaliação é excluída do modelo |
| 2 | Identificador | 6 | código de pedido/cliente/CEP não generaliza |
| 3 | Engenharia de atributos | 5 | datas brutas substituídas por derivadas (atraso, entrega, manuseio, hora/mês/dia) |
| 4 | Redundância | 5 | coluna duplicada ou versão agregada de outra feature |
| 5 | Evidência estatística | {len(FEATURES)} mantidas | Mann-Whitney (numéricas) e qui-quadrado (categóricas) contra o alvo |

### 3.2. Features mantidas — evidência estatística (α = 0,05)

**{n_sig} de {len(FEATURES)}** features são estatisticamente significativas contra
o alvo a α = 0,05; as outras duas são mantidas por decisão declarada:
`pagamento_tipo` (p = {pag_p} — variável clássica de negócio) e `hora_compra`
(p = {hora_p} — característica da compra, usada junto da sazonalidade).

{_md(sel_features)}

### 3.3. Colunas descartadas

{_md(sel_descartadas)}

### 3.4. Multicolinearidade

{_md(pares_md) if len(pares_md) else "Sem pares com |r| ≥ 0,80."}

Decisão: pares correlacionados **mantidos** — a regressão logística usa
regularização L2 (coeficientes estáveis apesar da colinearidade) e o gradient
boosting é insensível a correlação entre features; remover `valor_pedido` ou
`preco_total` não altera materialmente o AUC.

**Figuras de apoio da seleção** (as variáveis com maior efeito univariado):

{_fig('fig05_inssatisfacao_por_atraso.png', 'Dose-resposta: taxa de insatisfação por faixa de atraso')}

---

## 4. Pré-processamento

### 4.1. Fluxo de transformações com resultado numérico de cada passo

{_md(tab_prep)}

* **Ausências tratadas:** {_inte(info['n_missing'])} células nas features
  ({_inte(int(df['n_fotos'].isna().sum()))} em `n_fotos`,
  {_inte(int(df['distancia_km'].isna().sum()))} em `distancia_km` e valores
  residuais em `aprovacao_h`/`peso_g`) → mediana (numéricas) e moda (categóricas),
  aplicadas **somente no conjunto de treino** (pipeline do scikit-learn, sem vazamento).
* **Codificação:** {len(CATEGORICAL_FEATURES)} variáveis categóricas com
  {info['n_niveis_brutos']} níveis brutos → **{info['n_ohe']} colunas one-hot**
  (`min_frequency=30` agrupa níveis raros em {info['n_agrupados']} colunas
  "infrequent", evitando colunas espúrias).
* **Escala:** `StandardScaler` nas {len(NUMERIC_FEATURES)} numéricas apenas na
  regressão logística (não necessária para o boosting).
* **Divisão:** {_inte(info['n_train'])} treino / {_inte(info['n_test'])} teste
  (80/20 estratificada, `random_state=42`) — insatisfação
  {_pct(info['taxa_treino'])} (treino) e {_pct(info['taxa_teste'])} (teste).
* **Balanceamento:** `class_weight='balanced'` → pesos
  {_n(info['peso_pos'], 2)} (insatisfeito) e {_n(info['peso_neg'], 2)} (satisfeito),
  compensando a classe rara de {_pct(info['taxa_treino'])}.

---

## 5. Implementação do modelo

### 5.1. Algoritmos e configuração

| Modelo | Pré-processamento | Hiperparâmetros |
|---|---|---|
| Regressão logística | imputação + OHE + padronização | `C=1,0`, `class_weight=balanced`, `max_iter=2000`, penalidade L2 |
| HistGradientBoosting | imputação + OHE | `learning_rate=0,08`, `max_iter=300`, `max_leaf_nodes=31`, `class_weight=balanced` |

**Protocolo:** validação cruzada estratificada **{5} dobras** no conjunto de treino
(seleção) → ajuste final → avaliação única no conjunto de teste.
`random_state=42` em toda a pipeline.

### 5.2. Métricas no conjunto de teste (n = {_inte(info['n_test'])})

| Modelo | AUC (CV) | AUC (teste) | PR-AUC | Precisão | Recall | F1 | Acurácia | Brier |
|---|---|---|---|---|---|---|---|---|
| Regressão logística | {_n(m_lr['cv_roc_auc_mean'], 3)}±{_n(m_lr['cv_roc_auc_std'], 3)} | {_n(m_lr['roc_auc'], 3)} | {_n(m_lr['pr_auc'], 3)} | {_n(m_lr['precisao'], 3)} | {_n(m_lr['recall'], 3)} | {_n(m_lr['f1'], 3)} | {_n(m_lr['acuracia'], 3)} | {_n(m_lr['brier'], 3)} |
| Gradient boosting (HGB) | {_n(m_hgb['cv_roc_auc_mean'], 3)}±{_n(m_hgb['cv_roc_auc_std'], 3)} | **{_n(m_hgb['roc_auc'], 3)}** | {_n(m_hgb['pr_auc'], 3)} | {_n(m_hgb['precisao'], 3)} | {_n(m_hgb['recall'], 3)} | {_n(m_hgb['f1'], 3)} | {_n(m_hgb['acuracia'], 3)} | {_n(m_hgb['brier'], 3)} |

*Baseline do PR-AUC = taxa de insatisfação = {_n(modelo['taxa_positivos_pct'] / 100, 3)};
o HGB supera o baseline em ≈ {_n(m_hgb['pr_auc'] / (modelo['taxa_positivos_pct'] / 100), 1)}x.*

### 5.3. Matrizes de confusão (limiar 0,5)

**Regressão logística:**

| | Prev. satisfeito | Prev. insatisfeito |
|---|---|---|
| **Real satisfeito** | {conf_lr[0][0]} (TN) | {conf_lr[0][1]} (FP) |
| **Real insatisfeito** | {conf_lr[1][0]} (FN) | {conf_lr[1][1]} (TP) |

**Gradient boosting (melhor modelo):**

| | Prev. satisfeito | Prev. insatisfeito |
|---|---|---|
| **Real satisfeito** | {conf_hgb[0][0]} (TN) | {conf_hgb[0][1]} (FP) |
| **Real insatisfeito** | {conf_hgb[1][0]} (FN) | {conf_hgb[1][1]} (TP) |

Leitura: o HGB identifica **{conf_hgb[1][1]}** de {_inte(conf_hgb[1][0] + conf_hgb[1][1])}
clientes insatisfeitos (recall {_n(m_hgb['recall'], 3)}) com {conf_hgb[0][1]} falsos
positivos no conjunto de teste — ver figura abaixo.

{_fig('fig10_curvas_roc_pr.png', 'Curvas ROC e Precisão-Recall no conjunto de teste')}
{_fig('fig11_matriz_confusao.png', 'Matriz de confusão do melhor modelo')}
{_fig('fig12_calibracao.png', 'Calibração das probabilidades previstas')}

**Melhor modelo: `{melhor}`** — salvo em `outputs/modelo_insatisfacao.joblib`
(pronto para pontuar novos pedidos).

---

## 6. Avaliação e resultado final

### 6.1. Regressão logística — odds ratios

*Leitura: OR > 1 → **aumenta** a chance de nota 1-2; OR < 1 → **protege**.
Para variáveis numéricas o OR refere-se a 1 desvio-padrão; para categóricas,
compara-se à categoria de referência.*

**Top fatores de risco (OR > 1):**

{_md(or_risco[['variavel', 'odds_ratio']])}

### 6.2. Regressão logística — fatores protetores (OR < 1)

{_md(or_prot[['variavel', 'odds_ratio']])}

### 6.3. Gradient boosting — importância por permutação (queda média de AUC)

{_md(imp_hgb[['variavel', 'importancia_media', 'importancia_std']])}

{_fig('fig13_dependencia_parcial.png', 'Dependência parcial: atraso, frete e distância')}
{shap_md}
### 6.4. Síntese do resultado

1. **O atraso é o fator dominante:** OR = {_n(tab_or_atraso['odds_ratio'])}x
   (IC95% {_n(tab_or_atraso['ic95_inf'])}–{_n(tab_or_atraso['ic95_sup'])}) e relação
   dose-resposta monotônica nas faixas de atraso (Seção 3).
2. **`n_itens` é a variável mais importante no boosting** (permutação):
   {_pct(df.loc[df['n_itens'] == 1, TARGET].mean())} de insatisfação em pedidos com
   1 item vs. {_pct(df.loc[df['n_itens'] >= 2, TARGET].mean())} com 2+ itens.
3. **O modelo final ({melhor})** atinge AUC = {_n(m_hgb['roc_auc'], 3)} e
   PR-AUC = {_n(m_hgb['pr_auc'], 3)} (baseline {_n(modelo['taxa_positivos_pct'] / 100, 3)}),
   viabilizando um **score de risco de insatisfação por pedido**.
4. **A forma de pagamento não se associou** ao alvo
   (qui-quadrado, p = {pag_p})
   — exemplo de variável mantida por decisão de negócio, não por evidência.

### 6.5. Artefatos desta execução

| Arquivo | Conteúdo |
|---|---|
| `reports/relatorio_etapas.md` | este relatório |
| `reports/relatorio.md` | relatório executivo completo (EDA → recomendações) |
| `reports/tables/tab10_inventario_variaveis.csv` | inventário das {len(df.columns)} colunas |
| `reports/tables/tab11_descritivas_features.csv` | descritivas das {len(FEATURES)} features |
| `reports/tables/tab12_correlacoes_numericas.csv` | matriz de correlação completa |
| `reports/tables/tab13_selecao_atributos.csv` | decisão + evidência por variável |
| `reports/tables/tab14_preprocessamento.csv` | resultado numérico de cada transformação |
| `outputs/analysis_dataset.csv` | base final ({_inte(len(df))} × {len(df.columns)}) |
| `outputs/modelo_insatisfacao.joblib` | melhor modelo serializado |
| `outputs/resultados_modelo.json` | métricas e interpretações em JSON |

*Reprodutibilidade:* `python run_pipeline.py` (etapas `load → eda → tests → model →
etapas → report`) · seed 42 · dados: Kaggle `olistbr/brazilian-ecommerce`.
""")

    ETAPAS_MD.write_text("".join(partes), encoding="utf-8")
    print(f"[etapas] relatório -> {ETAPAS_MD}")


if __name__ == "__main__":
    gerar_relatorio_etapas()
