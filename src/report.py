
from __future__ import annotations

import json

import pandas as pd

from .config import (
    ANALYSIS_CSV, FIGURES, MODEL_JSON, REPORT_MD, SCORE_COL, STATUS_CSV, TABLES, TARGET,
)
from .features import carregar_analise

try:
    import tabulate
    _TEM_TABULATE = True
except ImportError:
    _TEM_TABULATE = False


def _md(df: pd.DataFrame, max_rows: int | None = None, colunas: list | None = None) -> str:
    d = df if colunas is None else df[colunas]
    if max_rows:
        d = d.head(max_rows)
    if _TEM_TABULATE:
        return d.to_markdown(index=False, floatfmt=".2f")
    cab = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    linhas = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in linha) + " |"
        for linha in d.itertuples(index=False, name=None)
    ]
    return "\n".join([cab, sep] + linhas)


def _fig(nome: str, legenda: str) -> str:
    return f"![{legenda}](figures/{nome})\n\n*Figura: {legenda}*"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def _inte(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _n(x, casas: int = 1) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def gerar_relatorio() -> None:
    analise = carregar_analise()
    status = pd.read_csv(STATUS_CSV)
    modelo = json.loads(MODEL_JSON.read_text())

    n = len(analise)
    taxa_neg = analise[TARGET].mean()
    media_nota = analise[SCORE_COL].mean()
    atrasados = analise["atraso_dias"] > 0
    pct_atraso = atrasados.mean()
    p_atraso_dado_neg = atrasados[analise[TARGET] == 1].mean()
    p_atraso_dado_pos = atrasados[analise[TARGET] == 0].mean()

    med = analise.groupby(TARGET)[
        ["atraso_dias", "entrega_dias", "frete_total", "frete_ratio", "distancia_km"]
    ].median()
    med_neg, med_pos = med.loc[1], med.loc[0]

    tab_testes = pd.read_csv(TABLES / "tab05_testes_hipotese.csv")
    tab_odds = pd.read_csv(TABLES / "tab06_odds_ratio_atraso.csv").iloc[0]
    tab_notas = pd.read_csv(TABLES / "tab01_resumo_notas.csv")
    tab_cat = pd.read_csv(TABLES / "tab03_insatisfacao_por_categoria.csv")
    tab_uf = pd.read_csv(TABLES / "tab04_insatisfacao_por_uf.csv")
    tab_coef = pd.read_csv(TABLES / "tab07_odds_ratios_logistica.csv")
    tab_imp = pd.read_csv(TABLES / "tab08_importancia_permutacao.csv")
    tab_metricas = pd.read_csv(TABLES / "tab09_metricas_modelos.csv")

    cats_top = tab_cat[(tab_cat["n_pedidos"] >= 100)].head(5)
    ufs_top = tab_uf[tab_uf["n_pedidos"] >= 300].head(3)

    p_neg_1_item = analise.loc[analise["n_itens"] == 1, TARGET].mean()
    p_neg_multi = analise.loc[analise["n_itens"] >= 2, TARGET].mean()
    _ef = tab_testes[tab_testes["comparacao"].isin(
        ["Frete total (R$)", "Participação do frete no pedido (%)",
         "Distância cliente-vendedor (km)"]
    )]["valor_efeito"].abs()
    max_abs_efeto_frete_dist = float(_ef.max())

    coment_neg = analise.loc[analise[TARGET] == 1, "tem_comentario"].mean()
    coment_pos = analise.loc[analise[TARGET] == 0, "tem_comentario"].mean()

    mensal = analise.groupby("mes_compra").agg(
        n=(TARGET, "size"), nota=(SCORE_COL, "mean"), neg=(TARGET, "mean")
    )
    mensal = mensal[mensal["n"] >= 500]
    pior_mes = mensal["neg"].idxmax()
    melhor_mes = mensal["neg"].idxmin()

    tab_coef_exib = tab_coef.copy()
    tab_coef_exib["variavel"] = (
        tab_coef_exib["variavel"]
        .str.replace("infrequent_sklearn", "outros (agrupados)", regex=False)
        .str.replace("categoria_grupo_", "categoria: ", regex=False)
        .str.replace("uf_cliente_", "UF cliente: ", regex=False)
        .str.replace("uf_vendedor_", "UF vendedor: ", regex=False)
        .str.replace("pagamento_tipo_", "pagamento: ", regex=False)
    )
    coef_risco = tab_coef_exib[tab_coef_exib["odds_ratio"] > 1].head(6)
    coef_prot = tab_coef_exib[tab_coef_exib["odds_ratio"] < 1].tail(6).iloc[::-1]
    imp_hgb = tab_imp[tab_imp["modelo"] == "gradient_boosting"].head(8)

    met = modelo["metricas"]
    melhor = modelo["melhor_modelo"]
    m_lr = met["regressao_logistica"]
    m_hgb = met["gradient_boosting"]

    pct_sem_comentario_neg = 1 - coment_neg

    partes: list[str] = []
    partes.append(f"""# Quais fatores estão associados à insatisfação dos clientes?

**Análise exploratória, testes de hipótese e modelo preditivo aplicados ao dataset
[Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**

> **Pergunta de negócio:** *Quais fatores estão associados à insatisfação dos
> clientes e como essas evidências podem apoiar decisões de melhoria operacional?*

---

## 1. Sumário executivo

* **{_inte(n)} pedidos entregues com avaliação** compõem a base de análise; a taxa de
  insatisfação (notas 1-2) é de **{_pct(taxa_neg)}** (nota média = {_n(media_nota, 2)}).
* **O atraso na entrega é o fator mais associado à insatisfação:** enquanto
  {_pct(p_atraso_dado_pos)} dos clientes satisfeitos receberam com atraso,
  **{_pct(p_atraso_dado_neg)} dos insatisfeitos** receberam com atraso.
  Pedidos atrasados têm **{_n(tab_odds['odds_ratio'])}x mais chances** de gerar
  nota 1-2 (IC95% {_n(tab_odds['ic95_inf'])}–{_n(tab_odds['ic95_sup'])}; p < 0,001).
* **Pedidos com múltiplos itens** são o segundo achado: {_pct(p_neg_1_item)} de
  insatisfação em pedidos com 1 item vs. **{_pct(p_neg_multi)} com 2+ itens**.
* **Frete, distância e categoria** associam-se à nota, porém com efeito **pequeno**
  (|r| ≤ {_n(max_abs_efeto_frete_dist, 2)} nos testes da Seção 5). Já a **forma de
  pagamento NÃO** se associou (p = 0,094).
* O modelo de classificação atinge **AUC = {_n(m_hgb['roc_auc'], 3)}** (gradient boosting)
  e **AUC = {_n(m_lr['roc_auc'], 3)}** (regressão logística), permitindo estimar a
  *probabilidade de avaliação negativa* por pedido e priorizar ações.
* A interpretação das variáveis (Seção 7) sustenta recomendações operacionais
  concretas (Seção 8): gestão do **prazo prometido**, correção da operação de
  **pedidos fracionados**, auditoria em **categorias críticas** e uso do
  **score de risco** como alerta proativo ao cliente.
""")

    partes.append(f"""---

## 2. Dados e método

O dataset da Olist (marketplace brasileiro) traz **9 arquivos** relacionados por
`order_id`: pedidos, clientes, itens, avaliações, pagamentos, produtos, vendedores,
geolocalização e tradução de categorias.

**Critérios adotados**

| Decisão | Detalhe |
|---|---|
| Recorte analítico | Pedidos **entregues**, com data de entrega e **com avaliação** (1 review por pedido — a mais antiga) |
| Variável-alvo | `insatisfeito = 1` quando `review_score <= 2` (notas 1-2 vs. 3-5) |
| Granularidade | 1 linha = 1 pedido (pedido com múltiplos itens: total de itens/preço/frete + item principal = maior preço) |
| Variáveis-chave | Atraso vs. prazo prometido, tempo de entrega/manuseio, frete, valor, distância (geodésica CEP cliente × CEP vendedor), categoria, UF, forma de pagamento, sazonalidade |
| Vazamento | **Comentário da avaliação é excluído do modelo** (faz parte da própria avaliação); fica apenas na EDA |
| Reprodutibilidade | `random_state=42` em separação, CV e permutações |

**Volume da base de análise:** {_inte(n)} pedidos · notas de 1 a 5 ·
{analise['categoria_grupo'].nunique()} grupos de categoria · {analise['uf_cliente'].nunique()} UFs.
""")

    partes.append(f"""---

## 3. EDA: pedidos e avaliações

A base completa tem **{_inte(len(status))} pedidos**; a maior parte chega ao cliente
(`delivered`), e pedidos cancelados/indisponíveis concentram avaliações ausentes.

{_fig('fig01_status_pedidos.png', 'Status dos pedidos na base completa')}

As notas concentram-se em **5** ({_pct(tab_notas.loc[tab_notas['nota'] == 5, 'pct'].iloc[0] / 100)}),
mas notas **1** respondem por {_pct(tab_notas.loc[tab_notas['nota'] == 1, 'pct'].iloc[0] / 100)} —
mais do que 2 e 3 somadas ({_pct((tab_notas.loc[tab_notas['nota'].isin([2, 3]), 'pct'].sum()) / 100)}),
padrão típico de avaliações "ou amo muito, ou odeio".

{_fig('fig02_distribuicao_notas.png', 'Distribuição das notas de avaliação')}

No tempo, a insatisfação oscila mês a mês: o pior mês da série foi **{pior_mes}**
({_pct(mensal.loc[pior_mes, 'neg'])} de notas 1-2) e o melhor **{melhor_mes}**
({_pct(mensal.loc[melhor_mes, 'neg'])}), o que sugere efeitos de sazonalidade/capacidade
a monitorar.

{_fig('fig03_evolucao_temporal.png', 'Evolução mensal da nota média e da insatisfação')}

Ainda na EDA: avaliações com comentário são **{_pct(coment_neg)}** entre insatisfeitos
vs. **{_pct(coment_pos)}** entre satisfeitos — evidência de que o cliente insatisfeito
tende a *se manifestar* (sinal útil para monitoramento, embora observado após a nota).

{_fig('fig09_comentario_por_nota.png', '% de avaliações com comentário por nota')}
""")

    partes.append(f"""---

## 4. Relação entre operação e nota

**Atraso.** A mediana do atraso piora conforme a nota cai: pedidos bem avaliados
chegam em média **{_n(abs(med_pos['atraso_dias']))} dias antes** do prazo prometido,
enquanto os insatisfeitos têm apenas **{_n(abs(med_neg['atraso_dias']))} dias de
antecedência** — e {_pct(p_atraso_dado_neg)} deles ultrapassam o prazo de fato.

{_fig('fig04_atraso_por_nota.png', 'Distribuição do atraso por nota de avaliação')}

A taxa de insatisfação sobe de **{_pct(analise.loc[analise['faixa_atraso'] == 'No prazo (até 7d antes)', TARGET].mean())}**
(no prazo) para **{_pct(analise.loc[analise['faixa_atraso'] == 'Atraso > 7 dias', TARGET].mean())}**
(em atraso superior a 7 dias) — monotônica nas quatro faixas:

{_fig('fig05_inssatisfacao_por_atraso.png', 'Taxa de insatisfação por faixa de atraso')}

**Pedidos com múltiplos itens.** Segundo achado em magnitude: pedidos com **1 item**
têm {_pct(p_neg_1_item)} de insatisfação, contra **{_pct(p_neg_multi)} em pedidos com
2 ou mais itens** — padrão compatível com envios fracionados/parciais.

**Frete, preço e distância.** Insatisfeitos pagam frete mais alto
(mediana R$ {_n(med_neg['frete_total'], 2)} vs. R$ {_n(med_pos['frete_total'], 2)}) e vêm de
pedidos mais distantes ({_inte(med_neg['distancia_km'])} km vs. {_inte(med_pos['distancia_km'])} km),
mas os efeitos são **pequenos**; a *participação* do frete no pedido é praticamente
igual entre os grupos ({_pct(med_neg['frete_ratio'])} vs. {_pct(med_pos['frete_ratio'])}).

{_fig('fig07_frete_preco_distancia.png', 'Frete, participação do frete e distância por nota')}

**Categorias e estados.** Existem categorias com insatisfação mais que o dobro da
média e estados com taxas distintas:

{_fig('fig06_categorias_inssatisfacao.png', 'Categorias com maior % de insatisfação')}
{_fig('fig08_insatisfacao_por_uf.png', 'Taxa de insatisfação por estado do cliente')}
""")

    partes.append(f"""---

## 5. Testes estatísticos de hipóteses

Com N ≈ {_inte(n)}, **pequenas diferenças se tornam "significativas"**; por isso
reportamos p-valor **e** tamanho de efeito. Hipóteses testadas (α = 0,05):

| # | Hipótese nula | Teste | Estatística | p-valor | Tamanho de efeito |
|---|---|---|---|---|---|
""")
    for i, r in tab_testes.iterrows():
        p_str = "< 1e-300" if r["p_valor"] == 0 else f"{r['p_valor']:.2e}"
        partes.append(
            f"| {i + 1} | {r['comparacao']} | {r['teste']} | {r['estatistica']} | "
            f"{p_str} | {r['medida_efeito']} = {r['valor_efeito']:.4f} |\n"
        )

    mwu_atraso = tab_testes[tab_testes["comparacao"].str.startswith("Atraso (dias)")].iloc[0]
    mwu_entrega = tab_testes[tab_testes["comparacao"].str.startswith("Tempo total")].iloc[0]
    p_or = "< 1e-300" if tab_odds["p_valor"] == 0 else f"{tab_odds['p_valor']:.2e}"
    or_df = pd.DataFrame([
        {"medida": "Taxa de insatisfação — com atraso", "valor": f"{tab_odds['taxa_inssatisf_atrasado_pct']:.2f}%"},
        {"medida": "Taxa de insatisfação — no prazo", "valor": f"{tab_odds['taxa_inssatisf_noprazo_pct']:.2f}%"},
        {"medida": "Odds ratio (atrasado vs. no prazo)", "valor": f"{tab_odds['odds_ratio']:.2f}"},
        {"medida": "IC 95% do odds ratio", "valor": f"{tab_odds['ic95_inf']:.2f} – {tab_odds['ic95_sup']:.2f}"},
        {"medida": "Risco relativo", "valor": f"{tab_odds['risco_relativo']:.2f}x"},
        {"medida": "Qui-quadrado / p-valor", "valor": f"{tab_odds['chi2']:.1f} / {p_or}"},
    ])
    partes.append(f"""
**Leitura dos principais resultados**

* **Atraso e lentidão (efeitos moderados):** Mann-Whitney p < 0,001 com
  r = {_n(mwu_atraso['valor_efeito'], 3)} no atraso (mediana
  {_n(mwu_atraso['mediana_inssatisf'])} vs. {_n(mwu_atraso['mediana_satisfeit'])} dias —
  insatisfeitos recebem com bem **menos antecedência**) e r = {_n(mwu_entrega['valor_efeito'], 3)}
  no tempo total de entrega ({_n(mwu_entrega['mediana_inssatisf'])} vs.
  {_n(mwu_entrega['mediana_satisfeit'])} dias).
* **Atraso × insatisfação (qui-quadrado + odds ratio):**

{_md(or_df)}

  → ter atrasado multiplica por **{_n(tab_odds['odds_ratio'])}x** as chances de nota 1-2
  ({_pct(tab_odds['taxa_inssatisf_atrasado_pct'] / 100)} de insatisfação entre atrasados vs.
  {_pct(tab_odds['taxa_inssatisf_noprazo_pct'] / 100)} no prazo) — o maior efeito do estudo
  (V de Cramér = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('Cumprimento'), 'valor_efeito'].iloc[0], 3)}).
* **Categoria e UF** associam-se à insatisfação (qui-quadrado, p < 0,001), porém com
  **V de Cramér pequeno** (categoria = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('Categoria'), 'valor_efeito'].iloc[0], 3)};
  UF = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('UF'), 'valor_efeito'].iloc[0], 3)}):
  efeitos reais, porém **diluídos** — servem para *onde* priorizar, não explicam o todo.
* **Forma de pagamento NÃO se associou** à insatisfação (p = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('Forma'), 'p_valor'].iloc[0], 3)};
  V = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('Forma'), 'valor_efeito'].iloc[0], 3)}) — hipótese não rejeitada.
* **Correlações de Spearman** com a nota: atraso (ρ = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Nota x atraso na entrega', 'valor_efeito'].iloc[0], 3)}) e
  tempo de entrega (ρ = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Nota x tempo total de entrega', 'valor_efeito'].iloc[0], 3)})
  lideram; distância (ρ = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Nota x distância cliente-vendedor', 'valor_efeito'].iloc[0], 3)}),
  frete-ratio (ρ = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Nota x participação do frete', 'valor_efeito'].iloc[0], 3)}) e
  valor (ρ = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Nota x valor do pedido', 'valor_efeito'].iloc[0], 3)}) têm
  associação **estatisticamente detectável, porém fraca**.
* **Kruskal-Wallis** entre faixas de atraso: p < 0,001 (epsilon² =
  {_n(tab_testes.loc[tab_testes['teste'] == 'Kruskal-Wallis', 'valor_efeito'].iloc[0], 4)}).

_Tabelas completas:_ `reports/tables/tab05_testes_hipotese.csv` e `tab06_odds_ratio_atraso.csv`.
""")

    partes.append(f"""---

## 6. Modelo: probabilidade de avaliação negativa

**Configuração:** separação estratificada 80/20, validação cruzada estratificada
({5} dobras), `class_weight='balanced'` para a desbalanceada base (positivos =
{_n(modelo['taxa_positivos_pct'], 1)}%). Features: {modelo['n_features_brutas']}
(variáveis operacionais + categoria/UF/pagamento/sazonealidade — **sem** texto de
comentário, para evitar vazamento).

| Modelo | AUC (CV) | AUC (teste) | PR-AUC (teste) | Precisão | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| Regressão logística | {m_lr['cv_roc_auc_mean']:.3f}±{m_lr['cv_roc_auc_std']:.3f} | {m_lr['roc_auc']:.3f} | {m_lr['pr_auc']:.3f} | {m_lr['precisao']:.3f} | {m_lr['recall']:.3f} | {m_lr['f1']:.3f} | {m_lr['brier']:.3f} |
| Gradient boosting (HGB) | {m_hgb['cv_roc_auc_mean']:.3f}±{m_hgb['cv_roc_auc_std']:.3f} | **{m_hgb['roc_auc']:.3f}** | {m_hgb['pr_auc']:.3f} | {m_hgb['precisao']:.3f} | {m_hgb['recall']:.3f} | {m_hgb['f1']:.3f} | {m_hgb['brier']:.3f} |

*Baseline do PR-AUC = taxa de positivos = {modelo['taxa_positivos_pct'] / 100:.3f}. Melhor
modelo: **{melhor.replace('_', ' ')}** (salvo em `outputs/modelo_insatisfacao.joblib`).*

{_fig('fig10_curvas_roc_pr.png', 'Curvas ROC e Precisão-Recall no conjunto de teste')}
{_fig('fig11_matriz_confusao.png', 'Matriz de confusão do melhor modelo (limiar 0.5)')}
{_fig('fig12_calibracao.png', 'Calibração das probabilidades previstas')}

A calibração é razoável — a probabilidade prevista é utilizável como **score de
risco** de insatisfação por pedido.
""")

    shap_ok = modelo.get("shap_disponivel", False)
    shap_secao = ""
    if shap_ok:
        shap_secao = f"""
{_fig('fig14_shap_importancia.png', 'Importância média |SHAP| — contribuição de cada variável')}
{_fig('fig15_shap_beeswarm.png', 'SHAP beeswarm — direção e magnitude do efeito de cada variável')}
"""

    partes.append(f"""---

## 7. Interpretação das variáveis mais relevantes

Esta é a etapa central da análise: entender **o que de fato move a nota**.

### 7.1. Regressão logística — odds ratios

Lendo a tabela abaixo: `odds_ratio > 1` → **aumenta** a chance de nota 1-2;
`< 1` → **protege**. Para numéricos, o OR refere-se a 1 desvio-padrão de aumento.

**Fatores que MAIS aumentam a insatisfação (OR por desvio-padrão):**

{_md(coef_risco[['variavel', 'odds_ratio']])}

**Fatores protetores (maior associação com nota alta):**

{_md(coef_prot[['variavel', 'odds_ratio']])}

*Tabela completa:* `reports/tables/tab07_odds_ratios_logistica.csv`.

### 7.2. Gradient boosting — importância por permutação

A permutação mede **quanto o AUC cai** quando a variável é embaralhada
(variação média sobre 10 repetições no conjunto de teste):

| # | Variável | Queda média de AUC | Desvio |
|---|---|---|---|
{chr(10).join(f"| {i + 1} | {r['variavel']} | {r['importancia_media']:.4f} | ±{r['importancia_std']:.4f} |" for i, (_, r) in enumerate(imp_hgb.iterrows()))}

{_fig('fig13_dependencia_parcial.png', 'Dependência parcial — atraso, frete e distância')}
{shap_secao}
### 7.3. Síntese das variáveis mais relevantes

1. **Prazo de entrega (atraso vs. prometido)** — maior efeito *univariado*:
   {_n(tab_odds['odds_ratio'])}x mais chances de nota 1-2 com atraso, relação
   dose-resposta ({_pct(analise.loc[analise['faixa_atraso'] == 'No prazo (até 7d antes)', TARGET].mean())}
   → {_pct(analise.loc[analise['faixa_atraso'] == 'Atraso > 7 dias', TARGET].mean())} nas faixas) e
   #2 em importância de permutação. A dependência parcial confirma o formato do efeito.
2. **Tempo total de entrega (lentidão percebida)** — #1 na permutação da regressão
   logística e #3 no boosting; maior efeito Mann-Whitney do estudo (r =
   {_n(mwu_entrega['valor_efeito'], 2)}): {_n(mwu_entrega['mediana_inssatisf'])} dias
   (insatisfeitos) vs. {_n(mwu_entrega['mediana_satisfeit'])} dias (satisfeitos);
   OR = {_n(tab_coef.loc[tab_coef['variavel'] == 'entrega_dias', 'odds_ratio'].iloc[0])} por desvio-padrão.
3. **Pedidos com múltiplos itens** — **#1 em importância de permutação** no boosting:
   {_pct(p_neg_1_item)} de insatisfação com 1 item vs. {_pct(p_neg_multi)} com 2+.
   Provável reflexo de envios fracionados/parciais — alavanca operacional direta.
4. **Categoria do produto** — #4 em permutação, com extremos nítidos na logística:
   `fashion_male_clothing` com OR = {_n(tab_coef.loc[tab_coef['variavel'] == 'categoria_grupo_fashion_male_clothing', 'odds_ratio'].iloc[0])}x
   e `food_drink` com OR = {_n(tab_coef.loc[tab_coef['variavel'] == 'categoria_grupo_food_drink', 'odds_ratio'].iloc[0], 2)}x
   (proteção).
5. **Frete total e distância** — associação **consistente, porém pequena**
   (r = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Frete total (R$)', 'valor_efeito'].iloc[0], 2)} e
   {_n(tab_testes.loc[tab_testes['comparacao'] == 'Distância cliente-vendedor (km)', 'valor_efeito'].iloc[0], 2)});
   a *participação* do frete no pedido é praticamente nula (r =
   {_n(tab_testes.loc[tab_testes['comparacao'] == 'Participação do frete no pedido (%)', 'valor_efeito'].iloc[0], 2)}).
6. **Manuseio (compra → postagem)** — efeito pequeno e residual (permutação 0,007;
   r = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Tempo até a postagem (dias)', 'valor_efeito'].iloc[0], 2)}).
7. **UF do cliente** — moderador com efeito pequeno (V = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('UF'), 'valor_efeito'].iloc[0], 2)};
   ex.: `uf_cliente_RJ` com OR = {_n(tab_coef.loc[tab_coef['variavel'] == 'uf_cliente_RJ', 'odds_ratio'].iloc[0])}x).
8. **O que NÃO explica a nota:** a **forma de pagamento** não teve associação
   (p = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('Forma'), 'p_valor'].iloc[0], 3)}) e a
   sazonalidade contribui de forma residual.
""")

    partes.append(f"""---

## 8. Recomendações gerenciais (tradução das evidências)

**R1 — Gestão ativa do prazo prometido (prioridade máxima).**
Com {_n(tab_odds['odds_ratio'])}x mais chances de insatisfação em atrasos e
{_pct(p_atraso_dado_neg)} dos insatisfeitos terem recebido com atraso, a alavanca de
maior retorno é: (i) revisar o *prometido x executado* por rota/transportadora,
(ii) antecipar aviso proativo ao cliente quando o risco de atraso aparecer
(usar o score do modelo, Seção 6) e (iii) negociar prazos realistas junto aos
vendedores/regiões mais lentos.

**R2 — Corrigir a operação de pedidos com múltiplos itens.**
Pedidos com 2+ itens têm {_pct(p_neg_multi)} de insatisfação vs. {_pct(p_neg_1_item)}
com 1 item — e é a variável de **maior importância no modelo**. Ações: rastrear
envios fracionados/parciais item a item, unificar envio quando possível e
comunicar proativamente quando uma parte do pedido atrasar.

**R3 — Atuação cirúrgica em categorias críticas.**
{chr(10).join(f"* **{r['categoria_grupo']}**: {_pct(r['pct_inssatisfacao'] / 100)} de insatisfação (n={_inte(r['n_pedidos'])})" for _, r in cats_top.iterrows())}
Auditar sellers, qualidade de embalagem e prazos dessas categorias; comparar com
a média geral ({_pct(taxa_neg)}).

**R4 — Frete: alavanca secundária (não priorizar sobre prazo/item).**
Insatisfeitos pagam frete um pouco mais alto (mediana R$ {_n(med_neg['frete_total'], 2)} vs.
R$ {_n(med_pos['frete_total'], 2)}), mas o efeito é **pequeno** e a participação do frete
no pedido é igual entre grupos — portanto, frete grátis isolado não resolve a
insatisfação; atuar apenas onde ele se cruza com atraso (rotas longas).

**R5 — Monitoramento contínuo por região e tempo.**
{chr(10).join(f"* **{r['uf_cliente']}**: {_pct(r['pct_inssatisfacao'] / 100)} (n={_inte(r['n_pedidos'])})" for _, r in ufs_top.iterrows())}
* Pior mês {pior_mes} vs. melhor mês {melhor_mes} — planejar capacidade nas janelas de pico.

**R6 — Usar o modelo como sistema de alerta.**
Com AUC ≈ {_n(max(m_lr['roc_auc'], m_hgb['roc_auc']), 2)} e calibração razoável, operar:
(a) score de risco por pedido **no momento da entrega**,
(b) atendimento proativo dos casos acima de limiar (priorizar por recall/precisão
conforme capacidade da equipe) e
(c) *fechamento do ciclo*: acompanhar a taxa de nota 1-2 dos pedidos intervidos.

**R7 — Escuta ativa pós-entrega.**
{_pct(coment_neg)} dos insatisfeitos escrevem comentário — tratar comentários/keywords
negativos como *sinal de alerta secundário* (monitoramento de reputação), sabendo
que o alerta mais barato é o **atraso**, que já é conhecido **antes** do cliente reclamar.
""")

    partes.append(f"""---

## 9. Limitações e próximos passos

* **Associação ≠ causalidade:** dados observacionais; confundidores possíveis
  (categoria × vendedor × região) — próximo passo: modelos com efeitos aleatórios
  ou experimentação (ex.: teste de promessa de prazo).
* **Sem conteúdo textual:** comentários não entram no modelo (vazamento); NLP
  (sentimento/tópicos) pode somar na triagem.
* **Sem info de devoluções/logística reversa** no dataset.
* Classificação com limiar 0,5 — calibrar conforme capacidade de atendimento
  (curva precisão×recall) na operação.
* Próximos passos sugeridos: monitorar drift mensal do modelo; adicionar
  tempo-de-vida do cliente (recompra) e avaliação por item; segmentar o
  score por transportadora.

## 10. Reprodutibilidade

```bash
pip install -r requirements.txt
python run_pipeline.py              # roda todas as etapas (load → eda → tests → model → report)
python run_pipeline.py --etapa eda  # executa apenas uma etapa
```

*Dados:* Kaggle — `olistbr/brazilian-ecommerce` (baixados automaticamente via
`kagglehub` ou em `data/raw/`). *Seed:* 42.

**Artefatos:** `reports/figures/` (15 figuras) · `reports/tables/` (9 tabelas) ·
`outputs/analysis_dataset.csv` · `outputs/resultados_modelo.json` ·
`outputs/modelo_insatisfacao.joblib`.
""")

    REPORT_MD.write_text("".join(partes), encoding="utf-8")
    print(f"[report] relatório -> {REPORT_MD}")


if __name__ == "__main__":
    gerar_relatorio()
