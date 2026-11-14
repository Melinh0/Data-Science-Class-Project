from __future__ import annotations
import json
import pandas as pd
from .config import (
    ANALYSIS_CSV, DICTIONARY_MD, FIGURES, MIN_N_RANKING, MODEL_JSON, PCA_JSON, REDUCAO_CSV,
    REPORT_MD, SCORE_COL, SELECAO_JSON, STATUS_CSV, TABLES, TARGET,
)
from .data_loader import carregar_dados
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
        texto = [i for i, c in enumerate(d.columns) if not pd.api.types.is_numeric_dtype(d[c])]
        return d.to_markdown(index=False, floatfmt=".2f", disable_numparse=texto)
    cab = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    linhas = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in linha) + " |"
        for linha in d.itertuples(index=False, name=None)
    ]
    return "\n".join([cab, sep] + linhas)
EXPLICACOES_FIGURAS: dict[str, str] = {
    "fig01_distribuicao_notas.png": (
        "A nota do cliente varia de 1 (pior) a 5 (melhor). A maioria das avaliações é "
        "nota 5 e existem poucos extremos (notas 1-2, que definem a insatisfação no "
        "estudo). A distribuição é assimétrica à esquerda, o que justifica tratar a "
        "insatisfação como a classe 'notas 1-2 vs. notas 3-5'."
    ),
    "fig02_evolucao_temporal.png": (
        "Evolução mensal da nota média e da taxa de insatisfação (notas 1-2): mostra que "
        "a percepção dos clientes varia ao longo do tempo e pode conter efeitos "
        "sazonais/capacidade a monitorar."
    ),
    "fig03_atraso_por_nota.png": (
        "Atraso em dias por nota de avaliação: insatisfeitos tendem a ter atraso maior e a "
        "receber com menos antecedência em relação ao prazo prometido. Mostra a relação "
        "entre atraso e o desfecho avaliativo."
    ),
    "fig04_insatisfacao_por_atraso.png": (
        "Taxa de insatisfação (notas 1-2) por faixa de atraso: relação de dose-resposta — "
        "quanto maior o atraso, maior a probabilidade de insatisfação, de forma consistente "
        "nas quatro faixas."
    ),
    "fig05_categorias.png": (
        "Percentual de insatisfação por categoria de produto (categorias com mais pedidos). "
        "Compara a taxa de notas 1-2 com a média geral (12,8%) para indicar categorias "
        "críticas onde a insatisfação se concentra."
    ),
    "fig06_insatisfacao_por_uf.png": (
        "Taxa de insatisfação por estado do cliente: destaca estados com risco de "
        "insatisfação acima da média, apontando onde priorizar ações regionais."
    ),
    "fig07_dispersao_frete_distancia.png": (
        "Scatterplot da distância cliente-vendedor contra o frete total e contra a "
        "participação do frete no pedido, com a cor indicando a nota. Quanto maior a "
        "distância, maior o frete; a nota não forma faixas separadas, ou seja, a "
        "associação existe mas é fraca."
    ),
    "fig08_ranking_spearman.png": (
        "Ranking das variáveis numéricas pela correlação de Spearman com a nota — indica "
        "quais variáveis se relacionam mais com a insatisfação. Barras em vermelho têm "
        "p < 0,05."
    ),
    "fig09_mutual_information.png": (
        "Informação mútua entre cada variável e a insatisfação — medida de associação que "
        "não assume linearidade. A linha vermelha é o maior valor obtido por variáveis-sombra "
        "(ruído embaralhado): só o que passa dela carrega sinal real."
    ),
    "fig10_consenso_selecao.png": (
        "Convergência dos cinco métodos de seleção (dois filtros, um wrapper e dois "
        "embutidos). Cada linha é uma variável e cada coluna um método; as variáveis "
        "marcadas em 3 ou mais de 5 métodos foram mantidas."
    ),
    "fig11_valores_impossiveis.png": (
        "Valores impossíveis ou incoerentes detectados nas regras de domínio (datas "
        "invertidas, distância fora do Brasil, preço zero). Foram transformados em ausentes "
        "antes da divisão treino/teste, para que a imputação seja aprendida só no treino."
    ),
    "fig12_pca_scree.png": (
        "Diagnóstico do PCA: à esquerda os autovalores caem abaixo do ruído simulado "
        "(análise paralela) e do critério de Kaiser; à direita a variância acumulada mostra "
        "quantos componentes são necessários para reter a informação."
    ),
    "fig13_pca_auc_vs_k.png": (
        "AUC de validação cruzada em função do número de componentes k: compara o "
        "desempenho do PCA com o modelo sem PCA. O k escolhido é o menor k cujo desempenho "
        "permanece dentro de um erro-padrão do modelo sem PCA."
    ),
    "fig14_pca_cp1_cp2.png": (
        "Projeção dos pedidos no plano dos dois primeiros componentes principais e as "
        "variáveis que mais definem esses eixos. Os dois grupos (satisfeito/insatisfeito) "
        "se sobrepõem no espaço reduzido, sinal de que o PCA sozinho não separa classes."
    ),
    "fig15_shap_importancia.png": (
        "Importância média |SHAP| por variável: contribuição média de cada variável para as "
        "decisões do modelo — baseada nos valores de Shapley, que distribuem o 'crédito' de "
        "forma justa entre as variáveis."
    ),
    "fig16_curvas_roc.png": (
        "Curvas ROC (sensibilidade × 1-especificidade) no conjunto de teste: a AUC resume o "
        "poder discriminativo do modelo (quanto mais próximo de 1, melhor); a diagonal "
        "tracejada é o classificador aleatório (AUC = 0,5)."
    ),
    "fig17_matriz_confusao.png": (
        "Matriz de confusão (real × previsto) com o limiar escolhido na validação cruzada: "
        "quantifica acertos (TN/TP) e erros (FP/FN); o recall (TP/(TP+FN)) mostra o quanto o "
        "modelo identificou os insatisfeitos."
    ),
    "fig18_calibracao.png": (
        "Curva de calibração: probabilidade prevista pelo modelo vs. frequência real "
        "observada. Se a curva acompanha a diagonal ideal, as probabilidades são confiáveis "
        "e podem ser usadas como score de risco."
    ),
    "fig19_comparacao_representacoes.png": (
        "AUC no teste para as três representações (original, selecionada e PCA) nos dois "
        "modelos, com o número de colunas de cada uma: mostra o custo em desempenho da "
        "redução de dimensionalidade."
    ),
    "fig20_roc_representacoes.png": (
        "Curvas ROC no teste comparando as três representações dentro de cada modelo: as "
        "curvas praticamente se sobrepõem, o que confirma que a seleção de variáveis e o "
        "PCA preservam a informação preditiva."
    ),
    "fig28_comparacao_representacoes.png": (
        "Comparação entre as três representações avaliadas (original, selecionada e PCA) em "
        "dimensionalidade e AUC no teste, para dois modelos. Mostra o que se ganha e o que se "
        "perde ao reduzir a base: uma representação só vale a pena se mantém o desempenho com "
        "menos colunas."
    ),
    "fig29_pca_cp1_cp2.png": (
        "Projeção dos pedidos no plano das duas primeiras componentes principais (CP1 × CP2), "
        "colorida pela insatisfação, com vetores de carga das variáveis. Os vetores indicam qual "
        "variável puxa cada direção; a sobreposição dos grupos mostra que essas duas componentes "
        "separam pouco o alvo."
    ),
    "fig30_pca_auc_vs_k.png": (
        "AUC obtida por validação cruzada em função do número de componentes k, comparada com o "
        "modelo sem PCA e com o k da regra dos 90%. Serve para escolher k pela informação "
        "preditiva preservada, e não apenas por uma porcentagem de variância."
    ),
    "fig31_consenso_selecao.png": (
        "Matriz de convergência dos cinco métodos de seleção de atributos (filtro MI e AUC, "
        "wrapper forward, embutido L1 e RF). Cada marca indica que o método escolheu a variável; "
        "o consenso (>= 3 de 5) define o subconjunto final."
    ),
}
NOMES_MODELO = {
    "regressao_logistica": "regressão logística",
    "gradient_boosting": "gradient boosting",
}
def _modelo_nome(chave: str) -> str:
    return NOMES_MODELO.get(chave, chave.replace("_", " "))
def _fig(nome: str, legenda: str) -> str:
    bloco = f"![{legenda}](../reports/figures/{nome})\n\n*Figura: {legenda}*"
    explicacao = EXPLICACOES_FIGURAS.get(nome)
    if explicacao:
        bloco += f"\n\n{explicacao}"
    return bloco
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")
def _inte(x) -> str:
    return f"{int(x):,}".replace(",", ".")
def _n(x, casas: int = 1) -> str:
    return f"{x:.{casas}f}".replace(".", ",")
def gerar_relatorio() -> None:
    analise = carregar_analise()
    status = pd.read_csv(STATUS_CSV)
    modelo = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
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
    tab_rank = pd.read_csv(TABLES / "tab12_ranking_features.csv")
    sel = json.loads(SELECAO_JSON.read_text(encoding="utf-8"))
    pca_info = json.loads(PCA_JSON.read_text(encoding="utf-8"))
    tab_red = pd.read_csv(REDUCAO_CSV)
    tab_grupos = pd.read_csv(TABLES / "tab20_importancia_grupos.csv")
    _caminho_shap = TABLES / "tab21_shap_por_feature.csv"
    tab_shap = pd.read_csv(_caminho_shap) if _caminho_shap.exists() else None
    adic = modelo["validacoes_adicionais"]
    exib_rank = tab_rank.head(12)[
        ["feature", "tipo", "n_metodos", "mi_insatisfeito", "auc_univariada", "selecionada"]
    ].copy()
    exib_rank["mi_insatisfeito"] = exib_rank["mi_insatisfeito"].map(lambda v: f"{v:.4f}")
    exib_rank["auc_univariada"] = exib_rank["auc_univariada"].map(lambda v: f"{v:.3f}")
    exib_rank["selecionada"] = exib_rank["selecionada"].map(lambda v: "sim" if v else "não")
    exib_rank.columns = [
        "Feature", "Tipo", "Métodos que a selecionam (de 5)", "MI com o alvo",
        "AUC univariada (CV)", "Selecionada",
    ]
    def _valor(feature: str, col: str) -> float:
        return float(tab_rank.loc[tab_rank["feature"] == feature, col].iloc[0])
    selecionadas = sel["selecionadas"]
    descartadas = sel["descartadas"]
    cats_top = tab_cat[tab_cat["n_pedidos"] >= MIN_N_RANKING].head(5)
    ufs_top = tab_uf[tab_uf["n_pedidos"] >= MIN_N_RANKING].head(3)
    ufs_vol = tab_uf.sort_values("excesso_insatisfeitos", ascending=False).head(3)
    taxa_faixa = analise.groupby("faixa_atraso")[TARGET].mean()
    p_pag = float(tab_testes.loc[tab_testes["comparacao"].str.startswith("Forma"), "p_valor"].iloc[0])
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
    coef_ok = tab_coef_exib[
        tab_coef_exib["n_treino"].isna() | (tab_coef_exib["n_treino"] >= MIN_N_RANKING)
    ].copy()
    coef_ok["n_treino"] = coef_ok["n_treino"].map(lambda v: "—" if pd.isna(v) else _inte(v))
    coef_risco = coef_ok[coef_ok["odds_ratio"] > 1].head(6)
    coef_prot = coef_ok[coef_ok["odds_ratio"] < 1].tail(6).iloc[::-1]
    imp_hgb = tab_imp[tab_imp["modelo"] == "gradient_boosting"].head(8)
    def _cat_extrema(risco: bool):
        d = coef_ok[coef_ok["variavel"].str.startswith("categoria: ")]
        d = d[d["odds_ratio"] > 1].head(1) if risco else d[d["odds_ratio"] < 1].tail(1)
        if len(d) == 0:
            return "—", float("nan")
        return d.iloc[0]["variavel"].replace("categoria: ", ""), float(d.iloc[0]["odds_ratio"])
    cat_risco, or_cat_risco = _cat_extrema(True)
    cat_prot, or_cat_prot = _cat_extrema(False)
    met = modelo["metricas"]
    melhor = modelo["melhor_modelo"]
    m_lr = met["regressao_logistica"]
    m_hgb = met["gradient_boosting"]
    pct_sem_comentario_neg = 1 - coment_neg
    m_best = met[melhor]
    calib_boa = m_best["brier_skill"] > 0 and m_best["ece"] <= 0.03
    calib_txt = (
        "As probabilidades são utilizáveis como estimativa de risco por pedido."
        if calib_boa else
        "As probabilidades **não** devem ser lidas como risco absoluto: use o escore apenas para "
        "ordenar pedidos ou recalibre antes de usá-lo como probabilidade."
    )
    g_gb = tab_grupos[tab_grupos["modelo"] == "gradient_boosting"].reset_index(drop=True)
    def _pos_grupo(nome: str) -> tuple[int, float]:
        i = int(g_gb.index[g_gb["grupo"] == nome][0])
        return i + 1, float(g_gb.loc[i, "queda_auc_media"])
    pos_g_prazo, queda_g_prazo = _pos_grupo("Prazo e tempo de entrega")
    pos_g_pedido, queda_g_pedido = _pos_grupo("Pedido (itens, preço, frete)")
    imp_full = tab_imp[tab_imp["modelo"] == "gradient_boosting"].reset_index(drop=True)
    pos_perm_nitens = int(imp_full.index[imp_full["variavel"] == "n_itens"][0]) + 1
    def _imp(v: str) -> float:
        return float(imp_full.loc[imp_full["variavel"] == v, "importancia_media"].iloc[0])
    if tab_shap is not None:
        shap_lider = tab_shap.iloc[0]["feature"]
        pos_shap_nitens = int(tab_shap.index[tab_shap["feature"] == "n_itens"][0]) + 1
        shap_txt = (f"No SHAP agregado por variável, **`{shap_lider}`** lidera e `n_itens` ocupa a "
                    f"posição {pos_shap_nitens}.")
    else:
        shap_txt = "O SHAP não estava disponível nesta execução."
    or_entrega_cond = float(tab_coef.loc[tab_coef["variavel"] == "entrega_dias", "odds_ratio"].iloc[0])
    or_atraso_cond = float(tab_coef.loc[tab_coef["variavel"] == "atraso_dias", "odds_ratio"].iloc[0])
    partes: list[str] = []
    partes.append(f"""# Quais fatores estão associados à insatisfação dos clientes?
**Análise exploratória, testes de hipótese e modelo preditivo aplicados ao dataset
[Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**
> **Pergunta de negócio:** *Quais fatores estão associados à insatisfação dos
> clientes e como essas evidências podem apoiar decisões de melhoria operacional?*
---
* **{_inte(n)} pedidos entregues com avaliação** compõem a base de análise; a taxa de
  insatisfação (notas 1-2) é de **{_pct(taxa_neg)}** (nota média = {_n(media_nota, 2)}).
* **A pontualidade da entrega é o fator mais associado à insatisfação** (atraso em relação ao
  prazo prometido e tempo total de entrega): enquanto {_pct(p_atraso_dado_pos)} dos clientes
  satisfeitos receberam com atraso, **{_pct(p_atraso_dado_neg)} dos insatisfeitos** receberam com
  atraso. Pedidos atrasados têm **{_n(tab_odds['odds_ratio'])}x mais chances** de gerar nota 1-2
  (IC95% {_n(tab_odds['ic95_inf'])}–{_n(tab_odds['ic95_sup'])}; p < 0,001).
* **Pedidos com múltiplos itens** têm mais insatisfação: {_pct(p_neg_1_item)} com 1 item vs.
  **{_pct(p_neg_multi)} com 2+ itens**. É uma associação bruta; a explicação por envios
  fracionados é hipótese, não verificável com estes dados.
* **Frete, distância e categoria** associam-se à nota, porém com efeito **pequeno**
  (|r| ≤ {_n(max_abs_efeto_frete_dist, 2)} nos testes da Seção 5). Já a **forma de
  pagamento NÃO** se associou (p = {_n(p_pag, 3)}).
* O melhor modelo (**{_modelo_nome(melhor)}**) atinge **AUC = {_n(m_best['roc_auc'], 3)}** no teste
  ({_n(adic['temporal']['roc_auc'], 3)} em validação temporal), com probabilidades
  {'calibradas' if calib_boa else 'pouco calibradas'}.
  Esse desempenho usa o atraso **realizado**, só conhecido na entrega; sem `atraso_dias` e
  `entrega_dias` (cenário pré-entrega) o AUC cai para **{_n(adic['pre_entrega']['roc_auc'], 3)}**.
* As recomendações (Seção 9): gestão do **prazo prometido**, investigação dos **pedidos com
  múltiplos itens**, auditoria em **categorias críticas** (com amostra suficiente) e uso do
  escore para **priorizar o atendimento pós-entrega**.
""")
    partes.append(f"""---
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
| Qualidade dos dados | Valores impossíveis (datas incoerentes, parcelas = 0, peso = 0, distância > 4.500 km) viram ausentes; `valor_pedido` e `preco_unitario_medio` (derivadas exatas) ficam fora do modelo; detalhes em `relatorio_etapas.md` |
| Treino/teste | Divisão **antes** de qualquer ajuste; imputação, escala, seleção de atributos, PCA e limiar usam só o treino |
**Volume da base de análise:** {_inte(n)} pedidos · notas de 1 a 5 ·
{analise['categoria_grupo'].nunique()} grupos de categoria · {analise['uf_cliente'].nunique()} UFs.
""")
    partes.append(f"""---
A base completa tem **{_inte(len(status))} pedidos**; a maior parte chega ao cliente
(`delivered`), e pedidos cancelados/indisponíveis concentram avaliações ausentes — o que
justifica o recorte de pedidos entregues com avaliação.
As notas concentram-se em **5** ({_pct(tab_notas.loc[tab_notas['nota'] == 5, 'pct'].iloc[0] / 100)}),
mas notas **1** respondem por {_pct(tab_notas.loc[tab_notas['nota'] == 1, 'pct'].iloc[0] / 100)} —
mais do que 2 e 3 somadas ({_pct((tab_notas.loc[tab_notas['nota'].isin([2, 3]), 'pct'].sum()) / 100)}),
padrão típico de avaliações "ou amo muito, ou odeio".
{_fig('fig01_distribuicao_notas.png', 'Distribuição das notas de avaliação')}
No tempo, a insatisfação oscila mês a mês: o pior mês da série foi **{pior_mes}**
({_pct(mensal.loc[pior_mes, 'neg'])} de notas 1-2) e o melhor **{melhor_mes}**
({_pct(mensal.loc[melhor_mes, 'neg'])}), o que sugere efeitos de sazonalidade/capacidade
a monitorar.
{_fig('fig02_evolucao_temporal.png', 'Evolução mensal da nota média e da insatisfação')}
Ainda na EDA: avaliações com comentário são **{_pct(coment_neg)}** entre insatisfeitos
vs. **{_pct(coment_pos)}** entre satisfeitos — evidência de que o cliente insatisfeito
tende a *se manifestar* (sinal útil para monitoramento, embora observado após a nota).
""")
    partes.append(f"""---
**Atraso.** A mediana do atraso piora conforme a nota cai: pedidos bem avaliados
chegam, na mediana, **{_n(abs(med_pos['atraso_dias']))} dias antes** do prazo prometido,
enquanto os insatisfeitos têm apenas **{_n(abs(med_neg['atraso_dias']))} dias de
antecedência** — e {_pct(p_atraso_dado_neg)} deles ultrapassam o prazo de fato.
{_fig('fig03_atraso_por_nota.png', 'Distribuição do atraso por nota de avaliação')}
A taxa de insatisfação cresce de forma **monotônica nas quatro faixas**:
**{_pct(taxa_faixa.get('Adiantado > 7 dias', float('nan')))}** (adiantado > 7 dias),
**{_pct(taxa_faixa.get('No prazo (até 7d antes)', float('nan')))}** (no prazo, até 7 dias antes),
**{_pct(taxa_faixa.get('Atraso 1-7 dias', float('nan')))}** (atraso de 1 a 7 dias) e
**{_pct(taxa_faixa.get('Atraso > 7 dias', float('nan')))}** (atraso > 7 dias). Agrupando todos os
pedidos sem atraso, a taxa é {_pct(tab_odds['taxa_inssatisf_noprazo_pct'] / 100)}, contra
{_pct(tab_odds['taxa_inssatisf_atrasado_pct'] / 100)} entre os atrasados.
{_fig('fig04_insatisfacao_por_atraso.png', 'Taxa de insatisfação por faixa de atraso')}
**Pedidos com múltiplos itens.** Pedidos com **1 item** têm {_pct(p_neg_1_item)} de
insatisfação, contra **{_pct(p_neg_multi)} em pedidos com 2 ou mais itens**. A explicação por envios
fracionados é uma hipótese: o dataset não registra remessas por item, então ela precisa ser
verificada com dados de logística.
**Frete, preço e distância.** Insatisfeitos pagam frete mais alto
(mediana R$ {_n(med_neg['frete_total'], 2)} vs. R$ {_n(med_pos['frete_total'], 2)}) e vêm de
pedidos mais distantes ({_inte(med_neg['distancia_km'])} km vs. {_inte(med_pos['distancia_km'])} km),
mas os efeitos são **pequenos**; a *participação* do frete no pedido é praticamente
igual entre os grupos (mediana {_pct(med_neg['frete_ratio'])} vs. {_pct(med_pos['frete_ratio'])}).
O scatterplot abaixo mostra a relação dispersa entre distância e frete:
{_fig('fig07_dispersao_frete_distancia.png', 'Scatterplot: distância × frete total e × participação do frete')}
**Categorias e estados.** Os gráficos consideram só grupos com pelo menos {MIN_N_RANKING}
pedidos (taxas de grupos pequenos são instáveis; os intervalos de confiança estão em `tab03` e `tab04`):
{_fig('fig05_categorias.png', 'Categorias com maior % de insatisfação')}
{_fig('fig06_insatisfacao_por_uf.png', 'Taxa de insatisfação por estado do cliente')}
**PCA (diagnóstico multivariado).** Para verificar se as variáveis numéricas são
realmente independentes entre si, aplicamos a análise de componentes principais (PCA):
os gráficos mostram onde o sinal cai para o ruído, quantos componentes são necessários
e o que cada um representa (ver `reports/tables/tab13_pca_variancia.csv` e
`tab14_pca_necessidade.csv`).
{_fig('fig12_pca_scree.png', 'Diagnóstico do PCA: autovalores e variância acumulada')}
{_fig('fig13_pca_auc_vs_k.png', 'Informação preditiva preservada pelo PCA em função de k')}
{_fig('fig14_pca_cp1_cp2.png', 'Projeção no plano CP1 × CP2 e cargas das variáveis')}
""")
    partes.append(f"""---
Com N ≈ {_inte(n)}, **pequenas diferenças se tornam "significativas"**; por isso
reportamos p-valor **e** tamanho de efeito. Hipóteses testadas (α = 0,05):
||---|---|---|---|---|---|
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
  {_pct(tab_odds['taxa_inssatisf_noprazo_pct'] / 100)} no prazo) — a maior associação entre as variáveis de prazo
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
    d_sel = tab_red.loc[(tab_red["representacao"] == "Selecionada") & (tab_red["modelo"] == "regressao_logistica"), "delta_auc_teste_vs_original"].iloc[0]
    d_pca = tab_red.loc[(tab_red["representacao"] == "PCA") & (tab_red["modelo"] == "regressao_logistica"), "delta_auc_teste_vs_original"].iloc[0]
    pca_sem_reducao = pca_info["k_escolhido"] >= pca_info["n_numericas"]
    txt_pca = (
        f"o PCA não permitiu reduzir (o critério preditivo exigiu as {pca_info['n_numericas']} componentes, "
        "ou seja, só rotaciona as variáveis)" if pca_sem_reducao
        else f"trocar as numéricas por {pca_info['k_escolhido']} componentes principais muda em {d_pca:+.4f}"
    )
    partes.append(f"""---
Os testes da Seção 5 medem associação univariada. Para decidir **quais variáveis manter**, a seleção foi feita
**só no treino**, com cinco métodos: dois filtros (informação mútua e AUC univariada), um wrapper (forward com
regressão logística) e dois embutidos (L1 e random forest). Todos pontuam as mesmas
{len(selecionadas) + len(descartadas)} features e comparam cada uma com **cópias embaralhadas dela mesma**
(features-sombra); a variável é selecionada se ao menos 3 de 5 métodos a escolhem. Critérios, custo do wrapper e
convergências estão no `relatorio_etapas.md` (Seção 3).
{_md(exib_rank)}
**Leitura**
1. **Selecionadas ({len(selecionadas)}):** {', '.join('`' + f + '`' for f in selecionadas)}.
2. **Descartadas ({len(descartadas)}):** {', '.join('`' + f + '`' for f in descartadas)}.
3. **Pagamento:** `pagamento_tipo` é escolhida por {int(_valor('pagamento_tipo', 'n_metodos'))} de 5 métodos,
   coerente com a Seção 5 (p = {_n(p_pag, 3)}).
4. **Variáveis de pouco sinal:** `hora_compra` é escolhida por {int(_valor('hora_compra', 'n_metodos'))} e
   `aprovacao_h` por {int(_valor('aprovacao_h', 'n_metodos'))} de 5 métodos.
5. **Informação preservada:** treinar só com as selecionadas muda o AUC de teste em {d_sel:+.4f} (regressão
   logística); {txt_pca}
   (`reports/tables/tab16_comparacao_representacoes.csv`).
{_fig('fig08_ranking_spearman.png', 'Correlação de Spearman de cada variável numérica com a nota (treino)')}
{_fig('fig09_mutual_information.png', 'Informação mútua com a insatisfação vs. variáveis-sombra')}
{_fig('fig10_consenso_selecao.png', 'Convergência dos cinco métodos de seleção')}
""")
    partes.append(f"""---
**Configuração:** separação estratificada 80/20 (a mesma usada na seleção e no PCA) e validação cruzada
estratificada (5 dobras) no treino. **Sem `class_weight`:** pesos de classe deslocam as probabilidades — numa
versão anterior com `class_weight='balanced'`, o modelo superestimava o risco e as probabilidades ficavam
piores que as de um preditor constante. O desbalanceamento (positivos = {_n(modelo['taxa_positivos_pct'], 1)}%) é tratado pelo
**limiar de decisão**, escolhido pelo F1 máximo nas previsões fora-da-dobra do treino. O melhor modelo é escolhido
pelo AUC de validação cruzada, não pelo teste. Features: {modelo['n_features_brutas']} (operacionais +
categoria/UF/pagamento/calendário; **sem** texto de comentário, para evitar vazamento; `valor_pedido` e
`preco_unitario_medio` ficaram de fora por serem derivadas exatas de outras variáveis).
**Escolha dos modelos**
A regressão logística é o baseline interpretável: P(insatisfeito|X) = 1/(1+exp(−z)); seus coeficientes dão odds
ratios **condicionais** (ajustados pelas demais variáveis). O HistGradientBoosting captura não linearidades e
interações (por exemplo, o salto de probabilidade quando o atraso passa de zero) com árvores sequenciais que
minimizam a log-loss.
| Modelo | AUC (CV) | AUC (teste) | Acurácia | Precisão | Recall | F1 | Limiar |
|---|---|---|---|---|---|---|---|
| Regressão logística | {m_lr['cv_roc_auc_mean']:.3f}±{m_lr['cv_roc_auc_std']:.3f} | {m_lr['roc_auc']:.3f} | {m_lr['acuracia']:.3f} | {m_lr['precisao']:.3f} | {m_lr['recall']:.3f} | {m_lr['f1']:.3f} | {m_lr['limiar']:.2f} |
| Gradient boosting (HGB) | {m_hgb['cv_roc_auc_mean']:.3f}±{m_hgb['cv_roc_auc_std']:.3f} | {m_hgb['roc_auc']:.3f} | {m_hgb['acuracia']:.3f} | {m_hgb['precisao']:.3f} | {m_hgb['recall']:.3f} | {m_hgb['f1']:.3f} | {m_hgb['limiar']:.2f} |
*Métricas no teste (n = 19.165). Acurácia, precisão, recall e F1 usam o limiar escolhido na
validação cruzada (F1 máximo); a AUC é independente de limiar. Melhor modelo:
**{_modelo_nome(melhor)}** (salvo em `outputs/modelo_insatisfacao.joblib`).*
{_fig('fig16_curvas_roc.png', 'Curvas ROC no conjunto de teste')}
{_fig('fig17_matriz_confusao.png', 'Matriz de confusão do melhor modelo (limiar escolhido na validação cruzada)')}
{_fig('fig18_calibracao.png', 'Calibração das probabilidades previstas')}
**Calibração.** A curva de calibração acompanha a diagonal: a probabilidade média prevista é
{_pct(m_best['prob_media_prevista'])} contra {_pct(m_best['taxa_observada'])} observada.
{calib_txt}
**Validações adicionais**
| Cenário | AUC | Acurácia | Recall | F1 |
|---|---|---|---|---|
| Teste aleatório (principal) | {_n(m_best['roc_auc'], 3)} | {_n(m_best['acuracia'], 3)} | {_n(m_best['recall'], 3)} | {_n(m_best['f1'], 3)} |
| Validação temporal (treina nos 80% mais antigos, testa nos 20% mais recentes) | {_n(adic['temporal']['roc_auc'], 3)} | {_n(adic['temporal']['acuracia'], 3)} | {_n(adic['temporal']['recall'], 3)} | {_n(adic['temporal']['f1'], 3)} |
| Modelo pré-entrega (sem `atraso_dias` e `entrega_dias`, que só existem após a entrega) | {_n(adic['pre_entrega']['roc_auc'], 3)} | {_n(adic['pre_entrega']['acuracia'], 3)} | {_n(adic['pre_entrega']['recall'], 3)} | {_n(adic['pre_entrega']['f1'], 3)} |
""")
    shap_ok = modelo.get("shap_disponivel", False)
    shap_secao = ""
    if shap_ok:
        shap_secao = f"""
{_fig('fig15_shap_importancia.png', 'Importância média |SHAP| por variável')}
"""
    g_exib = g_gb[["grupo", "features", "queda_auc_media", "queda_auc_dp"]].copy()
    g_exib["queda_auc_media"] = g_exib["queda_auc_media"].map(lambda v: f"{v:.4f}")
    g_exib["queda_auc_dp"] = g_exib["queda_auc_dp"].map(lambda v: f"±{v:.4f}")
    g_exib.columns = ["Grupo", "Variáveis do grupo", "Queda média de AUC", "Desvio"]
    partes.append(f"""---
`odds_ratio > 1` → **aumenta** a chance de nota 1-2; `< 1` → **protege**. Para numéricos, o OR refere-se a 1
desvio-padrão. São odds ratios **ajustados pelas demais variáveis**: por isso o OR de `atraso_dias`
({_n(or_atraso_cond, 2)}) é bem menor que o OR bruto de {_n(tab_odds['odds_ratio'])}x, já que `entrega_dias`
(OR = {_n(or_entrega_cond, 2)}), correlacionada com ela, absorve parte do efeito. Categorias com menos de
{MIN_N_RANKING} pedidos no treino ficam de fora (estimativas instáveis).
**Fatores que MAIS aumentam a insatisfação:**
{_md(coef_risco[['variavel', 'odds_ratio', 'n_treino']])}
**Fatores protetores:**
{_md(coef_prot[['variavel', 'odds_ratio', 'n_treino']])}
*Tabela completa:* `reports/tables/tab07_odds_ratios_logistica.csv`.
**Por variável (permutação individual):** queda do AUC no teste ao embaralhar uma coluna
(10 repetições). Atenção: variáveis correlacionadas **dividem o crédito** (embaralhar `atraso_dias` ainda deixa
`entrega_dias` informar o modelo), então variáveis independentes parecem relativamente mais importantes.
||---|---|---|---|
{chr(10).join(f"| {i + 1} | {r['variavel']} | {r['importancia_media']:.4f} | ±{r['importancia_std']:.4f} |" for i, (_, r) in enumerate(imp_hgb.iterrows()))}
**Por grupo (permutação conjunta):** as colunas de cada grupo são embaralhadas juntas, o que elimina a
divisão de crédito entre variáveis correlacionadas.
{_md(g_exib)}
{shap_txt}
{shap_secao}
1. **Pontualidade da entrega (atraso vs. prometido)** — associação *univariada* mais forte:
   {_n(tab_odds['odds_ratio'])}x mais chances de nota 1-2 com atraso e relação dose-resposta
   ({_pct(taxa_faixa.get('Adiantado > 7 dias', float('nan')))} → {_pct(taxa_faixa.get('Atraso > 7 dias', float('nan')))}
   nas faixas). O grupo "Prazo e tempo de entrega" é o **   {_n(queda_g_prazo, 3)}).
2. **Tempo total de entrega** — efeito Mann-Whitney de r = {_n(mwu_entrega['valor_efeito'], 2)}:
   {_n(mwu_entrega['mediana_inssatisf'])} dias (insatisfeitos) vs. {_n(mwu_entrega['mediana_satisfeit'])} dias
   (satisfeitos); OR condicional = {_n(or_entrega_cond)} por desvio-padrão.
3. **Pedidos com múltiplos itens** — associação bruta de {_pct(p_neg_1_item)} (1 item) contra
   {_pct(p_neg_multi)} (2+). Na permutação individual `n_itens` ocupa a posição {pos_perm_nitens} de
   {len(imp_full)}, mas isso exagera sua importância relativa: `atraso_dias` e `entrega_dias` dividem o crédito
   entre si. Na permutação por grupo, "Prazo e tempo de entrega" (queda de AUC {_n(queda_g_prazo, 3)}) pesa
   {_n(queda_g_prazo / max(queda_g_pedido, 1e-9), 1)}x mais que "Pedido (itens, preço, frete)"
   ({_n(queda_g_pedido, 3)},   grupo mais importante), mas **a causa** (envios fracionados) segue como hipótese a verificar.
4. **Categoria do produto** (com ao menos {MIN_N_RANKING} pedidos no treino) — maior OR: `{cat_risco}`
   ({_n(or_cat_risco)}x); maior efeito protetor: `{cat_prot}` ({_n(or_cat_prot, 2)}x).
5. **Frete total e distância** — associação **consistente, porém pequena**
   (r = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Frete total (R$)', 'valor_efeito'].iloc[0], 2)} e
   {_n(tab_testes.loc[tab_testes['comparacao'] == 'Distância cliente-vendedor (km)', 'valor_efeito'].iloc[0], 2)});
   a *participação* do frete no pedido é praticamente nula (r =
   {_n(tab_testes.loc[tab_testes['comparacao'] == 'Participação do frete no pedido (%)', 'valor_efeito'].iloc[0], 2)}).
6. **Manuseio (compra → postagem)** — efeito pequeno (permutação {_n(_imp('manuseio_dias'), 3)};
   r = {_n(tab_testes.loc[tab_testes['comparacao'] == 'Tempo até a postagem (dias)', 'valor_efeito'].iloc[0], 2)}).
7. **UF do cliente** — moderador de efeito pequeno (V = {_n(tab_testes.loc[tab_testes['comparacao'].str.startswith('UF'), 'valor_efeito'].iloc[0], 2)}).
8. **O que NÃO explica a nota:** a **forma de pagamento** não teve associação (p = {_n(p_pag, 3)}).
""")
    partes.append(f"""---
**R1 — Gestão ativa do prazo prometido (prioridade máxima).**
Com {_n(tab_odds['odds_ratio'])}x mais chances de insatisfação em atrasos e {_pct(p_atraso_dado_neg)} dos
insatisfeitos terem recebido com atraso, a alavanca de maior retorno é: (i) revisar o *prometido x executado* por
rota/transportadora, (ii) negociar prazos realistas com os vendedores/regiões mais lentos e (iii) **avisar o
cliente quando o atraso for provável**. Atenção a (iii): o escore do modelo principal usa `atraso_dias` e
`entrega_dias`, que só existem **depois** da entrega, então não serve para o aviso antecipado. O modelo pré-entrega
(sem essas variáveis, mas com `manuseio_dias`, conhecido na postagem) tem AUC = {_n(adic['pre_entrega']['roc_auc'], 3)},
contra {_n(m_best['roc_auc'], 3)} do modelo completo: serve como triagem, mas não prevê o atraso em si. O aviso antecipado
confiável exige dados de rastreio/ETA que não fazem parte do dataset.
**R2 — Investigar os pedidos com múltiplos itens (hipótese a verificar).**
Pedidos com 2+ itens têm {_pct(p_neg_multi)} de insatisfação contra {_pct(p_neg_1_item)} com 1 item. A
importância desse fator no modelo depende da métrica (permutação individual: posição {pos_perm_nitens} de
{len(imp_full)}; grupo "Pedido": posição {pos_g_pedido} de {len(g_gb)}). Antes de agir, confirmar com dados de
logística se há envios fracionados/vários vendedores; se confirmado, rastrear remessas item a item, unificar o
envio quando possível e comunicar proativamente quando uma parte atrasar.
**R3 — Auditar categorias críticas (com amostra suficiente: n ≥ {MIN_N_RANKING}).**
{chr(10).join(f"* **{r['categoria_grupo']}**: {_pct(r['pct_inssatisfacao'] / 100)} (IC95% {_pct(r['ic95_inf'] / 100)}–{_pct(r['ic95_sup'] / 100)}; n={_inte(r['n_pedidos'])})" for _, r in cats_top.iterrows())}
Auditar vendedores, embalagem e prazos dessas categorias e comparar com a média geral ({_pct(taxa_neg)}).
Intervalos que incluem a média geral indicam diferença ainda não confirmada.
**R4 — Frete: alavanca secundária (não priorizar sobre prazo).**
Insatisfeitos pagam frete um pouco mais alto (mediana R$ {_n(med_neg['frete_total'], 2)} vs.
R$ {_n(med_pos['frete_total'], 2)}), mas o efeito é **pequeno** e a participação do frete no pedido é igual entre
grupos; portanto, frete grátis isolado não resolve a insatisfação. Atuar apenas onde ele se cruza com atraso (rotas longas).
**R5 — Monitoramento contínuo por região e tempo.**
* Maiores taxas (n ≥ {MIN_N_RANKING}):
{chr(10).join(f"  * **{r['uf_cliente']}**: {_pct(r['pct_inssatisfacao'] / 100)} (IC95% {_pct(r['ic95_inf'] / 100)}–{_pct(r['ic95_sup'] / 100)}; n={_inte(r['n_pedidos'])})" for _, r in ufs_top.iterrows())}
* Maior excesso de insatisfeitos em volume (insatisfeitos observados − esperados pela taxa geral):
{chr(10).join(f"  * **{r['uf_cliente']}**: {_inte(r['excesso_insatisfeitos'])} pedidos a mais (taxa {_pct(r['pct_inssatisfacao'] / 100)}; n={_inte(r['n_pedidos'])})" for _, r in ufs_vol.iterrows())}
* Pior mês {pior_mes} vs. melhor mês {melhor_mes} (meses com n ≥ 500) — planejar capacidade nas janelas de pico.
**R6 — Usar o escore para priorizar o atendimento pós-entrega.**
Com AUC ≈ {_n(m_best['roc_auc'], 2)}: (a) calcular o escore **no momento da entrega**, quando atraso e tempo
realizados já são conhecidos; (b) acionar atendimento proativo para pedidos acima do limiar
({_n(m_best['limiar'], 2)}, escolhido pelo F1 em validação cruzada; no teste dá precisão
{_n(m_best['precisao'], 2)} e recall {_n(m_best['recall'], 2)}), ajustando-o à capacidade da equipe; e
(c) *fechar o ciclo*, acompanhando a taxa de nota 1-2 dos pedidos atendidos. {calib_txt}
**R7 — Escuta ativa pós-entrega.**
{_pct(coment_neg)} dos insatisfeitos escrevem comentário. Tratar comentários negativos como *sinal de alerta
secundário* (monitoramento de reputação); o alerta mais barato continua sendo o atraso.
""")
    partes.append(f"""---
* **Associação ≠ causalidade:** dados observacionais; confundidores possíveis (categoria × vendedor × região) —
  próximo passo: modelos com efeitos aleatórios ou experimentação (ex.: teste de promessa de prazo).
* **Escore pós-entrega:** o desempenho usa o atraso realizado. Sem ele (pré-entrega) o AUC é
  {_n(adic['pre_entrega']['roc_auc'], 3)}; antecipar o atraso exige dados de rastreio/ETA.
* **Estabilidade temporal:** na validação temporal o AUC é {_n(adic['temporal']['roc_auc'], 3)} contra
  {_n(m_best['roc_auc'], 3)} no teste aleatório; `mes_compra` captura eventos pontuais (ex.: fev–mar/2018) e
  meses novos entram como categoria desconhecida. {'A queda é relevante: o modelo **não** deve ser assumido estável; ' + 'retreinar e recalibrar periodicamente.' if m_best['roc_auc'] - adic['temporal']['roc_auc'] > 0.03 else 'Monitorar drift mensal.'}
* **Amostras pequenas:** categorias e UFs com menos de {MIN_N_RANKING} pedidos ficam fora dos rankings; os IC95%
  completos estão em `tab03` e `tab04`.
* **Dados e regras de limpeza:** valores impossíveis (datas incoerentes, distância fora do Brasil etc.) foram
  tratados como ausentes; a unificação de categorias `_2` é hipótese de domínio (`relatorio_etapas.md`, Seções 1 e 7).
* **Sem conteúdo textual:** comentários não entram no modelo (vazamento); NLP pode somar na triagem.
* **Sem info de devoluções/logística reversa** no dataset.
* Próximos passos: recalibrar probabilidades se forem usadas como risco absoluto; adicionar tempo-de-vida do
  cliente (recompra) e avaliação por item; segmentar o escore por transportadora.
```bash
pip install -r requirements.txt
python run_pipeline.pypython run_pipeline.py --etapa eda```
*Dados:* Kaggle — `olistbr/brazilian-ecommerce` (baixados automaticamente via `kagglehub` ou em `data/raw/`).
*Seed:* 42.
**Artefatos:** `../reports/figures/` ({len(list(FIGURES.glob('*.png')))} figuras) ·
`reports/tables/` ({len(list(TABLES.glob('*.csv')))} tabelas) · `outputs/prep/` (base pré-processada, treino e teste) ·
`outputs/resultados_modelo.json` · `outputs/modelo_insatisfacao.joblib`.
""")
    REPORT_MD.write_text("".join(partes), encoding="utf-8")
    print(f"[report] relatório -> {REPORT_MD}")
TABELAS_RAW_DOC: list[dict] = [
    {
        "chave": "orders",
        "arquivo": "olist_orders_dataset.csv",
        "rotulo": "orders — Pedidos",
        "descricao": "Tabela central: um registro por pedido.",
        "colunas": [
            ("order_id", "Identificador único do pedido."),
            ("customer_id", "Identificador do cliente que realizou o pedido."),
            ("order_status", "Situação do pedido (ex.: delivered, cancelled, unavailable)."),
            ("order_purchase_timestamp", "Data e hora da compra — origem das derivadas de sazonalidade (mês, dia da semana, hora)."),
            ("order_approved_at", "Data e hora em que o pagamento foi aprovado."),
            ("order_delivered_carrier_date", "Data e hora da coleta da encomenda pela transportadora."),
            ("order_delivered_customer_date", "Data e hora da entrega ao cliente."),
            ("order_estimated_delivery_date", "Data e hora estimada para a entrega — base do cálculo do atraso."),
        ],
    },
    {
        "chave": "customers",
        "arquivo": "olist_customers_dataset.csv",
        "rotulo": "customers — Clientes",
        "descricao": "Tabela de clientes (dados cadastrados).",
        "colunas": [
            ("customer_id", "Identificador do cliente."),
            ("customer_unique_id", "Identificador estável do cliente (sem pedido)."),
            ("customer_zip_code_prefix", "CEP do cliente — usado na distância cliente-vendedor."),
            ("customer_city", "Cidade do cliente."),
            ("customer_state", "UF do cliente (renomeada para uf_cliente na base final)."),
        ],
    },
    {
        "chave": "items",
        "arquivo": "olist_order_items_dataset.csv",
        "rotulo": "items — Itens dos pedidos",
        "descricao": "Tabela de itens comprados em cada pedido.",
        "colunas": [
            ("order_id", "Identificador do pedido."),
            ("order_item_id", "Identificador do item dentro do pedido."),
            ("product_id", "Identificador do produto comprado."),
            ("seller_id", "Identificador do vendedor que vendeu o item."),
            ("shipping_limit_date", "Data limite de envio do item."),
            ("price", "Preço unitário do item em R$."),
            ("freight_value", "Frete do item em R$."),
        ],
    },
    {
        "chave": "reviews",
        "arquivo": "olist_order_reviews_dataset.csv",
        "rotulo": "reviews — Avaliações de pedidos",
        "descricao": "Tabela de avaliações (reviews) dos clientes.",
        "colunas": [
            ("review_id", "Identificador da avaliação."),
            ("order_id", "Identificador do pedido avaliado."),
            ("review_score", "Nota dada pelo cliente (1 a 5)."),
            ("review_comment_title", "Título do comentário (opcional)."),
            ("review_comment_message", "Mensagem do comentário (opcional)."),
            ("review_creation_date", "Data e hora de criação da avaliação."),
            ("review_answer_timestamp", "Data e hora da resposta do vendedor."),
        ],
    },
    {
        "chave": "payments",
        "arquivo": "olist_order_payments_dataset.csv",
        "rotulo": "payments — Pagamentos",
        "descricao": "Tabela de pagamentos dos pedidos.",
        "colunas": [
            ("order_id", "Identificador do pedido."),
            ("payment_sequential", "Sequencial do pagamento dentro do pedido."),
            ("payment_type", "Tipo de pagamento (ex.: credit_card, boleto)."),
            ("payment_installments", "Número de parcelas."),
            ("payment_value", "Valor pago em R$."),
        ],
    },
    {
        "chave": "products",
        "arquivo": "olist_products_dataset.csv",
        "rotulo": "products — Produtos",
        "descricao": "Tabela de produtos avaliados.",
        "colunas": [
            ("product_id", "Identificador do produto."),
            ("product_category_name", "Categoria do produto."),
            ("product_name_lenght", "Comprimento do nome do produto."),
            ("product_description_lenght", "Comprimento da descrição do produto."),
            ("product_photos_qty", "Quantidade de fotos do produto."),
            ("product_weight_g", "Peso do produto em gramas."),
            ("product_length_cm", "Comprimento do produto em cm."),
            ("product_height_cm", "Altura do produto em cm."),
            ("product_width_cm", "Largura do produto em cm."),
        ],
    },
    {
        "chave": "sellers",
        "arquivo": "olist_sellers_dataset.csv",
        "rotulo": "sellers — Vendedores",
        "descricao": "Tabela de vendedores.",
        "colunas": [
            ("seller_id", "Identificador do vendedor."),
            ("seller_zip_code_prefix", "CEP do vendedor (usado na distância cliente-vendedor)."),
            ("seller_city", "Cidade do vendedor."),
            ("seller_state", "UF do vendedor (renomeada para uf_vendedor na base final)."),
        ],
    },
    {
        "chave": "geolocation",
        "arquivo": "olist_geolocation_dataset.csv",
        "rotulo": "geolocation — Geolocalização",
        "descricao": "Tabela de coordenadas geográficas.",
        "colunas": [
            ("geolocation_zip_code_prefix", "CEP associado à coordenada."),
            ("geolocation_lat", "Latitude da coordenada."),
            ("geolocation_lng", "Longitude da coordenada."),
            ("geolocation_city", "Cidade da coordenada."),
            ("geolocation_state", "UF da coordenada."),
        ],
    },
    {
        "chave": "category_translation",
        "arquivo": "product_category_name_translation.csv",
        "rotulo": "category_translation — Tradução de categorias",
        "descricao": "Tradução dos nomes das categorias de produtos.",
        "colunas": [
            ("product_category_name", "Nome original da categoria."),
            ("product_category_name_english", "Nome em inglês da categoria."),
        ],
    },
]
COLS_ANA_DOC: dict[str, tuple[str, str, str]] = {
    "order_id": ("descartada", "identificador", "Identificador único do pedido."),
    "customer_id": ("descartada", "identificador", "Identificador do cliente."),
    "order_status": ("descartada", "categórica", "Situação do pedido (pós-recorte: entregue)."),
    "order_purchase_timestamp": ("descartada", "data", "Data da compra (origem das derivadas de sazonalidade)."),
    "order_approved_at": ("descartada", "data", "Data da aprovação do pagamento (origem de aprovacao_h)."),
    "order_delivered_carrier_date": ("descartada", "data", "Data da coleta (origem de manuseio_dias)."),
    "order_delivered_customer_date": ("descartada", "data", "Data da entrega (origem de entrega_dias)."),
    "order_estimated_delivery_date": ("descartada", "data", "Data estimada (origem de atraso_dias)."),
    "customer_unique_id": ("descartada", "identificador", "Identificador estável do cliente (alta cardinalidade)."),
    "cep_cliente": ("descartada", "identificador", "CEP do cliente."),
    "cidade_cliente": ("descartada", "identificador", "Cidade do cliente (redundante com uf_cliente + distancia_km)."),
    "uf_cliente": ("feature", "categórica", "Unidade federativa do cliente."),
    "n_itens": ("feature", "numérica", "Número de itens do pedido."),
    "preco_total": ("feature", "numérica", "Soma dos preços dos itens do pedido."),
    "frete_total": ("feature", "numérica", "Soma dos fretes dos itens do pedido."),
    "categoria": ("descartada", "categórica", "Categoria do produto (redundante com categoria_grupo)."),
    "peso_g": ("feature", "numérica", "Peso do produto principal (em gramas)."),
    "n_fotos": ("feature", "numérica", "Número de fotos do produto principal."),
    "seller_zip": ("descartada", "identificador", "CEP do vendedor."),
    "uf_vendedor": ("feature", "categórica", "Unidade federativa do vendedor."),
    "n_parcelas": ("feature", "numérica", "Número de parcelas do pagamento."),
    "valor_pago": ("descartada", "numérica", "Valor total pago (redundante com valor_pedido/preco_total)."),
    "pagamento_tipo": ("feature", "categórica", "Tipo de pagamento."),
    "review_score": ("descartada", "numérica", "Nota dada pelo cliente (derivada do alvo)."),
    "review_creation_date": ("descartada", "data", "Data da avaliação (vazamento)."),
    "tem_comentario": ("descartada", "booleana", "Flag de comentário (vazamento)."),
    "comentario_chars": ("descartada", "numérica", "Tamanho do comentário (vazamento)."),
    "distancia_km": ("feature", "numérica", "Distância em km entre CEP cliente e CEP vendedor."),
    "atraso_dias": ("feature", "numérica", "Atraso em dias (entrega real - estimada)."),
    "entrega_dias": ("feature", "numérica", "Tempo de entrega em dias (compra -> entrega)."),
    "manuseio_dias": ("feature", "numérica", "Tempo de coleta em dias (compra -> coleta)."),
    "aprovacao_h": ("feature", "numérica", "Horas de aprovação do pagamento (compra -> aprovação)."),
    "valor_pedido": ("redundante", "numérica", "Valor total do pedido (preço + frete). Identidade exata: preco_total + frete_total (removida por redundancia)."),
    "frete_ratio": ("feature", "numérica", "Participação do frete no valor do pedido."),
    "preco_unitario_medio": ("redundante", "numérica", "Preço unitário médio do pedido. Derivada de preco_total / n_itens (removida por redundancia)."),
    "prazo_cumprido": ("descartada", "booleana", "Flag de prazo cumprido (redundante com atraso_dias)."),
    "faixa_atraso": ("descartada", "categórica", "Faixa agrupada de atraso (usada na EDA)."),
    "mes_compra": ("feature", "categórica", "Mês/ano da compra."),
    "dia_semana_compra": ("feature", "categórica", "Dia da semana da compra."),
    "hora_compra": ("feature", "numérica", "Hora da compra."),
    "categoria_grupo": ("feature", "categórica", "Categoria de produto agrupada."),
    "insatisfeito": ("alvo", "binária", "Insatisfação: review_score <= 2."),
}
def _tabela_colunas_md(colunas: list[tuple[str, str]]) -> str:
    linhas = [
        "| Atributo | Descrição |",
        "|---|---|",
    ]
    linhas += [f"| `{nome}` | {desc} |" for nome, desc in colunas]
    return "\n".join(linhas)
def gerar_dicionario() -> None:
    brutos = carregar_dados()
    analise = carregar_analise()
    n = len(analise)
    partes: list[str] = []
    partes.append(f"""# Dicionário de dados — Olist Brazilian E-commerce
**Dataset:** [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — marketplace brasileiro de e-commerce
**Fonte:** Kaggle
Esta documentação reúne os **atributos das tabelas do dataset** e as **features da
base final** usadas no estudo, com explicações simples — servindo de
contextualização ao leitor.
* **Junções:** as tabelas são relacionadas por `order_id` (pedidos, clientes, itens,
  avaliações, pagamentos, produtos, vendedores, geolocalização e tradução de
  categorias).
* **Atributos derivados:** o atraso (`atraso_dias`) é a diferença entre a data de
  entrega real e a estimada; o tempo de entrega (`entrega_dias`) entre a compra e a
  entrega; o frete (`frete_total`) soma os valores de frete dos itens; a participação
  do frete (`frete_ratio`) divide o frete pelo valor total do pedido; o valor do
  pedido (`valor_pedido`) soma preço e frete; etc.
* **Recorte:** pedidos **entregues**, com data de entrega e **com avaliação** —
  **{_inte(n)} pedidos** na base final.
""")
    for t in TABELAS_RAW_DOC:
        df = brutos.get(t["chave"])
        partes.append(
            f"### {t['rotulo']}\n"
            f"({_inte(len(df)) if df is not None else '?'} linhas × {len(t['colunas'])} colunas)\n\n"
            f"{t['descricao']}\n\n"
            f"{_tabela_colunas_md(t['colunas'])}\n\n"
        )
    linhas_tab = [
        "| Variável | Papel | Tipo | Descrição |",
        "|---|---|---|---|",
    ]
    for c in analise.columns:
        papel, tipo, desc = COLS_ANA_DOC.get(c, ("descartada", "desconhecido", "Sem descrição."))
        linhas_tab.append(f"| {c} | {papel} | {tipo} | {desc} |")
    partes.append(f"""## 2. Base final de análise — {_inte(len(analise))} pedidos × {len(analise.columns)} colunas
A base final consolida as tabelas por `order_id` e adiciona atributos derivados.
Cada coluna tem um **papel**: feature operacional, alvo (insatisfação) ou coluna
descartada.
""" + "\n".join(linhas_tab) + f"""
* **Insatisfação:** nota dada pelo cliente ≤ 2 (notas 1-2) vs. notas 3-5; a variável
  `insatisfeito` é o alvo do modelo.
* **Odds ratio (OR):** razão de chances; mostra quantas vezes o risco de insatisfação
  aumenta (OR > 1) ou diminui (OR < 1) com um desvio padrão da feature.
* **AUC:** área sob a curva ROC — poder discriminativo do modelo.
* **Calibração:** compara probabilidade prevista com frequência real observada.
* **Recall:** TP / (TP + FN) — poder de identificar insatisfeitos.
* **Mann-Whitney / Qui-quadrado / V de Cramér / Spearman / Informação Mútua:** testes
  e medidas de associação.
* **PCA:** análise de componentes principais — diagnóstico de dimensionalidade.
* **SHAP:** valores de Shapley para explicar contribuições das features.
""")
    DICTIONARY_MD.write_text("".join(partes), encoding="utf-8")
    print(f"[report] dicionário -> {DICTIONARY_MD}")
if __name__ == "__main__":
    gerar_relatorio()
