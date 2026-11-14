from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .config import (
    CATEGORICAL_FEATURES, DISTANCIA_MAX_KM, ETAPAS_MD, MODEL_JSON, NUMERIC_FEATURES,
    PCA_JSON, PREP_JSON, RAW_SHAPES_CSV, REDUCAO_CSV, SELECAO_JSON, TABLES, TARGET,
)
from .features import carregar_analise
from .report import _fig, _inte, _md, _modelo_nome, _n, _pct
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
METODOS = ["filtro_mi", "filtro_auc", "wrapper_forward", "embutido_l1", "embutido_rf"]
ROTULO = {
    "filtro_mi": "Filtro: MI",
    "filtro_auc": "Filtro: AUC univ.",
    "wrapper_forward": "Wrapper: forward",
    "embutido_l1": "Embutido: L1",
    "embutido_rf": "Embutido: RF",
}
MOTIVOS_DESCARTE = {
    "order_id": ("identificador", "identificador único do pedido (1:1 com a linha)"),
    "customer_id": ("identificador", "chave de ligação pedido-cliente"),
    "customer_unique_id": ("identificador", "identificador de cliente (alta cardinalidade)"),
    "cep_cliente": ("identificador", "identificador geográfico fino (equivalente à cidade)"),
    "cidade_cliente": ("identificador", "alta cardinalidade; informação geográfica já em uf_cliente e distancia_km"),
    "seller_zip": ("identificador", "identificador geográfico do vendedor"),
    "order_status": ("constante", "após o recorte todos os pedidos são 'delivered'"),
    "categoria": ("redundância", "versão sem agrupamento de categoria_grupo"),
    "valor_pago": ("redundância", "equivalente a valor do pedido (preço + frete) já representado"),
    "prazo_cumprido": ("redundância", "binarização de atraso_dias"),
    "faixa_atraso": ("redundância", "discretização de atraso_dias (usada só na EDA)"),
    "valor_pedido": ("redundância", "identidade exata: preco_total + frete_total"),
    "preco_unitario_medio": ("redundância", "derivada: preco_total / n_itens"),
    "review_score": ("vazamento", "o alvo é derivado dela"),
    "review_creation_date": ("vazamento", "conhecida só depois da avaliação"),
    "tem_comentario": ("vazamento", "faz parte da própria avaliação"),
    "comentario_chars": ("vazamento", "faz parte da própria avaliação"),
    "order_purchase_timestamp": ("engenharia", "substituída por hora_compra, mes_compra, dia_semana_compra"),
    "order_approved_at": ("engenharia", "substituída por aprovacao_h"),
    "order_delivered_carrier_date": ("engenharia", "substituída por manuseio_dias"),
    "order_delivered_customer_date": ("engenharia", "substituída por entrega_dias"),
    "order_estimated_delivery_date": ("engenharia", "substituída por atraso_dias (real − prometido)"),
}
def _sn(flag) -> str:
    return "sim" if bool(flag) else "—"
def _lista(itens: list[str]) -> str:
    return ", ".join(f"`{i}`" for i in itens) if itens else "—"
def _tabela_diagnostico(diag: pd.DataFrame, categoria: str,
                        cols=("item", "n", "pct", "decisao", "justificativa")) -> str:
    d = diag[diag["categoria"] == categoria][list(cols)].copy()
    if d.empty:
        return "*Nenhuma ocorrência.*"
    if "n" in d:
        d["n"] = d["n"].map(lambda v: "—" if pd.isna(v) else _inte(v))
    if "pct" in d:
        d["pct"] = d["pct"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}".replace(".", ",") + "%")
    return _md(d)
def gerar_relatorio_etapas() -> None:
    df = carregar_analise()
    prep = json.loads(PREP_JSON.read_text(encoding="utf-8"))
    modelo = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    sel = json.loads(SELECAO_JSON.read_text(encoding="utf-8"))
    pca = json.loads(PCA_JSON.read_text(encoding="utf-8"))
    brutas = pd.read_csv(RAW_SHAPES_CSV) if RAW_SHAPES_CSV.exists() else None
    diag = pd.read_csv(TABLES / "tab15_diagnostico_qualidade.csv")
    vif = pd.read_csv(TABLES / "tab17_vif.csv")
    rank = pd.read_csv(TABLES / "tab12_ranking_features.csv")
    pca_var = pd.read_csv(TABLES / "tab13_pca_variancia.csv")
    red = pd.read_csv(REDUCAO_CSV)
    tab_grupos = pd.read_csv(TABLES / "tab20_importancia_grupos.csv")
    met = modelo["metricas"]
    melhor = modelo["melhor_modelo"]
    m_best = met[melhor]
    adic = modelo["validacoes_adicionais"]
    sel_tab = rank[["feature", "tipo"] + [f"sel_{m}" for m in METODOS] + ["n_metodos", "selecionada"]].copy()
    for m in METODOS:
        sel_tab[f"sel_{m}"] = sel_tab[f"sel_{m}"].map(_sn)
    sel_tab["selecionada"] = sel_tab["selecionada"].map(lambda v: "**sim**" if v else "não")
    sel_tab.columns = ["Feature", "Tipo"] + [ROTULO[m] for m in METODOS] + ["Nº métodos", "Selecionada"]
    unanimes = rank.loc[rank["n_metodos"] == 5, "feature"].tolist()
    maioria = rank.loc[(rank["n_metodos"] >= 3) & (rank["n_metodos"] < 5), "feature"].tolist()
    divergentes = rank.loc[rank["n_metodos"].isin([1, 2]), "feature"].tolist()
    nenhum = rank.loc[rank["n_metodos"] == 0, "feature"].tolist()
    por_metodo = pd.DataFrame({
        "Método": [ROTULO[m] for m in METODOS],
        "Critério de corte": [
            "MI > maior MI entre features-sombra (colunas embaralhadas)",
            "AUC univariada (CV 3) > maior AUC de feature-sombra",
            f"forward com regressão logística; para quando o ganho de AUC < {sel['limiares']['ganho_min_wrapper']}",
            f"coeficiente ≠ 0 com C = {sel['limiares']['c_l1']} (menor C a 1 erro-padrão do melhor)",
            "importância (Gini) > maior importância de feature-sombra",
        ],
        "Nº selecionadas": [len(sel["por_metodo"][m]) for m in METODOS],
    })
    comp = pca_var.head(min(8, len(pca_var)))
    comp_exib = pd.DataFrame({
        "Componente": comp["componente"],
        "Autovalor": comp["autovalor"].map(lambda v: _n(v, 2)),
        "Var. explicada": comp["variancia_explicada"].map(_pct),
        "Var. acumulada": comp["variancia_acumulada"].map(_pct),
        "Limiar análise paralela (p95)": comp["autovalor_analise_paralela_p95"].map(lambda v: _n(v, 2)),
        "Kaiser (>1)": comp["kaiser_maior_que_1"].map(_sn),
        "Acima do ruído": comp["supera_ruido_paralela"].map(_sn),
    })
    criterios = pd.DataFrame({
        "Critério": ["Kaiser (autovalor > 1)", "Análise paralela (acima do ruído)",
                     "90% da variância (apenas referência)", "Validação cruzada preditiva (1 erro-padrão)"],
        "k": [pca["k_kaiser"], pca["k_paralela"], pca["k_90pct"], pca["k_cv_1se"]],
    })
    rotulos_pc = pd.DataFrame({
        "Componente": list(pca["rotulos_componentes"].keys()),
        "Variância": [_pct(pca["variancia_por_componente"][c]) for c in pca["rotulos_componentes"]],
        "Maiores cargas (sinal)": list(pca["rotulos_componentes"].values()),
        "AUC da CP isolada vs. alvo": [_n(pca["auc_componentes_vs_alvo"].get(c, np.nan), 3)
                                       for c in pca["rotulos_componentes"]],
    })
    red_exib = red[["representacao", "modelo", "n_features_originais", "n_colunas_modelo"]].copy()
    red_exib["modelo"] = red_exib["modelo"].map(_modelo_nome)
    red_exib["AUC CV (treino)"] = [f"{a:.4f}±{b:.4f}" for a, b in zip(red["auc_cv_medio"], red["auc_cv_dp"])]
    red_exib["AUC teste"] = red["auc_teste"].map(lambda v: f"{v:.4f}")
    red_exib["Acurácia"] = red["acuracia_teste"].map(lambda v: f"{v:.4f}")
    red_exib["Precisão"] = red["precisao_teste"].map(lambda v: f"{v:.4f}")
    red_exib["Recall"] = red["recall_teste"].map(lambda v: f"{v:.4f}")
    red_exib["F1"] = red["f1_teste"].map(lambda v: f"{v:.4f}")
    red_exib["Limiar"] = red["limiar_f1_oof"].map(lambda v: f"{v:.2f}")
    red_exib["Δ AUC vs. original"] = red["delta_auc_teste_vs_original"].map(lambda v: f"{v:+.4f}")
    red_exib = red_exib.drop(columns=["n_features_originais"]).rename(columns={
        "representacao": "Representação", "modelo": "Modelo",
        "n_colunas_modelo": "Colunas no modelo",
    })
    def _delta(rep: str, mod: str) -> float:
        return float(red[(red["representacao"] == rep) & (red["modelo"] == mod)]
                     ["delta_auc_teste_vs_original"].iloc[0])
    linhas_m = []
    for nome, m in met.items():
        linhas_m.append({
            "Modelo": _modelo_nome(nome),
            "AUC CV": f"{m['cv_roc_auc_mean']:.3f}±{m['cv_roc_auc_std']:.3f}",
            "AUC teste": f"{m['roc_auc']:.3f}", "Acurácia": f"{m['acuracia']:.3f}",
            "Precisão": f"{m['precisao']:.3f}",
            "Recall": f"{m['recall']:.3f}", "F1": f"{m['f1']:.3f}",
            "Limiar": f"{m['limiar']:.2f}",
        })
    g_gb = tab_grupos[tab_grupos["modelo"] == "gradient_boosting"].copy()
    g_gb["Queda de AUC"] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(g_gb["queda_auc_media"], g_gb["queda_auc_dp"])]
    g_exib = g_gb[["grupo", "features", "Queda de AUC"]].rename(
        columns={"grupo": "Grupo", "features": "Variáveis do grupo"})
    conf = m_best["matriz_confusao"]
    colunas_descartadas = [c for c in df.columns if c in MOTIVOS_DESCARTE]
    descarte = pd.DataFrame({
        "Coluna": colunas_descartadas,
        "Critério": [MOTIVOS_DESCARTE[c][0] for c in colunas_descartadas],
        "Motivo": [MOTIVOS_DESCARTE[c][1] for c in colunas_descartadas],
    })
    vif_exib = vif.copy()
    vif_exib["vif_antes"] = vif_exib["vif_antes"].map(lambda v: ">1.000.000" if v >= 1e6 else _n(v, 1))
    vif_exib["vif_depois"] = vif_exib["vif_depois"].map(lambda v: "removida" if pd.isna(v) else _n(v, 1))
    vif_exib.columns = ["Variável", "VIF antes", "VIF depois"]
    n_equiv = int((diag["categoria"] == "Categorias equivalentes").sum())
    prep_tab = pd.DataFrame([
        ["Recorte analítico (entregues, com avaliação)",
         "Sem entrega ou sem avaliação não há alvo observável",
         f"{_inte(prep['n_total'])} → {_inte(prep['n_base'])} pedidos"],
        ["Unificação de categorias equivalentes",
         "Grafias/versões duplicadas (ex.: `home_confort`/`home_comfort_2`) fragmentam níveis",
         f"{n_equiv} regras (hipótese de domínio, ver Seção 7)"],
        ["Valores impossíveis → ausente",
         "Datas incoerentes, parcelas = 0, peso = 0, distância fora do Brasil",
         f"{_inte(prep['n_invalidos_total'])} valores"],
        ["Remoção de variáveis derivadas",
         "Identidades exatas / colinearidade extrema (VIF)",
         f"VIF máx. {'> 1.000.000' if prep['vif_max_antes'] >= 1e6 else _n(prep['vif_max_antes'], 0)} → {_n(prep['vif_max_depois'], 1)}; "
         f"{_lista(list(prep['redundantes_removidas']))}"],
        ["Descarte de identificadores, constantes e vazamento",
         "Não generalizam ou usam informação posterior ao alvo", f"{len(colunas_descartadas)} colunas"],
        ["Divisão treino/teste estratificada (80/20) **antes** de qualquer ajuste",
         "Evitar que o teste influencie imputação, escala, seleção, PCA e limiar",
         f"{_inte(prep['n_train'])} / {_inte(prep['n_test'])}; positivos "
         f"{_pct(prep['taxa_treino'])} / {_pct(prep['taxa_teste'])}"],
        ["Imputação por mediana (numéricas) e moda (categóricas), ajustada só no treino",
         "Ausências residuais e valores tratados como inválidos",
         f"{_inte(prep['ausentes_depois_regras'])} células a imputar"],
        ["One-hot com `min_frequency=30`",
         "Categóricas nominais (sem ordem); níveis raros geram colunas instáveis",
         f"{len(CATEGORICAL_FEATURES)} variáveis → {prep['n_dim_ohe']} colunas"],
        ["Padronização (z-score) só para regressão logística e PCA",
         "Modelos lineares e PCA são sensíveis à escala; árvores não",
         f"{len(NUMERIC_FEATURES)} numéricas"],
        ["Sem `class_weight` nem reamostragem",
         "Pesos de classe distorcem as probabilidades; o desbalanceamento "
         f"({_pct(prep['taxa_treino'])} de positivos) é tratado pelo limiar escolhido por validação cruzada",
         f"limiar do melhor modelo = {_n(m_best['limiar'], 2)}"],
    ], columns=["Transformação", "Problema que a justifica", "Resultado"])
    tabela_brutas = _md(brutas.assign(linhas=brutas["linhas"].map(_inte))) if brutas is not None else ""
    pca_sem_reducao = pca["k_escolhido"] >= pca["n_numericas"]
    perda_k90 = pca["auc_cv_sem_pca"] - pca["auc_cv_k_90"]
    txt_pca_conclusao = (
        f"* **Conclusão:** com k = p = {pca['n_numericas']} o PCA apenas rotaciona as variáveis; **nenhuma redução "
        f"é justificada** neste problema. Reduzir para k = {pca['k_90pct']} (regra dos 90%) custaria "
        f"{_n(perda_k90, 4)} de AUC, porque parte do sinal preditivo está em "
        "componentes de baixa variância." if pca_sem_reducao
        else f"* **Conclusão:** manter {pca['k_escolhido']} de {pca['n_numericas']} componentes preserva o AUC "
             "dentro de 1 erro-padrão."
    )
    txt_pca_comp = (
        "Com todas as componentes retidas, a regressão logística é idêntica à original (rotação não muda o "
        "ajuste linear) e o boosting piora, porque as árvores perdem os eixos originais; **não há ganho de "
        "dimensionalidade e há perda de interpretabilidade**." if pca_sem_reducao else
        f"Reduz a dimensão das numéricas ({pca['n_numericas']} → {pca['k_escolhido']}), mas as categóricas seguem "
        "em one-hot, então o ganho de dimensionalidade total é pequeno; em troca, perde-se a leitura direta "
        "(os coeficientes passam a ser de componentes, não de atraso, frete etc.)."
    )
    l1_sem_corte = len(sel["por_metodo"]["embutido_l1"]) == len(FEATURES)
    txt_l1 = (
        "* **L1 sem poder de corte:** com o C escolhido o L1 manteve **todas** as variáveis; ele concorda com "
        "o consenso por construção, e a seleção efetiva vem dos outros quatro métodos.\n" if l1_sem_corte else ""
    )
    partes: list[str] = []
    partes.append(f"""# Relatório de etapas — preparação, redução de dimensionalidade e modelo
**Dataset:** [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(tema substituído pelo trabalho final da equipe, conforme a OBS da atividade).
> **Pergunta central:** *como preparar e reduzir a base de dados preservando a informação
> relevante para as etapas seguintes?*
>
> **Regra aplicada a todo o relatório:** toda transformação responde *"que problema nos
> dados a justifica?"* e toda redução responde *"que informação é preservada e o que é perdido?"*.
**Protocolo anti-vazamento:** a divisão treino/teste ocorre antes de qualquer ajuste.
Imputação, escala, seleção de atributos, PCA, limiar de decisão e escolha do modelo usam
**somente o treino** ({_inte(prep['n_train'])} pedidos); o teste ({_inte(prep['n_test'])}) é
usado uma única vez para a avaliação final.
| Seção | Conteúdo |
|---|---|
| 1 | Diagnóstico inicial da base |
| 2 | Decisões de pré-processamento |
| 3 | Seleção de atributos (filtro, wrapper, embutido) |
| 4 | PCA |
| 5 | Comparação das representações |
| 6 | Modelo final e avaliação |
| 7 | Limitações e decisões que mudariam com novo conhecimento de domínio |
---
{tabela_brutas}
Após o recorte (pedidos entregues, com data de entrega e uma avaliação por pedido — a mais antiga),
a base tem **{_inte(prep['n_base'])} pedidos × {prep['n_colunas_base']} colunas**
({prep['n_features_candidatas']} features candidatas). Taxa do alvo (`review_score ≤ 2`):
**{_pct(df[TARGET].mean())}**. A lista completa de verificações está em
`reports/tables/tab15_diagnostico_qualidade.csv`.
{_tabela_diagnostico(diag, 'Ausentes')}
{_tabela_diagnostico(diag, 'Duplicatas')}
{_tabela_diagnostico(diag, 'Categorias equivalentes', cols=('item', 'decisao', 'justificativa'))}
{_tabela_diagnostico(diag, 'Valores inválidos', cols=('item', 'n', 'pct', 'decisao'))}
{_fig('fig11_valores_impossiveis.png', 'Valores impossíveis ou incoerentes: tratados como ausentes antes da divisão treino/teste')}
Critério: fora de 3×IQR **e** avaliado pelo domínio e pela coerência com outras variáveis.
Extremos coerentes (entregas de 100+ dias em pedidos muito atrasados e mal avaliados, itens caros
com frete proporcional) **foram mantidos**; só viraram erro os valores que violam o domínio (1.3).
{_tabela_diagnostico(diag, 'Valores extremos', cols=('item', 'decisao', 'justificativa'))}
{_md(descarte)}
{_tabela_diagnostico(diag, 'Redundância', cols=('item', 'decisao', 'justificativa'))}
{_md(vif_exib)}
*Interpretação:* `valor_pedido` é exatamente `preco_total + frete_total` e `preco_unitario_medio`
é `preco_total / n_itens`; foram removidas porque são 100% reconstruíveis. `atraso_dias` e
`entrega_dias` têm correlação alta, mas **medem coisas diferentes** (o atraso depende do prazo
prometido) e foram mantidas; o efeito dessa colinearidade é tratado nas importâncias por **grupo** (Seção 6).
---
{_md(prep_tab)}
**Base final pré-processada** (treino e teste separados), em `outputs/prep/`:
`X_train_prep.csv.gz` ({_inte(prep['n_train'])} × {prep['n_dim_saida']}), `X_test_prep.csv.gz`
({_inte(prep['n_test'])} × {prep['n_dim_saida']}), `y_train.csv`, `y_test.csv` e as versões limpas antes
da codificação (`treino_limpo.csv.gz`, `teste_limpo.csv.gz`).
**PCA (diagnóstico multivariado).** Para avaliar se as variáveis numéricas carregam informação redundante entre si, aplicamos a análise de componentes principais (PCA): o gráfico abaixo mostra os autovalores contra o ruído simulado e a variância acumulada pelos componentes.
{_fig('fig12_pca_scree.png', 'Diagnóstico do PCA: autovalores e variância acumulada')}
O diagnóstico orienta a etapa seguinte de seleção/preprocessamento.
---
Para não comparar medidas incomparáveis entre numéricas e categóricas, **todos os métodos pontuam as
mesmas {len(FEATURES)} features** (categóricas como bloco) e usam o treino. O corte não é um número fixo
de variáveis: cada método compara a feature com **cópias embaralhadas dela mesma (features-sombra)**,
que têm a mesma distribuição e nenhuma relação com o alvo.
{_md(por_metodo)}
{_md(sel_tab)}
{_fig('fig10_consenso_selecao.png', 'Convergência dos métodos de seleção')}
* **Unânimes (5/5):** {_lista(unanimes)}.
* **Maioria (3–4/5):** {_lista(maioria)}.
* **Divergentes (1–2/5):** {_lista(divergentes)}.
* **Rejeitadas por todos:** {_lista(nenhum)}.
* **Custo:** o wrapper exigiu **{_inte(sel['custo_wrapper']['n_ajustes'])} ajustes** de regressão
  logística ({_n(sel['custo_wrapper']['segundos'], 0)} s), contra um único cálculo por variável nos
  filtros; o resultado depende do modelo usado (linear) e do critério de parada.
{txt_l1}* **Dependência do modelo:** o embutido L1 (linear) e o random forest (não linear) nem sempre concordam;
  variáveis com efeito só não linear ou em interação tendem a aparecer no RF e não no L1.
* **Associação descritiva:** `reports/tables/tab12_ranking_features.csv` traz também Spearman
  (numéricas) e V de Cramér corrigido (categóricas), calculados no treino.
{_fig('fig09_mutual_information.png', 'Informação mútua vs. ruído (features-sombra)')}
---
PCA ajustado **só no treino**, sobre as **{pca['n_numericas']} numéricas padronizadas**. Não se retém
componentes "até atingir X% da variância": os critérios abaixo são comparados e a escolha final é validada
pela **informação preditiva preservada**.
{_md(comp_exib)}
{_fig('fig12_pca_scree.png', 'Diagnóstico do PCA: autovalores, análise paralela e variância acumulada')}
{_md(criterios)}
* Critérios puramente de variância divergem ({min(pca['k_kaiser'], pca['k_paralela'], pca['k_90pct'])} a
  {max(pca['k_kaiser'], pca['k_paralela'], pca['k_90pct'])} componentes): alta variância não implica
  relevância para o alvo.
* Pela validação cruzada preditiva (regressão logística com PCA(k) + categóricas), o menor k cujo AUC fica a
  1 erro-padrão do modelo **sem** PCA é **k = {pca['k_escolhido']}** ({_pct(pca['variancia_k_escolhido'])} da
  variância; AUC {pca['auc_cv_k_escolhido']:.4f} contra {pca['auc_cv_sem_pca']:.4f} sem PCA). Com
  k = {pca['k_90pct']} (regra dos 90%) o AUC seria {pca['auc_cv_k_90']:.4f}.
{txt_pca_conclusao}
{_fig('fig13_pca_auc_vs_k.png', 'Informação preditiva preservada pelo PCA em função de k')}
{_md(rotulos_pc)}
Tabela completa de cargas: `reports/tables/tab18_pca_cargas.csv`.
{_fig('fig14_pca_cp1_cp2.png', 'Projeção no plano CP1 × CP2 e cargas das variáveis')}
A componente mais associada ao alvo isoladamente é **{pca['componente_mais_separadora']}**
(AUC = {_n(pca['auc_componentes_vs_alvo'][pca['componente_mais_separadora']], 3)}). Se a nuvem CP1 × CP2
não separa as classes, é porque o PCA maximiza variância e **não** usa o alvo: o sinal de atraso pode
estar em componentes de baixa variância.
---
Mesmos modelos, mesma divisão, mesma validação. As três representações: **Original** ({len(FEATURES)}
features), **Selecionada** ({len(sel['selecionadas'])} features do consenso) e **PCA** ({pca['k_escolhido']}
componentes no lugar das {pca['n_numericas']} numéricas, mais as categóricas{', ou seja, sem redução' if pca_sem_reducao else ''}).
{_md(red_exib)}
{_fig('fig19_comparacao_representacoes.png', 'Dimensionalidade e informação preservada')}
{_fig('fig20_roc_representacoes.png', 'Curvas ROC no teste: original × selecionada × PCA')}
**Informação preservada e perdida**
* **Selecionada:** Δ AUC de {_delta('Selecionada', 'regressao_logistica'):+.4f} (regressão logística) e
  {_delta('Selecionada', 'gradient_boosting'):+.4f} (boosting) com {len(sel['selecionadas'])} de
  {len(FEATURES)} features. Mantém a interpretabilidade: continuam valendo OR e SHAP por variável.
* **PCA:** Δ AUC de {_delta('PCA', 'regressao_logistica'):+.4f} (regressão logística) e
  {_delta('PCA', 'gradient_boosting'):+.4f} (boosting). {txt_pca_comp}
* **Seleção × extração:** a seleção descarta variáveis inteiras e preserva o significado das que ficam; o PCA
  mistura todas as numéricas, o que só compensa com colinearidade forte e sem interesse em interpretar por
  variável. Depois de remover as redundâncias exatas, a colinearidade restante é moderada
  (VIF máx. {_n(prep['vif_max_depois'], 1)}).
---
Divisão estratificada 80/20; validação cruzada de 5 dobras no treino para comparar modelos e escolher o
limiar (F1 máximo nas previsões fora-da-dobra). **Melhor modelo: `{melhor}`** (maior AUC de CV; não é
escolhido pelo teste).
{_md(pd.DataFrame(linhas_m))}
*Métricas no teste (limiar = {_n(m_best['limiar'], 2)}, escolhido na validação cruzada pelo F1 máximo).
Acurácia, precisão, recall e F1 usam esse limiar; AUC é independente de limiar.*
**Matriz de confusão ({_modelo_nome(melhor)}, limiar {_n(m_best['limiar'], 2)}):** {conf[0][0]} TN, {conf[0][1]} FP,
{conf[1][0]} FN, {conf[1][1]} TP.
{_fig('fig16_curvas_roc.png', 'Curvas ROC no conjunto de teste')}
{_fig('fig18_calibracao.png', 'Calibração das probabilidades previstas')}
**Importância por grupo de variáveis (gradient boosting, queda de AUC no teste).** As colunas de um grupo
são permutadas juntas; isso evita subestimar variáveis correlacionadas (ex.: `atraso_dias` e `entrega_dias`)
e superestimar variáveis independentes.
{_md(g_exib)}
**Validações adicionais**
| Cenário | AUC | Acurácia | Recall | F1 |
|---|---|---|---|---|
| Teste aleatório (principal) | {_n(m_best['roc_auc'], 3)} | {_n(m_best['acuracia'], 3)} | {_n(m_best['recall'], 3)} | {_n(m_best['f1'], 3)} |
| Validação temporal (treina nos 80% mais antigos, testa nos 20% mais recentes) | {_n(adic['temporal']['roc_auc'], 3)} | {_n(adic['temporal']['acuracia'], 3)} | {_n(adic['temporal']['recall'], 3)} | {_n(adic['temporal']['f1'], 3)} |
| Modelo pré-entrega (sem `atraso_dias` e `entrega_dias`) | {_n(adic['pre_entrega']['roc_auc'], 3)} | {_n(adic['pre_entrega']['acuracia'], 3)} | {_n(adic['pre_entrega']['recall'], 3)} | {_n(adic['pre_entrega']['f1'], 3)} |
---
* **Unificação de categorias** (`home_appliances_2` → `home_appliances`, `home_comfort_2` → `home_comfort`):
  é hipótese de domínio (versões da mesma família). Se forem linhas de produto distintas, deve ser revertida.
* **Limites de validade:** distância > {_inte(DISTANCIA_MAX_KM)} km e "postagem após a entrega" foram tratadas como
  erro. Se `order_delivered_carrier_date` tiver outro significado (ex.: pedidos com vários vendedores), a regra muda.
* **Imputação por mediana/moda:** simples e estável; se a ausência for informativa (ex.: `n_fotos` ausente),
  um indicador de ausência pode ajudar.
* **Seleção por consenso:** o critério "≥ 3 de 5" e a tolerância do wrapper ({sel['limiares']['ganho_min_wrapper']}
  de AUC) são escolhas; a seleção foi feita em uma amostra de {_inte(sel['n_amostra'])} pedidos do treino.
* **PCA só nas numéricas:** categóricas não entram na rotação; um método para dados mistos (MCA/FAMD) poderia
  mudar a conclusão.
* **Calendário:** `mes_compra` captura eventos pontuais (ex.: fev–mar/2018); a validação temporal mostra quanto
  o desempenho resiste fora da janela de treino.
* **Associação ≠ causalidade.** Sem dados de rastreio, o aviso *antes* da entrega depende do modelo
  pré-entrega (AUC {_n(adic['pre_entrega']['roc_auc'], 3)}), mais fraco que o modelo com atraso realizado.
*Reprodutibilidade:* `python run_pipeline.py` (etapas `load → prep → eda → tests → select → pca → model →
reducao → etapas → report`) · seed 42 · dados: Kaggle `olistbr/brazilian-ecommerce`.
""")
    ETAPAS_MD.write_text("".join(partes), encoding="utf-8")
    print(f"[etapas] relatório -> {ETAPAS_MD}")
if __name__ == "__main__":
    gerar_relatorio_etapas()
