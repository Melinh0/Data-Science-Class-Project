from __future__ import annotations

import json
from collections import defaultdict
import numpy as np
import pandas as pd
from .config import (
    CUSTO_FN_FP, CV_FOLDS, MODEL_JSON, MODELOS_JSON, RELATORIO_MODELOS_MD, TABLES,
)
from .modelos_supervisionados import FAMILIA, NOMES, ORDEM
from .report import _fig, _inte, _md, _n, _pct
LIMITE_OVERFIT = 0.03
def _ic(v, lo, hi, casas: int = 3) -> str:
    return f"{_n(v, casas)} [{_n(lo, casas)}–{_n(hi, casas)}]"
def _grade_avaliada(hp: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for modelo, g in hp.groupby("modelo", sort=False):
        valores: dict[str, list[str]] = defaultdict(list)
        for cfg in g["configuracao"]:
            for k, v in json.loads(cfg).items():
                if v not in valores[k]:
                    valores[k].append(v)
        linhas.append({
            "Modelo": modelo,
            "Hiperparâmetros avaliados": "; ".join(f"`{k}` ∈ {{{', '.join(v)}}}" for k, v in valores.items()),
            "Configurações": len(g),
        })
    return pd.DataFrame(linhas)
def gerar_relatorio_modelos() -> None:
    if not MODELOS_JSON.exists():
        raise FileNotFoundError("Rode a etapa 'modelos' antes de 'relatorio_modelos'.")
    js = json.loads(MODELOS_JSON.read_text(encoding="utf-8"))
    cmp = pd.read_csv(TABLES / "tab24_comparacao_modelos.csv").set_index("modelo")
    hp = pd.read_csv(TABLES / "tab22_hiperparametros_avaliados.csv")
    esc = pd.read_csv(TABLES / "tab23_hiperparametros_escolhidos.csv")
    lim = pd.read_csv(TABLES / "tab25_limiares_e_erros.csv")
    sens = pd.read_csv(TABLES / "tab26_sensibilidade_custo.csv")
    dif = pd.read_csv(TABLES / "tab27_diferenca_auc_vs_final.csv").set_index("modelo")
    dif.index = dif.index.map({v: k for k, v in NOMES.items()})
    final = js["modelo_final"]
    nf = NOMES[final]
    dummy = js["dummy_teste"]
    taxa = js["taxa_positivos_teste"]
    n_test_pos = int(js["metricas_teste"][final]["tp"] + js["metricas_teste"][final]["fn"])
    cf = cmp.loc[final]
    cand = js["candidatos"]
    crit_txt = (
        "tem o maior AUC de validação cruzada e é o único dentro de 1 erro-padrão do melhor"
        if cand == [final] else
        f"está entre os de AUC de CV equivalente ao melhor (1 erro-padrão) e tem o menor custo esperado com FN = {CUSTO_FN_FP} × FP"
    )
    taxa_pos = js["taxa_positivos_teste"]
    custo_nao_agir, custo_contatar_todos = CUSTO_FN_FP * taxa_pos, 1 - taxa_pos
    reducao_custo = 1 - cf["custo_medio"] / custo_nao_agir
    auc_min, auc_max = cmp["auc_teste"].min(), cmp["auc_teste"].max()
    d_abs = [abs(dif.loc[m, "delta_auc_medio"]) for m in ORDEM if m != final]
    linhas = []
    for m in ORDEM:
        r = cmp.loc[m]
        linhas.append({
            "Modelo": NOMES[m] + (" ★" if m == final else ""),
            "Limiar": _n(r["limiar"], 2), "Acurácia": _n(r["acuracia"], 3), "Precisão": _n(r["precisao"], 3),
            "Recall": _n(r["recall"], 3), "F1": _n(r["f1"], 3),
            "AUC (IC95% bootstrap)": _ic(r["auc_teste"], r["auc_ic95_inf"], r["auc_ic95_sup"]),
        })
    linhas.append({"Modelo": NOMES["dummy"] + " (baseline)", "Limiar": "0,50", "Acurácia": _n(dummy["acuracia"], 3),
                   "Precisão": "0,000", "Recall": _n(dummy["recall"], 3), "F1": "0,000", "AUC (IC95% bootstrap)": "0,500"})
    tab_principal = pd.DataFrame(linhas)
    lideres = {}
    for met, rot in (("acuracia", "acurácia"), ("precisao", "precisão"), ("recall", "recall"), ("f1", "F1"), ("auc_teste", "AUC")):
        lideres[rot] = cmp[met].idxmax()
    texto_lideres = "; ".join(f"**{rot}**: {NOMES[m]} ({_n(cmp.loc[m, met], 3)})" for (rot, m), met in zip(
        lideres.items(), ("acuracia", "precisao", "recall", "f1", "auc_teste")))
    lider_unico = len(set(lideres.values())) == 1
    arm = []
    for m in ORDEM:
        d = lim[lim["modelo"] == NOMES[m]].set_index("regra_limiar")
        k05, kc = d.loc["0,5 (padrão)"], d.loc[f"custo FN={CUSTO_FN_FP}×FP (CV)"]
        arm.append({"Modelo": NOMES[m], "Acurácia @0,5": _n(k05["acuracia"], 3), "Recall @0,5": _n(k05["recall"], 3),
                    f"Acurácia @custo {CUSTO_FN_FP}:1": _n(kc["acuracia"], 3), f"Recall @custo {CUSTO_FN_FP}:1": _n(kc["recall"], 3)})
    arm.append({"Modelo": NOMES["dummy"], "Acurácia @0,5": _n(dummy["acuracia"], 3), "Recall @0,5": "0,000",
                f"Acurácia @custo {CUSTO_FN_FP}:1": "—", f"Recall @custo {CUSTO_FN_FP}:1": "—"})
    tab_arm = pd.DataFrame(arm)
    acc_c = [float(cmp.loc[m, "acuracia"]) for m in ORDEM]
    acc_05 = [float(lim[(lim["modelo"] == NOMES[m]) & (lim["regra_limiar"] == "0,5 (padrão)")]["acuracia"].iloc[0]) for m in ORDEM]
    n_abaixo_trivial = sum(a <= dummy["acuracia"] for a in acc_c)
    erros = []
    for m in ORDEM:
        k = js["metricas_teste"][m]
        erros.append({"Modelo": NOMES[m] + (" ★" if m == final else ""), "VP": _inte(k["tp"]), "FN": _inte(k["fn"]),
                      "FP": _inte(k["fp"]), "VN": _inte(k["tn"]),
                      "Insatisfeitos identificados": _pct(k["recall"]),
                      "Contatos desnecessários por acerto": _n(k["fp"] / max(k["tp"], 1), 1)})
    tab_erros = pd.DataFrame(erros)
    menos_fn = min(ORDEM, key=lambda m: js["metricas_teste"][m]["fn"])
    menos_fp = min(ORDEM, key=lambda m: js["metricas_teste"][m]["fp"])
    sens_t = []
    for m in ORDEM:
        d = sens[sens["modelo"] == m].set_index("razao_custo")
        sens_t.append({"Modelo": NOMES[m] + (" ★" if m == final else ""),
                       **{f"FN={r}×FP (recall / precisão)": f"{_n(d.loc[r, 'recall'], 2)} / {_n(d.loc[r, 'precisao'], 2)}" for r in (1, 3, 10)}})
    tab_sens = pd.DataFrame(sens_t)
    of = []
    for m in ORDEM:
        r = cmp.loc[m]
        if m == "knn":
            flag, treino, gap = "n/a (treino inclui o próprio ponto)", "—", "—"
        else:
            treino, gap = _n(r["auc_treino"], 3), _n(r["gap_treino_cv"], 3)
            if r["gap_treino_cv"] > LIMITE_OVERFIT:
                flag = "treino > CV, mas generaliza" if abs(r["gap_cv_teste"]) < 0.02 else "sim"
            else:
                flag = "não"
        of.append({"Modelo": NOMES[m], "AUC treino": treino, "AUC CV": f"{_n(r['auc_cv'], 3)}±{_n(r['auc_cv_dp'], 3)}",
                   "AUC teste": _n(r["auc_teste"], 3), "Treino − CV": gap,
                   "CV − teste": _n(r["gap_cv_teste"], 3), "Overfitting?": flag})
    tab_of = pd.DataFrame(of)
    com_of = [NOMES[m] for m in ORDEM if m != "knn" and cmp.loc[m, "gap_treino_cv"] > LIMITE_OVERFIT and abs(cmp.loc[m, "gap_cv_teste"]) >= 0.02]
    treino_alto = [NOMES[m] for m in ORDEM if m != "knn" and cmp.loc[m, "gap_treino_cv"] > LIMITE_OVERFIT and abs(cmp.loc[m, "gap_cv_teste"]) < 0.02]
    maior_gap_teste = float(cmp["gap_cv_teste"].abs().max())
    a, g = cmp.loc["arvore_decisao"], cmp.loc["gradient_boosting"]
    ev = pd.DataFrame([{"Métrica": rot, "Árvore de decisão": _n(a[k], 3), "Gradient boosting": _n(g[k], 3),
                        "Diferença": f"{g[k] - a[k]:+.3f}".replace(".", ",")}
                       for rot, k in (("Acurácia", "acuracia"), ("Precisão", "precisao"), ("Recall", "recall"),
                                      ("F1", "f1"), ("AUC (teste)", "auc_teste"))])
    d_auc_ev = g["auc_teste"] - a["auc_teste"]
    ens_sig = bool(dif.loc["arvore_decisao", "diferenca_significativa"]) if final == "gradient_boosting" else None
    vl = pd.DataFrame([
        {"Modelo": NOMES["regressao_logistica"], "Vantagens": "simples, rápido, interpretável (odds ratio)",
         "Limitações observadas": f"linear: não captura o efeito de limiar do atraso; AUC teste {_n(cmp.loc['regressao_logistica', 'auc_teste'], 3)}"},
        {"Modelo": NOMES["arvore_decisao"], "Vantagens": "regras legíveis; captura limiares e interações; sem escala",
         "Limitações observadas": f"instável e propensa a overfitting sem limite de profundidade (ver curva); AUC teste {_n(a['auc_teste'], 3)}"},
        {"Modelo": NOMES["knn"], "Vantagens": "sem treino explícito; poucos hiperparâmetros",
         "Limitações observadas": f"sensível à dimensionalidade (one-hot), exige k alto, predição lenta ({_n(cmp.loc['knn', 'tempo_predicao_teste_s'], 1)} s no teste) e sem interpretação"},
        {"Modelo": NOMES["gradient_boosting"], "Vantagens": "melhor equilíbrio desempenho/custo; lida com não linearidade e categorias",
         "Limitações observadas": "menos interpretável (exige SHAP/importância); mais hiperparâmetros"},
        {"Modelo": NOMES["mlp"], "Vantagens": "flexível; predição rápida",
         "Limitações observadas": (f"sem interpretação direta; varia com a semente (AUC teste {_n(min(js['mlp_auc_teste_sementes']), 3)}–{_n(max(js['mlp_auc_teste_sementes']), 3)} em 3 sementes)"
                                   if js.get("mlp_auc_teste_sementes") else "sem interpretação direta; varia com a semente")},
    ])
    tab_cand = pd.DataFrame([{
        "Modelo": NOMES[m] + (" ★" if m == final else ""), "AUC CV": _n(js["cv"][m]["auc_medio"], 4),
        "Dentro de 1 erro-padrão do melhor": "sim" if m in cand else "não",
        "Custo esperado em CV (por cliente)": _n(js["cv"][m]["custo_oof"], 4),
    } for m in ORDEM])
    nao_signif = [NOMES[m] for m in ORDEM if m != final and not bool(dif.loc[m, "diferenca_significativa"])]
    signif_pior = [NOMES[m] for m in ORDEM if m != final and bool(dif.loc[m, "diferenca_significativa"])]
    temporal = ""
    if MODEL_JSON.exists():
        mj = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
        va = mj.get("validacoes_adicionais", {})
        if "temporal" in va:
            temporal = (f"* **Estabilidade temporal:** na etapa `model`, a validação temporal do modelo principal deu AUC "
                        f"{_n(va['temporal']['roc_auc'], 3)} (contra {_n(mj['metricas'][mj['melhor_modelo']]['roc_auc'], 3)} no holdout "
                        f"aleatório). A comparação deste relatório usa o holdout aleatório e **não** mede esse efeito.\n")
    borda = {NOMES[m]: v for m, v in js["na_borda_da_grade"].items() if v}
    txt_borda = ("; ".join(f"{k}: {', '.join(v)}" for k, v in borda.items())) if borda else None
    esc_exib = esc[["modelo", "hiperparametros_escolhidos", "no_limite_da_grade", "auc_cv_medio", "auc_cv_dp", "tempo_busca_s"]].copy()
    esc_exib["hiperparametros_escolhidos"] = esc_exib["hiperparametros_escolhidos"].map(
        lambda s: ", ".join(f"`{k}={v}`" for k, v in json.loads(s).items()))
    esc_exib["auc_cv_medio"] = [f"{a:.4f}±{b:.4f}" for a, b in zip(esc["auc_cv_medio"], esc["auc_cv_dp"])]
    esc_exib = esc_exib.drop(columns="auc_cv_dp")
    esc_exib.columns = ["Modelo", "Valores escolhidos", "No limite da grade?", "AUC CV", "Busca (s)"]
    md = f"""# Relatório de modelos supervisionados
**Tema:** predição de insatisfação do cliente no Olist (nota 1-2), em substituição ao tema de evasão, conforme a OBS da atividade.
**Pergunta central:** *qual modelo apresenta o desempenho mais adequado e que evidências sustentam essa escolha?*
> **Resposta curta:** o modelo escolhido é **{nf}** (★). Pelo critério declarado antes de olhar o teste, ele {crit_txt}.
> Com o falso negativo custando {CUSTO_FN_FP}× o falso positivo, no teste identifica **{_pct(cf['recall'])}** dos clientes insatisfeitos,
> com precisão de **{_pct(cf['precisao'])}** e AUC de **{_n(cf['auc_teste'], 3)}**.
---
* **O que se prevê:** `insatisfeito = 1` quando `review_score ≤ 2` (notas 1-2), por pedido entregue com avaliação.
* **Classe positiva:** insatisfeito. Prevalência de **{_pct(taxa)}** no teste ({_inte(n_test_pos)} de {_inte(js['n_teste'])} pedidos): classe rara.
* **Por que importa:** identificar o cliente insatisfeito permite ação corretiva (contato, compensação, correção operacional).
* **Tipos de erro:** *falso negativo* = cliente insatisfeito **não** identificado (o erro mais grave aqui); *falso positivo* = contato
  desnecessário com cliente que estava satisfeito.
* **Premissa de custo:** como o cliente insatisfeito pesa mais, adotou-se **custo(FN) = {CUSTO_FN_FP} × custo(FP)** para escolher o limiar de decisão.
  É uma premissa de negócio **não calibrada com custos reais**; por isso há uma análise de sensibilidade (razões de 1 a 10) na Seção 4.5.
* **Holdout:** divisão estratificada 80/20 (semente 42): **{_inte(js['n_treino'])}** pedidos de treino e **{_inte(js['n_teste'])}** de teste, com a mesma
  proporção de positivos ({_pct(js['taxa_positivos_treino'])} / {_pct(taxa)}). É a mesma partição usada em todo o pipeline.
* **O teste fica reservado:** hiperparâmetros, limiar de decisão e escolha do modelo usam **apenas o treino** (validação cruzada). O teste é usado
  uma vez, para a avaliação final.
* **k-fold estratificado (k = {CV_FOLDS}):** dentro do treino, cada dobra preserva a proporção de positivos. Estratificar é essencial com classe rara
  (12,8%): sem isso, dobras com poucos positivos tornariam AUC e recall instáveis.
* **Comparação justa:** mesmas dobras e mesma semente para todos os modelos; mesmo critério de seleção de hiperparâmetros (AUC de CV);
  mesma regra de limiar; pré-processamento **dentro** do pipeline (imputação, escala e one-hot ajustados apenas no treino de cada dobra,
  sem vazamento). Escala só para os modelos sensíveis a ela (regressão logística, KNN e MLP).
* **Limiar de decisão:** em vez do padrão 0,5, é o que **minimiza o custo esperado** (FN = {CUSTO_FN_FP} × FP) nas previsões fora-da-dobra do treino.
{_md(_grade_avaliada(hp))}
**Valores escolhidos** (maior AUC médio de validação cruzada; o ajuste final usa todo o treino):
{_md(esc_exib)}
* **KNN:** k controla o viés-variância (k pequeno = decisão local e ruidosa; k grande = suavização). A representação das variáveis (todas × selecionadas pela etapa `select`) entrou como hiperparâmetro.
* **Árvore:** a complexidade é controlada por `max_depth` e `min_samples_leaf`; a opção "sem limite" foi incluída para expor o overfitting.
* **Ensemble:** o `HistGradientBoosting` é um ensemble de árvores rasas em sequência (boosting), o que atende ao item de ensembles do enunciado.
* **MLP:** compararam-se arquiteturas (tamanho e profundidade) e regularização L2 (`alpha`), com parada antecipada.
{('* **Atenção:** em ' + txt_borda + ' o valor escolhido está no limite da grade; pode haver melhor valor fora dela.') if txt_borda else '* O valor escolhido de nenhum hiperparâmetro numérico ficou no limite da grade.'}
Todas as configurações avaliadas, com média e desvio da CV: `../reports/tables/tab22_hiperparametros_avaliados.csv`.
{_fig('fig24_curvas_validacao.png', 'Curvas de validação: complexidade × generalização')}
{_md(tab_principal)}
Líderes por métrica: {texto_lideres}.
{'Um mesmo modelo lidera todas as métricas.' if lider_unico else '**Nenhum modelo lidera todas as métricas**: a escolha depende de qual erro pesa mais.'}
Métricas complementares (PR-AUC, Brier skill, ECE, custo médio, F2) estão em `tab24_comparacao_modelos.csv`.
Não. Com {_pct(taxa)} de positivos, o classificador que **sempre prevê "satisfeito"** acerta **{_pct(dummy['acuracia'])}** e não identifica nenhum insatisfeito (recall 0).
{_md(tab_arm)}
No limiar de custo, a acurácia dos modelos fica entre **{_n(min(acc_c), 3)} e {_n(max(acc_c), 3)}**, praticamente a do classificador trivial
({_n(dummy['acuracia'], 3)}); em **{n_abaixo_trivial} de {len(ORDEM)}** modelos ela é menor ou igual à dele. Ao priorizar o cliente insatisfeito o limiar cai e a
acurácia diminui (média de {_n(float(np.mean(acc_05)), 3)} com limiar 0,5 para {_n(float(np.mean(acc_c)), 3)}): é o preço de capturar mais insatisfeitos.
Por isso a acurácia **não** distingue um modelo útil de um inútil e não pode ser o critério de escolha.
{_fig('fig22_matrizes_confusao_modelos.png', 'Matrizes de confusão no teste')}
{_md(tab_erros)}
* Menos falsos negativos (menos insatisfeitos perdidos): **{NOMES[menos_fn]}** ({_inte(js['metricas_teste'][menos_fn]['fn'])}).
* Menos falsos positivos (menos contatos desnecessários): **{NOMES[menos_fp]}** ({_inte(js['metricas_teste'][menos_fp]['fp'])}).
* Modelo final: {_inte(js['metricas_teste'][final]['fn'])} insatisfeitos não identificados e {_inte(js['metricas_teste'][final]['fp'])} contatos desnecessários.
{_fig('fig23_matriz_modelo_final.png', 'Matriz de confusão do modelo final')}
{_fig('fig21_roc_pr_modelos.png', 'Curvas ROC e precisão × recall no teste')}
A AUC não depende de limiar e resume a separação entre as classes. Os intervalos de confiança (bootstrap do teste, {_inte(js.get('n_bootstrap', 300))} reamostras) mostram
quais diferenças são reais. Diferença de AUC contra o modelo final: {'; '.join(f"{NOMES[m]} {dif.loc[m, 'delta_auc_medio']:+.3f}".replace('.', ',') + (' (significativa)' if bool(dif.loc[m, 'diferenca_significativa']) else ' (dentro do ruído)') for m in ORDEM if m != final)}.
O modelo de maior AUC {'é' if cmp['auc_teste'].idxmax() == final else '**não** é'} o modelo final.
Os AUCs dos {len(ORDEM)} modelos ficam numa faixa estreita ({_n(auc_min, 3)}–{_n(auc_max, 3)}): o teto parece vir da informação disponível nas
variáveis, e não da família de algoritmo.
Como o cliente insatisfeito pesa mais, o recall é priorizado, mas **a precisão não pode ser ignorada**: cada contato desnecessário tem custo.
O limiar sai do custo esperado, e a tabela mostra como recall e precisão mudam com a razão de custo (1 = erros equivalentes; 10 = FN muito mais grave):
{_md(tab_sens)}
{_fig('fig27_sensibilidade_custo.png', 'Sensibilidade à razão de custo FN/FP')}
{_md(tab_of)}
{_fig('fig25_overfitting_auc.png', 'AUC no treino, na validação cruzada e no teste')}
* Maior diferença |validação − teste| em AUC: **{_n(maior_gap_teste, 3)}**{' — os resultados da validação se mantêm no teste.' if maior_gap_teste < 0.02 else ' — atenção: a validação não se reproduz bem no teste.'}
* {('**Overfitting com perda de generalização** (treino − CV > ' + str(LIMITE_OVERFIT).replace('.', ',') + ' e validação ≠ teste): **' + ', '.join(com_of) + '**.') if com_of else 'Nenhum modelo perde desempenho entre a validação e o teste: **não há evidência de overfitting que prejudique a generalização**.'}
* {('**' + ', '.join(treino_alto) + '**: AUC de treino acima do de CV (diferença > ' + str(LIMITE_OVERFIT).replace('.', ',') + '), comportamento esperado de modelos flexíveis, sem prejuízo no teste.') if treino_alto else 'Nenhum modelo flexível mostra AUC de treino muito acima do de CV.'}
* **KNN:** o AUC de treino não é interpretável (cada ponto é seu próprio vizinho; com pesos por distância dá 1,0), então a generalização do KNN é avaliada só por CV × teste.
{_md(ev)}
O boosting {'supera' if d_auc_ev > 0 else 'não supera'} a árvore em AUC ({d_auc_ev:+.3f}{', diferença ' + ('significativa' if ens_sig else 'dentro do ruído') if ens_sig is not None else ''}).
Em troca, perde a leitura direta das regras.
{_fig('fig26_arvore_regras.png', 'Árvore de decisão: três primeiros níveis')}
```
{js['regras_arvore_texto'].strip()}
```
{_md(vl)}
**Critério (definido antes de ver o teste):** entre os modelos cujo AUC médio de CV está a até 1 erro-padrão do melhor
({NOMES[js['melhor_auc_cv']]}: {_n(js['cv'][js['melhor_auc_cv']]['auc_medio'], 4)}; 1 EP = {_n(js['erro_padrao_melhor'], 4)}), escolher o de **menor custo esperado** em CV (FN = {CUSTO_FN_FP} × FP).
{_md(tab_cand)}
{('Só o ' + nf + ' fica dentro de 1 erro-padrão do melhor; por isso o desempate por custo não foi necessário.') if cand == [final] else ''}
**Escolhido: {nf}.** Evidências:
* **Validação:** AUC de CV {_n(cf['auc_cv'], 3)} ± {_n(cf['auc_cv_dp'], 3)}; **teste** {_ic(cf['auc_teste'], cf['auc_ic95_inf'], cf['auc_ic95_sup'])}.
* **Erros no teste:** recall {_ic(cf['recall'], cf['recall_ic95_inf'], cf['recall_ic95_sup'])} e precisão {_ic(cf['precisao'], cf['precisao_ic95_inf'], cf['precisao_ic95_sup'])}.
* **Custo médio no teste** (FN = {CUSTO_FN_FP} × FP): {_n(cf['custo_medio'], 4)} por cliente, o menor entre {'os modelos' if cmp['custo_medio'].idxmin() == final else 'os candidatos, mas não entre todos os modelos'}.
* **Referência de custo:** não agir custa {_n(custo_nao_agir, 3)} por cliente (todos os insatisfeitos são falsos negativos) e contatar todos custa {_n(custo_contatar_todos, 3)}; o modelo final reduz o custo em **{_pct(reducao_custo)}** frente a não agir.
* **Generalização:** diferença treino − CV de {_n(cf['gap_treino_cv'], 3)} e CV − teste de {_n(cf['gap_cv_teste'], 3)}.
* **Diferenças para os demais:** o AUC do modelo final é maior em {_n(min(d_abs), 3)}–{_n(max(d_abs), 3)} de AUC; {('o IC95% pareado exclui zero para ' + ', '.join(signif_pior) + '.') if signif_pior else 'nenhuma diferença é significativa.'}{(' Dentro do ruído: ' + ', '.join(nao_signif) + '.') if nao_signif else ''} São diferenças **pequenas em termos práticos**: por isso a escolha também se apoia em erros, custo e interpretabilidade, e não só no AUC.
* **Não é o de maior acurácia:** {'o modelo final também tem a maior acurácia' if lideres['acurácia'] == final else f"a maior acurácia é de {NOMES[lideres['acurácia']]}, que prioriza menos o cliente insatisfeito"}.
* **Custo assumido:** a razão {CUSTO_FN_FP}:1 é premissa; com custos reais (contato, compensação, perda do cliente) o limiar e o modelo podem mudar. Veja a sensibilidade na Seção 4.5.
* **Um único holdout:** o teste dá uma estimativa; o intervalo por bootstrap só captura a variação de amostragem **do teste**, não a variabilidade do treino.
* **Grades finitas:** busca em grade com poucos valores; {('há hiperparâmetros no limite da grade (' + txt_borda + ').') if txt_borda else 'nenhum numérico ficou no limite.'}
* **Seleção de variáveis:** a representação "selecionadas" do KNN usa um consenso calculado em todo o treino, o que cria um vazamento leve entre as dobras de CV (não alcança o teste).
{temporal}* **Atraso realizado:** os modelos usam `atraso_dias` e `entrega_dias`, só conhecidos na entrega; para alerta **antes** da entrega o desempenho cai (ver relatório principal).
* **Calibração:** limiares de custo pressupõem probabilidades razoavelmente calibradas; ECE por modelo em `tab24_comparacao_modelos.csv`.
* **Próximos passos:** recalibrar probabilidades, buscar em grades mais finas (busca aleatória) nos modelos com hiperparâmetro na borda,
  repetir a CV com várias sementes, validar com corte temporal e usar custos reais para definir o limiar.
*Artefatos:* `../reports/tables/tab22` a `tab27`, `../reports/figures/fig21` a `fig27`, `outputs/resultados_modelos_supervisionados.json`, `outputs/modelo_supervisionado_final.joblib`.
"""
    RELATORIO_MODELOS_MD.write_text(md, encoding="utf-8")
    print(f"[relatorio_modelos] relatório -> {RELATORIO_MODELOS_MD}")
