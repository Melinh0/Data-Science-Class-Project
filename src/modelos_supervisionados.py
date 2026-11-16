from __future__ import annotations
import json
import os
import time
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from .config import (
    ANALYSIS_CSV, CATEGORICAL_FEATURES, CUSTO_FN_FP, CV_FOLDS, FIGURES,
    MODELOS_JOBLIB, MODELOS_JSON, NUMERIC_FEATURES, RANDOM_STATE, SELECAO_JSON, TABLES,
)
from .features import carregar_analise
from .preprocess import dividir, nomes_saida, pipeline_preprocess
warnings.filterwarnings("ignore")
RAPIDO = os.environ.get("MODELOS_RAPIDO") == "1"
N_BOOT = 50 if RAPIDO else 300
RAZOES_CUSTO = [1, 2, 3, 5, 10]
SEMENTES_MLP = [0, 1, 2]
N_TREINO_KNN_AUC = 10000
ORDEM = ["regressao_logistica", "arvore_decisao", "knn", "gradient_boosting", "mlp"]
NOMES = {
    "regressao_logistica": "Regressão logística",
    "arvore_decisao": "Árvore de decisão",
    "knn": "KNN",
    "gradient_boosting": "Gradient boosting",
    "mlp": "MLP",
    "dummy": "Classe majoritária",
}
CORES = {
    "regressao_logistica": "#7f7f7f", "arvore_decisao": "#2ca02c", "knn": "#ff7f0e",
    "gradient_boosting": "#1f77b4", "mlp": "#9467bd", "dummy": "#bbbbbb",
}
FAMILIA = {
    "regressao_logistica": "linear (baseline)", "arvore_decisao": "árvore única",
    "knn": "baseado em instâncias", "gradient_boosting": "ensemble de árvores (boosting)",
    "mlp": "rede neural",
}
def _registro(selecionadas: list[str]) -> dict[str, dict]:
    num_sel = [c for c in NUMERIC_FEATURES if c in selecionadas]
    cat_sel = [c for c in CATEGORICAL_FEATURES if c in selecionadas]
    pre_todas = pipeline_preprocess(escalar_numerico=True)
    pre_sel = pipeline_preprocess(escalar_numerico=True, numericas=num_sel, categoricas=cat_sel)
    def pre(escalar: bool):
        return pipeline_preprocess(escalar_numerico=escalar)
    reg = {
        "regressao_logistica": dict(
            pipe=Pipeline([("pre", pre(True)),
                           ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
            grade={"clf__C": [0.01, 0.1, 1.0, 10.0]},
            n_jobs=4, train_score=True),
        "arvore_decisao": dict(
            pipe=Pipeline([("pre", pre(False)),
                           ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            grade={"clf__max_depth": [3, 4, 5, 6, 8, 10, None],
                   "clf__min_samples_leaf": [20, 50, 100, 200],
                   "clf__criterion": ["gini", "entropy"]},
            n_jobs=8, train_score=True),
        "knn": dict(
            pipe=Pipeline([("pre", pre_todas), ("clf", KNeighborsClassifier())]),
            grade={"pre": [pre_todas, pre_sel],
                   "clf__n_neighbors": [5, 15, 51, 151, 301],
                   "clf__weights": ["uniform", "distance"]},
            n_jobs=8, train_score=False),
        "gradient_boosting": dict(
            pipe=Pipeline([("pre", pre(False)),
                           ("clf", HistGradientBoostingClassifier(max_iter=500, random_state=RANDOM_STATE))]),
            grade={"clf__learning_rate": [0.03, 0.08, 0.15],
                   "clf__max_leaf_nodes": [15, 31],
                   "clf__min_samples_leaf": [20, 100]},
            n_jobs=2, train_score=True),
        "mlp": dict(
            pipe=Pipeline([("pre", pre(True)),
                           ("clf", MLPClassifier(early_stopping=True, validation_fraction=0.1,
                                                 n_iter_no_change=10, max_iter=200,
                                                 random_state=RANDOM_STATE))]),
            grade={"clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                   "clf__alpha": [1e-4, 1e-3, 1e-2]},
            n_jobs=4, train_score=True),
    }
    if RAPIDO:
        reg["regressao_logistica"]["grade"] = {"clf__C": [1.0]}
        reg["arvore_decisao"]["grade"] = {"clf__max_depth": [4, None], "clf__min_samples_leaf": [100]}
        reg["knn"]["grade"] = {"pre": [pre_todas], "clf__n_neighbors": [51], "clf__weights": ["uniform"]}
        reg["gradient_boosting"]["grade"] = {"clf__learning_rate": [0.08], "clf__max_leaf_nodes": [31]}
        reg["mlp"]["grade"] = {"clf__hidden_layer_sizes": [(32,)], "clf__alpha": [1e-3]}
    return reg
def _rotulo_param(valor) -> str:
    if valor is None:
        return "sem limite"
    if isinstance(valor, tuple):
        return "(" + ",".join(str(v) for v in valor) + ")"
    if hasattr(valor, "transformers"):
        n = sum(len(t[2]) for t in valor.transformers)
        return f"{'todas' if n == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) else 'selecionadas'} ({n} variáveis)"
    return str(valor)
def _params_texto(params: dict) -> dict[str, str]:
    return {k.split("__")[-1]: _rotulo_param(v) for k, v in params.items()}
def _na_borda(grade: dict, params: dict) -> list[str]:
    saida = []
    for k, valores in grade.items():
        nums = [v for v in valores if isinstance(v, (int, float)) and not isinstance(v, bool)]
        v = params.get(k)
        if len(nums) >= 3 and isinstance(v, (int, float)) and (v == min(nums) or v == max(nums)):
            saida.append(k.split("__")[-1])
    return saida
def _custo_limiar(y, p, r: float) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    ordem = np.argsort(p, kind="mergesort")
    ps, ys = p[ordem], y[ordem]
    n, total_pos = len(y), int(ys.sum())
    pos_antes = np.concatenate([[0], np.cumsum(ys)])
    cand, primeiro = np.unique(ps, return_index=True)
    fn = pos_antes[primeiro]
    fp = (n - primeiro) - (total_pos - fn)
    custo = r * fn + fp
    if r * total_pos < custo.min():
        return float(np.nextafter(ps.max(), np.inf))
    return float(cand[int(np.argmin(custo))])
def _limiar_f1(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])
def _ece(y, p, bins: int = 10) -> float:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["b"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())
def _met(y, p, limiar: float, r: float = CUSTO_FN_FP) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    return {
        "acuracia": (tp + tn) / len(y), "precisao": prec, "recall": rec, "f1": f1, "f2": f2,
        "custo_medio": (r * fn + fp) / len(y), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "limiar": float(limiar),
    }
def _oof(pipe, X, y, cv) -> tuple[np.ndarray, list[float]]:
    oof = np.zeros(len(y))
    aucs = []
    for tr, va in cv.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
    return oof, aucs
def _salvar_fig(fig, nome: str) -> None:
    fig.savefig(FIGURES / nome)
    plt.close(fig)
    print(f"[modelos] figura -> {nome}")
def _fig_roc_pr(y_test, probs: dict, ponto: dict, final: str) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for nome in ORDEM:
        p = probs[nome]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        largura = 2.6 if nome == final else 1.5
        rotulo = f"{NOMES[nome]} (AUC={roc_auc_score(y_test, p):.3f})" + (" (final)" if nome == final else "")
        eixos[0].plot(fpr, tpr, color=CORES[nome], lw=largura, label=rotulo)
        eixos[1].plot(rec, prec, color=CORES[nome], lw=largura,
                      label=f"{NOMES[nome]} (AP={average_precision_score(y_test, p):.3f})")
        m = ponto[nome]
        eixos[0].scatter([m["fp"] / (m["fp"] + m["tn"])], [m["recall"]], color=CORES[nome], s=55, zorder=5,
                         edgecolor="black")
        eixos[1].scatter([m["recall"]], [m["precisao"]], color=CORES[nome], s=55, zorder=5, edgecolor="black")
    eixos[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    eixos[0].set_xlabel("Taxa de falsos positivos")
    eixos[0].set_ylabel("Taxa de verdadeiros positivos (recall)")
    eixos[0].set_title("Curvas ROC no teste (● = limiar de decisão)")
    eixos[0].legend(loc="lower right", fontsize=8)
    base = float(np.mean(y_test))
    eixos[1].axhline(base, ls="--", color="gray", lw=0.9, label=f"baseline = {base:.3f}")
    eixos[1].set_xlabel("Recall")
    eixos[1].set_ylabel("Precisão")
    eixos[1].set_title("Precisão × recall no teste (● = limiar de decisão)")
    eixos[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _salvar_fig(fig, "fig21_roc_pr_modelos.png")
def _desenha_matriz(ax, m: dict, titulo: str, destaque: bool = False) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.0%})".replace(",", "."), ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Prev. satisfeito", "Prev. insatisfeito"], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Real satisfeito", "Real insatisfeito"], fontsize=8)
    ax.set_title(titulo, fontsize=9, fontweight="bold", color="#b2182b" if destaque else "black")
    ax.grid(False)
    if destaque:
        for s in ax.spines.values():
            s.set_edgecolor("#b2182b")
            s.set_linewidth(2.5)
def _fig_matrizes(metricas: dict, dummy: dict, final: str) -> None:
    chaves = ORDEM + ["dummy"]
    fig, eixos = plt.subplots(2, 3, figsize=(12.5, 8))
    for ax, nome in zip(eixos.ravel(), chaves):
        m = dummy if nome == "dummy" else metricas[nome]
        titulo = f"{NOMES[nome]}{' (final)' if nome == final else ''}\nlimiar {m['limiar']:.2f} | recall {m['recall']:.0%} | precisão {m['precisao']:.0%}"
        _desenha_matriz(ax, m, titulo, destaque=nome == final)
    fig.suptitle(f"Matrizes de confusão no teste (percentuais por linha; falso negativo custa {CUSTO_FN_FP}× o falso positivo)",
                 fontweight="bold")
    fig.tight_layout()
    _salvar_fig(fig, "fig22_matrizes_confusao_modelos.png")
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    m = metricas[final]
    _desenha_matriz(ax, m, f"Modelo final: {NOMES[final]}\nlimiar {m['limiar']:.2f} | recall {m['recall']:.0%} | precisão {m['precisao']:.0%}", True)
    fig.tight_layout()
    _salvar_fig(fig, "fig23_matriz_modelo_final.png")
def _fig_validacao(resultados_cv: dict) -> None:
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
    ax = eixos[0, 0]
    d = resultados_cv["arvore_decisao"].copy()
    d["prof"] = d["param_clf__max_depth"].map(lambda v: 99 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v))
    g = d.groupby("prof").apply(lambda t: t.loc[t["mean_test_roc_auc"].idxmax()])
    x = [("sem\nlimite" if k == 99 else str(k)) for k in g.index]
    ax.plot(x, g["mean_train_roc_auc"], "o--", color="#2ca02c", label="treino")
    ax.plot(x, g["mean_test_roc_auc"], "o-", color="#d62728", label="validação (CV)")
    ax.set_title("Árvore: AUC × profundidade máxima")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("AUC")
    ax.legend()
    ax = eixos[0, 1]
    d = resultados_cv["knn"].copy()
    d["k"] = d["param_clf__n_neighbors"].astype(int)
    for w, t in d.groupby("param_clf__weights"):
        g = t.groupby("k")["mean_test_roc_auc"].max()
        ax.plot(g.index, g.values, "o-", label=f"pesos: {w}")
    ax.set_xscale("log")
    ax.set_title("KNN: AUC de validação × k")
    ax.set_xlabel("k (vizinhos)")
    ax.set_ylabel("AUC (CV)")
    ax.legend()
    ax = eixos[1, 0]
    d = resultados_cv["gradient_boosting"].copy()
    d["lr"] = d["param_clf__learning_rate"].astype(float)
    for leaves, t in d.groupby("param_clf__max_leaf_nodes"):
        g = t.groupby("lr").apply(lambda u: u.loc[u["mean_test_roc_auc"].idxmax()])
        ax.plot(g.index, g["mean_test_roc_auc"], "o-", label=f"folhas={leaves} (validação)")
        ax.plot(g.index, g["mean_train_roc_auc"], "x--", alpha=0.6, label=f"folhas={leaves} (treino)")
    ax.set_title("Boosting: AUC × taxa de aprendizado")
    ax.set_xlabel("learning_rate")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
    ax = eixos[1, 1]
    d = resultados_cv["mlp"].copy()
    d["alpha"] = d["param_clf__alpha"].astype(float)
    d["camadas"] = d["param_clf__hidden_layer_sizes"].map(_rotulo_param)
    for cam, t in d.groupby("camadas"):
        t = t.sort_values("alpha")
        ax.plot(t["alpha"], t["mean_test_roc_auc"], "o-", label=f"{cam} (validação)")
        ax.plot(t["alpha"], t["mean_train_roc_auc"], "x--", alpha=0.6, label=f"{cam} (treino)")
    ax.set_xscale("log")
    ax.set_title("MLP: AUC × regularização (alpha)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
    fig.suptitle("Curvas de validação: complexidade × generalização (treino só onde computado)",
                 fontweight="bold")
    fig.tight_layout()
    _salvar_fig(fig, "fig24_curvas_validacao.png")
def _fig_overfitting(tab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    larg = 0.26
    x = np.arange(len(tab))
    treino = tab["auc_treino"].where(tab["modelo"] != "knn")
    b1 = ax.bar(x - larg, treino, larg, label="treino", color="#bbbbbb")
    b2 = ax.bar(x, tab["auc_cv"], larg, yerr=tab["auc_cv_dp"], label="validação cruzada", color="#4c72b0")
    b3 = ax.bar(x + larg, tab["auc_teste"], larg, label="teste", color="#c44e52")
    ax.bar_label(b1, labels=["n/a" if np.isnan(v) else f"{v:.3f}" for v in treino], fontsize=7, padding=2)
    ax.bar_label(b3, labels=[f"{v:.3f}" for v in tab["auc_teste"]], fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([NOMES[m] for m in tab["modelo"]], fontsize=9)
    ax.set_ylim(min(0.6, tab["auc_teste"].min() - 0.05), min(1.0, treino.max() + 0.05))
    ax.set_ylabel("AUC")
    ax.set_title("Evidência de overfitting: AUC no treino, na validação cruzada e no teste\n(KNN: AUC de treino não se aplica, pois cada ponto é seu próprio vizinho)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _salvar_fig(fig, "fig25_overfitting_auc.png")
def _fig_arvore(est: Pipeline) -> str:
    pre, clf = est.named_steps["pre"], est.named_steps["clf"]
    nomes = nomes_saida(pre)
    fig, ax = plt.subplots(figsize=(17, 7.5))
    plot_tree(clf, max_depth=3, feature_names=nomes, class_names=["satisfeito", "insatisfeito"],
              filled=True, rounded=True, proportion=True, fontsize=8, ax=ax, impurity=False)
    ax.set_title("Árvore de decisão ajustada: três primeiros níveis (cores = classe majoritária; valores = proporção)")
    _salvar_fig(fig, "fig26_arvore_regras.png")
    return export_text(clf, feature_names=nomes, max_depth=3, decimals=2)
def _fig_sensibilidade(tab: pd.DataFrame, final: str) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8))
    for nome in ORDEM:
        d = tab[tab["modelo"] == nome].sort_values("razao_custo")
        lw = 2.6 if nome == final else 1.5
        eixos[0].plot(d["razao_custo"], d["recall"], "o-", color=CORES[nome], lw=lw, label=NOMES[nome])
        eixos[1].plot(d["razao_custo"], d["precisao"], "o-", color=CORES[nome], lw=lw, label=NOMES[nome])
    for ax, t in zip(eixos, ("Recall", "Precisão")):
        ax.set_xscale("log")
        ax.set_xticks(RAZOES_CUSTO)
        ax.set_xticklabels([str(r) for r in RAZOES_CUSTO])
        ax.axvline(CUSTO_FN_FP, color="red", ls=":", lw=1)
        ax.set_xlabel("Custo do falso negativo / custo do falso positivo")
        ax.set_ylabel(t)
        ax.set_title(f"{t} no teste ao priorizar o cliente insatisfeito")
    eixos[0].legend(fontsize=8)
    fig.tight_layout()
    _salvar_fig(fig, "fig27_sensibilidade_custo.png")
def executar_etapa_modelos() -> None:
    if not ANALYSIS_CSV.exists() or not SELECAO_JSON.exists():
        raise FileNotFoundError("Rode 'load' e 'select' antes de 'modelos'.")
    inicio = time.time()
    selecionadas = json.loads(SELECAO_JSON.read_text(encoding="utf-8"))["selecionadas"]
    df = carregar_analise()
    X_train, X_test, y_train, y_test = dividir(df)
    print(f"[modelos] treino {len(X_train):,} / teste {len(X_test):,} | positivos {y_train.mean():.1%} "
          f"| {'MODO RÁPIDO' if RAPIDO else 'grades completas'} | custo FN:FP = {CUSTO_FN_FP}:1")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    registro = _registro(selecionadas)
    saida: dict[str, dict] = {}
    cv_results: dict[str, pd.DataFrame] = {}
    linhas_hp, linhas_esc = [], []
    for nome in ORDEM:
        cfg = registro[nome]
        n_conf = int(np.prod([len(v) for v in cfg["grade"].values()]))
        print(f"[modelos] {NOMES[nome]}: {n_conf} configurações x {CV_FOLDS} dobras...")
        t0 = time.time()
        gs = GridSearchCV(
            cfg["pipe"], cfg["grade"], cv=cv, n_jobs=cfg["n_jobs"], refit="roc_auc",
            scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
            return_train_score=cfg["train_score"], error_score="raise",
        )
        gs.fit(X_train, y_train)
        t_busca = time.time() - t0
        res = pd.DataFrame(gs.cv_results_)
        cv_results[nome] = res
        est = gs.best_estimator_
        params = gs.best_params_
        print(f"[modelos]   melhor: {_params_texto(params)} | AUC CV {gs.best_score_:.4f} | busca {t_busca:.0f}s")
        for i, r in res.iterrows():
            linhas_hp.append({
                "modelo": NOMES[nome], "configuracao": json.dumps(_params_texto(r["params"]), ensure_ascii=False),
                "auc_cv_medio": r["mean_test_roc_auc"], "auc_cv_dp": r["std_test_roc_auc"],
                "pr_auc_cv_medio": r["mean_test_average_precision"],
                "auc_treino_medio": r["mean_train_roc_auc"] if cfg["train_score"] else np.nan,
                "rank_auc_cv": int(r["rank_test_roc_auc"]),
            })
        t0 = time.time()
        oof, aucs = _oof(cfg["pipe"].set_params(**params), X_train, y_train, cv)
        t_oof = time.time() - t0
        lim_custo, lim_f1 = _custo_limiar(y_train, oof, CUSTO_FN_FP), _limiar_f1(y_train, oof)
        t0 = time.time()
        p_test = est.predict_proba(X_test)[:, 1]
        t_pred = time.time() - t0
        if nome == "knn":
            amostra = X_train.sample(n=min(N_TREINO_KNN_AUC, len(X_train)), random_state=RANDOM_STATE)
            auc_treino = float(roc_auc_score(y_train.loc[amostra.index], est.predict_proba(amostra)[:, 1]))
        else:
            auc_treino = float(roc_auc_score(y_train, est.predict_proba(X_train)[:, 1]))
        extras = {}
        if nome == "mlp":
            auc_s = []
            for s in SEMENTES_MLP:
                m = clone(est).set_params(clf__random_state=s).fit(X_train, y_train)
                auc_s.append(float(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])))
            extras = {"auc_teste_sementes": auc_s}
        borda = _na_borda(cfg["grade"], params)
        saida[nome] = {
            "est": est, "params": params, "p_test": p_test, "oof": oof, "aucs": aucs,
            "lim_custo": lim_custo, "lim_f1": lim_f1, "auc_treino": auc_treino,
            "cv_mean": float(np.mean(aucs)), "cv_std": float(np.std(aucs, ddof=1)),
            "custo_oof": _met(y_train, oof, lim_custo)["custo_medio"],
            "t_busca": t_busca + t_oof, "t_pred": t_pred, "borda": borda, **extras,
        }
        linhas_esc.append({
            "modelo": NOMES[nome], "familia": FAMILIA[nome], "n_configuracoes": n_conf,
            "hiperparametros_escolhidos": json.dumps(_params_texto(params), ensure_ascii=False),
            "no_limite_da_grade": ", ".join(borda) if borda else "—",
            "auc_cv_medio": saida[nome]["cv_mean"], "auc_cv_dp": saida[nome]["cv_std"],
            "tempo_busca_s": round(t_busca + t_oof, 1),
        })
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    p_dummy = dummy.predict_proba(X_test)[:, 1]
    met_dummy = _met(y_test, p_dummy, 0.5)
    metricas, linhas_lim = {}, []
    for nome in ORDEM:
        s = saida[nome]
        metricas[nome] = _met(y_test, s["p_test"], s["lim_custo"])
        for regra, t in (("0,5 (padrão)", 0.5), ("F1 máximo (CV)", s["lim_f1"]),
                         (f"custo FN={CUSTO_FN_FP}×FP (CV)", s["lim_custo"])):
            m = _met(y_test, s["p_test"], t)
            linhas_lim.append({"modelo": NOMES[nome], "regra_limiar": regra, **{k: m[k] for k in
                ("limiar", "acuracia", "precisao", "recall", "f1", "f2", "custo_medio", "tn", "fp", "fn", "tp")}})
    m = met_dummy
    linhas_lim.append({"modelo": NOMES["dummy"], "regra_limiar": "0,5 (padrão)", **{k: m[k] for k in
        ("limiar", "acuracia", "precisao", "recall", "f1", "f2", "custo_medio", "tn", "fp", "fn", "tp")}})
    rng = np.random.default_rng(RANDOM_STATE)
    y_arr = y_test.to_numpy()
    idx = rng.integers(0, len(y_arr), size=(N_BOOT, len(y_arr)))
    boot = {}
    for nome in ORDEM:
        p, t = saida[nome]["p_test"], saida[nome]["lim_custo"]
        auc_b, rec_b, prec_b = [], [], []
        for b in idx:
            yb, pb = y_arr[b], p[b]
            auc_b.append(roc_auc_score(yb, pb))
            pred = pb >= t
            tp = int((pred & (yb == 1)).sum())
            rec_b.append(tp / max(int((yb == 1).sum()), 1))
            prec_b.append(tp / max(int(pred.sum()), 1))
        boot[nome] = {"auc": np.array(auc_b), "recall": np.array(rec_b), "precisao": np.array(prec_b)}
    cv_melhor = max(ORDEM, key=lambda n: saida[n]["cv_mean"])
    se = saida[cv_melhor]["cv_std"] / np.sqrt(CV_FOLDS)
    limite = saida[cv_melhor]["cv_mean"] - se
    candidatos = [n for n in ORDEM if saida[n]["cv_mean"] >= limite]
    final = min(candidatos, key=lambda n: saida[n]["custo_oof"])
    print(f"[modelos] melhor AUC de CV: {NOMES[cv_melhor]} ({saida[cv_melhor]['cv_mean']:.4f}, 1 EP = {se:.4f}); "
          f"candidatos (dentro de 1 EP): {[NOMES[n] for n in candidatos]}")
    print(f"[modelos] modelo final (menor custo esperado em CV entre os candidatos): {NOMES[final]}")
    linhas_cmp = []
    for nome in ORDEM:
        s, m = saida[nome], metricas[nome]
        linhas_cmp.append({
            "modelo": nome, "limiar": m["limiar"], "acuracia": m["acuracia"], "precisao": m["precisao"],
            "recall": m["recall"], "f1": m["f1"], "auc_teste": float(roc_auc_score(y_test, s["p_test"])),
            "auc_cv": s["cv_mean"], "auc_cv_dp": s["cv_std"], "auc_treino": s["auc_treino"],
            "gap_treino_cv": s["auc_treino"] - s["cv_mean"], "gap_cv_teste": s["cv_mean"] - float(roc_auc_score(y_test, s["p_test"])),
            "auc_ic95_inf": float(np.percentile(boot[nome]["auc"], 2.5)), "auc_ic95_sup": float(np.percentile(boot[nome]["auc"], 97.5)),
            "recall_ic95_inf": float(np.percentile(boot[nome]["recall"], 2.5)), "recall_ic95_sup": float(np.percentile(boot[nome]["recall"], 97.5)),
            "precisao_ic95_inf": float(np.percentile(boot[nome]["precisao"], 2.5)), "precisao_ic95_sup": float(np.percentile(boot[nome]["precisao"], 97.5)),
            "pr_auc_teste": float(average_precision_score(y_test, s["p_test"])),
            "brier_skill": float(1 - brier_score_loss(y_test, s["p_test"]) / (y_test.mean() * (1 - y_test.mean()))),
            "ece": _ece(y_test, s["p_test"]), "custo_medio": m["custo_medio"],
            "tempo_busca_s": round(s["t_busca"], 1), "tempo_predicao_teste_s": round(s["t_pred"], 2),
            "final": nome == final, "candidato_1ep": nome in candidatos,
        })
    tab_cmp = pd.DataFrame(linhas_cmp)
    deltas = []
    for nome in ORDEM:
        d = boot[nome]["auc"] - boot[final]["auc"]
        deltas.append({"modelo": NOMES[nome], "comparado_com": NOMES[final],
                       "delta_auc_medio": float(d.mean()), "ic95_inf": float(np.percentile(d, 2.5)),
                       "ic95_sup": float(np.percentile(d, 97.5)),
                       "diferenca_significativa": bool(nome != final and (np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0))})
    linhas_sens = []
    for nome in ORDEM:
        for r in RAZOES_CUSTO:
            t = _custo_limiar(y_train, saida[nome]["oof"], r)
            m = _met(y_test, saida[nome]["p_test"], t, r)
            linhas_sens.append({"modelo": nome, "razao_custo": r, "limiar": t, "recall": m["recall"],
                                "precisao": m["precisao"], "f1": m["f1"], "custo_medio": m["custo_medio"]})
    tab_sens = pd.DataFrame(linhas_sens)
    pd.DataFrame(linhas_hp).to_csv(TABLES / "tab22_hiperparametros_avaliados.csv", index=False)
    pd.DataFrame(linhas_esc).to_csv(TABLES / "tab23_hiperparametros_escolhidos.csv", index=False)
    tab_cmp.to_csv(TABLES / "tab24_comparacao_modelos.csv", index=False)
    pd.DataFrame(linhas_lim).to_csv(TABLES / "tab25_limiares_e_erros.csv", index=False)
    tab_sens.to_csv(TABLES / "tab26_sensibilidade_custo.csv", index=False)
    pd.DataFrame(deltas).to_csv(TABLES / "tab27_diferenca_auc_vs_final.csv", index=False)
    print("[modelos] tabelas -> tab22 a tab27")
    probs = {n: saida[n]["p_test"] for n in ORDEM}
    _fig_roc_pr(y_test, probs, metricas, final)
    _fig_matrizes(metricas, met_dummy, final)
    _fig_validacao(cv_results)
    _fig_overfitting(tab_cmp)
    regras = _fig_arvore(saida["arvore_decisao"]["est"])
    _fig_sensibilidade(tab_sens, final)
    joblib.dump(saida[final]["est"], MODELOS_JOBLIB)
    payload = {
        "modelo_final": final, "nome_final": NOMES[final], "custo_fn_fp": CUSTO_FN_FP,
        "modo_rapido": RAPIDO, "n_bootstrap": N_BOOT, "n_treino": int(len(X_train)), "n_teste": int(len(X_test)),
        "taxa_positivos_treino": float(y_train.mean()), "taxa_positivos_teste": float(y_test.mean()),
        "criterio_final": (f"entre os modelos com AUC de CV a até 1 erro-padrão do melhor "
                           f"({NOMES[cv_melhor]}: {saida[cv_melhor]['cv_mean']:.4f} - {se:.4f}), "
                           f"menor custo esperado em CV com FN = {CUSTO_FN_FP} x FP"),
        "melhor_auc_cv": cv_melhor, "erro_padrao_melhor": float(se), "candidatos": candidatos,
        "metricas_teste": {n: metricas[n] for n in ORDEM}, "dummy_teste": met_dummy,
        "cv": {n: {"auc_medio": saida[n]["cv_mean"], "auc_dp": saida[n]["cv_std"],
                   "auc_por_dobra": [float(a) for a in saida[n]["aucs"]], "custo_oof": saida[n]["custo_oof"],
                   "limiar_custo": saida[n]["lim_custo"], "limiar_f1": saida[n]["lim_f1"]} for n in ORDEM},
        "hiperparametros": {n: _params_texto(saida[n]["params"]) for n in ORDEM},
        "na_borda_da_grade": {n: saida[n]["borda"] for n in ORDEM},
        "mlp_auc_teste_sementes": saida["mlp"].get("auc_teste_sementes"),
        "regras_arvore_texto": regras,
        "tempo_total_s": round(time.time() - inicio, 1),
    }
    MODELOS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[modelos] resultados -> {MODELOS_JSON.name} | modelo final salvo -> {MODELOS_JOBLIB.name}")
    print(f"[modelos] etapa concluída em {time.time() - inicio:.0f}s.")
