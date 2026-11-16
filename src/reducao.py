from __future__ import annotations
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from .config import (
    ANALYSIS_CSV, CATEGORICAL_FEATURES, FIGURES, NUMERIC_FEATURES, PCA_JSON, RANDOM_STATE,
    REDUCAO_CSV, SELECAO_JSON,
)
from .features import carregar_analise
from .preprocess import dividir, pipeline_preprocess
CV_REDUCAO = 3
INTERPRETABILIDADE = {
    "Original": "Alta: cada coluna é uma variável de negócio (OR e SHAP por variável).",
    "Selecionada": "Alta: subconjunto das mesmas variáveis; leituras idênticas com menos colunas.",
    "PCA": "Baixa: componentes são combinações lineares das numéricas; a leitura passa pelas cargas.",
}
def _pipe(rep: str, modelo: str, numericas: list[str], categoricas: list[str], k: int | None) -> Pipeline:
    extra = ("pca", PCA(n_components=k, random_state=RANDOM_STATE)) if rep == "PCA" else None
    escalar = modelo == "regressao_logistica" or rep == "PCA"
    pre = pipeline_preprocess(escalar, numericas, categoricas, extra)
    if modelo == "regressao_logistica":
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, max_leaf_nodes=31, random_state=RANDOM_STATE,
        )
    return Pipeline([("pre", pre), ("clf", clf)])
def executar_etapa_reducao() -> None:
    for caminho in (ANALYSIS_CSV, SELECAO_JSON, PCA_JSON):
        if not caminho.exists():
            raise FileNotFoundError(f"{caminho.name} ausente: rode 'load', 'select' e 'pca' antes.")
    sel = json.loads(SELECAO_JSON.read_text(encoding="utf-8"))
    k = int(json.loads(PCA_JSON.read_text(encoding="utf-8"))["k_escolhido"])
    selecionadas = sel["selecionadas"]
    df = carregar_analise()
    X_train, X_test, y_train, y_test = dividir(df)
    cv = StratifiedKFold(CV_REDUCAO, shuffle=True, random_state=RANDOM_STATE)
    representacoes = {
        "Original": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "Selecionada": ([c for c in NUMERIC_FEATURES if c in selecionadas],
                        [c for c in CATEGORICAL_FEATURES if c in selecionadas]),
        "PCA": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    linhas, curvas = [], {}
    for rep, (num, cat) in representacoes.items():
        for modelo in ("regressao_logistica", "gradient_boosting"):
            colunas = num + cat
            pipe = _pipe(rep, modelo, num, cat, k)
            cvs = cross_val_score(pipe, X_train[colunas], y_train, cv=cv, scoring="roc_auc")
            oof = cross_val_predict(pipe, X_train[colunas], y_train, cv=5,
                                    method="predict_proba")[:, 1]
            p_o, r_o, t_o = precision_recall_curve(y_train, oof)
            f1_o = 2 * p_o[:-1] * r_o[:-1] / np.clip(p_o[:-1] + r_o[:-1], 1e-12, None)
            limiar = float(t_o[int(np.argmax(f1_o))])
            t0 = time.time()
            pipe.fit(X_train[colunas], y_train)
            t_fit = time.time() - t0
            prob = pipe.predict_proba(X_test[colunas])[:, 1]
            pred = (prob >= limiar).astype(int)
            n_dim = pipe.named_steps["pre"].transform(X_test[colunas].head(5)).shape[1]
            linhas.append({
                "representacao": rep, "modelo": modelo,
                "n_features_originais": len(colunas),
                "n_colunas_modelo": int(n_dim),
                "auc_cv_medio": float(cvs.mean()), "auc_cv_dp": float(cvs.std(ddof=1)),
                "auc_teste": float(roc_auc_score(y_test, prob)),
                "limiar_f1_oof": round(limiar, 4),
                "acuracia_teste": float(accuracy_score(y_test, pred)),
                "precisao_teste": float(precision_score(y_test, pred, zero_division=0)),
                "recall_teste": float(recall_score(y_test, pred, zero_division=0)),
                "f1_teste": float(f1_score(y_test, pred, zero_division=0)),
                "tempo_ajuste_s": round(t_fit, 1),
                "interpretabilidade": INTERPRETABILIDADE[rep],
            })
            curvas[(rep, modelo)] = roc_curve(y_test, prob)[:2]
            print(f"[reducao] {rep:12s} | {modelo:20s} | {n_dim:3d} colunas | "
                  f"AUC CV {cvs.mean():.4f} | AUC teste {linhas[-1]['auc_teste']:.4f} | "
                  f"F1 {linhas[-1]['f1_teste']:.3f} | acc {linhas[-1]['acuracia_teste']:.3f} | {t_fit:.1f}s")
    tab = pd.DataFrame(linhas)
    base = tab[tab["representacao"] == "Original"].set_index("modelo")["auc_teste"]
    tab["delta_auc_teste_vs_original"] = tab.apply(lambda r: r["auc_teste"] - base[r["modelo"]], axis=1)
    tab.to_csv(REDUCAO_CSV, index=False)
    print(f"[reducao] tabela -> {REDUCAO_CSV.name}")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    larg = 0.35
    reps = list(representacoes)
    nome_rep = {"Original": "Original", "Selecionada": "Selecionada", "PCA": f"PCA (k={k})"}
    nome_modelo = {"regressao_logistica": "Regressão logística",
                   "gradient_boosting": "Gradient boosting (HGB)"}
    for i, modelo in enumerate(("regressao_logistica", "gradient_boosting")):
        d = tab[tab["modelo"] == modelo].set_index("representacao").loc[reps]
        barras = ax.bar(np.arange(len(reps)) + (i - 0.5) * larg, d["auc_teste"], larg,
                        yerr=d["auc_cv_dp"], label=nome_modelo[modelo])
        ax.bar_label(barras, labels=[f"{a:.3f}\n({n} col.)" for a, n in
                                     zip(d["auc_teste"], d["n_colunas_modelo"])],
                     fontsize=8.5, padding=2)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([nome_rep[r] for r in reps], fontsize=11)
    ax.set_ylim(tab["auc_teste"].min() - 0.05, tab["auc_teste"].max() + 0.05)
    ax.set_ylabel("AUC no teste (barras de erro = DP da CV)", fontsize=11)
    ax.set_title("Original × selecionada × PCA: dimensionalidade e informação preservada",
                 fontsize=13, pad=10)
    ax.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig19_comparacao_representacoes.png")
    plt.close(fig)
    print("[reducao] figura -> fig19_comparacao_representacoes.png")
    cores = {"Original": "#1f77b4", "Selecionada": "#2ca02c", "PCA": "#d62728"}
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for ax, modelo in zip(eixos, ("regressao_logistica", "gradient_boosting")):
        for rep in representacoes:
            fpr, tpr = curvas[(rep, modelo)]
            auc = tab[(tab.representacao == rep) & (tab.modelo == modelo)]["auc_teste"].iloc[0]
            rot = f"PCA (k={k})" if rep == "PCA" else rep
            ax.plot(fpr, tpr, color=cores[rep], lw=2,
                    label=f"{rot} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="aleatório (AUC = 0,5)")
        ax.set_xlabel("Falso positivo (1 − especificidade)", fontsize=11)
        ax.set_ylabel("Verdadeiro positivo (sensibilidade)", fontsize=11)
        ax.set_title("Regressão logística" if modelo == "regressao_logistica"
                     else "Gradient boosting (HGB)", fontsize=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Curvas ROC no teste: original × selecionada × PCA", fontsize=14,
                 fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig20_roc_representacoes.png", bbox_inches="tight")
    plt.close(fig)
    print("[reducao] figura -> fig20_roc_representacoes.png")
    print("[reducao] etapa concluída.")
