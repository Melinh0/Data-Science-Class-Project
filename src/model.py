"""Etapa 3 — Modelo de classificação: probabilidade de avaliação negativa.

Dois modelos comparados por validação cruzada:
  * Regressão logística (interpretável -> odds ratios);
  * Gradient boosting para árvores (HGB) com importância por permutação,
    dependência parcial e (se disponível) SHAP.

Variáveis de texto/comentário são EXCLUÍDAS por vazamento (o comentário faz
parte da própria avaliação). Saídas em outputs/ e reports/.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ANALYSIS_CSV, CATEGORICAL_FEATURES, CV_FOLDS, FIGURES, MODEL_JOBLIB,
    MODEL_JSON, NUMERIC_FEATURES, RANDOM_STATE, TABLES, TARGET, TEST_SIZE,
)
from .features import carregar_analise

warnings.filterwarnings("ignore")


# ------------------------------------------------------------ pré-processamento --
def _preprocessador(escalar: bool) -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ] + ([("scaler", StandardScaler())] if escalar else []))
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                              sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num, NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def _modelos() -> dict[str, Pipeline]:
    return {
        "regressao_logistica": Pipeline([
            ("pre", _preprocessador(escalar=True)),
            ("clf", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", _preprocessador(escalar=False)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
    }


# ------------------------------------------------------------------- métricas --
def _metricas(y_true, y_prob, limiar: float = 0.5) -> dict:
    y_pred = (y_prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "precisao": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "acuracia": round(float((y_pred == y_true).mean()), 4),
        "matriz_confusao": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "limiar": limiar,
    }


# ----------------------------------------------------------------- interpretação --
def _odds_ratios(pipe: Pipeline) -> pd.DataFrame:
    """Extrai coeficientes da logística em escala original (odds ratios)."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    nomes = pre.get_feature_names_out(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    nomes = [n.split("__", 1)[-1] for n in nomes]

    coefs = clf.coef_.ravel()
    escala = np.ones(len(nomes))
    num_names = list(pre.transformers_[0][1].get_feature_names_out(NUMERIC_FEATURES))
    num_names = [n.split("__", 1)[-1] for n in num_names]
    scaler = pre.named_transformers_["num"].named_steps.get("scaler")
    if scaler is not None:
        for i, n in enumerate(nomes):
            if n in num_names:
                escala[i] = scaler.scale_[num_names.index(n)]

    df = pd.DataFrame({
        "variavel": nomes,
        "coef_padrao": coefs,
        # OR por 1 desvio-padrão (numéricas) ou vs. categoria de referência
        "odds_ratio": np.exp(coefs),
        # OR por unidade de medida original (numéricas)
        "odds_ratio_por_unidade": np.exp(coefs / escala),
        "escala_por_unidade": escala,
    })
    df["aumenta_risco"] = df["odds_ratio"] > 1
    return df.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


def _importancia_permutacao(pipes: dict, X_test, y_test) -> pd.DataFrame:
    linhas = []
    for nome, pipe in pipes.items():
        r = permutation_importance(
            pipe, X_test, y_test, n_repeats=10, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        tmp = pd.DataFrame({
            "modelo": nome,
            "variavel": X_test.columns[np.argsort(-r.importances_mean)],
            "importancia_media": np.sort(r.importances_mean)[::-1],
            "importancia_std": r.importances_std[np.argsort(-r.importances_mean)],
        })
        linhas.append(tmp)
    return pd.concat(linhas, ignore_index=True)


# ---------------------------------------------------------------------- figuras --
def fig_curvas_roc_pr(y_test, probs: dict) -> None:
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for nome, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        eixos[0].plot(fpr, tpr, label=f"{nome} (AUC={roc_auc_score(y_test, p):.3f})")
        eixos[1].plot(rec, prec, label=f"{nome} (AP={average_precision_score(y_test, p):.3f})")
    eixos[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    eixos[0].set_title("Curva ROC")
    eixos[0].set_xlabel("Taxa de falsos positivos")
    eixos[0].set_ylabel("Taxa de verdadeiros positivos")
    eixos[0].legend(loc="lower right")
    eixos[1].set_title("Curva Precisão-Recall")
    eixos[1].set_xlabel("Recall")
    eixos[1].set_ylabel("Precisão")
    base = y_test.mean()
    eixos[1].axhline(base, ls="--", lw=0.8, color="gray", label=f"baseline={base:.2f}")
    eixos[1].legend(loc="lower left")
    fig.suptitle("Desempenho dos modelos no conjunto de teste", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_curvas_roc_pr.png")
    plt.close(fig)
    print("[model] figura -> fig10_curvas_roc_pr.png")


def fig_matriz_confusao(y_test, y_prob, limiar=0.5) -> None:
    cm = confusion_matrix(y_test, (y_prob >= limiar).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Prev. satisfeito", "Prev. insatisfeito"],
                yticklabels=["Real satisfeito", "Real insatisfeito"])
    ax.set_title("Matriz de confusão (melhor modelo, limiar 0.5)")
    ax.set_ylabel("Real")
    ax.set_xlabel("Previsto")
    fig.savefig(FIGURES / "fig11_matriz_confusao.png")
    plt.close(fig)
    print("[model] figura -> fig11_matriz_confusao.png")


def fig_calibracao(y_test, probs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, p in probs.items():
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=nome)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfeita")
    ax.set_title("Calibração: probabilidade prevista x observada")
    ax.set_xlabel("Probabilidade média prevista (nota 1-2)")
    ax.set_ylabel("Fração observada de notas 1-2")
    ax.legend()
    fig.savefig(FIGURES / "fig12_calibracao.png")
    plt.close(fig)
    print("[model] figura -> fig12_calibracao.png")


def fig_pdp(pipe: Pipeline, X_test, alvos: list[str]) -> None:
    amostra = X_test.sample(min(5000, len(X_test)), random_state=RANDOM_STATE)
    fig, eixos = plt.subplots(1, len(alvos), figsize=(5 * len(alvos), 4))
    PartialDependenceDisplay.from_estimator(
        pipe, amostra, alvos, kind="average", grid_resolution=60, ax=eixos,
    )
    fig.suptitle("Dependência parcial: efeito marginal na probabilidade prevista",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_dependencia_parcial.png")
    plt.close(fig)
    print("[model] figura -> fig13_dependencia_parcial.png")


def fig_shap(pipe: Pipeline, X_test, topo: int = 12) -> bool:
    """SHAP (TreeExplainer) para o gradient boosting — opcional."""
    try:
        import shap
    except ImportError:
        print("[model] shap não instalado -> interpretar por permutação/PDP.")
        return False
    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        X_t = pd.DataFrame(
            pre.transform(X_test),
            columns=[n.split("__", 1)[-1] for n in pre.get_feature_names_out(
                NUMERIC_FEATURES + CATEGORICAL_FEATURES)],
        )
        amostra = X_t.sample(min(3000, len(X_t)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(amostra)
        if isinstance(sv, list):
            sv = sv[1]
        # bar (média |SHAP|)
        media = np.abs(sv).mean(axis=0)
        top = np.argsort(-media)[:topo]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(amostra.columns[top][::-1], media[top][::-1], color="#4c72b0")
        ax.set_title("Importância média |SHAP| (top variáveis)")
        ax.set_xlabel("Média |SHAP|")
        fig.savefig(FIGURES / "fig14_shap_importancia.png")
        plt.close(fig)
        # beeswarm
        shap.summary_plot(sv, amostra, max_display=topo, show=False, plot_size=(8, 5.5))
        plt.gcf().savefig(FIGURES / "fig15_shap_beeswarm.png")
        plt.close("all")
        print("[model] figuras -> fig14_shap_importancia.png, fig15_shap_beeswarm.png")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[model] SHAP indisponível ({type(exc).__name__}: {exc})")
        return False


# ------------------------------------------------------------------ execução --
def executar_etapa_modelo() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes do modelo.")
    df = carregar_analise()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)
    print(
        f"[model] {len(X):,} amostras | positivos (nota 1-2): {y.mean():.1%} "
        f"| {X.shape[1]} features brutas"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

    modelos = _modelos()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    resultados: dict[str, dict] = {}

    for nome, pipe in modelos.items():
        print(f"[model] validação cruzada ({CV_FOLDS}x) - {nome}...")
        cvres = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"],
            n_jobs=-1,
        )
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        m = _metricas(y_test, prob)
        m["cv_roc_auc_mean"] = round(float(cvres["test_roc_auc"].mean()), 4)
        m["cv_roc_auc_std"] = round(float(cvres["test_roc_auc"].std()), 4)
        m["cv_pr_auc_mean"] = round(float(cvres["test_average_precision"].mean()), 4)
        resultados[nome] = {"metricas": m, "prob_test": prob}
        print(
            f"[model] {nome}: AUC test={m['roc_auc']} | PR-AUC={m['pr_auc']} | "
            f"recall={m['recall']} | precisão={m['precisao']}"
        )

    probs = {n: r["prob_test"] for n, r in resultados.items()}
    melhor = max(resultados, key=lambda n: resultados[n]["metricas"]["roc_auc"])
    print(f"[model] melhor modelo por AUC: {melhor}")

    # ---------------- figuras de avaliação ----------------
    fig_curvas_roc_pr(y_test, probs)
    fig_matriz_confusao(y_test, probs[melhor])
    fig_calibracao(y_test, probs)

    # ---------------- interpretação ----------------
    pipe_lr = modelos["regressao_logistica"]
    pipe_hgb = modelos["gradient_boosting"]

    odds = _odds_ratios(pipe_lr)
    odds.to_csv(TABLES / "tab07_odds_ratios_logistica.csv", index=False)
    print(f"[model] tabela -> tab07_odds_ratios_logistica.csv ({len(odds)} variáveis)")

    imp = _importancia_permutacao(modelos, X_test, y_test)
    imp.to_csv(TABLES / "tab08_importancia_permutacao.csv", index=False)
    print("[model] tabela -> tab08_importancia_permutacao.csv")

    tab_metricas = pd.DataFrame(
        [dict(modelo=n, **r["metricas"]) for n, r in resultados.items()]
    )
    # matriz de confusao não cabe bem em CSV tabular -> string
    tab_metricas["matriz_confusao"] = tab_metricas["matriz_confusao"].apply(str)
    tab_metricas.to_csv(TABLES / "tab09_metricas_modelos.csv", index=False)
    print("[model] tabela -> tab09_metricas_modelos.csv")

    fig_pdp(pipe_hgb, X_test, ["atraso_dias", "frete_ratio", "distancia_km"])
    shap_ok = fig_shap(pipe_hgb, X_test)

    # ---------------- artefatos ----------------
    joblib.dump(modelos[melhor], MODEL_JOBLIB)
    print(f"[model] modelo salvo -> {MODEL_JOBLIB.name}")

    topo_odds = odds.head(15).to_dict("records")
    topo_odds_neg = odds.tail(15).to_dict("records")
    topo_imp = (
        imp[imp["modelo"] == "gradient_boosting"].head(15).to_dict("records")
    )
    payload = {
        "melhor_modelo": melhor,
        "n_amostras": int(len(X)),
        "n_features_brutas": int(X.shape[1]),
        "taxa_positivos_pct": round(float(y.mean()) * 100, 2),
        "metricas": {n: r["metricas"] for n, r in resultados.items()},
        "top_odds_ratio_aumentam_risco": topo_odds,
        "top_odds_ratio_protetores": topo_odds_neg,
        "top_importancia_permutacao": topo_imp,
        "shap_disponivel": shap_ok,
    }
    MODEL_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"[model] resultados -> {MODEL_JSON.name}")
    print("[model] etapa concluída.")
