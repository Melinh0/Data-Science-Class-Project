from __future__ import annotations
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from .config import (
    ANALYSIS_CSV, CATEGORICAL_FEATURES, CV_FOLDS, FIGURES, GRUPOS_FEATURES, MIN_N_RANKING,
    MODEL_JOBLIB, MODEL_JSON, NUMERIC_FEATURES, PRE_ENTREGA_EXCLUIR, RANDOM_STATE, TABLES,
    rotulo,
)
from .features import carregar_analise
from .preprocess import (
    dividir, dividir_temporal, features_do_modelo, nomes_saida, pipeline_preprocess,
)
warnings.filterwarnings("ignore")
def _preprocessador(escalar: bool, numericas: list[str] | None = None):
    return pipeline_preprocess(escalar_numerico=escalar, numericas=numericas)
def _modelos(numericas: list[str] | None = None) -> dict[str, Pipeline]:
    return {
        "regressao_logistica": Pipeline([
            ("pre", _preprocessador(True, numericas)),
            ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "gradient_boosting": Pipeline([
            ("pre", _preprocessador(False, numericas)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=300, max_leaf_nodes=31, random_state=RANDOM_STATE,
            )),
        ]),
    }
def _cv_oof(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> tuple[np.ndarray, list, list]:
    oof = np.zeros(len(y))
    aucs, aps = [], []
    for tr, va in cv.split(X, y):
        m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
    return oof, aucs, aps
def _limiar_f1(y_true, prob) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])
def _ece(y_true, prob, bins: int = 10) -> float:
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(prob)})
    d["b"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / g["n"].sum())
def _metricas(y_true, y_prob, limiar: float) -> dict:
    y_pred = (y_prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    taxa = float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    brier_ref = taxa * (1 - taxa)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(brier, 4),
        "brier_baseline": round(brier_ref, 4),
        "brier_skill": round(1 - brier / brier_ref, 4),
        "ece": round(_ece(y_true, y_prob), 4),
        "prob_media_prevista": round(float(np.mean(y_prob)), 4),
        "taxa_observada": round(taxa, 4),
        "precisao": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "acuracia": round(float((y_pred == y_true).mean()), 4),
        "matriz_confusao": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "limiar": round(float(limiar), 4),
    }
def _odds_ratios(pipe: Pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    nomes = nomes_saida(pre)
    coefs = clf.coef_.ravel()
    n_num = len(NUMERIC_FEATURES)
    escala = np.ones(len(nomes))
    escala[:n_num] = pre.named_transformers_["num"].named_steps["scaler"].scale_
    Xt = pre.transform(X_train)
    n_treino = np.full(len(nomes), np.nan)
    n_treino[n_num:] = Xt[:, n_num:].sum(axis=0)
    df = pd.DataFrame({
        "variavel": nomes,
        "coef_padrao": coefs,
        "odds_ratio": np.exp(coefs),
        "odds_ratio_por_unidade": np.exp(coefs / escala),
        "escala_por_unidade": escala,
        "n_treino": n_treino,
    })
    df["aumenta_risco"] = df["odds_ratio"] > 1
    return df.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
def _importancia_permutacao(pipes: dict, X_test, y_test) -> pd.DataFrame:
    linhas = []
    for nome, pipe in pipes.items():
        r = permutation_importance(
            pipe, X_test, y_test, n_repeats=10, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=1,
        )
        ordem = np.argsort(-r.importances_mean)
        linhas.append(pd.DataFrame({
            "modelo": nome,
            "variavel": X_test.columns[ordem],
            "importancia_media": r.importances_mean[ordem],
            "importancia_std": r.importances_std[ordem],
        }))
    return pd.concat(linhas, ignore_index=True)
def _importancia_grupos(pipes: dict, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    linhas = []
    for nome, pipe in pipes.items():
        base = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        for grupo, cols in GRUPOS_FEATURES.items():
            cols = [c for c in cols if c in X_test.columns]
            if not cols:
                continue
            rng = np.random.default_rng(RANDOM_STATE)
            quedas = []
            for _ in range(n_repeats):
                perm = rng.permutation(len(X_test))
                Xp = X_test.copy()
                for c in cols:
                    Xp[c] = X_test[c].to_numpy()[perm]
                quedas.append(base - roc_auc_score(y_test, pipe.predict_proba(Xp)[:, 1]))
            linhas.append({
                "modelo": nome, "grupo": grupo, "features": ", ".join(cols),
                "queda_auc_media": float(np.mean(quedas)), "queda_auc_dp": float(np.std(quedas)),
            })
    return pd.DataFrame(linhas).sort_values(
        ["modelo", "queda_auc_media"], ascending=[True, False]
    ).reset_index(drop=True)
def fig_curvas_roc(y_test, probs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for nome, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        rotulo = nome.replace("_", " ")
        ax.plot(fpr, tpr, lw=2, label=f"{rotulo} (AUC = {roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="aleatório (AUC = 0,5)")
    ax.set_title("Curvas ROC no conjunto de teste", fontsize=13, fontweight="bold")
    ax.set_xlabel("Falso positivo (1 − especificidade)", fontsize=11)
    ax.set_ylabel("Verdadeiro positivo (sensibilidade)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig16_curvas_roc.png")
    plt.close(fig)
    print("[model] figura -> fig16_curvas_roc.png")
def fig_matriz_confusao(y_test, y_prob, limiar) -> None:
    cm = confusion_matrix(y_test, (y_prob >= limiar).astype(int))
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Prev. satisfeito", "Prev. insatisfeito"],
                yticklabels=["Real satisfeito", "Real insatisfeito"], annot_kws={"fontsize": 13})
    ax.set_title(f"Matriz de confusão (melhor modelo, limiar {limiar:.3f})", fontsize=12.5,
                 pad=10)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_xlabel("Previsto", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig17_matriz_confusao.png")
    plt.close(fig)
    print("[model] figura -> fig17_matriz_confusao.png")
def fig_calibracao(y_test, probs: dict, metricas: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for nome, p in probs.items():
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o",
                label=f"{nome.replace('_', ' ')} (AUC = {metricas[nome]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfeita")
    ax.set_title("Calibração: probabilidade prevista x observada", fontsize=13)
    ax.set_xlabel("Probabilidade média prevista (nota 1-2)", fontsize=11)
    ax.set_ylabel("Fração observada de notas 1-2", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig18_calibracao.png")
    plt.close(fig)
    print("[model] figura -> fig18_calibracao.png")
def _feature_de(coluna: str) -> str:
    for c in CATEGORICAL_FEATURES:
        if coluna.startswith(c + "_"):
            return c
    return coluna
def fig_shap(pipe: Pipeline, X_test, topo: int = 12) -> tuple[bool, pd.DataFrame | None]:
    try:
        import shap
    except ImportError:
        print("[model] shap não instalado -> interpretar por permutação.")
        return False, None
    try:
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        X_t = pd.DataFrame(pre.transform(X_test), columns=nomes_saida(pre))
        amostra = X_t.sample(min(3000, len(X_t)), random_state=RANDOM_STATE)
        sv = shap.TreeExplainer(clf).shap_values(amostra)
        if isinstance(sv, list):
            sv = sv[1]
        media = np.abs(sv).mean(axis=0)
        por_feature = (
            pd.Series(media, index=amostra.columns).groupby(_feature_de).sum()
            .sort_values(ascending=False).rename("shap_medio_abs").reset_index()
            .rename(columns={"index": "feature"})
        )
        por_feature.columns = ["feature", "shap_medio_abs"]
        por_feature.to_csv(TABLES / "tab21_shap_por_feature.csv", index=False)
        topo_dados = por_feature.head(topo).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh([rotulo(f) for f in topo_dados["feature"]], topo_dados["shap_medio_abs"],
                color="#4c72b0")
        for i, v in enumerate(topo_dados["shap_medio_abs"]):
            ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
        ax.set_title("Importância média |SHAP| por variável (modelo final)", fontsize=13)
        ax.set_xlabel("Média |SHAP| (contribuição média na previsão)", fontsize=11)
        ax.set_xlim(0, topo_dados["shap_medio_abs"].max() * 1.18)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig15_shap_importancia.png")
        plt.close(fig)
        print("[model] figura -> fig15_shap_importancia.png")
        return True, por_feature
    except Exception as exc:
        print(f"[model] SHAP indisponível ({type(exc).__name__}: {exc})")
        return False, None
def _validacoes_adicionais(df: pd.DataFrame, melhor: str, limiar: float,
                           X_train, X_test, y_train, y_test) -> dict:
    saida: dict = {}
    X_tr, X_te, y_tr, y_te = dividir_temporal(df)
    m = clone(_modelos()[melhor]).fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    saida["temporal"] = {**_metricas(y_te, p, limiar), "n_treino": len(X_tr), "n_teste": len(X_te)}
    print(f"[model] validação temporal ({melhor}): AUC={saida['temporal']['roc_auc']}")
    num_pre = [c for c in NUMERIC_FEATURES if c not in PRE_ENTREGA_EXCLUIR]
    mp = clone(_modelos(num_pre)[melhor])
    mp.fit(X_train.drop(columns=PRE_ENTREGA_EXCLUIR), y_train)
    X_te_pre = X_test.drop(columns=PRE_ENTREGA_EXCLUIR)
    pp = mp.predict_proba(X_te_pre)[:, 1]
    saida["pre_entrega"] = {**_metricas(y_test, pp, limiar), "features_excluidas": PRE_ENTREGA_EXCLUIR}
    print(f"[model] modelo pré-entrega ({melhor}): AUC={saida['pre_entrega']['roc_auc']}")
    return saida
def executar_etapa_modelo() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes do modelo.")
    df = carregar_analise()
    X_train, X_test, y_train, y_test = dividir(df)
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    print(
        f"[model] {len(X):,} amostras | positivos (nota 1-2): {y.mean():.1%} "
        f"| {X.shape[1]} features brutas | treino {len(X_train):,} / teste {len(X_test):,}"
    )
    modelos = _modelos()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    resultados: dict[str, dict] = {}
    for nome, pipe in modelos.items():
        print(f"[model] validação cruzada ({CV_FOLDS}x) - {nome}...")
        oof, aucs, aps = _cv_oof(pipe, X_train, y_train, cv)
        limiar = _limiar_f1(y_train, oof)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        m = _metricas(y_test, prob, limiar)
        m["cv_roc_auc_mean"] = round(float(np.mean(aucs)), 4)
        m["cv_roc_auc_std"] = round(float(np.std(aucs)), 4)
        m["cv_pr_auc_mean"] = round(float(np.mean(aps)), 4)
        m["cv_brier_skill"] = round(1 - brier_score_loss(y_train, oof) / (y_train.mean() * (1 - y_train.mean())), 4)
        resultados[nome] = {"metricas": m, "prob_test": prob, "limiar": limiar}
        print(
            f"[model] {nome}: AUC test={m['roc_auc']} | limiar(F1, OOF)={m['limiar']} "
            f"| acurácia={m['acuracia']} | precisão={m['precisao']} | recall={m['recall']} | F1={m['f1']}"
        )
    probs = {n: r["prob_test"] for n, r in resultados.items()}
    metricas = {n: r["metricas"] for n, r in resultados.items()}
    melhor = max(resultados, key=lambda n: resultados[n]["metricas"]["cv_roc_auc_mean"])
    print(f"[model] melhor modelo por AUC de validação cruzada (não pelo teste): {melhor}")
    fig_curvas_roc(y_test, probs)
    fig_matriz_confusao(y_test, probs[melhor], resultados[melhor]["limiar"])
    fig_calibracao(y_test, probs, metricas)
    pipe_lr = modelos["regressao_logistica"]
    pipe_hgb = modelos["gradient_boosting"]
    odds = _odds_ratios(pipe_lr, X_train)
    odds.to_csv(TABLES / "tab07_odds_ratios_logistica.csv", index=False)
    print(f"[model] tabela -> tab07_odds_ratios_logistica.csv ({len(odds)} variáveis)")
    imp = _importancia_permutacao(modelos, X_test, y_test)
    imp.to_csv(TABLES / "tab08_importancia_permutacao.csv", index=False)
    grupos = _importancia_grupos(modelos, X_test, y_test)
    grupos.to_csv(TABLES / "tab20_importancia_grupos.csv", index=False)
    print("[model] tabelas -> tab08_importancia_permutacao.csv, tab20_importancia_grupos.csv")
    tab_metricas = pd.DataFrame([dict(modelo=n, **r["metricas"]) for n, r in resultados.items()])
    tab_metricas["matriz_confusao"] = tab_metricas["matriz_confusao"].apply(str)
    fora = ["pr_auc", "brier", "brier_baseline", "brier_skill", "ece",
            "cv_pr_auc_mean", "cv_brier_skill"]
    tab_metricas = tab_metricas.drop(columns=[c for c in fora if c in tab_metricas.columns])
    tab_metricas.to_csv(TABLES / "tab09_metricas_modelos.csv", index=False)
    print("[model] tabela -> tab09_metricas_modelos.csv")
    shap_ok, shap_tab = fig_shap(pipe_hgb, X_test)
    adicionais = _validacoes_adicionais(
        df, melhor, metricas[melhor]["limiar"], X_train, X_test, y_train, y_test
    )
    joblib.dump(modelos[melhor], MODEL_JOBLIB)
    print(f"[model] modelo salvo -> {MODEL_JOBLIB.name}")
    elegiveis = odds[odds["n_treino"].isna() | (odds["n_treino"] >= MIN_N_RANKING)]
    payload = {
        "melhor_modelo": melhor,
        "n_amostras": int(len(X)),
        "n_features_brutas": int(X.shape[1]),
        "taxa_positivos_pct": round(float(y.mean()) * 100, 2),
        "criterio_melhor_modelo": "AUC médio de validação cruzada no treino",
        "metricas": metricas,
        "validacoes_adicionais": adicionais,
        "top_odds_ratio_aumentam_risco": elegiveis.head(15).to_dict("records"),
        "top_odds_ratio_protetores": elegiveis.tail(15).to_dict("records"),
        "top_importancia_permutacao": imp[imp["modelo"] == "gradient_boosting"].head(15).to_dict("records"),
        "importancia_grupos": grupos.to_dict("records"),
        "shap_por_feature": None if shap_tab is None else shap_tab.head(15).to_dict("records"),
        "shap_disponivel": shap_ok,
    }
    MODEL_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[model] resultados -> {MODEL_JSON.name}")
    print("[model] etapa concluída.")
