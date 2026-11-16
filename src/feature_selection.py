from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import (
    ANALYSIS_CSV, CATEGORICAL_FEATURES, COR_NEG, FIGURES, GRUPOS_FEATURES, NUMERIC_FEATURES,
    RANDOM_STATE, SCORE_COL, SELECAO_JSON, TABLES, rotulo,
)
from .features import carregar_analise
from .preprocess import MIN_FREQ_OHE, dividir
ALPHA = 0.05
N_AMOSTRA = 40000
GANHO_MIN_WRAPPER = 0.002
MIN_METODOS = 3
METODOS = ["filtro_mi", "filtro_auc", "wrapper_forward", "embutido_l1", "embutido_rf"]
def _salvar_fig(fig, nome: str) -> None:
    fig.savefig(FIGURES / nome)
    plt.close(fig)
    print(f"[select] figura -> {nome}")
def _grupo_de(feature: str) -> str:
    for g, cols in GRUPOS_FEATURES.items():
        if feature in cols:
            return g
    return "Outros"
def _v_cramer_corrigido(a: pd.Series, y: pd.Series) -> float:
    tabela = pd.crosstab(a, y)
    chi2 = stats.chi2_contingency(tabela, correction=False)[0]
    n = tabela.to_numpy().sum()
    r, k = tabela.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    kc, rc = k - (k - 1) ** 2 / (n - 1), r - (r - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(min(kc - 1, rc - 1), 1e-9)))
def _com_sombras(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sombras = {f"sombra_{c}": X[c].to_numpy()[rng.permutation(len(X))] for c in X.columns}
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(sombras)], axis=1)
def _matriz_ordinal(X: pd.DataFrame, categoricas: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    M = pd.DataFrame(index=X.index)
    for c in X.columns:
        base = c.replace("sombra_", "")
        if base in categoricas:
            M[c] = pd.factorize(X[c].astype("object").where(X[c].notna(), "ausente"))[0]
        else:
            M[c] = X[c].astype(float)
    mascara = np.array([c.replace("sombra_", "") in categoricas for c in M.columns])
    return M, mascara
def _bloco(serie: pd.Series, numerica: bool) -> np.ndarray:
    if numerica:
        pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    else:
        pipe = Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=MIN_FREQ_OHE,
                                sparse_output=False)),
        ])
    return pipe.fit_transform(serie.astype("object").to_frame() if not numerica
                              else serie.astype(float).to_frame())
def _blocos(X: pd.DataFrame, categoricas: list[str]) -> tuple[np.ndarray, dict[str, slice]]:
    partes, grupos, ini = [], {}, 0
    for c in X.columns:
        m = _bloco(X[c], c.replace("sombra_", "") not in categoricas)
        partes.append(m)
        grupos[c] = slice(ini, ini + m.shape[1])
        ini += m.shape[1]
    return np.hstack(partes), grupos
def _filtro_mi(Xs: pd.DataFrame, y: np.ndarray, categoricas: list[str]) -> pd.Series:
    M, discreta = _matriz_ordinal(Xs, categoricas)
    M = M.fillna(M.median(numeric_only=True))
    mi = mutual_info_classif(M, y, discrete_features=discreta, random_state=RANDOM_STATE)
    return pd.Series(mi, index=Xs.columns)
def _filtro_auc(Xs: pd.DataFrame, y: np.ndarray, categoricas: list[str]) -> pd.Series:
    M, discreta = _matriz_ordinal(Xs, categoricas)
    cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    saida = {}
    for c, cat in zip(M.columns, discreta):
        clf = HistGradientBoostingClassifier(
            max_iter=60, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE,
            categorical_features=[bool(cat)], early_stopping=False,
        )
        saida[c] = float(cross_val_score(clf, M[[c]].to_numpy(), y, cv=cv, scoring="roc_auc").mean())
    return pd.Series(saida)
def _wrapper_forward(M: np.ndarray, grupos: dict[str, slice], y: np.ndarray) -> tuple[list[dict], int]:
    idx_tr, idx_va = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=RANDOM_STATE,
    )
    restantes, escolhidas, historico, n_fits = list(grupos), [], [], 0
    auc_atual = 0.5
    while restantes:
        melhor, auc_melhor = None, -1.0
        for c in restantes:
            cols = np.r_[tuple(np.arange(M.shape[1])[grupos[f]] for f in escolhidas + [c])]
            clf = LogisticRegression(max_iter=300, solver="lbfgs")
            clf.fit(M[idx_tr][:, cols], y[idx_tr])
            n_fits += 1
            auc = roc_auc_score(y[idx_va], clf.predict_proba(M[idx_va][:, cols])[:, 1])
            if auc > auc_melhor:
                melhor, auc_melhor = c, auc
        ganho = auc_melhor - auc_atual
        if ganho < GANHO_MIN_WRAPPER:
            break
        escolhidas.append(melhor)
        restantes.remove(melhor)
        historico.append({"ordem": len(escolhidas), "feature": melhor,
                          "auc_validacao": auc_melhor, "ganho": ganho})
        auc_atual = auc_melhor
        print(f"[select]   wrapper passo {len(escolhidas)}: +{melhor} "
              f"(AUC val. {auc_melhor:.4f}, ganho {ganho:+.4f})")
    return historico, n_fits
def _embutido_l1(M: np.ndarray, grupos: dict[str, slice], y: np.ndarray) -> tuple[pd.Series, float]:
    cs = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    cv = LogisticRegressionCV(
        Cs=cs, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=300,
        random_state=RANDOM_STATE, refit=False,
    ).fit(M, y)
    scores = cv.scores_[1]
    media, ep = scores.mean(axis=0), scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    melhor = int(np.argmax(media))
    elegiveis = np.where(media >= media[melhor] - ep[melhor])[0]
    c_escolhido = float(cs[elegiveis.min()])
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=c_escolhido, max_iter=300).fit(M, y)
    coef = np.abs(clf.coef_.ravel())
    return pd.Series({f: float(coef[s].max()) for f, s in grupos.items()}), c_escolhido
def _embutido_rf(M: np.ndarray, grupos: dict[str, slice], y: np.ndarray) -> pd.Series:
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=50, n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(M, y)
    imp = rf.feature_importances_
    return pd.Series({f: float(imp[s].sum()) for f, s in grupos.items()})
ROTULO_METODO_CURTO = {
    "filtro_mi": "Filtro:\ninformação\nmútua",
    "filtro_auc": "Filtro:\nAUC\nunivariada",
    "wrapper_forward": "Wrapper:\nforward\n(log. linear)",
    "embutido_l1": "Embutido:\nreg. log.\nL1",
    "embutido_rf": "Embutido:\nrandom\nforest",
}
def _fig_ranking_spearman(tab: pd.DataFrame) -> None:
    d = tab.dropna(subset=["spearman_rho"]).assign(a=lambda t: t["spearman_rho"].abs()).sort_values("a")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    barras = ax.barh([rotulo(f) for f in d["feature"]], d["a"],
                     color=[COR_NEG if p < ALPHA else "gray" for p in d["spearman_p"]])
    ax.bar_label(barras, labels=[f"{r:+.3f}" for r in d["spearman_rho"]], padding=3, fontsize=8)
    ax.set_title("Correlação de Spearman de cada variável numérica com a nota (treino)",
                 fontsize=13)
    ax.set_xlabel("|ρ| de Spearman com a nota", fontsize=11)
    ax.set_xlim(0, d["a"].max() * 1.25)
    fig.tight_layout()
    _salvar_fig(fig, "fig08_ranking_spearman.png")
def _fig_mi(tab: pd.DataFrame, limiar: float) -> None:
    d = tab.sort_values("mi_insatisfeito")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    cores = ["#4c72b0" if t == "numérica" else "#dd8452" for t in d["tipo"]]
    ax.barh([rotulo(f) for f in d["feature"]], d["mi_insatisfeito"], color=cores)
    ax.axvline(limiar, color="red", ls="--", lw=1.2,
               label=f"maior MI de variável-sombra = {limiar:.4f}")
    ax.set_title("Informação mútua de cada variável com a insatisfação (treino)", fontsize=13)
    ax.set_xlabel("Informação mútua (nats)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _salvar_fig(fig, "fig09_mutual_information.png")
def _fig_consenso(tab: pd.DataFrame) -> None:
    d = tab.sort_values(["n_metodos", "auc_univariada"], ascending=[True, True])
    mat = d[[f"sel_{m}" for m in METODOS]].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1.4)
    ax.set_xticks(range(len(METODOS)))
    ax.set_xticklabels([ROTULO_METODO_CURTO[m] for m in METODOS], fontsize=8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{rotulo(f)}  ({k}/5)" for f, k in
                        zip(d["feature"], d["n_metodos"])], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "x" if mat[i, j] else "", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
    ax.set_title(f"Convergência dos cinco métodos de seleção\n"
                 f"(selecionada se ≥ {MIN_METODOS} de 5)", fontsize=13, pad=12)
    ax.grid(False)
    fig.tight_layout()
    _salvar_fig(fig, "fig10_consenso_selecao.png")
def executar_etapa_selecao() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes da seleção de features.")
    df = carregar_analise()
    X_train, _, y_train, _ = dividir(df)
    nota_train = df.loc[X_train.index, SCORE_COL]
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    print(f"[select] seleção usa SOMENTE o treino: {len(X_train):,} pedidos, {len(features)} features")
    amostra = X_train.sample(n=min(N_AMOSTRA, len(X_train)), random_state=RANDOM_STATE)
    y_amostra = y_train.loc[amostra.index].to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    Xs = _com_sombras(amostra, rng)
    t0 = time.time()
    mi = _filtro_mi(Xs, y_amostra, CATEGORICAL_FEATURES)
    print(f"[select] filtro MI: {time.time() - t0:.0f}s")
    t0 = time.time()
    auc_uni = _filtro_auc(Xs, y_amostra, CATEGORICAL_FEATURES)
    print(f"[select] filtro AUC univariada: {time.time() - t0:.0f}s")
    sombras = [c for c in Xs.columns if c.startswith("sombra_")]
    lim_mi, lim_auc = float(mi[sombras].max()), float(auc_uni[sombras].max())
    M_orig, grupos_orig = _blocos(amostra.reset_index(drop=True), CATEGORICAL_FEATURES)
    t0 = time.time()
    historico, n_fits = _wrapper_forward(M_orig, grupos_orig, y_amostra)
    t_wrapper = time.time() - t0
    print(f"[select] wrapper: {n_fits} ajustes em {t_wrapper:.0f}s")
    t0 = time.time()
    l1_coef, c_l1 = _embutido_l1(M_orig, grupos_orig, y_amostra)
    print(f"[select] embutido L1: C={c_l1} em {time.time() - t0:.0f}s")
    M_sh, grupos_sh = _blocos(Xs, CATEGORICAL_FEATURES)
    rf_imp = _embutido_rf(M_sh, grupos_sh, y_amostra)
    lim_rf = float(rf_imp[sombras].max())
    wrapper_ordem = {h["feature"]: h for h in historico}
    linhas = []
    for f in features:
        num = f in NUMERIC_FEATURES
        a = X_train[f]
        linha = {
            "feature": f, "tipo": "numérica" if num else "categórica", "grupo": _grupo_de(f),
            "spearman_rho": np.nan, "spearman_p": np.nan, "rank_biserial": np.nan,
            "cramers_v_corrigido": np.nan,
        }
        if num:
            ok = a.notna()
            rho, p = stats.spearmanr(a[ok], nota_train[ok])
            u = stats.mannwhitneyu(a[ok & (y_train == 1)], a[ok & (y_train == 0)])[0]
            linha.update(spearman_rho=float(rho), spearman_p=float(p),
                         rank_biserial=float(2 * u / ((ok & (y_train == 1)).sum() * (ok & (y_train == 0)).sum()) - 1))
        else:
            linha["cramers_v_corrigido"] = _v_cramer_corrigido(a.fillna("ausente"), y_train)
        linha.update(
            mi_insatisfeito=float(mi[f]), mi_sombra_max=lim_mi,
            auc_univariada=float(auc_uni[f]), auc_sombra_max=lim_auc,
            wrapper_ordem=wrapper_ordem[f]["ordem"] if f in wrapper_ordem else np.nan,
            wrapper_ganho_auc=wrapper_ordem[f]["ganho"] if f in wrapper_ordem else np.nan,
            l1_coef_abs_max=float(l1_coef[f]),
            rf_importancia=float(rf_imp[f]), rf_sombra_max=lim_rf,
            sel_filtro_mi=bool(mi[f] > lim_mi),
            sel_filtro_auc=bool(auc_uni[f] > lim_auc),
            sel_wrapper_forward=f in wrapper_ordem,
            sel_embutido_l1=bool(l1_coef[f] > 0),
            sel_embutido_rf=bool(rf_imp[f] > lim_rf),
        )
        linhas.append(linha)
    tab = pd.DataFrame(linhas)
    tab["n_metodos"] = tab[[f"sel_{m}" for m in METODOS]].sum(axis=1)
    tab["selecionada"] = tab["n_metodos"] >= MIN_METODOS
    tab = tab.sort_values(["n_metodos", "auc_univariada"], ascending=[False, False]).reset_index(drop=True)
    tab.to_csv(TABLES / "tab12_ranking_features.csv", index=False)
    print("[select] tabela -> tab12_ranking_features.csv")
    selecionadas = tab.loc[tab["selecionada"], "feature"].tolist()
    SELECAO_JSON.write_text(json.dumps({
        "selecionadas": selecionadas,
        "descartadas": tab.loc[~tab["selecionada"], "feature"].tolist(),
        "criterio": f">= {MIN_METODOS} de {len(METODOS)} métodos",
        "por_metodo": {m: tab.loc[tab[f"sel_{m}"], "feature"].tolist() for m in METODOS},
        "limiares": {"mi_sombra_max": lim_mi, "auc_sombra_max": lim_auc, "rf_sombra_max": lim_rf,
                     "c_l1": c_l1, "ganho_min_wrapper": GANHO_MIN_WRAPPER},
        "custo_wrapper": {"n_ajustes": n_fits, "segundos": round(t_wrapper, 1)},
        "n_amostra": int(len(amostra)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _fig_ranking_spearman(tab)
    _fig_mi(tab, lim_mi)
    _fig_consenso(tab)
    print(f"[select] consenso (>= {MIN_METODOS}/5): {len(selecionadas)} de {len(features)} features")
    print(f"[select]   mantidas:   {selecionadas}")
    print(f"[select]   descartadas: {tab.loc[~tab['selecionada'], 'feature'].tolist()}")
    print("[select] etapa concluída.")
