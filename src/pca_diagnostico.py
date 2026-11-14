from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import (
    ANALYSIS_CSV, COR_NEG, COR_POS, FIGURES, NUMERIC_FEATURES, PCA_JSON, RANDOM_STATE,
    TABLES, rotulo,
)
from .features import carregar_analise
from .preprocess import dividir, pipeline_preprocess
N_PERMUTACOES = 20
CV_PCA = 3
def _autovalores_paralelos(Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eigs = []
    for _ in range(N_PERMUTACOES):
        Zp = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        eigs.append(np.linalg.eigvalsh(np.corrcoef(Zp, rowvar=False))[::-1])
    return np.percentile(np.array(eigs), 95, axis=0)
def _auc_cv_pca(X_train: pd.DataFrame, y_train: pd.Series, k: int | None) -> tuple[float, float]:
    extra = None if k is None else ("pca", PCA(n_components=k, random_state=RANDOM_STATE))
    pipe = Pipeline([
        ("pre", pipeline_preprocess(escalar_numerico=True, etapa_extra_numerica=extra)),
        ("clf", LogisticRegression(max_iter=300)),
    ])
    cv = StratifiedKFold(CV_PCA, shuffle=True, random_state=RANDOM_STATE)
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    return float(s.mean()), float(s.std(ddof=1))
def _rotulo_componente(cargas: pd.Series, n: int = 3) -> str:
    top = cargas.reindex(cargas.abs().sort_values(ascending=False).index)[:n]
    return ", ".join(f"{rotulo(f)} ({v:+.2f})" for f, v in top.items())
def executar_pca() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes do PCA.")
    df = carregar_analise()
    X_train, _, y_train, _ = dividir(df)
    p = len(NUMERIC_FEATURES)
    print(f"[pca] ajustado SOMENTE no treino ({len(X_train):,} pedidos), {p} numéricas padronizadas")
    pre_num = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    Z = pre_num.fit_transform(X_train[NUMERIC_FEATURES])
    pca = PCA(random_state=RANDOM_STATE).fit(Z)
    eig = pca.explained_variance_
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    rng = np.random.default_rng(RANDOM_STATE)
    paralelo = _autovalores_paralelos(Z, rng)
    nomes_pc = [f"PC{i + 1}" for i in range(p)]
    k_kaiser = int((eig > 1).sum())
    k_paralela = int((eig > paralelo).sum())
    k_90 = int(np.argmax(cum >= 0.90) + 1)
    rank_efetivo = int((eig > 1e-8).sum())
    tab_var = pd.DataFrame({
        "componente": nomes_pc, "autovalor": eig, "variancia_explicada": var,
        "variancia_acumulada": cum, "autovalor_analise_paralela_p95": paralelo,
        "kaiser_maior_que_1": eig > 1, "supera_ruido_paralela": eig > paralelo,
    })
    tab_var.to_csv(TABLES / "tab13_pca_variancia.csv", index=False)
    print("[pca] validando k por desempenho preditivo (LR, CV 3 dobras)...")
    base_m, base_s = _auc_cv_pca(X_train, y_train, None)
    curva = []
    for k in range(1, p + 1):
        m, s = _auc_cv_pca(X_train, y_train, k)
        curva.append({"k": k, "auc_cv_medio": m, "auc_cv_dp": s})
        print(f"[pca]   k={k:2d}: AUC CV = {m:.4f} ± {s:.4f}")
    curva = pd.DataFrame(curva)
    curva.to_csv(TABLES / "tab19_pca_cv.csv", index=False)
    limite = base_m - base_s / np.sqrt(CV_PCA)
    k_cv = int(curva.loc[curva["auc_cv_medio"] >= limite, "k"].min())
    k_escolhido = k_cv
    n_pc_carga = max(k_escolhido, 3)
    cargas = pd.DataFrame(pca.components_[:n_pc_carga].T, index=NUMERIC_FEATURES,
                          columns=nomes_pc[:n_pc_carga])
    cargas.index.name = "variavel"
    cargas.round(4).reset_index().to_csv(TABLES / "tab18_pca_cargas.csv", index=False)
    escores = pca.transform(Z)
    y = y_train.to_numpy()
    auc_pc = {nomes_pc[i]: float(roc_auc_score(y, escores[:, i])) for i in range(min(5, p))}
    sep_pc = {k: abs(v - 0.5) + 0.5 for k, v in auc_pc.items()}
    melhor_pc = max(sep_pc, key=sep_pc.get)
    tab_nec = pd.DataFrame({
        "metricas": [
            "n_features_numericas", "rank_efetivo_autovalor>1e-8", "k_criterio_kaiser",
            "k_analise_paralela", "k_90pct_variancia_(apenas_referencia)", "k_validacao_cruzada_1se",
            "k_escolhido", "variancia_acumulada_k_escolhido", "auc_cv_sem_pca",
            "auc_cv_k_escolhido", "auc_cv_k_90pct",
        ],
        "valor": [
            p, rank_efetivo, k_kaiser, k_paralela, k_90, k_cv, k_escolhido,
            float(cum[k_escolhido - 1]), base_m,
            float(curva.loc[curva["k"] == k_escolhido, "auc_cv_medio"].iloc[0]),
            float(curva.loc[curva["k"] == k_90, "auc_cv_medio"].iloc[0]),
        ],
    })
    tab_nec.to_csv(TABLES / "tab14_pca_necessidade.csv", index=False)
    fig, eixos = plt.subplots(1, 2, figsize=(13.5, 5))
    ax1, ax2 = eixos
    xs = range(1, p + 1)
    ax1.bar(xs, eig, color="#4c72b0", label="Autovalor")
    ax1.plot(xs, paralelo, "k--", marker="x", ms=5, label="Análise paralela (p95 do ruído)")
    ax1.axhline(1, color="gray", ls=":", label="Kaiser (autovalor = 1)")
    ax1.axvline(k_escolhido + 0.5, color=COR_NEG, lw=1.5, label=f"k escolhido = {k_escolhido}")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Autovalor")
    ax1.set_title("Autovalores: onde o sinal cai para o ruído", fontsize=13)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.set_xlim(0.3, p + 0.7)
    ax2.plot(xs, cum, color="#b2182b", marker="o", ms=4, label="Variância acumulada")
    ax2.axhline(0.90, color="gray", ls=":", label="90% da variância")
    ax2.axvline(k_escolhido, color=COR_NEG, lw=1.5, label=f"k escolhido = {k_escolhido}")
    ax2.axvline(k_90, color="orange", ls="--", label=f"k para 90% = {k_90}")
    ax2.scatter([k_escolhido], [cum[k_escolhido - 1]], s=70, color=COR_NEG, zorder=5)
    ax2.annotate(f"k = {k_escolhido}\n{cum[k_escolhido - 1] * 100:.1f}% da var.",
                 (k_escolhido, cum[k_escolhido - 1]), textcoords="offset points",
                 xytext=(12, -22), fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax2.set_xlabel("Número de componentes (k)")
    ax2.set_ylabel("Variância acumulada explicada")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Variância acumulada retida pelo PCA", fontsize=13)
    ax2.legend(fontsize=8.5, loc="center right")
    ax2.set_xlim(0.3, p + 0.7)
    fig.suptitle(f"Diagnóstico do PCA ({p} variáveis numéricas padronizadas, só treino)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "fig12_pca_scree.png")
    plt.close(fig)
    amostra = np.random.default_rng(RANDOM_STATE).choice(len(y), size=min(8000, len(y)), replace=False)
    top_carga = (
        cargas.iloc[:, :2].abs().sum(axis=1).sort_values(ascending=False).index[:6]
    )
    fig, eixos = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = eixos[0]
    for cls, cor, rot in ((0, COR_POS, "satisfeito"), (1, COR_NEG, "insatisfeito")):
        m = amostra[y[amostra] == cls]
        ax.scatter(escores[m, 0], escores[m, 1], s=6, alpha=0.3, color=cor, label=rot)
    esc = np.percentile(np.abs(escores[:, :2]), 99)
    for i, f in enumerate(top_carga):
        j = list(NUMERIC_FEATURES).index(f)
        cx, cy = cargas.iloc[j, 0] * esc * 0.9, cargas.iloc[j, 1] * esc * 0.9
        ax.arrow(0, 0, cx, cy, color="black", alpha=0.7, head_width=0.12, lw=1.2)
        mult = 1.12 + 0.22 * (i % 2)
        ax.text(cx * mult, cy * mult, rotulo(f), fontsize=8.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.4, alpha=0.95))
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlim(-esc, esc)
    ax.set_ylim(-esc, esc)
    ax.set_xlabel(f"Componente 1 ({var[0] * 100:.1f}% da variância)")
    ax.set_ylabel(f"Componente 2 ({var[1] * 100:.1f}%)")
    ax.set_title("Projeção dos pedidos no plano CP1 × CP2", fontsize=13)
    ax.legend(loc="upper right", markerscale=3, fontsize=9)
    ax = eixos[1]
    larg = 0.38
    idx = np.arange(len(top_carga))
    v1 = [cargas.loc[f, "PC1"] for f in top_carga]
    v2 = [cargas.loc[f, "PC2"] for f in top_carga]
    ax.barh(idx + larg / 2, v1, larg, color="#4c72b0", label="Carga no CP1")
    ax.barh(idx - larg / 2, v2, larg, color="#dd8452", label="Carga no CP2")
    ax.set_yticks(idx)
    ax.set_yticklabels([rotulo(f) for f in top_carga], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Carga (correlação variável × componente)")
    ax.set_title("Cargas das 6 variáveis que mais definem o CP1 e o CP2", fontsize=13)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False)
    ax.invert_yaxis()
    fig.suptitle("PCA: quem separa os pedidos no espaço reduzido (treino)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIGURES / "fig14_pca_cp1_cp2.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.fill_between(curva["k"], curva["auc_cv_medio"] - curva["auc_cv_dp"],
                    curva["auc_cv_medio"] + curva["auc_cv_dp"], alpha=0.2, color="#4c72b0",
                    label="± 1 erro-padrão")
    ax.plot(curva["k"], curva["auc_cv_medio"], marker="o", ms=4, color="#4c72b0",
            label="PCA(k) + categóricas, regressão logística")
    ax.axhline(base_m, color="gray", ls="--", label=f"sem PCA ({base_m:.4f})")
    ax.axhline(limite, color="gray", ls=":", label="limite 1-SE: sem PCA − 1 erro-padrão")
    ax.axvline(k_escolhido, color=COR_NEG, label=f"k escolhido = {k_escolhido}")
    ax.annotate(f"k = {k_escolhido}", (k_escolhido, curva.loc[curva["k"] == k_escolhido,
                 "auc_cv_medio"].iloc[0]), textcoords="offset points", xytext=(-38, -26),
                fontsize=9, color=COR_NEG,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COR_NEG))
    ax.set_xlabel("Número de componentes (k)")
    ax.set_ylabel("AUC (validação cruzada no treino)")
    ax.set_title("Informação preditiva preservada pelo PCA", fontsize=13)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig13_pca_auc_vs_k.png")
    plt.close(fig)
    PCA_JSON.write_text(json.dumps({
        "n_numericas": p, "rank_efetivo": rank_efetivo, "k_kaiser": k_kaiser,
        "k_paralela": k_paralela, "k_90pct": k_90, "k_cv_1se": k_cv, "k_escolhido": k_escolhido,
        "variancia_k_escolhido": float(cum[k_escolhido - 1]),
        "variancia_k_90": float(cum[k_90 - 1]),
        "auc_cv_sem_pca": base_m, "auc_cv_sem_pca_dp": base_s,
        "auc_cv_k_escolhido": float(curva.loc[curva["k"] == k_escolhido, "auc_cv_medio"].iloc[0]),
        "auc_cv_k_90": float(curva.loc[curva["k"] == k_90, "auc_cv_medio"].iloc[0]),
        "rotulos_componentes": {nomes_pc[i]: _rotulo_componente(cargas[nomes_pc[i]])
                                for i in range(min(5, n_pc_carga))},
        "variancia_por_componente": {nomes_pc[i]: float(var[i]) for i in range(min(5, p))},
        "auc_componentes_vs_alvo": auc_pc, "componente_mais_separadora": melhor_pc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pca] Kaiser: {k_kaiser} | análise paralela: {k_paralela} | 90% da variância: {k_90} | "
          f"validação cruzada (1 EP): {k_cv}")
    print(f"[pca] k escolhido = {k_escolhido} "
          f"({cum[k_escolhido - 1] * 100:.1f}% da variância; AUC CV {curva.loc[curva['k'] == k_escolhido, 'auc_cv_medio'].iloc[0]:.4f} "
          f"vs. {base_m:.4f} sem PCA)")
    print("[pca] etapa concluída.")
