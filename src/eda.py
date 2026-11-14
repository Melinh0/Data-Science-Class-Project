from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import (
    ANALYSIS_CSV, COR_NEG, COR_POS, FIGURES, MIN_N_RANKING, PALETTE_NOTAS, SCORE_COL,
    TABLES, TARGET,
)
from .features import carregar_analise
MIN_N_MES = 500
def _wilson(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    p = k / n
    centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centro - meia, centro + meia
MIN_N_MES = 500
def _wilson(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    p = k / n
    centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    meia = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centro - meia, centro + meia
def _salvar(fig, nome: str) -> None:
    caminho = FIGURES / nome
    fig.savefig(caminho)
    plt.close(fig)
    print(f"[eda] figura -> {caminho.name}")
def _salvar_tabela(df: pd.DataFrame, nome: str) -> None:
    caminho = TABLES / nome
    df.to_csv(caminho, index=False)
    print(f"[eda] tabela -> {caminho.name}")
def fig_distribuicao_notas(analise: pd.DataFrame) -> None:
    contagem = analise[SCORE_COL].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    pct = contagem / len(analise) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    barras = ax.bar(
        [str(int(s)) for s in contagem.index], contagem.values,
        color=[PALETTE_NOTAS[s] for s in contagem.index],
    )
    ax.bar_label(barras, labels=[f"{p:.1f}%" for p in pct], padding=3, fontsize=12)
    ax.set_title("Distribuição das notas de avaliação (1 a 5)", fontsize=16)
    ax.set_xlabel("Nota da avaliação", fontsize=14)
    ax.set_ylabel("Nº de pedidos", fontsize=14)
    ax.tick_params(labelsize=13)
    taxa_neg = analise[TARGET].mean() * 100
    ax.axhline(0, color="gray", lw=0.8)
    ax.text(
        0.98, 0.92,
        f"Notas 1-2 (insatisfeitos): {taxa_neg:.1f}%\n"
        f"Média = {analise[SCORE_COL].mean():.2f} | Mediana = {analise[SCORE_COL].median():.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )
    _salvar(fig, "fig01_distribuicao_notas.png")
def fig_evolucao_temporal(analise: pd.DataFrame) -> None:
    serie = (
        analise.groupby("mes_compra")
        .agg(n=(TARGET, "size"), nota_media=(SCORE_COL, "mean"),
             pct_insatisfeito=(TARGET, "mean"))
        .reset_index()
    )
    serie = serie[serie["n"] >= MIN_N_MES]
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(serie["mes_compra"], serie["nota_media"], marker="o", color="#2166ac",
             label="Nota média")
    ax1.set_ylabel("Nota média")
    ax1.set_ylim(1, 5)
    ax1.tick_params(axis="x", rotation=60)
    ax2 = ax1.twinx()
    ax2.bar(serie["mes_compra"], serie["pct_insatisfeito"] * 100, alpha=0.25,
            color=COR_NEG, label="% insatisfeitos")
    ax2.set_ylabel("% de avaliações 1-2")
    ax2.set_ylim(0, None)
    for eixo in (ax1, ax2):
        for rotulo_tick in eixo.get_xticklabels():
            rotulo_tick.set_horizontalalignment("right")
    ax1.set_title(f"Evolução mensal: nota média e % de avaliações negativas (meses com n ≥ {MIN_N_MES})")
    linhas, rotulos = [], []
    for ax in (ax1, ax2):
        l, r = ax.get_legend_handles_labels()
        linhas += l
        rotulos += r
    ax1.legend(linhas, rotulos, loc="lower left")
    _salvar(fig, "fig02_evolucao_temporal.png")
def fig_atraso_por_nota(analise: pd.DataFrame) -> None:
    dados = analise.copy()
    dados["atraso_rec"] = dados["atraso_dias"].clip(-45, 60)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=dados, x=SCORE_COL, y="atraso_rec", hue=SCORE_COL, palette=PALETTE_NOTAS,
        legend=False, showfliers=False, ax=ax,
    )
    medias = dados.groupby(SCORE_COL)["atraso_rec"].median()
    for i, nota in enumerate(medias.index):
        ax.text(i, medias[nota], f"med={medias[nota]:.1f}d", ha="center", va="bottom",
                fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.axhline(0, color=COR_NEG, ls="--", lw=1.2, label="Prazo estimado (0 dia)")
    ax.set_title("Atraso na entrega por nota de avaliação", fontsize=14)
    ax.set_xlabel("Nota da avaliação", fontsize=12)
    ax.set_ylabel("Atraso em relação ao prazo (dias, recortado em [-45, 60])", fontsize=11)
    ax.legend(loc="lower left")
    _salvar(fig, "fig03_atraso_por_nota.png")
def fig_inssatisfacao_por_faixa_atraso(analise: pd.DataFrame) -> None:
    ordem = ["Adiantado > 7 dias", "No prazo (até 7d antes)",
             "Atraso 1-7 dias", "Atraso > 7 dias"]
    dados = (
        analise.dropna(subset=["faixa_atraso"])
        .groupby("faixa_atraso", observed=True)
        .agg(n=(TARGET, "size"), pct=(TARGET, "mean"))
        .reindex(ordem)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cores = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]
    barras = ax.bar(dados["faixa_atraso"], dados["pct"] * 100, color=cores)
    ax.bar_label(
        barras,
        labels=[f"{p:.1f}%\n(n={n:,})" for p, n in zip(dados["pct"] * 100, dados["n"])],
        padding=3, fontsize=9,
    )
    taxa_geral = analise[TARGET].mean() * 100
    ax.axhline(taxa_geral, color="gray", ls="--", lw=1.2,
               label=f"Insatisfação geral = {taxa_geral:.1f}%")
    ax.set_title("Taxa de insatisfação (notas 1-2) por faixa de atraso", fontsize=14)
    ax.set_xlabel("Faixa de atraso na entrega", fontsize=12)
    ax.set_ylabel("% de avaliações insatisfeitas", fontsize=12)
    ax.set_ylim(0, dados["pct"].max() * 100 * 1.3)
    ax.tick_params(axis="x", rotation=12)
    ax.legend()
    _salvar(fig, "fig04_insatisfacao_por_atraso.png")
def fig_categorias(analise: pd.DataFrame) -> None:
    dados = (
        analise.groupby("categoria_grupo")
        .agg(n=(TARGET, "size"), pct=(TARGET, "mean"))
        .reset_index()
    )
    dados = dados[dados["n"] >= MIN_N_RANKING].sort_values("pct", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cores = [COR_NEG if p >= analise[TARGET].mean() else COR_POS for p in dados["pct"]]
    barras = ax.barh(dados["categoria_grupo"][::-1], dados["pct"][::-1] * 100,
                     color=cores[::-1])
    ax.bar_label(barras, labels=[f"{p:.1f}%" for p in dados["pct"][::-1]],
                 padding=3, fontsize=8)
    taxa_geral = analise[TARGET].mean() * 100
    ax.axvline(taxa_geral, color="gray", ls="--", lw=1.2,
               label=f"Insatisfação geral = {taxa_geral:.1f}%")
    ax.set_title(f"Categorias com maior % de insatisfação (mín. {MIN_N_RANKING} pedidos)",
                 fontsize=14)
    ax.set_xlabel("% de avaliações insatisfeitas (1-2)", fontsize=12)
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    _salvar(fig, "fig05_categorias.png")
def fig_uf(analise: pd.DataFrame) -> None:
    dados = (
        analise.groupby("uf_cliente")
        .agg(n=(TARGET, "size"), pct=(TARGET, "mean"))
        .reset_index()
    )
    dados = dados[dados["n"] >= MIN_N_RANKING].sort_values("pct", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    cores = [COR_NEG if p >= analise[TARGET].mean() else COR_POS for p in dados["pct"]]
    barras = ax.bar(dados["uf_cliente"], dados["pct"] * 100, color=cores)
    ax.bar_label(barras, labels=[f"{p:.1f}%" for p in dados["pct"]], padding=2, fontsize=8)
    ax.axhline(analise[TARGET].mean() * 100, color="gray", ls="--", lw=1.2,
               label=f"Insatisfação geral = {analise[TARGET].mean() * 100:.1f}%")
    ax.set_title(f"Taxa de insatisfação por estado do cliente (mín. {MIN_N_RANKING} pedidos)",
                 fontsize=14)
    ax.set_xlabel("UF do cliente", fontsize=12)
    ax.set_ylabel("% de avaliações insatisfeitas", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend()
    _salvar(fig, "fig06_insatisfacao_por_uf.png")
def fig_dispersao_frete_distancia(analise: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    dados = analise.dropna(subset=["distancia_km", "frete_total", "frete_ratio"]).copy()
    dados = dados.iloc[rng.choice(len(dados), size=min(6000, len(dados)), replace=False)]
    dados["distancia_km"] = dados["distancia_km"].clip(upper=dados["distancia_km"].quantile(0.99))
    dados["frete_total"] = dados["frete_total"].clip(upper=dados["frete_total"].quantile(0.99))
    dados["frete_ratio"] = dados["frete_ratio"].clip(upper=dados["frete_ratio"].quantile(0.99))
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    pares = [
        (eixos[0], "frete_total", "Frete total (R$)"),
        (eixos[1], "frete_ratio", "Frete / valor do pedido"),
    ]
    for ax, col, ylabel in pares:
        for nota in sorted(dados[SCORE_COL].unique()):
            d = dados[dados[SCORE_COL] == nota]
            ax.scatter(d["distancia_km"], d[col], s=8, alpha=0.45,
                       color=PALETTE_NOTAS[nota], label=f"Nota {int(nota)}")
        ax.set_xlabel("Distância cliente-vendedor (km)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    eixos[0].set_title("Frete total × distância", fontsize=13)
    eixos[1].set_title("Participação do frete × distância", fontsize=13)
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, title="Nota da avaliação", loc="lower center", ncol=5,
               fontsize=10, title_fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Dispersão: quanto mais distante, maior o frete (amostra de 6.000 pedidos)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _salvar(fig, "fig07_dispersao_frete_distancia.png")
def tabelas_resumo(analise: pd.DataFrame) -> None:
    resumo_notas = (
        analise[SCORE_COL].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        .rename_axis("nota").reset_index(name="n_pedidos")
    )
    resumo_notas["pct"] = (resumo_notas["n_pedidos"] / len(analise) * 100).round(2)
    _salvar_tabela(resumo_notas, "tab01_resumo_notas.csv")
    kpis = pd.DataFrame(
        {
            "metrica": [
                "Pedidos na base completa",
                "Pedidos entregues com avaliação (base de análise)",
                "Taxa de insatisfação (notas 1-2)",
                "Nota média",
                "% de pedidos com atraso na entrega",
                "Mediana do atraso (dias)",
                "Mediana do tempo de entrega (dias)",
                "Mediana do frete (R$)",
                "Mediana do valor do pedido (R$)",
                "Mediana da distância cliente-vendedor (km)",
            ],
            "valor": [
                None,
                len(analise),
                round(analise[TARGET].mean() * 100, 2),
                round(analise[SCORE_COL].mean(), 3),
                round((analise["atraso_dias"] > 0).mean() * 100, 2),
                round(analise["atraso_dias"].median(), 2),
                round(analise["entrega_dias"].median(), 2),
                round(analise["frete_total"].median(), 2),
                round(analise["valor_pedido"].median(), 2),
                round(analise["distancia_km"].median(), 1),
            ],
        }
    )
    _salvar_tabela(kpis, "tab02_kpis_gerais.csv")
    taxa_geral = analise[TARGET].mean()
    for chave, arquivo in (("categoria_grupo", "tab03_insatisfacao_por_categoria.csv"),
                           ("uf_cliente", "tab04_insatisfacao_por_uf.csv")):
        g = (
            analise.groupby(chave)
            .agg(n_pedidos=(TARGET, "size"), n_insatisfeitos=(TARGET, "sum"))
            .reset_index()
        )
        g["pct_inssatisfacao"] = g["n_insatisfeitos"] / g["n_pedidos"] * 100
        inf, sup = _wilson(g["n_insatisfeitos"], g["n_pedidos"])
        g["ic95_inf"], g["ic95_sup"] = inf * 100, sup * 100
        g["excesso_insatisfeitos"] = g["n_insatisfeitos"] - g["n_pedidos"] * taxa_geral
        g = g.round(3).sort_values("pct_inssatisfacao", ascending=False)
        _salvar_tabela(g, arquivo)
def executar_etapa_eda() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes da EDA.")
    analise = carregar_analise()
    print(f"[eda] analisando {len(analise):,} pedidos...")
    fig_distribuicao_notas(analise)
    fig_evolucao_temporal(analise)
    fig_atraso_por_nota(analise)
    fig_inssatisfacao_por_faixa_atraso(analise)
    fig_categorias(analise)
    fig_uf(analise)
    fig_dispersao_frete_distancia(analise)
    tabelas_resumo(analise)
    print("[eda] etapa concluída.")
