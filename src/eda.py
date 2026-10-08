"""Etapa 1 — Análise exploratória de pedidos e avaliações.

Gera figuras em ``reports/figures/`` e tabelas-resumo em ``reports/tables/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    ANALYSIS_CSV, COR_NEG, COR_POS, FIGURES, PALETTE_NOTAS, SCORE_COL, STATUS_CSV, TABLES,
    TARGET,
)
from .features import carregar_analise


def _salvar(fig, nome: str) -> None:
    caminho = FIGURES / nome
    fig.savefig(caminho)
    plt.close(fig)
    print(f"[eda] figura -> {caminho.name}")


def _salvar_tabela(df: pd.DataFrame, nome: str) -> None:
    caminho = TABLES / nome
    df.to_csv(caminho, index=False)
    print(f"[eda] tabela -> {caminho.name}")


# ------------------------------------------------------------------ figuras --
def fig_status_pedidos(status: pd.DataFrame) -> None:
    contagem = status["order_status"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    barras = ax.bar(contagem.index, contagem.values, color="#4c72b0")
    ax.bar_label(barras, fmt="%d", padding=2, fontsize=9)
    ax.set_title("Distribuição dos pedidos por status (base completa)")
    ax.set_xlabel("Status do pedido")
    ax.set_ylabel("Nº de pedidos")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=20)
    _salvar(fig, "fig01_status_pedidos.png")


def fig_distribuicao_notas(analise: pd.DataFrame) -> None:
    contagem = analise[SCORE_COL].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    pct = contagem / len(analise) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    barras = ax.bar(
        [str(int(s)) for s in contagem.index], contagem.values,
        color=[PALETTE_NOTAS[s] for s in contagem.index],
    )
    ax.bar_label(barras, labels=[f"{p:.1f}%" for p in pct], padding=2, fontsize=9)
    ax.set_title("Distribuição das notas de avaliação (1-5)")
    ax.set_xlabel("Nota da avaliação")
    ax.set_ylabel("Nº de pedidos")
    taxa_neg = analise[TARGET].mean() * 100
    ax.axhline(0, color="gray", lw=0.8)
    ax.text(
        0.98, 0.92,
        f"Notas 1-2 (insatisfeitos): {taxa_neg:.1f}%\n"
        f"Média = {analise[SCORE_COL].mean():.2f} | Mediana = {analise[SCORE_COL].median():.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
    )
    _salvar(fig, "fig02_distribuicao_notas.png")


def fig_evolucao_temporal(analise: pd.DataFrame) -> None:
    serie = (
        analise.groupby("mes_compra")
        .agg(n=(TARGET, "size"), nota_media=(SCORE_COL, "mean"),
             pct_insatisfeito=(TARGET, "mean"))
        .reset_index()
    )
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
        for rotulo in eixo.get_xticklabels():
            rotulo.set_horizontalalignment("right")
    ax1.set_title("Evolução mensal: nota média e % de avaliações negativas")
    linhas, rotulos = [], []
    for ax in (ax1, ax2):
        l, r = ax.get_legend_handles_labels()
        linhas += l
        rotulos += r
    ax1.legend(linhas, rotulos, loc="lower left")
    _salvar(fig, "fig03_evolucao_temporal.png")


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
                fontsize=8, color="black")
    ax.axhline(0, color=COR_NEG, ls="--", lw=1.2, label="Prazo estimado (0 dia)")
    ax.set_title("Atraso na entrega por nota de avaliação")
    ax.set_xlabel("Nota da avaliação")
    ax.set_ylabel("Atraso em relação ao prazo (dias, recortado em [-45, 60])")
    ax.legend(loc="lower left")
    _salvar(fig, "fig04_atraso_por_nota.png")


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
    ax.set_title("Taxa de insatisfação (notas 1-2) por faixa de atraso")
    ax.set_xlabel("Faixa de atraso na entrega")
    ax.set_ylabel("% de avaliações insatisfeitas")
    ax.set_ylim(0, dados["pct"].max() * 100 * 1.25)
    ax.tick_params(axis="x", rotation=12)
    ax.legend()
    _salvar(fig, "fig05_inssatisfacao_por_atraso.png")


def fig_categorias(analise: pd.DataFrame) -> None:
    dados = (
        analise.groupby("categoria_grupo")
        .agg(n=(TARGET, "size"), pct=(TARGET, "mean"))
        .reset_index()
    )
    dados = dados[dados["n"] >= 100].sort_values("pct", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cores = [COR_NEG if p >= analise[TARGET].mean() else COR_POS for p in dados["pct"]]
    barras = ax.barh(dados["categoria_grupo"][::-1], dados["pct"][::-1] * 100,
                     color=cores[::-1])
    ax.bar_label(barras, labels=[f"{p:.1f}%" for p in dados["pct"][::-1]],
                 padding=3, fontsize=8)
    taxa_geral = analise[TARGET].mean() * 100
    ax.axvline(taxa_geral, color="gray", ls="--", lw=1.2,
               label=f"Insatisfação geral = {taxa_geral:.1f}%")
    ax.set_title("Categorias com maior % de insatisfação (mín. 100 pedidos)")
    ax.set_xlabel("% de avaliações insatisfeitas (1-2)")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    _salvar(fig, "fig06_categorias_inssatisfacao.png")


def fig_frete_preco_distancia(analise: pd.DataFrame) -> None:
    dados = analise.copy()
    dados["distancia_rec"] = dados["distancia_km"].clip(upper=dados["distancia_km"].quantile(0.995))
    paineis = [
        ("frete_total", "Frete total (R$)", True),
        ("frete_ratio", "Frete / valor do pedido", False),
        ("distancia_rec", "Distância cliente-vendedor (km)", False),
    ]
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, rotulo, log) in zip(eixos, paineis):
        sns.boxplot(data=dados, x=SCORE_COL, y=col, hue=SCORE_COL,
                    palette=PALETTE_NOTAS, legend=False, showfliers=False, ax=ax)
        med = dados.groupby(SCORE_COL)[col].median()
        for i, nota in enumerate(med.index):
            ax.text(i, med[nota], f"{med[nota]:,.2f}".replace(",", "."), ha="center",
                    va="bottom", fontsize=8)
        ax.set_xlabel("Nota da avaliação")
        ax.set_ylabel(rotulo)
        if log:
            ax.set_yscale("log")
    fig.suptitle("Frete, participação do frete e distância por nota de avaliação",
                 fontweight="bold")
    fig.tight_layout()
    _salvar(fig, "fig07_frete_preco_distancia.png")


def fig_uf(analise: pd.DataFrame) -> None:
    dados = (
        analise.groupby("uf_cliente")
        .agg(n=(TARGET, "size"), pct=(TARGET, "mean"))
        .reset_index()
    )
    dados = dados[dados["n"] >= 300].sort_values("pct", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cores = [COR_NEG if p >= analise[TARGET].mean() else COR_POS for p in dados["pct"]]
    barras = ax.bar(dados["uf_cliente"], dados["pct"] * 100, color=cores)
    ax.bar_label(barras, labels=[f"{p:.1f}%" for p in dados["pct"]], padding=2, fontsize=8)
    ax.axhline(analise[TARGET].mean() * 100, color="gray", ls="--", lw=1.2,
               label=f"Insatisfação geral = {analise[TARGET].mean() * 100:.1f}%")
    ax.set_title("Taxa de insatisfação por estado do cliente (mín. 300 pedidos)")
    ax.set_xlabel("UF do cliente")
    ax.set_ylabel("% de avaliações insatisfeitas")
    ax.tick_params(axis="x", rotation=0)
    ax.legend()
    _salvar(fig, "fig08_insatisfacao_por_uf.png")


def fig_comentario(analise: pd.DataFrame) -> None:
    dados = analise.groupby(SCORE_COL)["tem_comentario"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    cores = [PALETTE_NOTAS[s] for s in dados[SCORE_COL]]
    barras = ax.bar(dados[SCORE_COL].astype(str), dados["tem_comentario"] * 100, color=cores)
    ax.bar_label(barras, labels=[f"{v:.1f}%" for v in dados["tem_comentario"] * 100],
                 padding=2, fontsize=9)
    ax.set_title("% de avaliações com comentário, por nota")
    ax.set_xlabel("Nota da avaliação")
    ax.set_ylabel("% com comentário")
    _salvar(fig, "fig09_comentario_por_nota.png")


# ------------------------------------------------------------------ tabelas --
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

    cat = (
        analise.groupby("categoria_grupo")
        .agg(n_pedidos=(TARGET, "size"), pct_inssatisfacao=(TARGET, "mean"))
        .round(4)
        .reset_index()
        .sort_values("pct_inssatisfacao", ascending=False)
    )
    cat["pct_inssatisfacao"] *= 100
    _salvar_tabela(cat, "tab03_insatisfacao_por_categoria.csv")

    uf = (
        analise.groupby("uf_cliente")
        .agg(n_pedidos=(TARGET, "size"), pct_inssatisfacao=(TARGET, "mean"))
        .round(4)
        .reset_index()
        .sort_values("pct_inssatisfacao", ascending=False)
    )
    uf["pct_inssatisfacao"] *= 100
    _salvar_tabela(uf, "tab04_insatisfacao_por_uf.csv")


# ------------------------------------------------------------------ execução --
def executar_etapa_eda() -> None:
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes da EDA.")
    analise = carregar_analise()
    status = pd.read_csv(STATUS_CSV)

    print(f"[eda] analisando {len(analise):,} pedidos...")
    fig_status_pedidos(status)
    fig_distribuicao_notas(analise)
    fig_evolucao_temporal(analise)
    fig_atraso_por_nota(analise)
    fig_inssatisfacao_por_faixa_atraso(analise)
    fig_categorias(analise)
    fig_frete_preco_distancia(analise)
    fig_uf(analise)
    fig_comentario(analise)
    tabelas_resumo(analise)
    print("[eda] etapa concluída.")
