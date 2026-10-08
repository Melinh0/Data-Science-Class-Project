
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ANALYSIS_CSV, SCORE_COL, TABLES, TARGET
from .features import carregar_analise

ALPHA = 0.05


def _mwu(col: str, rotulo: str) -> dict:
    dados = df_alvo[[col, TARGET]].dropna()
    x = dados.loc[dados[TARGET] == 1, col]
    y = dados.loc[dados[TARGET] == 0, col]
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    r_rb = 2 * stat / (len(x) * len(y)) - 1
    return {
        "teste": "Mann-Whitney U",
        "comparacao": rotulo,
        "estatistica": round(float(stat), 1),
        "p_valor": float(p),
        "significativo_005": "sim" if p < ALPHA else "não",
        "medida_efeito": "correlação rank-bisserial (r)",
        "valor_efeito": round(float(r_rb), 4),
        "n_inssatisf": len(x),
        "n_satisfeit": len(y),
        "mediana_inssatisf": round(float(x.median()), 2),
        "mediana_satisfeit": round(float(y.median()), 2),
    }


def _spearman(col: str, rotulo: str) -> dict:
    dados = df_alvo[[SCORE_COL, col]].dropna()
    rho, p = stats.spearmanr(dados[SCORE_COL], dados[col])
    return {
        "teste": "Spearman (correlação de postos)",
        "comparacao": f"Nota x {rotulo}",
        "estatistica": round(float(rho), 4),
        "p_valor": float(p),
        "significativo_005": "sim" if p < ALPHA else "não",
        "medida_efeito": "rho de Spearman",
        "valor_efeito": round(float(rho), 4),
        "n_inssatisf": int((df_alvo[TARGET] == 1).sum()),
        "n_satisfeit": int((df_alvo[TARGET] == 0).sum()),
        "mediana_inssatisf": np.nan,
        "mediana_satisfeit": np.nan,
    }


def _chi2(x_col: pd.Series, y_col: pd.Series, rotulo: str) -> dict:
    tabela = pd.crosstab(x_col, y_col)
    chi2, p, dof, _ = stats.chi2_contingency(tabela)
    n = tabela.to_numpy().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabela.shape) - 1)))
    return {
        "teste": "Qui-quadrado de independência",
        "comparacao": rotulo,
        "estatistica": round(float(chi2), 2),
        "p_valor": float(p),
        "significativo_005": "sim" if p < ALPHA else "não",
        "medida_efeito": "V de Cramér",
        "valor_efeito": round(float(v_cramer), 4),
        "n_inssatisf": int((y_col == 1).sum()),
        "n_satisfeit": int((y_col == 0).sum()),
        "mediana_inssatisf": np.nan,
        "mediana_satisfeit": np.nan,
    }


def _odds_ratio_atraso() -> dict:
    late = df_alvo["atraso_dias"] > 0
    neg = df_alvo[TARGET] == 1
    a = int((late & neg).sum())
    b = int((late & ~neg).sum())
    c = int((~late & neg).sum())
    d = int((~late & ~neg).sum())

    odds_ratio = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ic_inf = np.exp(np.log(odds_ratio) - 1.96 * se)
    ic_sup = np.exp(np.log(odds_ratio) + 1.96 * se)
    chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, d]])
    return {
        "a_atrasado_inssatisf": a,
        "b_atrasado_satisfeit": b,
        "c_noprazo_inssatisf": c,
        "d_noprazo_satisfeit": d,
        "taxa_inssatisf_atrasado_pct": round(a / (a + b) * 100, 2),
        "taxa_inssatisf_noprazo_pct": round(c / (c + d) * 100, 2),
        "odds_ratio": round(float(odds_ratio), 3),
        "ic95_inf": round(float(ic_inf), 3),
        "ic95_sup": round(float(ic_sup), 3),
        "risco_relativo": round((a / (a + b)) / (c / (c + d)), 3),
        "chi2": round(float(chi2), 2),
        "p_valor": float(p),
    }


def _kruskal() -> dict:
    grupos = [
        g.dropna().to_numpy()
        for _, g in df_alvo.groupby("faixa_atraso", observed=True)[SCORE_COL]
    ]
    grupos = [g for g in grupos if len(g) > 0]
    h, p = stats.kruskal(*grupos)
    k, n = len(grupos), sum(len(g) for g in grupos)
    eps2p = (h - k + 1) / (n - k)
    return {
        "teste": "Kruskal-Wallis",
        "comparacao": "Nota x faixa de atraso",
        "estatistica": round(float(h), 2),
        "p_valor": float(p),
        "significativo_005": "sim" if p < ALPHA else "não",
        "medida_efeito": "epsilon²",
        "valor_efeito": round(float(eps2p), 4),
        "n_inssatisf": int((df_alvo[TARGET] == 1).sum()),
        "n_satisfeit": int((df_alvo[TARGET] == 0).sum()),
        "mediana_inssatisf": np.nan,
        "mediana_satisfeit": np.nan,
    }


df_alvo: pd.DataFrame = None


def executar_etapa_testes() -> None:
    global df_alvo
    if not ANALYSIS_CSV.exists():
        raise FileNotFoundError("Rode a etapa 'load' antes dos testes.")
    df_alvo = carregar_analise()

    linhas: list[dict] = []

    for col, rotulo in [
        ("atraso_dias", "Atraso (dias) — insatisfeitos x satisfeitos"),
        ("entrega_dias", "Tempo total de entrega (dias)"),
        ("frete_total", "Frete total (R$)"),
        ("frete_ratio", "Participação do frete no pedido (%)"),
        ("valor_pedido", "Valor do pedido (R$)"),
        ("n_itens", "Nº de itens do pedido"),
        ("distancia_km", "Distância cliente-vendedor (km)"),
        ("manuseio_dias", "Tempo até a postagem (dias)"),
    ]:
        linhas.append(_mwu(col, rotulo))

    for col, rotulo in [
        ("atraso_dias", "atraso na entrega"),
        ("entrega_dias", "tempo total de entrega"),
        ("frete_ratio", "participação do frete"),
        ("distancia_km", "distância cliente-vendedor"),
        ("valor_pedido", "valor do pedido"),
    ]:
        linhas.append(_spearman(col, rotulo))

    linhas.append(_chi2(df_alvo["prazo_cumprido"], df_alvo[TARGET],
                        "Cumprimento do prazo (sim/não) x insatisfação"))
    linhas.append(_chi2(df_alvo["categoria_grupo"], df_alvo[TARGET],
                        "Categoria do produto x insatisfação"))
    linhas.append(_chi2(df_alvo["uf_cliente"], df_alvo[TARGET],
                        "UF do cliente x insatisfação"))
    linhas.append(_chi2(df_alvo["pagamento_tipo"], df_alvo[TARGET],
                        "Forma de pagamento x insatisfação"))
    linhas.append(_chi2(df_alvo["faixa_atraso"], df_alvo[TARGET],
                        "Faixa de atraso x insatisfação"))

    linhas.append(_kruskal())

    resumo = pd.DataFrame(linhas)
    resumo["p_valor"] = resumo["p_valor"].apply(lambda p: float(f"{p:.3g}"))
    caminho = TABLES / "tab05_testes_hipotese.csv"
    resumo.to_csv(caminho, index=False)
    print(f"[testes] tabela -> {caminho.name} ({len(resumo)} testes)")

    odds = _odds_ratio_atraso()
    caminho_or = TABLES / "tab06_odds_ratio_atraso.csv"
    pd.DataFrame([odds]).to_csv(caminho_or, index=False)
    print(f"[testes] tabela -> {caminho_or.name}")
    print(
        f"[testes] OR do atraso = {odds['odds_ratio']} "
        f"(IC95% {odds['ic95_inf']}-{odds['ic95_sup']})"
    )
    print("[testes] etapa concluída.")
