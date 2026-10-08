# Análise da Insatisfação do Cliente — Olist Brazilian E-commerce

Projeto de **Análise Exploratória de Dados (EDA) + Inferência estatística + Modelagem
preditiva** que responde à pergunta:

> **Quais fatores estão associados à insatisfação dos clientes e como essas
> evidências podem apoiar decisões de melhoria operacional?**

Dataset: [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle).

---

## Resultados principais

| Achado | Evidência |
|---|---|
| **Atraso na entrega** é o fator mais associado à nota baixa | Odds ratio **11,6x** (IC95% 11,0–12,2); insatisfação de 9,2% (no prazo) → 54,1% (atrasado) → **78,4%** (atraso > 7 dias) |
| **Pedidos com múltiplos itens** | 11,3% de insatisfação (1 item) vs. **26,8%** (2+ itens) — **#1 em importância no modelo** |
| **Tempo total de entrega** | 15,7 dias (insatisfeitos) vs. 9,9 dias (satisfeitos) — maior efeito Mann-Whitney (r = 0,37) |
| **Categorias críticas** | `fashion_male_clothing` 22,9%, `office_furniture` 22,1%, `audio` 21,4% (média geral: 12,8%) |
| Frete / distância | efeitos reais, porém **pequenos** (\|r\| ≤ 0,16) |
| Forma de pagamento | **sem associação** (p = 0,094) |
| Modelo | Gradient boosting **AUC = 0,767**; regressão logística AUC = 0,746 (baseline PR-AUC = 0,128) |

Relatório completo com figuras, tabelas e **recomendações gerenciais**: [`reports/relatorio.md`](reports/relatorio.md).

Resultados **etapa por etapa** (análise de variáveis → seleção de atributos →
pré-processamento → modelo): [`reports/relatorio_etapas.md`](reports/relatorio_etapas.md).

---

## Como executar

```bash
pip install -r requirements.txt
python run_pipeline.py                 # todas as etapas (load → eda → tests → model → etapas → report)
python run_pipeline.py --etapa eda     # uma etapa específica
python run_pipeline.py --etapas load tests   # sequência específica
```

Se `data/raw/` estiver vazio, os dados são baixados automaticamente via `kagglehub`
(requer token do Kaggle em `~/.kaggle/`).

## Estrutura

```
├── run_pipeline.py            # orquestrador das 6 etapas
├── requirements.txt
├── data/raw/                  # 9 CSVs do dataset
├── src/
│   ├── config.py              # caminhos, constantes, estilo, target (nota <= 2)
│   ├── data_loader.py         # carga dos CSVs (datas, CEPs, BOM)
│   ├── features.py            # 1 linha = 1 pedido: atraso, frete, distância, etc.
│   ├── eda.py                 # etapa eda   → reports/figures + tables
│   ├── statistics_tests.py    # etapa tests → Mann-Whitney, Spearman, qui-quadrado,
│   │                          #               Kruskal-Wallis, odds ratio
│   ├── model.py               # etapa model → LR + gradient boosting, SHAP, PDP
│   ├── etapas.py              # etapa etapas → reports/relatorio_etapas.md (passo a passo)
│   └── report.py              # etapa report → reports/relatorio.md
├── reports/
│   ├── relatorio.md           # relatório final (gerado)
│   ├── relatorio_etapas.md    # resultados por etapa (gerado)
│   ├── figures/               # 16 figuras (geradas)
│   └── tables/                # 14 tabelas CSV (geradas)
└── outputs/
    ├── analysis_dataset.csv   # base de análise (95.824 pedidos entregues c/ review)
    ├── resultados_modelo.json # métricas + interpretação
    └── modelo_insatisfacao.joblib
```

## Decisões metodológicas

* **Alvo:** `insatisfeito = review_score <= 2` (notas 1-2 vs. 3-5).
* **Recorte:** pedidos `delivered` com data de entrega e 1 review por pedido (a mais antiga).
* **Sem vazamento:** comentário da avaliação fica **fora** do modelo (faz parte da própria avaliação).
* **N grande ⇒ p-valor fácil de obter:** todos os testes reportam também **tamanho de efeito**
  (rank-biserial, V de Cramér, epsilon²).
* **Reprodutibilidade:** `random_state=42` em separação, CV, permutações e SHAP.
* Modelos com `class_weight='balanced'` (positivos ≈ 12,8%); avaliação por ROC-AUC,
  PR-AUC, Brier e curva de calibração.
