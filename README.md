# Análise da Insatisfação do Cliente — Olist Brazilian E-commerce

Projeto de **Análise Exploratória de Dados (EDA) + Inferência estatística + Preparação/Redução de
dimensionalidade + Modelagem preditiva** que responde à pergunta:

> **Quais fatores estão associados à insatisfação dos clientes e como essas
> evidências podem apoiar decisões de melhoria operacional?**

Dataset: [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle).

Três relatórios são gerados pelo pipeline:

* [`docs/relatorio_etapas.md`](docs/relatorio_etapas.md) — **preparação e redução de dimensionalidade**
  (diagnóstico de qualidade, pré-processamento, seleção de atributos por filtro/wrapper/embutido, PCA,
  comparação das representações, limitações).
* [`docs/relatorio.md`](docs/relatorio.md) — **análise de negócio** (EDA, testes de hipótese, modelo,
  interpretação e recomendações).
* [`docs/relatorio_modelos.md`](docs/relatorio_modelos.md) — **modelos supervisionados e avaliação**
  (KNN, árvore, boosting e MLP comparados com validação cruzada, matrizes de confusão, ROC e escolha do modelo final).

---

## Resultados principais

Números da última execução completa sobre a base real (95.824 pedidos); o detalhe está nos relatórios.

| Achado | Evidência |
|---|---|
| **Pontualidade da entrega** é o fator mais associado à nota baixa | Odds ratio bruto **11,6x** (IC95% 11,0–12,2); insatisfação 9,2% (sem atraso) → 54,1% (atrasado); por faixa: 9,0% → 10,2% → 36,1% → **78,4%** (atraso > 7 dias) |
| **Importância por grupo (gradient boosting)** | Prazo e tempo de entrega (queda de AUC 0,196) ≫ itens/preço/frete (0,062) > produto (0,017) > geografia (0,007); calendário e pagamento ≈ 0 |
| **Pedidos com múltiplos itens** | 11,3% (1 item) vs. 26,8% (2+); associação robusta (2º grupo mais importante), mas a causa (envios fracionados) é **hipótese** |
| Frete / distância | efeitos reais, porém **pequenos** (\|r\| ≤ 0,16) |
| Forma de pagamento | **sem associação** (p = 0,094) |
| **Modelo** | Gradient boosting AUC = **0,766** (CV 0,759), F1 0,478, recall 0,462, precisão 0,495, acurácia 0,871, limiar 0,22 (**calibrado**, sem `class_weight`) |
| **Validações** | Temporal (treina no passado, testa nos 20% mais recentes): AUC **0,690**; pré-entrega (sem atraso/tempo de entrega): AUC **0,692** |
| **Seleção de atributos** | 5 métodos, consenso ≥ 3/5: 11 de 19 features; `entrega_dias` e `n_itens` unânimes; pagamento, hora e dia da semana descartados |
| **PCA** | k escolhido = **13 de 13**: nenhuma redução se justifica (k = 10, pela regra dos 90%, custaria 0,006 de AUC) |
| **Representações** | Selecionada (109 colunas) ≈ original (Δ AUC −0,0001 no boosting); PCA piora o boosting em 0,009 |

**Dicionário de dados** — atributos das tabelas brutas e features da base final, com explicações simples: [`docs/dicionario_dados.md`](docs/dicionario_dados.md).

---

## Tipo de modelo e o que ele retorna

**A tarefa é classificação binária, não regressão.** O alvo `insatisfeito` é categórico
(0 = notas 3-5, 1 = notas 1-2) e o que se entrega é uma **decisão de ação** (contatar ou não o
cliente) — é isso que define classificação. Retornar uma probabilidade **não** faz do modelo uma
regressão: em classificação, a probabilidade é a etapa interna do modelo (o "quão provável" é a
classe positiva) e a classificação final sai do corte dessa probabilidade no limiar de decisão.
Uma regressão de verdade retornaria um valor contínuo estimado (por exemplo, prever a nota de 1 a
5) e seria medida por RMSE/MAE — não é o caso aqui, onde as métricas são AUC, precisão, recall e F1.

Apesar do nome, a **regressão logística é um classificador**: ela ajusta uma reta sobre o log-odds
(log(p/(1−p))) e aplica a função sigmoid para transformar esse número em probabilidade; o que se
entrega é a classe.

**O que cada modelo retorna (duas saídas):**

1. **Probabilidade** (`predict_proba`): a chance estimada de o pedido ser insatisfeito (score de 0
   a 1). É a base do AUC, da curva de calibração e da priorização da fila de atendimento.
2. **Classificação 0/1** (`predict`): a probabilidade cortada no **limiar de decisão** (aqui, o que
   minimiza o custo esperado com FN = 3 × FP). É **essa** que a gestão aciona: 1 = contatar o
   cliente, 0 = não agir. A probabilidade complementa, permitindo atender primeiro os de maior risco.

**Como cada modelo calcula a probabilidade de insatisfação:**

| Modelo | Lógica no contexto do projeto |
|---|---|
| Regressão logística | combinação linear das variáveis (z = β₀ + β₁x₁ + …) passa pela sigmoid 1/(1+e⁻ᶻ) → probabilidade; os coeficientes viram odds ratios interpretáveis |
| Árvore de decisão | partições sucessivas por limiares (ex.: `atraso_dias ≤ 2,74`) agrupam pedidos parecidos; a probabilidade da folha = fração de insatisfeitos nela |
| KNN | para um pedido novo, encontra os k vizinhos mais próximos e devolve a fração de insatisfeitos entre eles (ponderada pela distância) |
| Gradient boosting | várias árvores rasas em sequência, cada uma corrigindo o erro da anterior; as saídas somam em escala de log-odds e viram probabilidade pela sigmoid |
| MLP | rede neural com camadas ocultas e não-linearidades (ReLU) transformam as variáveis; a saída sigmoid entrega a probabilidade |
| Classe majoritária (baseline) | sempre devolve 0 — serve de régua: qualquer modelo útil precisa superá-lo |

**Papel na pipeline:** as etapas `model` e `modelos` são o **objetivo final** do projeto — convertem
as evidências da EDA em uma ferramenta de ação. A etapa `model` ajusta o modelo de negócio
(regressão logística + gradient boosting) com interpretação (odds ratios, SHAP, importância por
grupo) e validações (temporal e pré-entrega); a etapa `modelos` compara 5 famílias com validação
cruzada, escolhe o limiar pelo custo esperado e grava o modelo final
(`outputs/modelo_supervisionado_final.joblib`), usado para pontuar pedidos novos. Detalhamento:
[`docs/relatorio_modelos.md`](docs/relatorio_modelos.md).

---

## Como executar

```bash
pip install -r requirements.txt
python run_pipeline.py                      # todas as etapas
python run_pipeline.py --etapa select       # uma etapa específica
python run_pipeline.py --etapas load prep   # sequência específica
```

Etapas, na ordem: `load → prep → eda → tests → select → pca → model → reducao → modelos → etapas → report → relatorio_modelos → dicionario`.

A etapa `modelos` é a mais longa (busca de hiperparâmetros de 5 famílias de modelos): cerca de 20 minutos.
Para um teste rápido com grades mínimas: `MODELOS_RAPIDO=1 python run_pipeline.py --etapa modelos`.

Se `data/raw/` estiver vazio, os dados são baixados automaticamente via `kagglehub`
(requer token do Kaggle em `~/.kaggle/`). Também dá para baixar manualmente com
`python scripts/download_dataset.py`.

### Usando o notebook (entrega da atividade)

A mesma análise também roda por notebook — é a **entrega executável** do projeto de
pré-processamento e redução de dimensionalidade:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/analise_preprocessamento_reducao.ipynb
# ou: abra no Jupyter e use "Run All"
```

[`notebooks/analise_preprocessamento_reducao.ipynb`](notebooks/analise_preprocessamento_reducao.ipynb)
cobre diagnóstico da base, decisões de pré-processamento, seleção de atributos, PCA e comparação das
representações — **cada etapa chama as funções do `src/`**, sem código duplicado. Defina
`FORCE_ETAPAS = False` no primeiro bloco para reaproveitar os artefatos já gerados pelo pipeline.

O [`notebooks/reducao_dimensionalidade.ipynb`](notebooks/reducao_dimensionalidade.ipynb) é uma
**exploração de referência** (construção da base e investigação de qualidade passo a passo), e a
auditoria das 9 tabelas brutas está em [`docs/DATASET_REVIEW.md`](docs/DATASET_REVIEW.md).

## Estrutura

```
├── run_pipeline.py            # orquestrador das etapas
├── requirements.txt
├── data/raw/                  # 9 CSVs do dataset
├── scripts/
│   └── download_dataset.py    # download manual do dataset (kagglehub)
├── presentation/              # decks da equipe (pptx + pdf)
├── docs/                      # relatórios gerados — os .md vivem aqui
│   ├── DATASET_REVIEW.md      # auditoria das 9 tabelas brutas (diagnóstico inicial)
│   ├── relatorio.md           # análise de negócio (EDA, testes, modelo, recomendações)
│   ├── relatorio_etapas.md    # preparação e redução de dimensionalidade
│   ├── relatorio_modelos.md   # modelos supervisionados e avaliação
│   └── dicionario_dados.md    # dicionário de dados
├── notebooks/
│   ├── analise_preprocessamento_reducao.ipynb  # entrega executável (mesmo código do src/)
│   ├── pipeline_completo.ipynb                 # pipeline completo ponta a ponta (todas as etapas)
│   └── reducao_dimensionalidade.ipynb          # exploração de referência
├── src/
│   ├── __init__.py
│   ├── config.py              # caminhos, constantes, features, redundâncias, grupos de variáveis
│   ├── data_loader.py         # carga dos CSVs (datas, CEPs, BOM)
│   ├── features.py            # etapa load   → 1 linha = 1 pedido (atraso, frete, distância...)
│   ├── qualidade.py           # etapa prep   → diagnóstico, regras de domínio, VIF, base final treino/teste
│   ├── preprocess.py          #                pré-processamento único + divisão treino/teste compartilhada
│   ├── eda.py                 # etapa eda    → figuras e tabelas descritivas
│   ├── statistics_tests.py    # etapa tests  → Mann-Whitney, Spearman, qui-quadrado, Kruskal-Wallis, OR (+ Holm)
│   ├── feature_selection.py   # etapa select → filtro (MI, AUC), wrapper (forward), embutido (L1, RF) + consenso
│   ├── pca_diagnostico.py     # etapa pca    → Kaiser, análise paralela, validação preditiva, cargas, CP1×CP2
│   ├── model.py               # etapa model  → LR + gradient boosting, calibração, importância por grupo, SHAP
│   ├── reducao.py             # etapa reducao→ original × selecionada × PCA
│   ├── modelos_supervisionados.py  # etapa modelos → KNN, árvore, boosting, MLP: busca por CV, limiar por custo, teste
│   ├── relatorio_modelos.py   # etapa relatorio_modelos → docs/relatorio_modelos.md
│   ├── etapas.py              # etapa etapas → docs/relatorio_etapas.md
│   └── report.py              # etapa report → docs/relatorio.md + docs/dicionario_dados.md
├── reports/                   # figuras (fig01–fig27) e tabelas (tab01–tab27) geradas
└── outputs/                   # artefatos gerados
    ├── analysis_dataset.csv   # base de análise (gerada pela etapa load)
    ├── prep/                  # base final pré-processada: X/y de treino e teste
    ├── tabelas_brutas.csv, orders_status_dataset.csv  # diagnósticos das etapas load/prep
    ├── pca_info.json, selecao_atributos.json          # diagnósticos de PCA e seleção
    ├── resultados_modelo.json              # métricas e interpretações (etapa model)
    ├── resultados_modelos_supervisionados.json  # métricas (etapa modelos)
    ├── modelo_insatisfacao.joblib          # modelo principal de negócio
    ├── modelo_supervisionado_final.joblib  # modelo final escolhido (5 famílias)
    └── treino_*.parquet / teste_*.parquet  # bases comparadas pela etapa reducao
```

## Decisões metodológicas

* **Alvo:** `insatisfeito = review_score <= 2` (notas 1-2 vs. 3-5).
* **Recorte:** pedidos `delivered` com data de entrega e 1 review por pedido (a mais antiga).
* **Modelos supervisionados:** holdout 80/20 reservado; hiperparâmetros por k-fold estratificado (5) só no treino, com
  as mesmas dobras para todos os modelos; limiar de decisão pelo **custo esperado com falso negativo = 3 × falso positivo**
  (o cliente insatisfeito pesa mais; razão ajustável em `CUSTO_FN_FP`, com análise de sensibilidade de 1 a 10);
  modelo final = menor custo esperado entre os de AUC de CV a até 1 erro-padrão do melhor; IC95% por bootstrap no teste.
* **Qualidade dos dados:** valores impossíveis (datas incoerentes, parcelas = 0, peso = 0, distância > 4.500 km)
  viram ausentes; categorias equivalentes são unificadas; extremos coerentes com o domínio são mantidos.
* **Redundância:** `valor_pedido` (= `preco_total + frete_total`) e `preco_unitario_medio` (= `preco_total / n_itens`)
  ficam fora do modelo.
* **Sem vazamento de treino/teste:** a divisão ocorre antes de qualquer ajuste; imputação, escala, seleção de
  atributos, PCA e limiar usam só o treino. O modelo é escolhido pela validação cruzada, não pelo teste.
* **Sem vazamento do alvo:** o comentário da avaliação fica fora do modelo.
* **Desbalanceamento:** sem `class_weight` (distorce as probabilidades); o limiar vem de validação cruzada.
  Avaliação por acurácia, precisão, recall, F1, ROC-AUC e curva de calibração.
* **Uso do escore:** o modelo principal usa o atraso realizado (só conhecido na entrega). Um modelo
  **pré-entrega** (sem `atraso_dias`/`entrega_dias`) é avaliado à parte, assim como a validação **temporal**.
* **Importância:** permutação individual e **por grupo** (variáveis correlacionadas dividem o crédito) e SHAP.
* **Seleção de atributos:** 5 métodos comparados com *features-sombra* (cópias embaralhadas); consenso ≥ 3 de 5.
* **PCA:** o número de componentes não vem de uma porcentagem fixa de variância; compara Kaiser, análise
  paralela e a informação preditiva preservada (CV).
* **N grande ⇒ p-valor fácil de obter:** todos os testes reportam tamanho de efeito e p ajustado por Holm.
* **Reprodutibilidade:** `random_state=42` em separação, CV, permutações e SHAP.
