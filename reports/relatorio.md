# Preparação e redução de dimensionalidade: dataset Olist

Relatório analítico do projeto de Ciência de Dados (UECE). Base: Olist Brazilian E-Commerce (9 tabelas em `data/raw/`).
Todos os números vêm do notebook `notebooks/reducao_dimensionalidade.ipynb`; os problemas da base antes do
pré-processamento estão catalogados em `docs/DATASET_REVIEW.md`.

**Pergunta central:** como preparar e reduzir a base preservando a informação relevante para as etapas seguintes?

**Resposta em resumo.** A base final recomendada tem **78 colunas** (de 92 pré-processadas), obtidas por seleção de
11 das 19 variáveis candidatas. O PCA foi aplicado e interpretado, mas oferece pouco ganho: comprime só as
13 variáveis numéricas, dilui as variáveis mais informativas e perde interpretabilidade. Nenhuma das estratégias reduz
a dimensão total de forma expressiva (no máximo −15%), porque 79 das 92 colunas vêm do one-hot de variáveis
categóricas.

**Unidade de análise e alvo.** Cada linha é um **pedido**. Como o dataset não traz um alvo oficial, usamos
`insatisfeito = nota de avaliação ≤ 2` (12,8% dos pedidos). O alvo serve para orientar a seleção supervisionada
e para interpretar os componentes; **não treinamos modelos preditivos** (fora do escopo).

---

## 1. Diagnóstico inicial da base

**Dimensões:** `orders` 99.441, `customers` 99.441, `items` 112.650, `reviews` 99.224, `payments` 103.886,
`products` 32.951, `sellers` 3.095, `geolocation` 1.000.163, `category_translation` 71.

**Recorte (funil):**

| Etapa | Pedidos | Perda | Motivo |
|---|---|---|---|
| Todos | 99.441 | | |
| `order_status = delivered` | 96.478 | 2.963 | só pedidos finalizados têm prazo real de entrega |
| com data de entrega | 96.470 | 8 | inconsistência: "entregue" sem data |
| com avaliação | 95.824 | 646 (0,67%) | sem nota não há alvo |

**Problemas de qualidade encontrados** (cada item foi medido no notebook):

| Tipo | Problema | Evidência |
|---|---|---|
| Ausentes legítimos | Datas de aprovação, postagem e entrega ausentes | explicados pelo status do pedido (cancelado, indisponível, em trânsito...) em 146, 1.781 e 2.957 casos |
| Ausentes inconsistentes | Pedidos `delivered` sem datas | 8 sem entrega, 14 sem aprovação, 2 sem postagem |
| Ausentes por cadastro | Produtos sem categoria, nome, descrição e fotos | 610 (1,85%), sempre as mesmas linhas |
| Duplicatas | `review_id` repetido; pedidos com 2+ avaliações; linhas idênticas em `geo` | 814; 547 (em 202 as notas diferem); 261.831 (26%) |
| Inconsistência de cadastro | 2 categorias sem tradução para o inglês | `pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos` |
| Valores impossíveis | Postagem antes da compra; 0 parcelas; peso 0 g | 164 pedidos (só 2 acima de 1 dia: −171 e −4,2 dias); 2 pedidos; 4 produtos |
| Erro de geolocalização | Coordenadas fora do Brasil geram distâncias de até 8.678 km | 33 coordenadas em `geo` |
| Extremos legítimos | Frete até R$ 1.795, peso até 184 kg, entrega até 208 dias | coerentes com outras variáveis (112 kg, 6 itens e 2.134 km; 10 itens; atraso de 189 dias) |
| Extremos incertos | Postagem antes da aprovação; aprovação de até 31 dias | 1.343 pedidos; 32 de 72.331 pagamentos em cartão acima de 7 dias |
| Assimetria | Caudas longas | assimetria de 13,5 (`frete_total`), 10,1 (`preco_total`), 7,2 (`n_itens`) |
| Redundância | `valor_pedido = preco_total + frete_total` | diferença máxima de 4e-13; Spearman entre `preco_unitario_medio` e `preco_total` = 0,95 |
| Constante | `order_status` | 100% `delivered` após o recorte |
| Baixa variabilidade | Nenhuma variável quase constante | o maior valor dominante é `n_itens` com 90,1%, que carrega informação real |

---

## 2. Decisões de pré-processamento

Regra aplicada a cada item: **que problema justifica a transformação?** Tudo o que aprende dos dados
(medianas, médias, categorias, seleção, PCA) foi ajustado **somente no treino**.

| Problema | Transformação | Observação |
|---|---|---|
| Datas como texto | Conversão para data; variáveis derivadas (`atraso_dias`, `entrega_dias`, `manuseio_dias`, `aprovacao_h`, hora, dia da semana, mês) | datas brutas saem das features |
| Pedidos sem desfecho ou sem alvo | Recorte descrito acima | viés de seleção registrado |
| 2+ avaliações por pedido | Mantida a **mais recente** | muda a classe de só 96 pedidos (0,097%); escolha quase irrelevante |
| Duplicatas e erros de geolocalização | `drop_duplicates` em `geo`; descartar coordenadas fora do Brasil; mediana por CEP | distância máxima caiu de 8.678 para 3.398 km; 476 pedidos (0,5%) sem coordenada |
| Valores impossíveis | Viram ausentes: `manuseio_dias` < 0 (164), `n_parcelas` = 0 (2), peso 0 g (4 produtos) | regras de domínio, sem parâmetro aprendido, aplicadas antes da divisão |
| Item principal × pedido | `peso_total_g` = soma dos itens, `n_fotos_medio` = média, `categoria_principal` com nome explícito | corrige mistura de escalas (item × pedido) |
| Categoria sem tradução ou sem cadastro | Nome original em português; `sem_categoria` só para os 610 sem cadastro | não confunde "não traduzida" com "ausente" |
| Redundância | Removidas `valor_pedido` (identidade exata) e `preco_unitario_medio` (= `preco_total / n_itens`) | critério: identidade exata ou Spearman ≥ 0,90 |
| Assimetria | `log1p` em 7 variáveis (assimetria ≥ 3 no treino, sem negativos) | `frete_total` 13,5 → 0,5; `preco_total` 10,1 → 0,3; `n_itens` só 7,2 → 4,0 |
| Escalas diferentes | `StandardScaler` em todas as numéricas | ajustado no treino |
| Ausentes restantes | Mediana (numéricas) e moda (categóricas) | ajustadas no treino |
| Categóricas sem ordem | One-hot com `min_frequency` = 0,5% do treino | 146 colunas sem agrupar → **79**; total de 92 com as 13 numéricas |
| Vazamento | Excluídos `review_score`, `review_creation_date`, comentários, identificadores e CEPs | `customer_unique_id` só serve para agrupar a divisão |
| Treino/teste | 80/20, estratificada pelo alvo e **agrupada por cliente** | 76.658 / 19.166; insatisfação de 12,81% nos dois; 0 clientes em comum |

**Não há variável ordinal:** mês e dia da semana são nominais, e UFs e categorias não têm ordem.
**Mantidos de propósito:** extremos legítimos (tratados por log e padronização, não removidos) e os casos incertos
(postagem antes da aprovação, aprovação demorada).

---

## 3. Seleção de atributos

**Partimos de 19 variáveis** (13 numéricas e 6 categóricas), todas medidas no treino. Existe alvo adequado,
então métodos supervisionados são aplicáveis. Os modelos usados (regressão logística) servem **apenas como
instrumento de ranqueamento**; não avaliamos desempenho preditivo.

| Família | Método | Critério de voto "sim" (definido antes de ver os resultados) | Custo (ajustes de modelo) |
|---|---|---|---|
| Filtro | Informação mútua (MI) | acima do **ruído próprio** da variável (alvo embaralhado) e ≥ 0,5% da incerteza do alvo | 0 |
| Filtro | Qui-quadrado + V de Cramér | só categóricas (complemento) | 0 |
| Embutido | Regressão logística com L1, 5 valores de `C` | sobrevive em pelo menos 3 dos 5 | 5 |
| Wrapper | RFE (regressão logística) | posição ≤ 8 de 19 | ~30 |

O qui-quadrado só se aplica a categóricas; não foi forçado sobre as numéricas. Resultado (ordenado por votos):

| Variável | MI (% da incerteza do alvo) | L1 (de 5) | RFE (posição) | Votos |
|---|---|---|---|---|
| `atraso_dias` | 15,3 | 5 | 3 | **3** |
| `entrega_dias` | 12,5 | 5 | 1 | **3** |
| `n_itens` | 3,6 | 5 | 2 | **3** |
| `manuseio_dias` | 2,5 | 4 | 4 | **3** |
| `mes_compra` | 1,2 | 3 | 4 | **3** |
| `n_parcelas` | 0,9 | 4 | 7 | **3** |
| `uf_cliente` | 0,9 | 4 | 3 | **3** |
| `categoria_principal` | 0,7 | 4 | 5 | **3** |
| `distancia_km` | 0,4 | 3 | 8 | 2 |
| `uf_vendedor` | 0,2 | 3 | 8 | 2 |
| `pagamento_tipo` | 0,0 | 4 | 3 | 2 |
| `frete_total` | 2,6 | 1 | 32 | 1 |
| `frete_ratio` | 2,4 | 0 | 9 | 1 |
| `preco_total` | 1,6 | 2 | 9 | 1 |
| `peso_total_g` | 1,3 | 1 | 24 | 1 |
| `n_fotos_medio` | 0,7 | 0 | 31 | 0 |
| `aprovacao_h` | 0,1 | 2 | 23 | 0 |
| `dia_semana_compra` | 0,0 | 2 | 13 | 0 |
| `hora_compra` | 0,0 | 2 | 15 | 0 |

**Decisão:** ficam as 11 variáveis com **≥ 2 votos**. Os cortes são julgamentos, declarados antes de ver os resultados.

**Convergências.** Oito variáveis são sustentadas pelos três métodos. `atraso_dias` e `entrega_dias` concentram a informação
sobre o alvo (15,3% e 12,5% da incerteza; a terceira, `n_itens`, tem 3,6%), com salto grande para as demais.

**Divergências e o que as explica:**
- **`frete_total`, `frete_ratio`, `preco_total`, `peso_total_g`** (1 voto, só o filtro): cada uma, isolada, se associa ao alvo
  (1,3 a 2,6%), mas L1 e RFE não encontram contribuição **adicional** depois das demais variáveis. Elas **não são redundantes**:
  só 0 a 44% da variação das variáveis descartadas é recuperável a partir das mantidas. Foram descartadas por falta de
  contribuição ao alvo, o que implica perda real de informação (ver seção 6).
- **`pagamento_tipo`** (MI ≈ 0, V de Cramér 0,013, mas 4 de 5 no L1 e 3º no RFE): efeito pequeno que os modelos captam em
  parte por interação com `n_parcelas` (cartão em 2+ parcelas tem insatisfação ~13,7% contra 11,3% em 1 parcela). A causa
  exata não está estabelecida.
- **Filtro × modelos** medem coisas diferentes: o filtro olha a relação marginal de cada variável; L1 e RFE avaliam o efeito
  condicional às demais.

**Três correções de método feitas no projeto** (afetam a validade da comparação):
(1) a MI das categóricas precisa de `discrete_features`; sem isso, `pagamento_tipo` aparecia com MI 0,0081 em vez de 0,0001;
(2) o ruído da MI é medido **por variável**, porque o estimador de variáveis contínuas é bem mais ruidoso que o de discretas;
(3) as colunas one-hot foram **padronizadas** para L1 e RFE, que comparam tamanho de coeficientes; antes disso, a `categoria_principal`
aparecia em 2º no RFE e passou para 5º.

---

## 4. PCA

**Dados:** somente as **13 variáveis numéricas** já limpas, com log e padronizadas. Combinações lineares só fazem sentido
para variáveis contínuas; as 79 colunas one-hot ficam de fora. O PCA foi **ajustado no treino**; o teste só foi projetado.
A matriz tem posto 13 de 13 (sem dependência exata, ao contrário do PCA do projeto anterior, em que o posto era 14 de 15).

| Componente | Autovalor | Variância | Acumulada |
|---|---|---|---|
| CP1 | 2,70 | 20,7% | 20,7% |
| CP2 | 2,02 | 15,5% | 36,2% |
| CP3 | 1,52 | 11,7% | 47,9% |
| CP4 | 1,20 | 9,2% | 57,2% |
| CP5 | 1,05 | 8,1% | **65,2%** |
| CP6 | 0,99 | 7,6% | 72,9% |
| CP7–CP9 | 0,94 a 0,67 | 18,1% | **91,0%** (em CP9) |
| CP10–CP13 | 0,62 a 0,04 | 9,1% | 100% |

![Cotovelo e análise paralela](figures/fig01_pca_cotovelo_paralela.png)

**Número de componentes: 5.** Kaiser (autovalor > 1) e **análise paralela** (autovalores reais contra dados embaralhados,
100 repetições) concordam em 5. A variância acumulada decai de forma gradual, sem cotovelo nítido: 5 componentes
preservam 65% e 9 preservam 91%. Não escolhemos o número por um percentual; 9 componentes (≥ 90%) fica como alternativa
de sensibilidade.

![Cargas](figures/fig02_pca_cargas.png)

**Interpretação das cargas** (leitura nossa a partir das cargas; o sinal de cada componente é arbitrário):

| Componente | Variáveis de maior carga | Leitura |
|---|---|---|
| CP1 | `preco_total` 0,51, `frete_total` 0,46, `peso_total_g` 0,44, `n_parcelas` 0,32 | porte e valor do pedido |
| CP2 | `entrega_dias` 0,51, `frete_ratio` 0,46, `distancia_km` 0,39, `manuseio_dias` 0,32 | logística: distância e tempo |
| CP3 | `atraso_dias` 0,48, `manuseio_dias` 0,41, `frete_ratio` −0,41 | atraso, misturado com frete |
| CP4 | `aprovacao_h` 0,53, `n_itens` 0,48, `distancia_km` −0,38 | mistura, difícil de nomear |
| CP5 | `n_fotos_medio` 0,61, `hora_compra` −0,42, `aprovacao_h` 0,38 | mistura, difícil de nomear |

A última componente (CP13, 0,29% da variância) é formada por `preco_total` (0,68), `frete_ratio` (0,60) e `frete_total` (0,42):
`frete_ratio` é quase uma função das outras duas, o que reforça sua exclusão.

![CP1 x CP2](figures/fig03_pca_cp1_cp2.png)

**Projeção CP1 × CP2:** uma única nuvem, **sem separação visível** entre satisfeitos e insatisfeitos (o alvo só colore os pontos).
Há apenas uma leve tendência de insatisfeitos em valores altos de CP2.

**Informação sobre o alvo:** a MI com o alvo é 2,3% (CP1), 4,3% (CP2), **5,6% (CP3)**, 2,0% (CP4) e 3,4% (CP5). Nenhuma
componente se aproxima das melhores variáveis originais (`atraso_dias` 15,3%): a mistura de variáveis **dilui** o sinal.

**Informação preservada** (R² de reconstrução de cada variável no teste): com k=5, média 0,65; preserva bem as variáveis redundantes entre si
(`preco_total` 0,89, `frete_ratio` 0,87, `frete_total` 0,86) e mal as isoladas (`hora_compra` 0,21, `n_fotos_medio` 0,43) e
`atraso_dias`, a mais relevante, só 0,58. Com k=9, média 0,91.

---

## 5. Comparação das representações

Comparação sem modelos preditivos: dimensão, informação preservada e interpretabilidade. O "R² ponderado" pesa cada
variável numérica pela sua informação sobre o alvo.

| Representação | Colunas | Redução | R² médio (numéricas) | R² ponderado | Interpretabilidade |
|---|---|---|---|---|---|
| A. Pré-processada completa | 92 | 0% | 1,00 | 1,00 | total |
| **B. Seleção de atributos** | **78** | 15% | 0,54 | **0,85** | total: variáveis originais |
| C1. PCA (k=5) + one-hot | 84 | 9% | 0,65 | 0,70 | só nas numéricas; componentes mistas |
| C2. PCA (k=9) + one-hot | 88 | 4% | **0,91** | **0,90** | só nas numéricas; componentes mistas |

No **bloco numérico**: seleção 13 → 6 variáveis (−54%); PCA 13 → 5 (−62%) ou 13 → 9 (−31%). Com o mesmo orçamento de 6 dimensões,
o PCA tem R² médio de 0,73 e ponderado de 0,71, contra 0,54 e 0,85 da seleção.

**Leitura crítica:**
- Os dois critérios **divergem**: o PCA preserva mais a variância total (ganha no R² simples); a seleção preserva melhor as
  variáveis relevantes (ganha no ponderado). Cuidado: a seleção foi escolhida a partir do alvo e o peso do R² ponderado também
  vem do alvo, então a vantagem da seleção nessa métrica é **em parte circular**.
- **Seleção ganha** em interpretabilidade (variáveis originais, com significado direto) e **perde** a informação própria das
  variáveis descartadas (R² de recuperação entre 0 e 0,44).
- **PCA ganha** em compressão do bloco numérico, mas **perde** a leitura direta das variáveis, dilui `atraso_dias` e `entrega_dias`
  (as mais informativas) e **não atua** nas colunas one-hot, que são 86% da dimensionalidade.
- Nenhuma estratégia reduz a base total além de 15%. A redução mais efetiva exigiria **agrupar categorias** (ver seção 6).

**Representação recomendada: B (seleção de atributos, 78 colunas).** Mantém o significado das variáveis e a informação mais
relevante, e o PCA traz ganho de compressão pequeno para o custo em interpretação. O PCA permanece documentado como alternativa,
e as componentes (k=5) estão salvas.

---

## 6. Limitações e próximos passos

O detalhamento está na seção 9 do notebook. Principais pontos:

- **Alvo definido por nós** (nota ≤ 2); com outra definição o ranking de variáveis pode mudar.
- **Viés de seleção:** 646 pedidos sem avaliação foram excluídos. Clientes recorrentes foram mantidos no mesmo conjunto, mas seus
  pedidos continuam correlacionados.
- **Distância com precisão de CEP**, usando o vendedor do item mais caro.
- **Casos incertos mantidos** (postagem antes da aprovação, aprovações de até 31 dias): sem conhecimento do processo da empresa
  não dá para dizer se são erro ou prática normal.
- **Seleção sem análise de estabilidade** (uma partição, uma semente), com cortes que são julgamentos; L1 e RFE agregados por
  variável favorecem variáveis com muitos níveis; a divergência entre filtro e modelos não foi totalmente explicada.
- **Perda real de informação** ao descartar `frete_total`, `preco_total`, `peso_total_g` e `frete_ratio`: uma política de frete
  ou de valor que a empresa considere uma alavanca poderia justificar mantê-las.
- **PCA** só nas numéricas, linear e não supervisionado (ordena por variância, não por relevância).
- **Informação preservada medida por reconstrução e MI**, não por desempenho em modelagem.

**Decisões que mudariam com novas informações de domínio:** a regra da avaliação duplicada; o limiar do alvo (nota ≤ 3, por exemplo);
tratar os `manuseio_dias` negativos por truncamento se logística confirmar diferença de fuso; **agrupar `categoria_principal`
(74 níveis) em macrocategorias e as UFs em regiões**, o que reduziria dezenas de colunas e é a redução de dimensionalidade mais
efetiva possível; codificação cíclica de `hora_compra`.

**Para as etapas de modelagem:** refazer a seleção dentro de validação cruzada agrupada por cliente (estabilidade), tratar o
desbalanceamento (12,8% de insatisfeitos) com métricas apropriadas, comparar as três bases salvas no problema real, e investigar
a divergência em `pagamento_tipo`, `frete_total` e `mes_compra`.

---

## Entregáveis

| Entregável | Local |
|---|---|
| Notebook executável (executado do zero sem erros, em cerca de 1 minuto) | `notebooks/reducao_dimensionalidade.ipynb` |
| Relatório analítico | `reports/relatorio.md` |
| Base final recomendada, treino e teste separados (78 colunas + `order_id` + alvo) | `outputs/treino_selecionada.parquet`, `outputs/teste_selecionada.parquet` |
| Base pré-processada completa (92 colunas) | `outputs/treino_completa.parquet`, `outputs/teste_completa.parquet` |
| Componentes do PCA (k=5) | `outputs/treino_pca5.parquet`, `outputs/teste_pca5.parquet` |
| Catálogo de problemas da base e decisões | `docs/DATASET_REVIEW.md` |
| Download dos dados | `scripts/download_dataset.py` |

**Reprodução:** `uv sync`, colocar os CSVs em `data/raw/` (ou `uv run --with kagglehub python scripts/download_dataset.py`) e
`uv run jupyter nbconvert --to notebook --execute notebooks/reducao_dimensionalidade.ipynb`. Semente fixa (`random_state=42`).
