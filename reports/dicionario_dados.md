# Dicionário de dados — Olist Brazilian E-commerce

**Dataset:** [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — marketplace brasileiro de e-commerce
**Fonte:** Kaggle

Esta documentação reúne os **atributos das tabelas do dataset** e as **features da
base final** usadas no estudo, com explicações simples — servindo de
contextualização ao leitor.

* **Junções:** as tabelas são relacionadas por `order_id` (pedidos, clientes, itens,
  avaliações, pagamentos, produtos, vendedores, geolocalização e tradução de
  categorias).
* **Atributos derivados:** o atraso (`atraso_dias`) é a diferença entre a data de
  entrega real e a estimada; o tempo de entrega (`entrega_dias`) entre a compra e a
  entrega; o frete (`frete_total`) soma os valores de frete dos itens; a participação
  do frete (`frete_ratio`) divide o frete pelo valor total do pedido; o valor do
  pedido (`valor_pedido`) soma preço e frete; etc.
* **Recorte:** pedidos **entregues**, com data de entrega e **com avaliação** —
  **95.824 pedidos** na base final.

### orders — Pedidos
(99.441 linhas × 8 colunas)

Tabela central: um registro por pedido.

| Atributo | Descrição |
|---|---|
| `order_id` | Identificador único do pedido. |
| `customer_id` | Identificador do cliente que realizou o pedido. |
| `order_status` | Situação do pedido (ex.: delivered, cancelled, unavailable). |
| `order_purchase_timestamp` | Data e hora da compra — origem das derivadas de sazonalidade (mês, dia da semana, hora). |
| `order_approved_at` | Data e hora em que o pagamento foi aprovado. |
| `order_delivered_carrier_date` | Data e hora da coleta da encomenda pela transportadora. |
| `order_delivered_customer_date` | Data e hora da entrega ao cliente. |
| `order_estimated_delivery_date` | Data e hora estimada para a entrega — base do cálculo do atraso. |

### customers — Clientes
(99.441 linhas × 5 colunas)

Tabela de clientes (dados cadastrados).

| Atributo | Descrição |
|---|---|
| `customer_id` | Identificador do cliente. |
| `customer_unique_id` | Identificador estável do cliente (sem pedido). |
| `customer_zip_code_prefix` | CEP do cliente — usado na distância cliente-vendedor. |
| `customer_city` | Cidade do cliente. |
| `customer_state` | UF do cliente (renomeada para uf_cliente na base final). |

### items — Itens dos pedidos
(112.650 linhas × 7 colunas)

Tabela de itens comprados em cada pedido.

| Atributo | Descrição |
|---|---|
| `order_id` | Identificador do pedido. |
| `order_item_id` | Identificador do item dentro do pedido. |
| `product_id` | Identificador do produto comprado. |
| `seller_id` | Identificador do vendedor que vendeu o item. |
| `shipping_limit_date` | Data limite de envio do item. |
| `price` | Preço unitário do item em R$. |
| `freight_value` | Frete do item em R$. |

### reviews — Avaliações de pedidos
(99.224 linhas × 7 colunas)

Tabela de avaliações (reviews) dos clientes.

| Atributo | Descrição |
|---|---|
| `review_id` | Identificador da avaliação. |
| `order_id` | Identificador do pedido avaliado. |
| `review_score` | Nota dada pelo cliente (1 a 5). |
| `review_comment_title` | Título do comentário (opcional). |
| `review_comment_message` | Mensagem do comentário (opcional). |
| `review_creation_date` | Data e hora de criação da avaliação. |
| `review_answer_timestamp` | Data e hora da resposta do vendedor. |

### payments — Pagamentos
(103.886 linhas × 5 colunas)

Tabela de pagamentos dos pedidos.

| Atributo | Descrição |
|---|---|
| `order_id` | Identificador do pedido. |
| `payment_sequential` | Sequencial do pagamento dentro do pedido. |
| `payment_type` | Tipo de pagamento (ex.: credit_card, boleto). |
| `payment_installments` | Número de parcelas. |
| `payment_value` | Valor pago em R$. |

### products — Produtos
(32.951 linhas × 9 colunas)

Tabela de produtos avaliados.

| Atributo | Descrição |
|---|---|
| `product_id` | Identificador do produto. |
| `product_category_name` | Categoria do produto. |
| `product_name_lenght` | Comprimento do nome do produto. |
| `product_description_lenght` | Comprimento da descrição do produto. |
| `product_photos_qty` | Quantidade de fotos do produto. |
| `product_weight_g` | Peso do produto em gramas. |
| `product_length_cm` | Comprimento do produto em cm. |
| `product_height_cm` | Altura do produto em cm. |
| `product_width_cm` | Largura do produto em cm. |

### sellers — Vendedores
(3.095 linhas × 4 colunas)

Tabela de vendedores.

| Atributo | Descrição |
|---|---|
| `seller_id` | Identificador do vendedor. |
| `seller_zip_code_prefix` | CEP do vendedor (usado na distância cliente-vendedor). |
| `seller_city` | Cidade do vendedor. |
| `seller_state` | UF do vendedor (renomeada para uf_vendedor na base final). |

### geolocation — Geolocalização
(1.000.163 linhas × 5 colunas)

Tabela de coordenadas geográficas.

| Atributo | Descrição |
|---|---|
| `geolocation_zip_code_prefix` | CEP associado à coordenada. |
| `geolocation_lat` | Latitude da coordenada. |
| `geolocation_lng` | Longitude da coordenada. |
| `geolocation_city` | Cidade da coordenada. |
| `geolocation_state` | UF da coordenada. |

### category_translation — Tradução de categorias
(71 linhas × 2 colunas)

Tradução dos nomes das categorias de produtos.

| Atributo | Descrição |
|---|---|
| `product_category_name` | Nome original da categoria. |
| `product_category_name_english` | Nome em inglês da categoria. |

## 2. Base final de análise — 95.824 pedidos × 42 colunas

A base final consolida as tabelas por `order_id` e adiciona atributos derivados.
Cada coluna tem um **papel**: feature operacional, alvo (insatisfação) ou coluna
descartada.

| Variável | Papel | Tipo | Descrição |
|---|---|---|---|
| order_id | descartada | identificador | Identificador único do pedido. |
| customer_id | descartada | identificador | Identificador do cliente. |
| order_status | descartada | categórica | Situação do pedido (pós-recorte: entregue). |
| order_purchase_timestamp | descartada | data | Data da compra (origem das derivadas de sazonalidade). |
| order_approved_at | descartada | data | Data da aprovação do pagamento (origem de aprovacao_h). |
| order_delivered_carrier_date | descartada | data | Data da coleta (origem de manuseio_dias). |
| order_delivered_customer_date | descartada | data | Data da entrega (origem de entrega_dias). |
| order_estimated_delivery_date | descartada | data | Data estimada (origem de atraso_dias). |
| customer_unique_id | descartada | identificador | Identificador estável do cliente (alta cardinalidade). |
| cep_cliente | descartada | identificador | CEP do cliente. |
| cidade_cliente | descartada | identificador | Cidade do cliente (redundante com uf_cliente + distancia_km). |
| uf_cliente | feature | categórica | Unidade federativa do cliente. |
| n_itens | feature | numérica | Número de itens do pedido. |
| preco_total | feature | numérica | Soma dos preços dos itens do pedido. |
| frete_total | feature | numérica | Soma dos fretes dos itens do pedido. |
| categoria | descartada | categórica | Categoria do produto (redundante com categoria_grupo). |
| peso_g | feature | numérica | Peso do produto principal (em gramas). |
| n_fotos | feature | numérica | Número de fotos do produto principal. |
| seller_zip | descartada | identificador | CEP do vendedor. |
| uf_vendedor | feature | categórica | Unidade federativa do vendedor. |
| n_parcelas | feature | numérica | Número de parcelas do pagamento. |
| valor_pago | descartada | numérica | Valor total pago (redundante com valor_pedido/preco_total). |
| pagamento_tipo | feature | categórica | Tipo de pagamento. |
| review_score | descartada | numérica | Nota dada pelo cliente (derivada do alvo). |
| review_creation_date | descartada | data | Data da avaliação (vazamento). |
| tem_comentario | descartada | booleana | Flag de comentário (vazamento). |
| comentario_chars | descartada | numérica | Tamanho do comentário (vazamento). |
| distancia_km | feature | numérica | Distância em km entre CEP cliente e CEP vendedor. |
| atraso_dias | feature | numérica | Atraso em dias (entrega real - estimada). |
| entrega_dias | feature | numérica | Tempo de entrega em dias (compra -> entrega). |
| manuseio_dias | feature | numérica | Tempo de coleta em dias (compra -> coleta). |
| aprovacao_h | feature | numérica | Horas de aprovação do pagamento (compra -> aprovação). |
| valor_pedido | redundante | numérica | Valor total do pedido (preço + frete). Identidade exata: preco_total + frete_total (removida por redundancia). |
| frete_ratio | feature | numérica | Participação do frete no valor do pedido. |
| preco_unitario_medio | redundante | numérica | Preço unitário médio do pedido. Derivada de preco_total / n_itens (removida por redundancia). |
| prazo_cumprido | descartada | booleana | Flag de prazo cumprido (redundante com atraso_dias). |
| faixa_atraso | descartada | categórica | Faixa agrupada de atraso (usada na EDA). |
| mes_compra | feature | categórica | Mês/ano da compra. |
| dia_semana_compra | feature | categórica | Dia da semana da compra. |
| hora_compra | feature | numérica | Hora da compra. |
| categoria_grupo | feature | categórica | Categoria de produto agrupada. |
| insatisfeito | alvo | binária | Insatisfação: review_score <= 2. |

* **Insatisfação:** nota dada pelo cliente ≤ 2 (notas 1-2) vs. notas 3-5; a variável
  `insatisfeito` é o alvo do modelo.
* **Odds ratio (OR):** razão de chances; mostra quantas vezes o risco de insatisfação
  aumenta (OR > 1) ou diminui (OR < 1) com um desvio padrão da feature.
* **AUC:** área sob a curva ROC — poder discriminativo do modelo.
* **Calibração:** compara probabilidade prevista com frequência real observada.
* **Recall:** TP / (TP + FN) — poder de identificar insatisfeitos.
* **Mann-Whitney / Qui-quadrado / V de Cramér / Spearman / Informação Mútua:** testes
  e medidas de associação.
* **PCA:** análise de componentes principais — diagnóstico de dimensionalidade.
* **SHAP:** valores de Shapley para explicar contribuições das features.

