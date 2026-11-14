# DATASET_REVIEW — Problemas da base antes do pré-processamento

Registro dos problemas de qualidade identificados nos 9 CSVs do Olist (`data/raw/`) **antes** de qualquer
transformação. Serve de base para o relatório: cada decisão de pré-processamento deve apontar para um
problema desta lista ("que problema nos dados justifica esta transformação?").

**Legenda de status**
- ✅ **Confirmado**: medido no notebook ou em verificação própria.
- 🔎 **Hipótese**: explicação provável, ainda não verificada.
- ⏳ **A verificar**: aparece na auditoria, precisa ser reproduzido no notebook.

**Coluna "Decisão"**: preenchida só depois de investigar. "—" = ainda não decidido.

---

## 1. Tabelas e dimensões

| Tabela | Linhas | Colunas |
|---|---|---|
| orders | 99.441 | 8 |
| customers | 99.441 | 5 |
| items | 112.650 | 7 |
| reviews | 99.224 | 7 |
| payments | 103.886 | 5 |
| products | 32.951 | 9 |
| sellers | 3.095 | 4 |
| geo | 1.000.163 | 5 |
| cat_tr | 71 | 2 |

Não há variável-alvo "oficial" no dataset. O projeto usa `review_score` (nota 1–5) como possível alvo.
Nenhuma variável ordinal que exija preservar ordem foi identificada até aqui.

### Recorte aplicado (funil de pedidos)

| Etapa | Pedidos | Perda | Justificativa |
|---|---|---|---|
| Todos | 99.441 | — | — |
| `order_status = delivered` | 96.478 | 2.963 | Só pedidos finalizados têm prazo de entrega real |
| + com data de entrega | 96.470 | 8 | Remove a inconsistência I1 |
| + com avaliação | 95.824 | 646 (0,67%) | Sem nota não há alvo. Pequena perda, mas é um viés de seleção a citar nas limitações |

### Divisão treino/teste

| Conjunto | Pedidos | Insatisfação |
|---|---|---|
| Treino | 76.658 (80,0%) | 12,81% |
| Teste | 19.166 (20,0%) | 12,81% |

Estratificada pelo alvo e **agrupada por `customer_unique_id`** (0 clientes nos dois conjuntos), `random_state=42`.
Resolve D5. Tudo que aprende dos dados (imputação, escala, seleção supervisionada, PCA) é ajustado só no treino.

---

## 2. Tipos de dados

| ID | Problema | Evidência | Status | Decisão |
|---|---|---|---|---|
| T1 | Colunas de data lidas como texto (`str`) | `order_*_date`, `review_*_date`, `shipping_limit_date` | ✅ | — (converter para datetime) |
| T2 | CEPs lidos como inteiro: os que começam com 0 perdem o zero | `int64` em `customers`, `sellers` e `geo` | ✅ | **Sem ação**: os três têm o mesmo tipo, então a junção é consistente. |
| T3 | Nome de coluna com erro de grafia na fonte | `product_name_lenght`, `product_description_lenght` | ✅ | — (informativo) |

---

## 3. Valores ausentes

| ID | Tabela.coluna | Ausentes | Status | Explicação / Decisão |
|---|---|---|---|---|
| A1 | orders.`order_approved_at` | 160 (0,16%) | ✅ | Ausência **com motivo** em 146 (`canceled` 141, `created` 5). Os outros **14 são `delivered` sem aprovação** (ver I4). — |
| A2 | orders.`order_delivered_carrier_date` | 1.783 (1,79%) | ✅ | Ausência **com motivo** em 1.781 (`unavailable` 609, `canceled` 550, `invoiced` 314, `processing` 301, `created` 5, `approved` 2). Os outros **2 são `delivered` sem postagem** (ver I4). — |
| A3 | orders.`order_delivered_customer_date` | 2.965 (2,98%) | ✅ | Ausência **com motivo** em 2.957: pedidos não finalizados (`shipped` 1.107, `canceled` 619, `unavailable` 609, `invoiced` 314, `processing` 301, `created` 5, `approved` 2). Não imputar. Os outros **8 são `delivered` sem data** (ver I1). — |
| A4 | reviews.`review_comment_title` | 87.656 (88,34%) | 🔎 | Ausência provavelmente significa "cliente não comentou", não erro. Texto não entra na análise numérica. — |
| A5 | reviews.`review_comment_message` | 58.247 (58,70%) | 🔎 | Idem A4. — |
| A6 | products.`product_category_name`, `product_name_lenght`, `product_description_lenght`, `product_photos_qty` | 610 (1,85%) cada | ✅ | São **as mesmas 610 linhas** nas 4 colunas: um único problema (produtos sem cadastro de metadados). — |
| A7 | products.`product_weight_g` e dimensões (cm) | 2 cada | ✅ | Casos isolados. — |

| A9 | Ausentes após agregar itens e pagamentos (95.824 pedidos) | `peso_total_g` 22; `n_fotos_medio` 1.324; `n_parcelas` 1; `pagamento_tipo` 1 | ✅ | Fotos: ligado aos 610 produtos sem cadastro (A6). 1 pedido sem registro de pagamento. Imputar no pipeline (mediana / moda do treino). |
| A8 | CEP de cliente/vendedor sem coordenada em `geo` | 278 clientes (0,28%) e 7 vendedores (0,23%) | ✅ | **476 pedidos** (0,5%) ficam com `distancia_km` ausente. Imputar com a mediana do treino (no pipeline). |

Observação de verificação própria (base final, 95.824 pedidos): todos os 1.349 pedidos com `n_fotos`
ausente tinham categoria `sem_categoria`, e esse grupo tem insatisfação maior (16,6% contra 12,7%).
Ou seja, a ausência pode **não ser aleatória**. ⏳ Reproduzir no notebook.

---

## 4. Duplicatas e unicidade

| ID | Problema | Evidência | Status | Decisão |
|---|---|---|---|---|
| D1 | `review_id` repetido | 814 duplicados em `reviews` (sem linhas idênticas) | ✅ | — |
| D2 | Pedidos com mais de uma avaliação | 547 pedidos (1.098 linhas); em **202 pedidos (37%) as notas diferem** | ✅ | **Decisão: manter a avaliação mais recente** (última opinião do cliente). Sensibilidade medida: a classe (nota ≤ 2) muda em apenas 96 de 98.673 pedidos (0,097%) entre "mais antiga" e "mais recente"; a escolha é quase irrelevante, mas fica documentada. O projeto antigo usava a mais antiga, sem justificar. |
| D3 | Linhas completamente duplicadas em `geo` | 261.831 de 1.000.163 (26%) | ✅ | **Tratado**: `drop_duplicates()` → 738.332 linhas únicas; depois mediana por CEP. |
| D4 | Sem duplicatas em `orders`, `customers`, `products`, `sellers` | 0 linhas e 0 chaves repetidas | ✅ | Nada a tratar |
| D5 | Clientes recorrentes: mesmo `customer_unique_id` em vários pedidos | 3.077 pedidos repetidos na base final (≈3%) | ⏳ | — (limitação: observações não independentes entre treino e teste) |

| I1 | Pedidos `delivered` sem data de entrega | 8 pedidos | ✅ | — (inconsistência real; sem a data não há prazo de entrega, então tendem a sair do recorte) |
| I2 | Pedidos `canceled` que têm data de entrega | 6 de 625 cancelados | ✅ | — (não filtrar só por `order_status`; exigir também a data de entrega) |
| I4 | Pedidos `delivered` com datas intermediárias ausentes | 14 sem `order_approved_at`; 2 sem `order_delivered_carrier_date` | ✅ | Inconsistência real, mas minúscula (≈0,015% dos pedidos). Vão gerar `aprovacao_h` e `manuseio_dias` ausentes; **imputar com a mediana do treino** (no pipeline) e registrar. |
| I3 | Categorias de produto sem tradução em inglês | `pc_gamer`, `portateis_cozinha_e_preparadores_de_alimentos` | ✅ | — (usar o nome em português quando faltar a tradução, para não confundir com "sem categoria") |

---

## 5. Valores incoerentes ou extremos (da auditoria, ainda não reproduzidos no notebook)

Medidos na base final de 95.824 pedidos entregues com avaliação. ⏳ Precisam ser reproduzidos
sobre as tabelas brutas.

| ID | Problema | Evidência | Domínio / suspeita | Decisão |
|---|---|---|---|---|
| X1 | Data de postagem anterior à compra | **164 pedidos** com `manuseio_dias` < 0. Destes, só **2 passam de −1 dia** (−171 d e −4,2 d); 132 são diferenças de minutos/horas; 30 são menores que 7 min | Impossível no processo logístico. Hipótese anterior (134 vs 164 por arredondamento da auditoria) **confirmada**: 134 = negativos < −0,005 dia. | **Decisão: todos os 164 → ausente** (0,17% dos pedidos), imputados depois no pipeline. Regra de domínio, sem parâmetro aprendido dos dados, então pode ser aplicada antes da divisão treino/teste sem vazamento. |
| X2 | Distância cliente–vendedor maior que o território do Brasil | 4 casos > 4.400 km (máximo 8.678 km, ex.: PR→SC) | Causa confirmada: `geo` tem **33 coordenadas fora do Brasil** | **Corrigido**: coordenadas fora do retângulo do Brasil (lat −33,75 a 5,27; lng −73,99 a −34,79) são descartadas antes de calcular a mediana por CEP. Resultado: 0 casos > 4.400 km, máximo 3.398 km. |
| X3 | Distância igual a zero | 22 casos | Cliente e vendedor no mesmo prefixo de CEP → mesmo centro | **Legítimo, mantido.** Limitação: a precisão é de CEP, não de endereço. |
| X4 | Peso do produto igual a zero | 4 produtos na tabela `products` (a auditoria contou 6 *pedidos* na base final) | Peso 0 g é fisicamente impossível | **Decisão: transformar em ausente** antes de agregar (feito no notebook). Imputar depois, no pipeline. |
| X5 | Valores extremos de preço/frete/prazo | `preco_total` até R$ 13.440; frete até R$ 1.795; entrega até 208 dias | Coerentes com outras variáveis (ver X9): frete alto ↔ peso alto e distância longa; entrega ~200 d ↔ atraso ~170 d | **Mantidos (legítimos).** A escala é tratada no pré-processamento (log, escalonamento), não removendo linhas. |
| X6 | Avaliação criada antes da entrega | 8.281 reviews (8,6%) com `review_creation_date` < data de entrega | Não sabemos o motivo. A nota pode não refletir a experiência de entrega. **Não concluir sem mais evidência.** | — |
| X7 | Atributos do "item principal" vs. agregados do pedido | `peso_g`, `n_fotos`, categoria vinham do item mais caro; `n_itens`, `preco_total`, `frete_total` agregavam o pedido | Em ~10% dos pedidos (2+ itens) o peso representava só um item | **Corrigido**: `peso_total_g` = soma dos itens; `n_fotos_medio` = média; `categoria_principal` = categoria do item mais caro (nome explícito). Pesos totalmente ausentes continuam ausentes (`sum(min_count=1)`), e não viram 0. |
| X8 | Parcelas de pagamento iguais a 0 | 2 pedidos com `n_parcelas` = 0 | Mínimo possível é 1 parcela | **Decisão: → ausente** (2 pedidos). |
| X9 | Peso total muito alto | `peso_total_g` máx. 184.400 g (184 kg), com `n_itens` = 10 e frete R$ 355 | ~18 kg por item, frete proporcional | **Legítimo, mantido.** |
| X10 | Postagem antes da aprovação do pagamento | 1.343 pedidos (1,4%) | **Incerto**: pode ser prática normal (vendedor despacha antes da confirmação). Não sabemos o significado exato. | **Mantido**, sem tratamento. Limitação a registrar. |
| X11 | Aprovação muito demorada | `aprovacao_h` até 741 h. Por tipo: `credit_card` mediana 0,3 h e só 32 de 72.331 (0,04%) acima de 7 dias; `boleto` mediana 29 h, 25 de 19.048 (0,13%) acima de 7 dias | Raro em todos os tipos; o motivo dos casos de cartão é desconhecido (talvez revisão antifraude) | **Mantido** como extremo; a assimetria é tratada com log. Limitação a registrar. |

---

## 6. Redundância entre variáveis (da auditoria)

| ID | Problema | Evidência | Decisão |
|---|---|---|---|
| R1 | Identidade exata: `valor_pedido = preco_total + frete_total` | Treino: maior diferença 4e-13; Spearman `valor_pedido`↔`preco_total` = 0,99 | **Remover `valor_pedido`** (sem perda de informação). |
| R2 | Variáveis derivadas de outras: `frete_ratio`, `preco_unitario_medio` | Spearman (treino): `preco_unitario_medio`↔`preco_total` 0,95; ↔`valor_pedido` 0,93; `frete_ratio`↔`preco_unitario_medio` −0,82; ↔`preco_total` −0,78; ↔`valor_pedido` −0,70 | Critério: ρ ≥ 0,90 ou identidade exata. **Remover `preco_unitario_medio`** (= `preco_total/n_itens`, ambos mantidos). **Manter `frete_ratio`** (ρ máx. 0,82; razão não linear que expressa o peso do frete); caso limítrofe, reavaliar nas cargas do PCA e na seleção. O limiar 0,90 é julgamento. |
| R3 | `entrega_dias` e `atraso_dias` | Spearman no treino abaixo de 0,70 (Pearson na auditoria: 0,60) | **Mantidas**: correlacionadas, mas não redundantes (prazo prometido ≠ tempo real). |

**Baixa variabilidade (treino):** nenhuma variável constante ou quase constante. O maior valor dominante é `n_itens` com 90,1% (os 10% restantes são pedidos com 2+ itens, informação real); depois `pagamento_tipo` 75,4% e `uf_vendedor` 70,9%. Nenhuma sai por esse critério.

---

## 7. Assimetria e escala

| ID | Problema | Evidência (assimetria) | Decisão |
|---|---|---|---|
| S1 | Distribuições muito assimétricas, com caudas longas | Assimetria no treino: `frete_total` 13,5; `preco_total` 10,1; `n_itens` 7,2; `peso_total_g` 6,8; `manuseio_dias` 4,7; `aprovacao_h` 4,2; `entrega_dias` 3,8 | **Decisão**: `log1p` nessas 7 (regra: assimetria ≥ 3 no treino e sem valores negativos). Resultado: `frete_total` 0,5; `preco_total` 0,3; `peso_total_g` 0,5; `entrega_dias` 0,0; `manuseio_dias` 0,6; `aprovacao_h` 1,0. **Exceção: `n_itens` só caiu de 7,2 para 4,0** (contagem com 90% de valor 1; o log não a simetriza). Mantida; limitação. `atraso_dias` (tem negativos), `frete_ratio`, `distancia_km`, `n_parcelas`, `n_fotos_medio` e `hora_compra` só padronizam. |
| S2 | `hora_compra` é cíclica (23h é vizinha de 0h) | 0 a 23 | — |

---

## 8. Cardinalidade e codificação

| ID | Problema | Evidência | Decisão |
|---|---|---|---|
| C1 | Variáveis categóricas com muitos níveis | `categoria_grupo` 53; `mes_compra` 23; `uf_cliente` 27; `uf_vendedor` 22 | — |
| C2 | One-hot gera a maior parte da dimensionalidade | Níveis no treino: `categoria_principal` 74, `uf_cliente` 27, `uf_vendedor` 22, `mes_compra` 12, `dia_semana_compra` 7, `pagamento_tipo` 4. Colunas one-hot na comparação: 146 sem agrupar; 129 (`min_frequency=30`); 80 (0,5%); 67 (1%). | **Decisão: `min_frequency=0.005`** (nível precisa de ~383 pedidos no treino). **Resultado no pipeline real: 79 colunas one-hot + 13 numéricas = 92 colunas** (a comparação deu 80 por artefato de `fillna("ausente")`: o único pedido sem `pagamento_tipo` está no treino e criou um nível extra; **confirmado**: 3 colunas `infrequent`, sem a de pagamento). As categóricas geram mais colunas que todas as numéricas juntas. |
| C3 | Níveis muito raros e mês-ano como categoria | `2016-09` com 1 pedido (100% insatisfeito); `2016-12` com 1 pedido; `ano-mês` tem 23 níveis | **Decisão**: usar o **mês do ano (1–12)** em vez de ano-mês. Captura sazonalidade, evita níveis raros e não deixa o modelo "decorar" meses específicos. |

---

## 9. Possível vazamento de dados (data leakage)

| ID | Variável | Motivo | Decisão |
|---|---|---|---|
| L1 | `review_score`, `review_creation_date`, comentários | Derivam da própria avaliação (alvo) | **Excluídos das features.** `review_score` só serve para definir o alvo `insatisfeito` (nota ≤ 2). |
| L2 | Identificadores: `order_id`, `customer_id`, `customer_unique_id`, CEPs, `vendedor_principal` | Não generalizam e identificam a linha | **Excluídos.** `customer_unique_id` é usado apenas como grupo na divisão treino/teste. |
| L3 | Datas brutas (`order_*`) | Viram variáveis derivadas (atraso, tempo de entrega etc.) | **Excluídas**; substituídas por `atraso_dias`, `entrega_dias`, `manuseio_dias`, `aprovacao_h`, `hora_compra`, `dia_semana_compra`, `mes_compra`. |
| L4 | `order_status` é constante | Todos os 95.824 pedidos são `delivered` (consequência do recorte) | **Excluída** (atributo constante, sem informação). |

---

## 10. Pendências de investigação

1. ~~Cruzar A1 e A2 com `order_status`~~ (feito: ausência com motivo, mais 14 + 2 inconsistências em `delivered`).
2. ~~Confirmar se os 610 produtos de A6 são as mesmas linhas~~ (feito, confirmado).
3. ~~Regra de escolha da avaliação (D2)~~ (feito: mais recente; impacto 0,097%).
4. Reproduzir X1–X7 e R1–R3 no notebook, sobre as tabelas brutas.
5. ~~Checar o formato dos CEPs (T2)~~ (feito: todos `int64`, junção consistente; 0,28% de clientes sem coordenada).
6. Preencher a coluna "Decisão" ao longo da parte de pré-processamento.
