from pathlib import Path
import matplotlib
matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
DOCS = ROOT / "docs"
REPORT_MD = DOCS / "relatorio.md"
ETAPAS_MD = DOCS / "relatorio_etapas.md"
DICTIONARY_MD = DOCS / "dicionario_dados.md"
ANALYSIS_CSV = OUTPUTS / "analysis_dataset.csv"
STATUS_CSV = OUTPUTS / "orders_status_dataset.csv"
MODEL_JSON = OUTPUTS / "resultados_modelo.json"
MODEL_JOBLIB = OUTPUTS / "modelo_insatisfacao.joblib"
RAW_SHAPES_CSV = OUTPUTS / "tabelas_brutas.csv"
PREP_DIR = OUTPUTS / "prep"
PREP_JSON = PREP_DIR / "prep_info.json"
SELECAO_JSON = OUTPUTS / "selecao_atributos.json"
PCA_JSON = OUTPUTS / "pca_info.json"
REDUCAO_CSV = TABLES / "tab16_comparacao_representacoes.csv"
MODELOS_JSON = OUTPUTS / "resultados_modelos_supervisionados.json"
MODELOS_JOBLIB = OUTPUTS / "modelo_supervisionado_final.joblib"
RELATORIO_MODELOS_MD = DOCS / "relatorio_modelos.md"
CUSTO_FN_FP = 3
for _d in (OUTPUTS, FIGURES, TABLES, PREP_DIR, DOCS):
    _d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORE_COL = "review_score"
TARGET = "insatisfeito"
NEGATIVE_MAX = 2
DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_creation_date",
]
NUMERIC_CANDIDATAS = [
    "atraso_dias",
    "entrega_dias",
    "manuseio_dias",
    "aprovacao_h",
    "frete_total",
    "preco_total",
    "valor_pedido",
    "frete_ratio",
    "n_itens",
    "preco_unitario_medio",
    "distancia_km",
    "n_parcelas",
    "peso_g",
    "n_fotos",
    "hora_compra",
]
REDUNDANTES = {
    "valor_pedido": "identidade exata: preco_total + frete_total",
    "preco_unitario_medio": "derivada: preco_total / n_itens (r > 0,9 com preco_total)",
}
NUMERIC_FEATURES = [c for c in NUMERIC_CANDIDATAS if c not in REDUNDANTES]
PRE_ENTREGA_EXCLUIR = ["atraso_dias", "entrega_dias"]
GRUPOS_FEATURES = {
    "Prazo e tempo de entrega": ["atraso_dias", "entrega_dias", "manuseio_dias"],
    "Pedido (itens, preço, frete)": ["n_itens", "preco_total", "frete_total", "frete_ratio"],
    "Produto": ["categoria_grupo", "peso_g", "n_fotos"],
    "Geografia": ["uf_cliente", "uf_vendedor", "distancia_km"],
    "Pagamento e aprovação": ["pagamento_tipo", "n_parcelas", "aprovacao_h"],
    "Calendário": ["mes_compra", "dia_semana_compra", "hora_compra"],
}
DISTANCIA_MAX_KM = 4500
MIN_N_RANKING = 300
CATEGORICAL_FEATURES = [
    "categoria_grupo",
    "uf_cliente",
    "uf_vendedor",
    "pagamento_tipo",
    "mes_compra",
    "dia_semana_compra",
]
ROTULOS_PT = {
    "atraso_dias": "Atraso na entrega (dias)",
    "entrega_dias": "Tempo de entrega (dias)",
    "manuseio_dias": "Tempo de postagem (dias)",
    "aprovacao_h": "Aprovação do pagamento (horas)",
    "frete_total": "Frete total (R$)",
    "preco_total": "Preço total dos itens (R$)",
    "valor_pedido": "Valor do pedido (R$)",
    "frete_ratio": "Frete / valor do pedido",
    "n_itens": "Nº de itens no pedido",
    "preco_unitario_medio": "Preço médio por item (R$)",
    "distancia_km": "Distância cliente-vendedor (km)",
    "n_parcelas": "Nº de parcelas",
    "peso_g": "Peso do produto (g)",
    "n_fotos": "Nº de fotos do anúncio",
    "hora_compra": "Hora da compra",
    "categoria_grupo": "Categoria do produto",
    "uf_cliente": "UF do cliente",
    "uf_vendedor": "UF do vendedor",
    "pagamento_tipo": "Tipo de pagamento",
    "mes_compra": "Mês da compra",
    "dia_semana_compra": "Dia da semana da compra",
    "review_score": "Nota da avaliação",
    "insatisfeito": "Insatisfeito (nota 1-2)",
}
def rotulo(col: str) -> str:
    base = col.replace("sombra_", "")
    return ROTULOS_PT.get(base, base.replace("_", " "))
import seaborn as sns
sns.set_theme(style="whitegrid", palette="viridis")
mpl_rc = {
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.frameon": False,
}
import matplotlib.pyplot as plt
plt.rcParams.update(mpl_rc)
PALETTE_NOTAS = {1: "#b2182b", 2: "#ef8a62", 3: "#fddbc7", 4: "#67a9cf", 5: "#2166ac"}
COR_NEG, COR_POS = "#b2182b", "#2166ac"
