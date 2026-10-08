"""Configuração central do projeto: caminhos, constantes e estilo."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (pipeline de scripts)

# ---------------------------------------------------------------- caminhos --
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
REPORT_MD = ROOT / "reports" / "relatorio.md"

ANALYSIS_CSV = OUTPUTS / "analysis_dataset.csv"       # pedidos entregues + review
STATUS_CSV = OUTPUTS / "orders_status_dataset.csv"     # todos os pedidos (status)
MODEL_JSON = OUTPUTS / "resultados_modelo.json"        # métricas/interpretação do modelo
MODEL_JOBLIB = OUTPUTS / "modelo_insatisfacao.joblib"  # melhor modelo salvo

for _d in (OUTPUTS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- constantes --
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

SCORE_COL = "review_score"
TARGET = "insatisfeito"          # review_score <= NEGATIVE_MAX
NEGATIVE_MAX = 2                 # notas 1-2 = insatisfeito; 3-5 = satisfeito

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_creation_date",
]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "categoria_grupo",
    "uf_cliente",
    "uf_vendedor",
    "pagamento_tipo",
    "mes_compra",
    "dia_semana_compra",
]

# ------------------------------------------------------------------ estilo --
import seaborn as sns  # noqa: E402

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
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(mpl_rc)

PALETTE_NOTAS = {1: "#b2182b", 2: "#ef8a62", 3: "#fddbc7", 4: "#67a9cf", 5: "#2166ac"}
COR_NEG, COR_POS = "#b2182b", "#2166ac"
