
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
REPORT_MD = ROOT / "reports" / "relatorio.md"
ETAPAS_MD = ROOT / "reports" / "relatorio_etapas.md"

ANALYSIS_CSV = OUTPUTS / "analysis_dataset.csv"
STATUS_CSV = OUTPUTS / "orders_status_dataset.csv"
MODEL_JSON = OUTPUTS / "resultados_modelo.json"
MODEL_JOBLIB = OUTPUTS / "modelo_insatisfacao.joblib"

for _d in (OUTPUTS, FIGURES, TABLES):
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
