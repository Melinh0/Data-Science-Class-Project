from __future__ import annotations
import argparse
import os
import sys
import time
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from src.config import (
    ANALYSIS_CSV, DICTIONARY_MD, ETAPAS_MD, MODEL_JSON, MODELOS_JSON, PCA_JSON, PREP_JSON, REDUCAO_CSV,
    REPORT_MD, SELECAO_JSON, STATUS_CSV, FIGURES, TABLES,
)
ETAPAS = ["load", "prep", "eda", "tests", "select", "pca", "model", "reducao", "modelos",
          "etapas", "report", "relatorio_modelos", "dicionario"]
PREREQUISITOS = {
    "prep": [ANALYSIS_CSV, STATUS_CSV],
    "eda": [ANALYSIS_CSV, STATUS_CSV],
    "tests": [ANALYSIS_CSV],
    "select": [ANALYSIS_CSV],
    "pca": [ANALYSIS_CSV],
    "model": [ANALYSIS_CSV],
    "reducao": [ANALYSIS_CSV, SELECAO_JSON, PCA_JSON],
    "modelos": [ANALYSIS_CSV, SELECAO_JSON],
    "relatorio_modelos": [MODELOS_JSON],
    "etapas": [ANALYSIS_CSV, STATUS_CSV, MODEL_JSON, PREP_JSON, SELECAO_JSON, PCA_JSON, REDUCAO_CSV,
               TABLES / "tab05_testes_hipotese.csv",
               TABLES / "tab09_metricas_modelos.csv"],
    "report": [ANALYSIS_CSV, MODEL_JSON, SELECAO_JSON, PCA_JSON, REDUCAO_CSV],
    "dicionario": [ANALYSIS_CSV],
}
def _verificar_prerequisitos(etapa: str) -> None:
    faltando = [p for p in PREREQUISITOS.get(etapa, []) if not p.exists()]
    if faltando:
        nomes = ", ".join(p.name for p in faltando)
        sys.exit(f"[pipeline] etapa '{etapa}' requer: {nomes}. Rode as etapas anteriores (a partir de 'load').")
def executar(etapa: str) -> None:
    _verificar_prerequisitos(etapa)
    print(f"\n{'=' * 70}\n[etapa: {etapa}]\n{'=' * 70}")
    inicio = time.time()
    if etapa == "load":
        from src.features import salvar_bases
        salvar_bases()
    elif etapa == "prep":
        from src.qualidade import executar_etapa_prep
        executar_etapa_prep()
    elif etapa == "eda":
        from src.eda import executar_etapa_eda
        executar_etapa_eda()
    elif etapa == "tests":
        from src.statistics_tests import executar_etapa_testes
        executar_etapa_testes()
    elif etapa == "select":
        from src.feature_selection import executar_etapa_selecao
        executar_etapa_selecao()
    elif etapa == "pca":
        from src.pca_diagnostico import executar_pca
        executar_pca()
    elif etapa == "model":
        from src.model import executar_etapa_modelo
        executar_etapa_modelo()
    elif etapa == "reducao":
        from src.reducao import executar_etapa_reducao
        executar_etapa_reducao()
    elif etapa == "modelos":
        from src.modelos_supervisionados import executar_etapa_modelos
        executar_etapa_modelos()
    elif etapa == "relatorio_modelos":
        from src.relatorio_modelos import gerar_relatorio_modelos
        gerar_relatorio_modelos()
    elif etapa == "etapas":
        from src.etapas import gerar_relatorio_etapas
        gerar_relatorio_etapas()
    elif etapa == "report":
        from src.report import gerar_relatorio
        gerar_relatorio()
    elif etapa == "dicionario":
        from src.report import gerar_dicionario
        gerar_dicionario()
    print(f"[etapa: {etapa}] concluída em {time.time() - inicio:.1f}s")
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--etapa", choices=ETAPAS, help="executar uma única etapa")
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS,
                        help="sequência de etapas a executar")
    args = parser.parse_args()
    sequencia = [args.etapa] if args.etapa else (args.etapas or ETAPAS)
    total = time.time()
    for etapa in sequencia:
        executar(etapa)
    print(f"\n[pipeline] concluído em {time.time() - total:.1f}s")
    print(f"[pipeline] relatório: {REPORT_MD}")
    print(f"[pipeline] etapas:    {ETAPAS_MD}")
    if DICTIONARY_MD.exists():
        print(f"[pipeline] dicionário: {DICTIONARY_MD}")
    print(f"[pipeline] figuras:   {FIGURES}/  ({len(list(FIGURES.glob('*.png')))} arquivos)")
    print(f"[pipeline] tabelas:   {TABLES}/  ({len(list(TABLES.glob('*.csv')))} arquivos)")
if __name__ == "__main__":
    main()
