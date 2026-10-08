#!/usr/bin/env python3
"""Pipeline de análise da insatisfação de clientes (Olist).

Etapas:
  load   -> carga dos CSVs + feature engineering  (outputs/)
  eda    -> análise exploratória                  (reports/figures, reports/tables)
  tests  -> testes estatísticos de hipóteses      (reports/tables)
  model  -> classificação + interpretação         (outputs/, reports/)
  report -> relatório final em Markdown           (reports/relatorio.md)

Uso:
  python run_pipeline.py                 # todas as etapas, em ordem
  python run_pipeline.py --etapa eda     # uma etapa específica
  python run_pipeline.py --etapas load eda
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config import (
    ANALYSIS_CSV, MODEL_JSON, REPORT_MD, STATUS_CSV, FIGURES, TABLES,
)

ETAPAS = ["load", "eda", "tests", "model", "report"]

PREREQUISITOS = {
    "eda": [ANALYSIS_CSV, STATUS_CSV],
    "tests": [ANALYSIS_CSV],
    "model": [ANALYSIS_CSV],
    "report": [ANALYSIS_CSV, MODEL_JSON],
}


def _verificar_prerequisitos(etapa: str) -> None:
    faltando = [p for p in PREREQUISITOS.get(etapa, []) if not p.exists()]
    if faltando:
        nomes = ", ".join(p.name for p in faltando)
        sys.exit(f"[pipeline] etapa '{etapa}' requer: {nomes}. Rode a etapa 'load' antes.")


def executar(etapa: str) -> None:
    _verificar_prerequisitos(etapa)
    print(f"\n{'=' * 70}\n[etapa: {etapa}]\n{'=' * 70}")
    inicio = time.time()
    if etapa == "load":
        from src.features import salvar_bases
        salvar_bases()
    elif etapa == "eda":
        from src.eda import executar_etapa_eda
        executar_etapa_eda()
    elif etapa == "tests":
        from src.statistics_tests import executar_etapa_testes
        executar_etapa_testes()
    elif etapa == "model":
        from src.model import executar_etapa_modelo
        executar_etapa_modelo()
    elif etapa == "report":
        from src.report import gerar_relatorio
        gerar_relatorio()
    print(f"[etapa: {etapa}] concluída em {time.time() - inicio:.1f}s")


def main() -> None:
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
    print(f"[pipeline] figuras:   {FIGURES}/  ({len(list(FIGURES.glob('*.png')))} arquivos)")
    print(f"[pipeline] tabelas:   {TABLES}/  ({len(list(TABLES.glob('*.csv')))} arquivos)")


if __name__ == "__main__":
    main()
