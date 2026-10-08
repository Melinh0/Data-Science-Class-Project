#!/usr/bin/env python3
"""Gera os slides do projeto (apresentacao_olist.pptx + PDF).

Uso:
  python3 apresentacoes/gerar_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

AQUI = Path(__file__).resolve().parent
SAIDA_PPTX = AQUI / "apresentacao_olist.pptx"

AZUL = RGBColor(0x21, 0x66, 0xAC)
AZUL_ESC = RGBColor(0x14, 0x3D, 0x66)
VERMELHO = RGBColor(0xB2, 0x18, 0x2B)
CINZA = RGBColor(0x44, 0x44, 0x44)
CINZA_CLARO = RGBColor(0x77, 0x77, 0x77)
FUNDO_CAIXA = RGBColor(0xEF, 0xF4, 0xFA)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)
FONTE = "Calibri"


def _tf(shape):
    tf = shape.text_frame
    tf.word_wrap = True
    return tf


def texto(slide, l, t, w, h, corpo, tamanho=20, cor=CINZA, negrito=False,
          alinhamento=PP_ALIGN.LEFT, italico=False):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = _tf(box)
    p = tf.paragraphs[0]
    p.alignment = alinhamento
    r = p.add_run()
    r.text = corpo
    r.font.size = Pt(tamanho)
    r.font.color.rgb = cor
    r.font.bold = negrito
    r.font.italic = italico
    r.font.name = FONTE
    return box


def bullet(slide, l, t, w, h, destaque, resto, tamanho=19):
    """Bullet com trecho em negrito seguido de texto normal."""
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = _tf(box)
    p = tf.paragraphs[0]
    p.space_after = Pt(10)
    r0 = p.add_run()
    r0.text = "▪  "
    r0.font.size = Pt(tamanho)
    r0.font.color.rgb = AZUL
    r0.font.bold = True
    r0.font.name = FONTE
    if destaque:
        r1 = p.add_run()
        r1.text = destaque
        r1.font.size = Pt(tamanho)
        r1.font.color.rgb = AZUL_ESC
        r1.font.bold = True
        r1.font.name = FONTE
    r2 = p.add_run()
    r2.text = resto
    r2.font.size = Pt(tamanho)
    r2.font.color.rgb = CINZA
    r2.font.name = FONTE
    return box


def cabecalho(slide, titulo):
    barra = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                   Inches(13.333), Inches(0.16))
    barra.fill.solid()
    barra.fill.fore_color.rgb = AZUL
    barra.line.fill.background()
    texto(slide, 0.7, 0.35, 12.0, 0.8, titulo, tamanho=32, cor=AZUL_ESC,
          negrito=True)


def rodape(slide, num):
    texto(slide, 0.7, 6.95, 9.0, 0.4,
          "Olist Brazilian E-commerce · Kaggle · análise: 95.824 pedidos",
          tamanho=12, cor=CINZA_CLARO)
    texto(slide, 12.4, 6.95, 0.6, 0.4, str(num), tamanho=12, cor=CINZA_CLARO,
          alinhamento=PP_ALIGN.RIGHT)


def cartao(slide, l, t, w, h, numero, rotulo, cor=AZUL):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(l), Inches(t),
                                 Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = FUNDO_CAIXA
    shp.line.color.rgb = cor
    shp.line.width = Pt(1.25)
    tf = _tf(shp)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = numero
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = cor
    r.font.name = FONTE
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = rotulo
    r2.font.size = Pt(14)
    r2.font.color.rgb = CINZA
    r2.font.name = FONTE


# --------------------------------------------------------------- slides -----
def montar() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    branco = prs.slide_layouts[6]

    # ---- 1. Capa ----------------------------------------------------------
    s = prs.slides.add_slide(branco)
    faixa = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                               Inches(13.333), Inches(7.5))
    faixa.fill.solid()
    faixa.fill.fore_color.rgb = AZUL_ESC
    faixa.line.fill.background()
    texto(s, 1.0, 1.5, 11.3, 0.6, "PROJETO DE CIÊNCIA DE DADOS", tamanho=18,
          cor=RGBColor(0x9E, 0xC5, 0xE8), negrito=True)
    texto(s, 1.0, 2.2, 11.3, 2.0,
          "Quais fatores estão associados\nà insatisfação dos clientes?",
          tamanho=42, cor=BRANCO, negrito=True)
    texto(s, 1.0, 4.4, 11.3, 1.0,
          "Análise exploratória, testes estatísticos e modelo preditivo "
          "no e-commerce brasileiro (Olist)", tamanho=22,
          cor=RGBColor(0xCD, 0xDC, 0xEA))
    texto(s, 1.0, 6.3, 11.3, 0.5,
          "Dataset: Kaggle — olistbr/brazilian-ecommerce", tamanho=15,
          cor=RGBColor(0x9E, 0xC5, 0xE8), italico=True)

    # ---- 2. Base de dados -------------------------------------------------
    s = prs.slides.add_slide(branco)
    cabecalho(s, "Base de dados")
    bullet(s, 0.7, 1.45, 8.0, 1.0, "Olist Brazilian E-commerce (Kaggle): ",
           "marketplace brasileiro com pedidos de 2017–2018.")
    bullet(s, 0.7, 2.35, 8.0, 1.2,
           "9 arquivos relacionados por order_id: ",
           "pedidos, clientes, itens, avaliações, pagamentos, produtos, "
           "vendedores, geolocalização e categorias.")
    bullet(s, 0.7, 3.55, 8.0, 1.0, "Avaliação do cliente: ",
           "nota de 1 a 5 por pedido (variável-alvo da análise).")
    bullet(s, 0.7, 4.45, 8.0, 1.2, "Recorte analítico: ",
           "95.824 pedidos entregues com avaliação — 1 linha = 1 pedido, "
           "com atraso, frete, categoria, UF e pagamento.")
    cartao(s, 9.2, 1.6, 3.4, 1.3, "9", "arquivos CSV")
    cartao(s, 9.2, 3.2, 3.4, 1.3, "95.824", "pedidos na análise")
    cartao(s, 9.2, 4.8, 3.4, 1.3, "12,8%", "com nota 1–2 (insatisfeitos)",
           cor=VERMELHO)
    rodape(s, 2)

    # ---- 3. Objetivo e pergunta -------------------------------------------
    s = prs.slides.add_slide(branco)
    cabecalho(s, "Objetivo e pergunta de pesquisa")
    texto(s, 0.7, 1.5, 11.9, 0.4, "OBJETIVO DO TRABALHO", tamanho=15,
          cor=CINZA_CLARO, negrito=True)
    texto(s, 0.7, 1.95, 11.9, 1.1,
          "Identificar os fatores operacionais associados à insatisfação dos "
          "clientes (atraso, frete, categoria, região etc.) e traduzir as "
          "evidências em recomendações de melhoria operacional.",
          tamanho=22, cor=CINZA)

    caixa = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7),
                               Inches(3.5), Inches(11.9), Inches(2.4))
    caixa.fill.solid()
    caixa.fill.fore_color.rgb = FUNDO_CAIXA
    caixa.line.color.rgb = VERMELHO
    caixa.line.width = Pt(2)
    tf = _tf(caixa)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r0 = p.add_run()
    r0.text = "PERGUNTA QUE BUSCA RESPONDER\n"
    r0.font.size = Pt(15)
    r0.font.bold = True
    r0.font.color.rgb = VERMELHO
    r0.font.name = FONTE
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r1 = p2.add_run()
    r1.text = ("“Quais fatores estão associados à insatisfação dos clientes "
               "e como essas evidências podem apoiar decisões de melhoria "
               "operacional?”")
    r1.font.size = Pt(24)
    r1.font.bold = True
    r1.font.color.rgb = AZUL_ESC
    r1.font.name = FONTE
    rodape(s, 3)

    return prs


def main() -> None:
    prs = montar()
    prs.save(SAIDA_PPTX)
    print(f"slides -> {SAIDA_PPTX}")
    print("Para PDF: soffice --headless --convert-to pdf "
          f"--outdir {AQUI} {SAIDA_PPTX}")


if __name__ == "__main__":
    main()
