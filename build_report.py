"""Render the checkpoint report from report_content.json; no experiment is rerun."""
from pathlib import Path
import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Aldawah_COSC4371_Midsemester_Checkpoint.pdf"
for name, filename in [("TimesLocal", "times.ttf"), ("TimesLocal-Bold", "timesbd.ttf"), ("TimesLocal-Italic", "timesi.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / filename)))
pdfmetrics.registerFontFamily("TimesLocal", normal="TimesLocal", bold="TimesLocal-Bold", italic="TimesLocal-Italic", boldItalic="TimesLocal-Bold")
styles = {
    "title": ParagraphStyle("title", fontName="TimesLocal-Bold", fontSize=16, leading=18, spaceAfter=7),
    "heading": ParagraphStyle("heading", fontName="TimesLocal-Bold", fontSize=12, leading=14, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="TimesLocal", fontSize=10.5, leading=12.8, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="TimesLocal", fontSize=9, leading=10.7, spaceAfter=4),
    "caption": ParagraphStyle("caption", fontName="TimesLocal-Italic", fontSize=9.4, leading=11, spaceAfter=5),
}
blocks = json.loads((ROOT / "report_content.json").read_text(encoding="utf-8"))
story = []
for block in blocks:
    kind = block["kind"]
    if kind == "break":
        story.append(PageBreak())
    elif kind == "table":
        font = block.get("font", 9)
        cell = ParagraphStyle("cell", fontName="TimesLocal", fontSize=font, leading=font + 1.5)
        head = ParagraphStyle("head", parent=cell, fontName="TimesLocal-Bold")
        data = [[Paragraph(value, head if i == 0 else cell) for value in row] for i, row in enumerate(block["rows"])]
        table = Table(data, colWidths=block["widths"], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([table, Spacer(1, 5)])
    elif kind == "image":
        image_path = ROOT / block["path"]
        if not image_path.exists() and block["path"].startswith("repository/"):
            image_path = ROOT / block["path"].removeprefix("repository/")
        story.append(Image(str(image_path), width=block["width"], height=block["height"], hAlign="CENTER"))
    else:
        story.append(Paragraph(block["text"], styles[kind]))

def footer(canvas, doc):
    canvas.setFont("TimesLocal", 9)
    canvas.drawString(43, 26, "COSC 4371 | Project 15 | Zainulabdeen Aldawah")
    canvas.drawRightString(569, 26, str(doc.page))

SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=43, rightMargin=43, topMargin=36, bottomMargin=40,
                  title="Project 15: Midsemester Checkpoint", author="Zainulabdeen Aldawah",
                  subject="Prompt-injection detection across datasets",
                  creator="[5] OpenAI. ChatGPT/Codex, October 29, 2026. https://chatgpt.com/. AI assistance was used for formatting and verification purposes.").build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
