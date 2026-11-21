#!/usr/bin/env python3
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from pathlib import Path
import re
here=Path(__file__).resolve().parent.parent
src=here/'Q1_Initial_Manuscript_Draft.md'
out=here/'Q1_Initial_Manuscript_Draft.docx'
doc=Document()
sec=doc.sections[0]
sec.page_height,sec.page_width=Inches(11.69),Inches(8.27)
sec.top_margin,sec.bottom_margin=Inches(.84),Inches(.77)
sec.left_margin,sec.right_margin=Inches(.90),Inches(.90)
normal=doc.styles['Normal'];normal.font.name='Aptos';normal.font.size=Pt(10.5);normal.font.color.rgb=RGBColor(34,43,56)
normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.14
for name,size,space_before,space_after in [('Title',16,4,11),('Heading 1',13,16,7),('Heading 2',11.3,12,5),('Heading 3',10.5,8,4)]:
 s=doc.styles[name];s.font.name='Aptos Display';s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor(29,64,99)
 s.paragraph_format.space_before=Pt(space_before);s.paragraph_format.space_after=Pt(space_after)
 s.paragraph_format.keep_with_next=True
# header
head=sec.header.paragraphs[0];head.text='UAV ELEPHANT FORECASTING  |  WORKING DRAFT — NOT FOR SUBMISSION'
head.style=doc.styles['Caption'];head.alignment=WD_ALIGN_PARAGRAPH.RIGHT
head.runs[0].font.size=Pt(8);head.runs[0].font.color.rgb=RGBColor(125,133,145)
# footer
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
foot.add_run('Q1 preparation • 16 November 2026   |   Page ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
for r in foot.runs:r.font.size=Pt(8);r.font.color.rgb=RGBColor(115,125,135)

def add_fmt(p, text):
 # split on **bold** and *italic*; no HTML parsing
 for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)',text):
  if not part:continue
  if part.startswith('**') and part.endswith('**'):
   r=p.add_run(part[2:-2]);r.bold=True
  elif part.startswith('*') and part.endswith('*'):
   r=p.add_run(part[1:-1]);r.italic=True
  else:p.add_run(part)

lines=src.read_text(encoding='utf8').splitlines()
for raw in lines:
 line=raw.strip()
 if not line:continue
 if line.startswith('# '):
  p=doc.add_paragraph(style='Title');add_fmt(p,line[2:]);p.alignment=WD_ALIGN_PARAGRAPH.LEFT
 elif line.startswith('## '):
  p=doc.add_paragraph(style='Heading 1');add_fmt(p,line[3:])
 elif line.startswith('### '):
  p=doc.add_paragraph(style='Heading 2');add_fmt(p,line[4:])
 elif re.match(r'^\d+\. ',line):
  p=doc.add_paragraph(style='List Number');add_fmt(p,re.sub(r'^\d+\. ','',line))
 elif line.startswith('- '):
  p=doc.add_paragraph(style='List Bullet');add_fmt(p,line[2:])
 else:
  p=doc.add_paragraph();add_fmt(p,line)
  if line.startswith('**WORKING MANUSCRIPT'):
   for r in p.runs:r.font.color.rgb=RGBColor(150,72,35);r.bold=True;r.font.size=Pt(9)
  if line.startswith('**Keywords:**'):p.paragraph_format.space_after=Pt(10)
  if line.startswith('[') and 'https://' in line:
   for r in p.runs:r.font.size=Pt(9)

doc.core_properties.title='Camera-Motion-Corrected Elephant Trajectory Forecasting — Working manuscript'
doc.core_properties.subject='Evidence-bounded Q1 research manuscript draft; not submission-ready'
doc.core_properties.author='Research team'
doc.save(out)
print(out)
