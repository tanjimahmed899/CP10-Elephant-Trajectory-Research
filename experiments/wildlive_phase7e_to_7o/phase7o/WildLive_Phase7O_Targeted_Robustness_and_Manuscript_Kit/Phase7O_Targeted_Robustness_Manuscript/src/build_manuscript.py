#!/usr/bin/env python3
"""Integrate a separately scoped WildLive supplementary report into the supplied main manuscript.

Does not modify original 2/5-second BIRDSAI/WildBox numerical results or figures.
"""
from pathlib import Path
import sys
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE

KIT=Path(__file__).resolve().parents[1]
BASE=KIT.parent
SOURCE=Path(sys.argv[1]).expanduser() if len(sys.argv)>1 else BASE/'Final Demo paper.docx'
if not SOURCE.is_file(): raise FileNotFoundError(f'Original supervisor manuscript required: {SOURCE}; run python src/build_manuscript.py /path/to/original.docx')
OUT=KIT/'manuscript'/'Final_Demo_paper_Phase7O_integrated.docx'
FIG_DIR=KIT/'results'/'figures'
DOC=Document(SOURCE)

def find_exact(text):
    hits=[p for p in DOC.paragraphs if p.text.strip()==text]
    if len(hits)!=1:raise ValueError(f'expected 1 paragraph for {text}; got {len(hits)}')
    return hits[0]

def newpara(anchor,txt='',style='Normal'):
    p=anchor.insert_paragraph_before(style=style)
    p.add_run(txt)
    return p

def heading(anchor,title):
    return newpara(anchor,title,'Heading 2')

def paragraph(anchor,txt):
    p=newpara(anchor,txt)
    p.paragraph_format.space_after=Pt(6)
    return p

def figure(anchor,filename,caption):
    p=anchor.insert_paragraph_before(style='Normal')
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG_DIR/filename),width=Inches(6.15))
    cap=newpara(anchor,caption, 'Figure Caption')
    return p,cap

def add_table_before(anchor,headers,rows):
    t=DOC.add_table(rows=1,cols=len(headers))
    t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,name in enumerate(headers):
        t.rows[0].cells[i].text=name
        t.rows[0].cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        t.rows[0].cells[i]._tc.get_or_add_tcPr().append(_shade('E9EFF4'))
        for rr in t.rows[0].cells[i].paragraphs[0].runs:rr.bold=True;rr.font.size=Pt(8)
    for row in rows:
        cells=t.add_row().cells
        for c,value in zip(cells,row):
            c.text=str(value)
            for rr in c.paragraphs[0].runs:rr.font.size=Pt(8)
    t._tbl.getparent().remove(t._tbl)
    anchor._p.addprevious(t._tbl)
    return t

def _shade(color):
    s=OxmlElement('w:shd');s.set(qn('w:fill'),color);return s

# Short abstract qualification without rewriting the supplied historical results.
p=find_exact('Keywords: aerial thermal imaging; WildBox; BIRDSAI; elephant trajectory forecasting; affine camera-motion correction; Transformer; conformal prediction; ByteTrack; virtual-boundary simulation; retrospective validation.')
p.text=p.text[:-1]+'; WildLive; held-out image registration.'
abstracts=[x for x in DOC.paragraphs if x.style.name=='Normal' and x.text.startswith('The results provide auditable evidence for the predictive utility')]
if len(abstracts)==1:
    abstracts[0].add_run(' A separate retrospective WildLive supplement identifies scene-dependent fixed-expert ranking and improved held-out background feature alignment using a projective mapping; this is not a verified gain in stabilized-input trajectory forecasting.')

# Distinct supplementary data population, placed under Data.
a=find_exact('4. Experimental methods')
heading(a,'3.4. Supplementary WildLive imagery and evidence units (Phases 7E–7O)')
paragraph(a,'To diagnose robustness outside the BIRDSAI and WildBox populations, we audited four elephant-labelled WildLive UAV video sequences [18]. These 3840 × 2160 RGB annotation sequences contain 3,008 annotated frames and 7,230 elephant bounding boxes. The available source ZIPs support 12,948 overlapping trajectory windows: 20 observed frame-indexed positions followed by 20 or 50 future frames. Frames cannot be converted into seconds without independently established sampling cadence. These normalized image-plane metrics are not pooled with the 10-Hz, 2-/5-second BIRDSAI/WildBox tasks.')
paragraph(a,'The DJI_0204_video1 and DJI_0204_video2 sequences have substantial observed background-scene overlap and are not treated as independent capture groups. DJI_0207 and DJI_0204_video2 were development material, DJI_0204_video1 was examined as a related-scene stress test, and DJI_0161_video1 was previously evaluated and is no longer untouched. Video names and feature-matching non-detections do not establish flight independence. The WildLive analysis is an exploratory supplement, not a separately verified external-flight holdout.')

# Method.
a=find_exact('5. Experimental results')
heading(a,'4.9. Supplementary held-out registration and annotation-QA exposure (Phase 7O)')
paragraph(a,'We retained the previous eight targeted consecutive-frame RGB pairs, six from DJI_0207 and two from DJI_0204_video2. An animal-exclusion mask expanded each box by a factor of 1.4. ORB/Hamming ratio-tested background matches were formed at 1536 × 864; a spatially stratified 75% training subset in 4 × 4 grid cells was used to fit translation-only, RANSAC similarity, or RANSAC projective homography mappings. Remaining matched background features were never used to estimate these transforms. The prespecified-for-this-code held-out metric is median reprojection distance in resized-image pixels. This pair selection and QA threshold are retrospective; results are not estimates of general population performance.')
paragraph(a,'The homography was mapped to native 3840 × 2160 image pixels before comparing transformed earlier-frame elephant bounding-box centres to later-frame annotations. Five track-specific transitions with at least 30 native-pixel residual were flagged for manual annotation, pose and identity review; flagging is not evidence of actual error. An archived prediction window was marked exposed if it contained any such adjacent-frame event within its history or future interval for the same track. Model ADE values were read from frozen archived errors, never re-estimated from newly generated predictions. The full unfiltered scores remain the primary descriptions.')

# Results, figure, two tables.
a=find_exact('6. Discussion')
heading(a,'5.9. Supplementary WildLive model transfer and targeted geometry sensitivity (Phase 7O)')
paragraph(a,'Archived 20/50-frame WildLive forecasting results exhibit recording-dependent expert rankings: Transformer wins the 20-frame ADE comparison for DJI_0207 and DJI_0161_video1; CV-OLS wins both horizons for DJI_0204_video2 and the 20-frame horizon for DJI_0204_video1; Stationary wins the remaining 50-frame cases. The locked adaptive gate improves relative to CV-OLS on DJI_0161_video1 but is marginally worse on related-scene DJI_0204_video1. Its superiority to the best fixed expert is not established. No final locked adaptive result exists for the two development recordings, so those missing results are not interpolated.')
cap=newpara(a,'Supplementary Table S1. Retrospective camera-registration comparison on eight targeted sampled image pairs; median of each pair’s median held-out matched-background reprojection errors at 1536 × 864.', 'Table Caption')
add_table_before(a,['2D transformation','Median error (resized px)','Worst pair median (resized px)'],[['Translation only','1.2626','1.4654'],['Similarity','0.9501','1.4894'],['Projective homography','0.7520','0.9332']])
paragraph(a,'Projective homography gave lower median error than the other two transforms on all eight selected frame pairs, but this result measures background alignment accuracy, not true egomotion, animal velocity, or corrected forecast quality. Limited camera baselines, surface parallax, moving trees, and a shared image field can violate the geometric assumptions of any single global mapping [19].')
figure(a,'01_heldout_registration_model_comparison.png','Supplementary Figure S1. Within-pair held-out keypoint reprojection error for translation, partial-affine similarity and projective homography. Bar heights are resized-image pixels, not camera-corrected forecasting errors.')
cap=newpara(a,'Supplementary Table S2. Descriptive WildLive DJI_0207 forecast-window overlap with five high-residual track-specific image pairs. The groups are overlapping and selected retrospectively.', 'Table Caption')
add_table_before(a,['Forecast horizon','Exposed windows','Other windows','Exposed winner','Other winner'],[['20 future frames','59','2,428','Stationary','Transformer'],['50 future frames','89','2,308','Stationary','Stationary']])
paragraph(a,'Five post-registration native-image centre residuals exceeded 30 px: DJI_0207 ID 2 at 389→390 (59.04 px), 400→401 (65.89 px), 401→402 (67.86 px), and ID 0 at 859→860 (39.33 px) and 860→861 (34.19 px). These are candidates for annotation/pose/camera-parallax inspection, not confirmed annotation mistakes. In the 59 event-exposed 20-frame windows, Stationary had ADE 0.010026 versus Transformer 0.010205; in 2,428 other windows Transformer had ADE 0.004384 versus Stationary 0.004700. None of these conditional differences are independent-flight statistics. All results are retrospective and full-sample scores are unchanged.')
figure(a,'02_QA_exposure_sensitivity.png','Supplementary Figure S2. Archived mean ADE by whether a prediction window intersects one or more high-residual observed image pairs. Exposure defines a diagnostic subgroup; no data were removed and no new model predictions were generated.')

# Discussion and validity.
a=find_exact('7. Limitations, validity threats, and ethics')
heading(a,'6.6. Cross-dataset contrast: stabilization diagnostics versus forecasting gains')
paragraph(a,'The main Stage 4 experiment and the supplementary WildLive experiment test different scientific propositions. On BIRDSAI TestReal, a matched future-target ablation evaluates learned forecasting from estimated affine-corrected versus raw observed histories under the study’s established cadence. In WildLive, no full-length stabilized observation or future path was generated for the archived fixed-expert results. The Phase 7O projection comparison only checks sparse out-of-fit background-feature geometry. It cannot be used as an additional proof of Stage 4’s learned-model improvement, nor can its pixel residuals be combined with 2-/5-second ADE. The variable expert rankings and high scale-floor activation instead reinforce the need to test preprocessing transfer and annotation quality under explicit source-group controls.')
a=find_exact('8. Reproducibility and proposed confirmatory study')
heading(a,'7.5. Supplementary WildLive camera and annotation threats')
paragraph(a,'The held-out background keypoints were drawn from the same image pairs as the fit points and can be correlated; eight preselected pairs do not represent a random sample of recording conditions. A homography registers a dominant image plane and may be biased by parallax at animal locations or moving backgrounds. Five high post-registration centre residuals are unadjudicated; pose changes, annotation jitter, true displacement, and local geometric misregistration remain alternative explanations. Treating 59 or 89 exposed overlapping windows as independent tests would be invalid. Flight events, FPS, camera pose, wildlife position in physical units and a new independent confirmation recording are not established. The targeted 30-pixel analysis was post hoc and does not justify exclusion or model retuning.')

# Add the new evidence to conclusion without modifying earlier conclusions.
a=find_exact('References')
paragraph(a,'In a separate, supplementary WildLive frame-indexed analysis, we additionally observed non-uniform fixed-expert performance across four elephant-labelled sequences. Among eight targeted RGB image pairs, projective homography provided the lowest held-out background alignment error of three tested 2D models, while five high-residual animal-centre transitions were flagged for further review. These observations support image-registration quality assurance and methodological failure analysis but do not establish camera-corrected forecasting gains or adaptive superiority on independent capture events.')

# New sources before the existing Appendix A.
a=find_exact('Appendix A. Evidence-to-claim reporting guardrails')
paragraph(a,'[18] Nguyen Ngoc, D., Richardson, T., Watson, M., Meier, K., Kline, J. M., Reid, S., Maalouf, G., Hine, D., Mirmehdi, M., and Burghardt, T. (2026). WildLive [Data set]. University of Bristol. https://doi.org/10.5523/bris.ewnuoroebuae20vei0cu6zu69')
paragraph(a,'[19] Ljubičić, R., Strelnikova, D., Perks, M. T., Eltner, A., Peña-Haro, S., Pizarro, A., Dal Sasso, S. F., Scherling, U., Vuono, P., and Manfreda, S. (2021). A comparison of tools and techniques for stabilising unmanned aerial system (UAS) imagery for surface flow observations. Hydrology and Earth System Sciences, 25, 5105–5132. https://doi.org/10.5194/hess-25-5105-2021')

# Update only metadata/covers from the supplied 21 November draft. Preserve page-number fields.
DOC.paragraphs[1].text='REVISED MANUSCRIPT DRAFT  |  STAGE 8 + SUPPLEMENTARY WILDLIVE PHASE 7O'
DOC.paragraphs[2].text='23 November 2026  ·  For supervisor review; journal format and bibliography require final verification'
for section in DOC.sections:
    for fp in section.footer.paragraphs:
        for run in fp.runs:
            if '21 November 2026' in run.text:
                run.text=run.text.replace('21 November 2026','23 November 2026')

OUT.parent.mkdir(parents=True,exist_ok=True)
DOC.save(OUT)
print('Saved',OUT,'paragraphs',len(DOC.paragraphs),'tables',len(DOC.tables))
for item in ('3.4. Supplementary WildLive imagery and evidence units (Phases 7E–7O)',
             '4.9. Supplementary held-out registration and annotation-QA exposure (Phase 7O)',
             '5.9. Supplementary WildLive model transfer and targeted geometry sensitivity (Phase 7O)',
             '6.6. Cross-dataset contrast: stabilization diagnostics versus forecasting gains',
             '7.5. Supplementary WildLive camera and annotation threats'):
    assert sum(p.text==item for p in DOC.paragraphs)==1
