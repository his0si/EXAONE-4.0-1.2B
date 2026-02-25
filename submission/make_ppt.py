#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""솔루션 PPT 생성 스크립트"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
DARK = RGBColor(0x1a, 0x1a, 0x2e)
BLUE = RGBColor(0x00, 0x66, 0xff)
WHITE = RGBColor(0xff, 0xff, 0xff)
LIGHT_GRAY = RGBColor(0xf0, 0xf0, 0xf5)
GREEN = RGBColor(0x00, 0xaa, 0x55)
RED = RGBColor(0xcc, 0x33, 0x33)
GRAY = RGBColor(0x66, 0x66, 0x66)

def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text(slide, left, top, width, height, text, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return tf

def add_para(tf, text, size=16, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    p = tf.add_paragraph()
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return p

def add_table(slide, left, top, width, height, rows, cols, data, header_color=BLUE):
    table_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height))
    table = table_shape.table

    for j in range(cols):
        cell = table.cell(0, j)
        cell.text = str(data[0][j])
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color

    for i in range(1, rows):
        for j in range(cols):
            cell = table.cell(i, j)
            cell.text = str(data[i][j])
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.alignment = PP_ALIGN.CENTER
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY
    return table

# ============================================================
# Slide 1: Title
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide, DARK)
add_text(slide, 1, 1.5, 11, 1.5, "EXAONE-4.0-1.2B 모델 경량화", size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(slide, 1, 3.2, 11, 1, "QuantizationModifier W8A8 + Context Window 축소", size=24, color=RGBColor(0x88, 0xbb, 0xff), align=PP_ALIGN.CENTER)
add_text(slide, 1, 5.0, 11, 0.8, "Public Score: 0.6295", size=28, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
add_text(slide, 1, 6.0, 11, 0.8, "모델 크기: 2.4GB → 1.4GB (42% 감소)", size=20, color=RGBColor(0xaa, 0xaa, 0xbb), align=PP_ALIGN.CENTER)

# ============================================================
# Slide 2: 과제 개요
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text(slide, 0.8, 0.4, 11, 0.8, "과제 개요", size=32, bold=True, color=BLUE)

tf = add_text(slide, 0.8, 1.4, 5.5, 5, "평가 지표", size=20, bold=True)
add_para(tf, "")
add_para(tf, "Score = 0.5 × PerfNorm + 0.5 × SpeedNorm", size=16, bold=True, color=BLUE)
add_para(tf, "")
add_para(tf, "• PerfNorm = 압축모델 성능 / 베이스모델 성능", size=14)
add_para(tf, "• SpeedNorm = 1 − (압축모델 spt / 베이스모델 spt)", size=14)
add_para(tf, "")
add_para(tf, "평가 환경", size=20, bold=True)
add_para(tf, "")
add_para(tf, "• 서버 GPU: L4 (24GB, Ada Lovelace)", size=14)
add_para(tf, "• vLLM 기반 추론", size=14)
add_para(tf, "• W8A8 → CutlassScaledMM 커널 사용", size=14)

tf2 = add_text(slide, 7, 1.4, 5.5, 5, "벤치마크", size=20, bold=True)
add_para(tf2, "")
add_para(tf2, "• KMMLU-Pro (MCQA, 2822문항)", size=14)
add_para(tf2, "• KMMLU-Redux (MCQA, 2587문항)", size=14)
add_para(tf2, "• Ko-LongRAG (QA, 600문항)", size=14)
add_para(tf2, "• KoMT-Bench (생성, 80문항)", size=14)
add_para(tf2, "")
add_para(tf2, "핵심 제약", size=20, bold=True)
add_para(tf2, "")
add_para(tf2, "• 1.2B 소형 모델 → INT4 양자화 시 품질 붕괴", size=14)
add_para(tf2, "• 로컬(RTX 4090)과 서버(L4)의 최적 방식 상이", size=14)
add_para(tf2, "• L4에서 W8A8만 유효 (W8A16, FP8 등 비효율)", size=14)

# ============================================================
# Slide 3: 최종 솔루션
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text(slide, 0.8, 0.4, 11, 0.8, "최종 솔루션: QuantizationModifier W8A8", size=32, bold=True, color=BLUE)

data = [
    ["항목", "설정"],
    ["양자화 도구", "llmcompressor QuantizationModifier"],
    ["양자화 스킴", "W8A8 (INT8 weight + INT8 activation)"],
    ["Weight", "per-channel, symmetric, static (minmax)"],
    ["Activation", "per-token, symmetric, dynamic"],
    ["제외 레이어", "embed_tokens, lm_head"],
    ["max_position_embeddings", "65536 → 16384"],
    ["캘리브레이션", "MANTA-1M, 2048 샘플, seq=2048"],
]
add_table(slide, 1.5, 1.5, 10, 4.5, len(data), 2, data)

tf = add_text(slide, 1, 6.2, 11, 1, "", size=14)
add_para(tf, "✓ Data-Free 양자화 (Hessian 계산 없음) → 캘리브레이션 과적합 방지", size=14, color=GREEN)
add_para(tf, "✓ Dynamic activation → 입력별 최적 스케일 → 품질 유지", size=14, color=GREEN)

# ============================================================
# Slide 4: 실험 결과 비교
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text(slide, 0.8, 0.4, 11, 0.8, "양자화 방식별 서버 제출 결과", size=32, bold=True, color=BLUE)

data = [
    ["방법", "서버 Score", "비고"],
    ["QuantizationModifier W8A8", "0.6295 ✅", "최종 선택"],
    ["GPTQ W8A8 + SparseGPT 10%", "0.6246", "비정형 스파시티"],
    ["GPTQ W8A8 (damp=0.02)", "0.6208", "Hessian 캘리브레이션"],
    ["GPTQ W8A16", "~0.52", "L4에서 느린 커널"],
    ["FP8 Dynamic", "~0.50", "FP8 커널 비효율"],
    ["GPTQ W4A16", "~0.49", "1.2B에 INT4 과도한 손실"],
]
add_table(slide, 1.5, 1.5, 10, 3.5, len(data), 3, data)

tf = add_text(slide, 1, 5.5, 11, 1.5, "", size=14)
add_para(tf, "핵심 발견: 단순한 min-max 양자화(QuantMod)가 복잡한 GPTQ보다 서버에서 우수", size=15, bold=True)
add_para(tf, "→ GPTQ의 Hessian 최적화가 캘리브레이션 데이터에 과적합되어 실제 평가에서 일반화 저하", size=14, color=GRAY)

# ============================================================
# Slide 5: 시도한 방법들
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text(slide, 0.8, 0.4, 11, 0.8, "시도했으나 효과 없었던 방법들", size=32, bold=True, color=BLUE)

data = [
    ["방법", "결과", "원인"],
    ["SFT (LoRA) → 양자화", "perf 하락", "1.2B에 MANTA SFT 오히려 성능 저하"],
    ["Post-quant SFT (norm만)", "변화 없음", "학습 파라미터 <5%로 불충분"],
    ["레이어 프루닝 (30→28L)", "perf 0.325→0.308", "속도 이득 대비 품질 손실 과도"],
    ["FFN 폭 프루닝 (25%)", "perf 0.273", "너무 공격적인 구조 변경"],
    ["Vocab 프루닝 (102k→80k)", "서버 0.48", "byte-fallback으로 토큰 수 폭증"],
    ["Self-KD + QuantMod", "perf 소폭 하락", "minmax에 KD 가중치 민감"],
    ["KV Cache FP8", "변화 없음", "L4에서 메모리가 병목 아님"],
]
add_table(slide, 0.8, 1.5, 11.5, 4.5, len(data), 3, data)

tf = add_text(slide, 1, 6.3, 11, 0.8, "", size=13)
add_para(tf, "총 110+ 모델 실험, 다양한 양자화/프루닝/파인튜닝 조합 탐색", size=14, color=GRAY)

# ============================================================
# Slide 6: 핵심 인사이트
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text(slide, 0.8, 0.4, 11, 0.8, "핵심 인사이트", size=32, bold=True, color=BLUE)

tf = add_text(slide, 0.8, 1.5, 5.5, 5, "1. 로컬 ≠ 서버", size=22, bold=True)
add_para(tf, "")
add_para(tf, "• RTX 4090: Marlin 커널 → W4A16 최적", size=14)
add_para(tf, "• L4: Cutlass INT8 커널 → W8A8 최적", size=14)
add_para(tf, "• 로컬 점수만 보고 제출하면 실패", size=14, color=RED)
add_para(tf, "")
add_para(tf, "2. 소형 모델의 양자화 한계", size=22, bold=True)
add_para(tf, "")
add_para(tf, "• 1.2B 파라미터 = INT4에 너무 민감", size=14)
add_para(tf, "• INT8 (W8A8)이 품질/속도 최적점", size=14)
add_para(tf, "• SFT/KD로 복구 시도도 한계", size=14)

tf2 = add_text(slide, 7, 1.5, 5.5, 5, "3. 단순함의 승리", size=22, bold=True)
add_para(tf2, "")
add_para(tf2, "• QuantMod(minmax) > GPTQ(Hessian)", size=14)
add_para(tf2, "• 복잡한 캘리브레이션이 과적합 유발", size=14)
add_para(tf2, "• Data-Free가 오히려 일반화 우수", size=14)
add_para(tf2, "")
add_para(tf2, "4. Context Window = 숨은 속도 변수", size=22, bold=True)
add_para(tf2, "")
add_para(tf2, "• 65536 → 16384 축소", size=14)
add_para(tf2, "• KV Cache 4배 절감 → 배치 효율 향상", size=14)
add_para(tf2, "• 벤치마크 최대 ~16k이므로 품질 무손실", size=14)

# ============================================================
# Slide 7: 모델 비교
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_text(slide, 0.8, 0.4, 11, 0.8, "모델 크기 비교", size=32, bold=True, color=BLUE)

data = [
    ["", "베이스 모델", "경량화 모델"],
    ["정밀도", "BF16", "INT8 (W8A8)"],
    ["모델 크기", "2.4 GB", "1.4 GB"],
    ["압축률", "-", "42% 감소"],
    ["max_position", "65536", "16384"],
    ["서버 Score", "0.5 (기준)", "0.6295"],
]
add_table(slide, 2.5, 1.5, 8, 3.5, len(data), 3, data)

tf = add_text(slide, 1, 5.5, 11, 1.5, "", size=16)
add_para(tf, "재현 방법:", size=18, bold=True)
add_para(tf, "pip install torch vllm llmcompressor transformers", size=14, color=GRAY)
add_para(tf, "python reproduce.py  # ~10초, 12GB GPU", size=14, color=GRAY)

# ============================================================
# Save
# ============================================================
out_path = os.path.join(os.path.dirname(__file__), "solution.pptx")
prs.save(out_path)
print(f"PPT saved to {out_path}")
