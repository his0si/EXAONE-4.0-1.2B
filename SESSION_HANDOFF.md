# EXAONE-4.0-1.2B 압축 프로젝트 — 세션 핸드오프

**최종 업데이트**: 2026-02-08  
**목적**: 다음 세션에서 바로 이어서 진행할 수 있도록 전체 상황·진행 내용·앞으로 할 일을 한 문서에 정리.

---

## 1. 프로젝트 목표와 제약

### 목표
- **모델**: EXAONE-4.0-1.2B
- **제출물**: Hugging Face 표준 형식 (가중치 + config만) → `submit.zip` 내 `model/`
- **평가**: 고정 vLLM(v0.14.1) 서빙, L4 GPU(22.4GiB VRAM) / 로컬 개발: RTX 5060 Ti 16.6GiB

### 채점 공식
```
Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)
PerfNorm = Perf_model / Perf_baseline
SpeedNorm = 1 - (Time_model/Tokens_model) / (Time_baseline/Tokens_baseline)
```
- **Time/Tokens**: 모든 벤치마크에서의 **순수 생성 시간**과 **생성 토큰 수** (대회 기준).
- 로컬에서는 `05_rank_and_package.py`가 **bench_sec_per_token** 기준으로 SpeedNorm 계산 (전용 짧은 프롬프트 속도가 아님).

### 벤치마크 (동일 가중)
- KMMLU-Pro, KMMLU-Redux (객관식)
- Ko-LongRAG (QA, F1)
- KoMT-Bench (Judge; 로컬에서는 스킵, 대회는 GPT-4 Judge)

### 제약
- vLLM/서버 코드 수정 불가, **제출은 모델 파일만**.
- Kernel fusion 등 엔진 쪽 최적화는 불가. **모델 구조·양자화만**으로 속도/성능 조절.

---

## 2. 디렉터리 구조 (경로: `/home/lgaimers/final/0208`)

```
0208/
├── configs/lanes.yaml       # 레인 정의, vLLM/학습/벤치 설정
├── scripts/
│   ├── 00_setup_check.py    # 환경·GPU 검증
│   ├── 01_prepare_data.py   # MANTA-1M + 벤치 데이터
│   ├── 02_finetune_sft.py   # SFT(LoRA) / Post-SFT / KD
│   ├── 03_quantize.py       # GPTQ/FP8 (llmcompressor)
│   ├── 04_eval_vllm.py      # vLLM 벤치 + 속도 측정 (--gemini-key 지원)
│   ├── 05_rank_and_package.py  # Score_proxy, 순위, top-N 패키징
│   ├── run_all.py           # 전체 오케스트레이터 (--force 지원)
│   ├── aggressive_prune_kd.py  # BI 기반 레이어/FFN 프루닝 + KD + (선택) FP8
│   ├── run_eval_detached.sh # nohup 평가 (세션 끊겨도 유지)
│   └── check_eval_status.sh   # 실행 중 프로세스·로그·results 요약
├── data/manta/              # train.json, train_50k.json, val.json
├── checkpoints/             # pruned22_kd, pruned26_kd, pruned28_kd 등
├── models/                  # lane01~12, pruned22_kd_fp8static 등
├── results/
│   ├── baseline.json
│   ├── <lane_id>/metrics.json
│   ├── summary.csv, summary.md
├── logs/                    # run_eval_detached 로그 (eval_*.log, eval_*.pid)
├── submissions/             # top1~3, submit_*.zip
└── SESSION_HANDOFF.md       # 이 문서
```

**베이스 모델 경로**: `configs/lanes.yaml`의 `project.base_model` → `./base_model` (즉 `/home/lgaimers/final/0208/base_model`).

---

## 3. 지금까지 진행한 작업 요약

### 3.1 파이프라인 스크립트
- **00~05, run_all.py**: 레인 빌드 → 평가 → 순위 → 패키징. `--force`로 기존 결과 덮어쓰기.
- **04_eval_vllm.py**  
  - MCQA 파싱·Ko-LongRAG 프롬프트·max_tokens·F1 등 수정으로 **로컬 정확도** 보정.  
  - **SpeedNorm**: 대회와 맞추기 위해 **bench_sec_per_token** 사용 (전체 벤치 생성 시간/토큰).  
  - KoMT-Bench: **Gemini Judge** 옵션 추가 (`--gemini-key`). (현재 키 429로 미사용)
- **05_rank_and_package.py**: `bench_sec_per_token` 기준으로 SpeedNorm 계산하도록 수정됨.

### 3.2 레인 구성 (configs/lanes.yaml)
- 12개 레인: quant only, SFT→quant, quant→post-SFT, structural(prune+KD+quant).
- lane12: **FP8 Static**.
- vLLM: max_model_len 16384, max_tokens 16384, apply_chat_template true.

### 3.3 구조적 압축 (0.65 목표 대비)
- **aggressive_prune_kd.py**  
  - Block Influence(BI)로 레이어 중요도 계산 → 레이어 제거.  
  - (선택) FFN width 프루닝.  
  - MANTA-1M으로 KD → (선택) llmcompressor로 FP8 Static.  
- **실험한 것**: 22레이어 프루닝 → KD(4,750샘플, 5에폭) → FP8 Static → `models/pruned22_kd_fp8static`.  
- **결과**: 성능 크게 하락(perf 0.206), Ko-LongRAG에서 과도한 토큰 생성으로 오히려 느려짐. **KD 데이터/에폭 부족**으로 판단.

### 3.4 로컬 vs 대회 점수 보정
- 확인된 대회 점수: lane01(W4A16) 0.4678, FP8_DYNAMIC 0.4936.
- 로컬 대비 보정 계수 사용: W4A16 약 0.82, FP8 약 0.92 (KoMT 미반영 등).
- **FP8_STATIC** 제출: 대회 점수 **0.4929** (13분 12초).

### 3.5 기타
- **토크나이저 경고**(Mistral regex): EXAONE 토크나이저에 대한 무해한 경고로 무시해도 됨.
- **세션 끊김 대비**: `scripts/run_eval_detached.sh`(nohup), `scripts/check_eval_status.sh`로 상태 확인.

---

## 4. 전체 모델 결과 및 예상(실제) 점수 (한 번에 보기)

상세 표는 **`results/전체모델_결과_예상_실제점수.md`** 에 있으며, 요약만 아래에 둡니다.

| Lane | 설명 | perf | bench_s/tok | Score_proxy(로컬) | 예상 대회 | 실제 대회 |
|------|------|------|-------------|-------------------|-----------|-----------|
| lane01_gptq_w4a16 | GPTQ W4A16 basic | 0.3286 | 0.000236 | 0.5724 | - | **0.4678** |
| lane02_gptq_w4a16_damp | GPTQ W4A16 dampened | 0.3172 | 0.000235 | 0.5567 | ~0.45 | - |
| lane03_gptq_w8a16 | GPTQ W8A16 | 0.3226 | 0.000269 | 0.5027 | ~0.40 | - |
| lane04_fp8_dynamic | FP8 Dynamic | 0.3263 | 0.000252 | 0.5395 | - | **0.4936** |
| lane05_sft_fp16 | LoRA SFT → FP16 | 0.2981 | 0.000237 | 0.5237 | ~0.42 | - |
| lane06_sft_gptq_w4a16 | LoRA SFT → GPTQ W4A16 | 0.2862 | 0.000217 | 0.5421 | - | **0.3202** |
| lane07_sft_gptq_w4a16_damp | LoRA SFT → W4 damp | 0.2909 | 0.000220 | 0.5438 | ~0.44 | - |
| lane08_w4a16_postsft | W4A16 → Post-SFT | 0.3247 | 0.000240 | 0.5591 | ~0.45 | - |
| lane09_w4a16_damp_postsft | W4 damp → Post-SFT | 0.3154 | 0.000238 | 0.5484 | ~0.44 | - |
| lane10_prune28_kd_w4a16 | Prune 28L → KD → W4 | 0.3095 | 0.000262 | 0.4954 | ~0.40 | - |
| lane11_prune26_kd_w4a16 | Prune 26L → KD → W4 | 0.2564 | 0.000244 | 0.4468 | ~0.36 | - |
| lane12_fp8_static | FP8 Static | 0.3303 | 0.000227 | 0.5915 | - | **0.4929** |
| fp8_dynamic_submitted | FP8 Dynamic (제출용) | 0.3247 | 0.000251 | 0.5389 | - | **0.4936** |
| pruned22_kd_fp8static | Prune 22L → KD → FP8 (실험) | 0.2061 | 0.000289 | 0.2872 | ~0.23 | 미제출 |
| unknown | unknown | 0.3302 | 0.000225 | 0.595 | ~0.48 | - |

- **실제 대회 점수** 확인된 것: lane01 **0.4678**, lane04/FP8_DYNAMIC **0.4936**, lane12/FP8_STATIC **0.4929**, lane06 **0.3202** (SFT+W4 제출 시 KoMT 하락 반영).
- **예상 대회 점수**는 로컬 Score_proxy × 0.80 추정(미제출 레인). KoMT·환경 차이로 오차 가능.

---

## 5. 현재 결과 요약 (상세)

### 5.1 베이스라인 (results/baseline.json)
- perf_aggregate: **0.3256**
- bench_sec_per_token: **0.000273**
- kmmlu_pro 0.3512, kmmlu_redux 0.3711, ko_longrag 0.2544, komt_bench null

### 5.2 lane12_fp8_static (로컬 최고 성능 유지 후보)
- perf_aggregate: **0.3303**
- bench_sec_per_token: **0.000227** (베이스 대비 약 17% 빠름)
- 로컬 Score_proxy는 summary.csv/summary.md 기준; 대회에서는 FP8_STATIC 0.4929로 확인됨

### 5.3 pruned22_kd_fp8static (실험)
- perf_aggregate: **0.2061** (하락)
- bench_sec_per_token: **0.000289**
- Ko-LongRAG 토큰 362,097(베이스 30,314 대비 과다 생성), F1 0.1253

### 5.4 대회에 제출해 본 것
- lane06: 약 0.32 (낮음).
- FP8_DYNAMIC: 0.4936.
- **FP8_STATIC**: **0.4929** (13분 12초).

---

## 6. 앞으로 할 일

### 6.1 단기 (바로 이어서 가능)
1. **05_rank_and_package 재실행**  
   - `python scripts/05_rank_and_package.py`  
   - 최신 baseline + 모든 results/*/metrics.json 반영한 순위·summary 갱신.
2. **제출용 모델 결정**  
   - 현재 대회 점수 기준으로는 **FP8 Static(lane12)**가 안정적.  
   - 원하면 top-3 패키징 후 submit_*.zip 재생성.
3. **Gemini Judge (KoMT)**  
   - 429 해소 후 `--gemini-key <KEY>`로 평가 시 KoMT 점수 포함 가능.  
   - 키는 코드/문서에 넣지 말고 환경변수 등으로만 전달 권장.

### 6.2 0.65 목표 대비 (중장기)
1. **Pruning + KD 강화**  
   - KD 데이터: `data/manta/train_50k.json` 사용 (이미 있음).  
   - aggressive_prune_kd.py에서 num_samples 20k~50k, 에폭 5~10 등으로 재실험.  
   - 22L보다 완만한 26L/28L부터 재시도할 수 있음.
2. **FFN width 프루닝 (Phase 2)**  
   - aggressive_prune_kd.py에 FFN intermediate 4096→3072 등 옵션 있음.  
   - 24L + FFN 3072 + KD + FP8 조합 실험.
3. **순수 양자화만**  
   - 성능 0.5 근처 유지 + 30~50% 속도 향상 목표면, lane12 FP8 Static 유지·재제출 또는 W8A16 등 추가 실험.

### 6.3 유지보수
- 새 레인 추가 시 `configs/lanes.yaml` 수정 후 `run_all.py` 또는 개별 스크립트로 빌드·평가.
- 긴 평가/학습은 `run_eval_detached.sh` 또는 `nohup python scripts/...`로 실행 후 `check_eval_status.sh`로 확인.

---

## 7. 자주 쓰는 명령어

```bash
# 환경
conda activate lgaimers
cd /home/lgaimers/final/0208

# 베이스라인 평가 (한 번만)
python scripts/04_eval_vllm.py --model ../base_model --baseline

# 단일 레인 평가 (덮어쓰기)
python scripts/04_eval_vllm.py --model models/lane12_fp8_static --lane-id lane12_fp8_static --force

# 세션 끊겨도 계속 실행
./scripts/run_eval_detached.sh models/lane12_fp8_static lane12_fp8_static
# 로그: tail -f logs/eval_*.log

# 상태 확인 (실행 중인 평가, 최근 로그, results 요약)
./scripts/check_eval_status.sh

# 순위·요약·top-3 패키징
python scripts/05_rank_and_package.py
```

---

## 8. 알려진 이슈

| 이슈 | 설명 | 대응 |
|------|------|------|
| KoMT 로컬 스킵 | Judge 없음 | --gemini-key 사용(할당량 있을 때) |
| Gemini 429 | API 할당량 초과 | 키 교체 또는 시간 두고 재시도 |
| pruned22 성능 하락 | KD 데이터·에폭 부족 | train_50k, 에폭 증가로 재학습 |
| 토크나이저 regex 경고 | EXAONE/Mistral 패턴 유사 | 무시 가능 |
| summary.md 구버전 | 이전 baseline/로컬 점수 기준 | 05_rank_and_package 재실행으로 갱신 |

---

## 9. 다음 세션 시작 시 체크리스트

- [ ] `conda activate lgaimers` 후 `python scripts/00_setup_check.py` 로 환경 확인
- [ ] `./scripts/check_eval_status.sh` 로 실행 중인 작업 여부 확인
- [ ] 제출이 목적이면 `results/summary.csv`·`summary.md` 확인 후 `05_rank_and_package.py` 필요 시 재실행
- [ ] 0.65 목표 계속이면 `SESSION_HANDOFF.md` §6.2 항목부터 진행

이 문서는 `SESSION_HANDOFF.md` 한 파일만 보면 이어서 작업할 수 있도록 정리한 요약본입니다. 세부 구현은 각 스크립트와 `configs/lanes.yaml`, `README.md`를 참고하면 됩니다.
