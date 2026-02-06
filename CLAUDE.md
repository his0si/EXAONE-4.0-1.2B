# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a model quantization project for compressing the EXAONE 4.0-1.2B language model using GPTQ (Gradient-based Post-Training Quantization). The project uses LG AI Research's EXAONE model and applies W4A16 quantization scheme (4-bit weights, 16-bit activations) to reduce model size from ~2.4GB to ~1.3GB while maintaining performance.

**Competition**: LG Aimers Phase 2 - LLM 경량화 해커톤

## Key Commands

### Run Model Quantization
```bash
# Activate environment
source /home/his0si/anaconda3/etc/profile.d/conda.sh && conda activate lgaimers

# Run quantization (best settings)
python quantize_w4a16_512.py   # W4A16 with 512 calibration samples (recommended)
python quantize_w8a16.py       # W8A16 for higher accuracy
python quantize.py             # Original W4A16 with 256 samples
```

### Evaluate Models
```bash
python evaluate_simple.py      # Evaluate all models (PPL + speed)
python estimate_scores.py      # Estimate vLLM scores
```

## Project Structure

```
/home/lgaimers/
├── base_model/              # Source EXAONE 4.0-1.2B model (~2.4GB)
├── model_W4A16_n512/        # Best quantized model (512 samples, ~1.3GB) ★
├── model_W4A16_n1024/       # W4A16 with 1024 samples
├── model_W8A16/             # W8A16 quantized (~1.8GB)
├── model_quantized/         # W4A16 with 256 samples
├── model/                   # Legacy output directory
│
├── quantize.py              # Main quantization script (configurable)
├── quantize_w4a16_512.py    # W4A16 with 512 samples
├── quantize_w4a16_1024.py   # W4A16 with 1024 samples
├── quantize_w8a16.py        # W8A16 quantization
├── evaluate_simple.py       # Model evaluation (PPL + speed)
├── estimate_scores.py       # Score estimation for vLLM
│
├── submit_W4A16_n512.zip    # Best submission file ★
├── RESULTS_SUMMARY.md       # Experiment results summary
└── eval_results.json        # Evaluation results data
```

## Quantization Experiment Results

### Performance Comparison

| Model | Size | PPL | PerfNorm | Est. Score |
|-------|------|-----|----------|------------|
| Base Model | 2.38GB | 70.53 | 1.0000 | - |
| **W4A16_n512** | **1.30GB** | **72.75** | **0.9694** | **~0.71** ★ |
| W4A16_n1024 | 1.30GB | 73.84 | 0.9551 | ~0.70 |
| W8A16 | 1.78GB | 70.83 | 0.9958 | ~0.64 |
| W4A16 (256) | 1.30GB | 81.83 | 0.8618 | ~0.65 |

### Key Findings

1. **Calibration samples matter significantly**: 256→512 samples improves PPL from 81.83 to 72.75 (12% improvement)
2. **512 samples is optimal**: 1024 samples shows slight degradation (possible overfitting)
3. **W4A16 beats W8A16 in total score**: Lower accuracy but much better speed gains
4. **Model size**: W4A16 achieves 45% size reduction (2.38GB → 1.30GB)

### Scoring Formula

```
Score = max(0.5 × PerfNorm + 0.5 × SpeedNorm, 0)
PerfNorm = base_ppl / model_ppl
SpeedNorm = 1 - (model_time_per_token / base_time_per_token)
```

## Evaluation Environment

### Competition Server
- GPU: L4 (22.4GB VRAM)
- vLLM 0.14.1 for inference
- torch==2.9.0+cu128
- transformers==4.57.3
- compressed-tensors==0.13.0

### Local Environment (lgaimers conda)
- GPU: RTX 5060 Ti (15.5GB) - vLLM 호환성 문제 있음
- torch==2.9.0+cu128
- transformers==4.57.3
- llmcompressor==0.9.0.1

**Note**: vLLM 0.14.1 has compatibility issues with RTX 5060 Ti (Blackwell architecture). Use transformers for local testing.

## Quantization Configuration

### Recommended Settings (W4A16_n512)
```python
SCHEME = "W4A16"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]
NUM_CALIBRATION_SAMPLES = 512  # Key parameter!
MAX_SEQUENCE_LENGTH = 512
```

### Alternative: W8A16 (Higher Accuracy)
```python
SCHEME = "W8A16"  # 8-bit weights, 16-bit activations
# Same other settings
```

## Model Loading Pattern

```python
# Required for EXAONE 4.0
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True  # For evaluation server
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
```

## Submission Format

```bash
# Create submission zip
mkdir -p temp_submit/model
cp -r model_W4A16_n512/* temp_submit/model/
cd temp_submit && zip -r ../submit.zip model && cd ..
rm -rf temp_submit
```

Structure: `submit.zip/model/` containing:
- config.json
- model.safetensors
- tokenizer files
- recipe.yaml

**Constraints**:
- Compressed: max 10GB
- Uncompressed: max 32GB
- Must be compatible with vLLM 0.14.1

## Dependencies

```
torch==2.9.0+cu128
transformers==4.57.3
llmcompressor==0.9.0.1
compressed-tensors==0.13.0
datasets
accelerate==1.10.1
safetensors==0.7.0
```

## Model License

EXAONE AI Model License Agreement 1.2 - NC (Non-Commercial)
- Research and educational purposes allowed
- Cannot be used to develop competing models
