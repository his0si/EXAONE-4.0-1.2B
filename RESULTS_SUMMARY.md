# EXAONE 4.0-1.2B 양자화 실험 결과

## 실험 환경
- 로컬: RTX 5060 Ti (vLLM 호환성 문제로 transformers만 사용)
- 평가 서버: L4 GPU + vLLM 0.14.1

## 양자화 결과 비교

| 모델 | 크기 | PPL | PerfNorm | 비고 |
|------|------|-----|----------|------|
| Base Model | 2.38GB | 70.53 | 1.0000 | 기준 |
| **W4A16_n512** | **1.30GB** | **72.75** | **0.9694** | **추천** |
| W4A16_n1024 | 1.30GB | 73.84 | 0.9551 | |
| W8A16 | 1.78GB | 70.83 | 0.9958 | 높은 정확도 |
| W4A16 (256 samples) | 1.30GB | 81.83 | 0.8618 | 기존 |

## 예상 점수 (vLLM 기준)

점수 공식: `Score = 0.5 × PerfNorm + 0.5 × SpeedNorm`
- PerfNorm = base_ppl / model_ppl
- SpeedNorm = 1 - (model_time / base_time)

### W4A16_n512 (추천)
- PerfNorm: 0.9694
- 예상 SpeedNorm (1.8x 속도 향상 가정): 0.4444
- **예상 Score: ~0.71**

### W8A16
- PerfNorm: 0.9958
- 예상 SpeedNorm (1.4x 속도 향상 가정): 0.2857
- **예상 Score: ~0.64**

## 결론

1. **W4A16_n512**가 최적 선택
   - 높은 PerfNorm (0.97)
   - 작은 모델 크기 (1.30GB, 45% 감소)
   - vLLM에서 좋은 속도 향상 기대

2. 캘리브레이션 샘플 수가 품질에 큰 영향
   - 256 → 512로 PPL 81.83 → 72.75 (12% 개선)
   - 1024는 오히려 약간 저하 (과적합 가능성)

3. W8A16은 정확도는 높지만 속도 이득이 적음

## 제출 파일 생성

```bash
# W4A16_n512 모델로 제출 파일 생성
cd /home/lgaimers
mkdir -p temp_submit/model
cp -r model_W4A16_n512/* temp_submit/model/
cd temp_submit && zip -r ../submit_W4A16_n512.zip model && cd ..
rm -rf temp_submit
```
