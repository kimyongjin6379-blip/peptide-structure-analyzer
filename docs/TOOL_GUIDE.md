# Peptide Structure Analyzer - Tool Guide

> **Version**: 2025.04
> **Environment**: Streamlit + Railway (CPU)
> **DB**: BIOPEP-UWM 기반 4,162개 생리활성 펩타이드 / 25가지 활성 유형
> **AI Model**: ESM-2 (Protein Language Model), ESMFold (3D Structure)

---

## 전체 구조 요약

| Page | 이름 | 핵심 기능 | AI 모델 |
|------|------|----------|---------|
| 1 | Peptide Analysis | 아미노산 조성/MW/물성 분석 | - |
| 2 | Sequence Prediction | Markov 서열 생성 + ESM-2 재순위 | ESM-2 |
| 3 | Bioactive Search | Markov→ESM-2→DB 매칭 파이프라인 | ESM-2 |
| 4 | 2D Structure | 서열 2D 시각화 (소수성, 물성) | - |
| 5 | 3D Structure | ESMFold 3D 구조 예측 | ESMFold |
| 6 | AI Analysis | ESM-2 임베딩/변이/생성/활성 예측 | ESM-2 (모델 선택 가능) |
| 7 | Fine-tuning | ESM-2 펩톤 특화 학습 | ESM-2 |

---

## Page 1. Peptide Composition Analysis

> 펩톤 샘플의 아미노산 조성, 분자량 분포, 물리화학적 특성을 분석합니다.

### 모드 선택
- **Single Sample**: 단일 샘플 상세 분석
- **Compare Samples**: 최대 5개 샘플 비교

### Single Sample - 4개 탭

| 탭 | 내용 |
|---|---|
| **Composition** | TAA 조성 막대그래프, Top 10 아미노산, 필수 AA 비율 |
| **Molecular Weight** | MW 분포 히스토그램, 평균 MW, 우세 MW 구간, 펩타이드 길이 추정 |
| **Properties** | 물리화학적 성질 레이더 차트, 소수성/전하/극성 비율 |
| **Statistics** | TAA 통계, MW 통계, 필수 AA 수치 요약 |

### Compare Samples
- 그룹 조성 비교 차트
- 샘플 간 유사도 매트릭스
- 상세 조성 테이블

### 기술 스택
- 데이터: TAA/FAA 조성 데이터 (CompositionLoader)
- AI 모델: **사용하지 않음** (순수 데이터 분석)
- 다른 페이지 연동: 없음 (독립적 분석 진입점)

---

## Page 2. Sequence Prediction

> 샘플의 아미노산 조성 데이터를 기반으로 가능한 펩타이드 서열을 생성하고, ESM-2로 재순위합니다.

### 생성 파라미터
- 길이 범위 (최소~최대)
- 생성 서열 수
- 생성 방법: `Markov` / `Random` / `Frequent`

### 결과 - 4개 탭

| 탭 | 내용 |
|---|---|
| **Top Sequences** | 상위 20개 서열 테이블 (순위, 서열, 길이, 우도 점수, MW), 1위 서열 다이어그램 + 소수성 프로파일 |
| **By Length** | 길이별 그룹화 → 각 그룹 상위 5개 서열 |
| **Detailed Analysis** | 생성 통계 (총 수, 평균/최소/최대 길이), Top 10 AA 조성, 우도 점수 분포 히스토그램 |
| **Export** | 상위 50개 서열 FASTA / CSV 다운로드 |

### ESM-2 Re-ranking (탭 아래 확장 패널)
- ESM-2 Fitness 점수로 재순위
- 결합 점수: `combined = likelihood x 0.4 + ESM-2_fitness x 0.6`
- Markov 단독 vs ESM-2 재순위 Top 10 비교 테이블
- 상위 3개 서열: 잔기 수준 AA 치환 개선 제안

### 기술 스택
- Markov Chain (SequenceGenerator)
- ESM-2 esm2_t6_8M (CPU)

### 다른 페이지로 전송
- **Top 서열 → Page 6** (임베딩 분석)
- **배치 서열 → Page 6 Tab 5** (배치 분석)

---

## Page 3. Bioactive Peptide Search

> Markov 생성 → ESM-2 필터링 → 4,162개 DB 매칭으로 생리활성 펩타이드를 예측합니다.
> 이 Tool의 핵심 파이프라인입니다.

### Tab 1 - Activity Profile

#### 파라미터
| 파라미터 | 범위 | 설명 |
|---------|------|------|
| 초기 서열 수 | 100 ~ 1,000 | Markov로 생성할 서열 수 |
| 길이 범위 | 3 ~ 20 AA | 생성 서열 길이 |
| ESM-2 상위 N | 50 ~ 500 | Fitness 기준 상위 N개만 DB 매칭 |

#### 3단계 파이프라인
```
Step 1: Markov 서열 생성 (빠름, ~1초)
   ↓
Step 2: ESM-2 임베딩+로짓 기반 Fitness 평가 → 상위 N개 필터링 (~2-3분, CPU)
   ↓
Step 3: 4,162개 생리활성 펩타이드 DB 매칭 (빠름, ~1초)
```

#### 출력 결과
1. **파이프라인 요약 메트릭** - 초기 서열 수, ESM-2 필터 후, DB 매칭 수, 감지 활성 유형, 고유 모티프
2. **레이더 차트** - Top 12 활성 유형별 정규화 점수 (0~1)
3. **활성 상세 테이블** - 활성명, 점수, DB 히트 수, 고유 모티프 수, 대표 매칭 펩타이드
4. **Top 3 활성 확장 분석** - 각 활성별 관련 모티프와 설명
5. **Hit 서열 테이블** - ESM-2 Fitness, Combined Score, 매칭 모티프 수, 활성 유형
6. **서열별 모티프 상세** - 각 서열 클릭 시 모티프, 위치, 활성, IC50 정보

#### 전송 버튼

| 버튼 | 목적지 | 설명 |
|------|--------|------|
| 🤖 → AI 활성 분석 | Page 6 Tab 4 | 선택 서열을 ESM-2 모델 선택 가능한 정밀 분석으로 |
| 🔬 → 3D 구조 예측 | Page 5 | 선택 서열의 ESMFold 3D 구조 예측 |
| 📦 전체 → AI 배치 | Page 6 Tab 5 | 모든 매칭 서열을 한번에 배치 분석 |

### Tab 2 - Compare Samples

- 최대 5개 샘플 선택
- 각 샘플별 200개 서열 생성 → DB 매칭
- 샘플별 활성 프로파일 레이더 차트 비교
- 활성별 Best Sample 테이블
- 상세 점수 매트릭스

### 기술 스택
- Markov Chain (서열 생성)
- ESM-2 esm2_t6_8M Fitness (임베딩+로짓 기반, 배치 16)
- BIOPEP-UWM DB: 4,162 peptides, 25 activity types

### DB 활성 유형 (상위 10)
| 활성 | DB 내 펩타이드 수 |
|------|------------------|
| ACE_inhibitor | 1,374 |
| antioxidant | 850 |
| DPP_IV_inhibitor | 757 |
| antithrombotic | 200+ |
| antimicrobial | 150+ |
| alpha_glucosidase_inhibitor | 100+ |
| immunomodulating | 80+ |
| opioid | 70+ |
| anti_amnestic | 60+ |
| hypotensive | 50+ |

---

## Page 4. 2D Structure Visualization

> 개별 펩타이드 서열의 2D 시각화 및 물리화학적 특성 분석을 수행합니다.

### 입력 방식
- **Generate from Sample**: 샘플 선택 → Markov/Random/Frequent로 생성
- **Custom Sequence**: 직접 서열 입력

### 3개 탭

| 탭 | 내용 |
|---|---|
| **Sequence Diagram** | 잔기별 색상 코드 시각화 (소수성/양전하/음전하/극성/기타) |
| **Hydrophobicity** | Kyte-Doolittle 소수성 프로파일 차트 + 통계 (평균/최대/최소) |
| **Properties** | 물리화학적 성질 레이더 차트, AA 조성 비율 상세 |

### 내보내기
- FASTA 파일 다운로드

### 기술 스택
- Kyte-Doolittle 소수성 스케일
- 물성 계산 (calculate_property_ratios)
- AI 모델: **사용하지 않음**
- 다른 페이지 연동: 없음 (독립)

---

## Page 5. 3D Structure Prediction

> ESMFold API를 이용하여 펩타이드의 3D 접힘 구조를 예측하고, py3Dmol로 시각화합니다.

### 입력 방식
- **Generate from Sample**: 샘플에서 Markov/Random 생성
- **Custom Sequence**: 직접 입력
- **Page 3에서 전송**: Bioactive Search 결과에서 자동 수신 (상단 알림 배너)

### 기능

| 단계 | 설명 |
|------|------|
| 서열 정보 | 길이, 분자량 (MW), 구조 예측 상태 |
| 구조 예측 | ESMFold API 호출 → PDB 구조 (2~10초, 캐시 지원) |
| 메타데이터 | 원자 수, 잔기 수, 체인 정보, 소요 시간 |
| 3D 시각화 | py3Dmol 뷰어 (스타일: cartoon/stick/sphere/line, 색상: spectrum/residue/chain) |
| 내보내기 | PDB 파일 다운로드 |

### 제한 사항
- 최대 30 AA (데모 환경)
- 인터넷 연결 필요 (ESMFold API)
- 첫 예측 시 2~10초, 캐시된 예측은 즉시

### 기술 스택
- **ESMFold** (Meta AI) - ESM-2와는 별개의 구조 예측 전용 모델
- py3Dmol (3D 시각화)

### ESM-2 vs ESMFold 차이

| | ESM-2 | ESMFold |
|---|---|---|
| **목적** | 서열 임베딩, 활성/Fitness 예측 | 3D 접힘 구조 예측 |
| **출력** | 벡터 (수치) | PDB 좌표 (3D 구조) |
| **사용처** | Page 2, 3, 6 | Page 5 |

---

## Page 6. AI Deep Analysis

> ESM-2 Protein Language Model 기반 고급 분석 도구 모음입니다.
> Railway 배포 환경 전용 (PyTorch 필요)

### 사이드바 설정

| 설정 | 옵션 | 설명 |
|------|------|------|
| ESM-2 모델 | esm2_t6_8M / esm2_t12_35M / esm2_t30_150M | 크기↑ = 정확도↑, 속도↓ |
| Fine-tuned 모델 | 범용 ESM-2 / 사용자 학습 모델 | Page 7에서 학습한 모델 자동 로드 |

#### ESM-2 모델별 스펙

| 모델 | 파라미터 | 임베딩 차원 | 특징 |
|------|----------|------------|------|
| esm2_t6_8M | 800만 | 320 | 빠름, Railway CPU 적합 |
| esm2_t12_35M | 3,500만 | 480 | 균형잡힌 성능 |
| esm2_t30_150M | 1.5억 | 1,280 | 고정밀, 느림 |

### Tab 1 - 서열 임베딩

| 기능 | 설명 |
|------|------|
| 임베딩 추출 | 잔기별 ESM-2 임베딩 히트맵 (50차원 시각화) |
| 잔기 중요도 | 각 잔기의 기능적 중요도 막대 차트 (높을수록 치환 시 영향 큼) |
| 유사도 비교 | 2개 서열 코사인 유사도 → 기능적 유사성 판단 |

**유사도 해석**: >0.9 매우 유사 / 0.7~0.9 중간 / <0.7 상이

### Tab 2 - 변이 예측 (Zero-shot)

| 항목 | 설명 |
|------|------|
| 입력 | 야생형 서열 + 변이 목록 (예: A5G, L10V, K15R) |
| 방법 | ESM-2 Zero-shot scoring (추가 학습 없이 진화적 정보 활용) |
| 출력 | 변이별 Score, 효과 분류, WT/MT 확률 |

**효과 해석**: Score > 0.5 = 유리한 변이 / -0.5~0.5 = 중립 / < -0.5 = 유해한 변이

### Tab 3 - DL 서열 생성

| 방법 | 설명 | 용도 |
|------|------|------|
| **ESM-2 Masked** | 특정 위치 마스크 → ESM-2가 최적 AA 제안 | 특정 위치 최적화 |
| **ESM-2 Iterative** | 반복적 마스크 채우기로 점진적 최적화 | 서열 전체 개선 |
| **ProtGPT2** | 사전학습 LM으로 자유 서열 생성 | 완전히 새로운 서열 |
| **VAE** | 자체 데이터 학습 후 잠재 공간에서 생성 | 학습 데이터 유사 서열 |

### Tab 4 - 활성 예측 (DB + ESM-2)

| 단계 | 설명 |
|------|------|
| DB 매칭 | 4,162개 DB에서 입력 서열 내 모티프 검색 (25가지 활성) |
| ESM-2 Fitness | 선택된 모델로 Fitness 점수 평가 |
| 레이더 차트 | Top 12 활성 유형별 정규화 점수 |
| 활성 테이블 | 활성, Hit 수, Score, Level (높음/중간/낮음) |
| 모티프 상세 | 매칭 모티프, 위치, 활성, IC50 |
| Top 3 분석 | 상위 3개 활성의 관련 모티프 상세 |

**Page 3 Tab 1과의 차이**:
- Page 3: Markov 생성부터 시작하는 **전체 파이프라인** (샘플 → 서열 생성 → 필터 → 매칭)
- Page 6 Tab 4: **개별 서열**에 대한 정밀 분석 + ESM-2 모델 크기 선택 가능

### Tab 5 - 배치 분석

| 분석 | 설명 |
|------|------|
| ESM-2 Fitness | 전체 서열 일괄 Fitness 평가 + 최고 서열 추천 |
| 유사도 매트릭스 | 서열 간 코사인 유사도 히트맵 |
| 클러스터링 | PCA 차원 축소 + K-means 산점도 |
| DB 활성 예측 | 4,162 DB 기반 배치 활성 예측 + 활성별 Top 후보 |

**배치 서열 소스**: Page 2/3에서 전송 or 직접 입력

---

## Page 7. ESM-2 Fine-tuning

> 범용 ESM-2를 우리 펩톤 데이터에 특화시켜 예측 정확도를 향상시킵니다.
> Railway 배포 환경 전용

### Tab 1 - 데이터 준비 & 학습

#### Step 1: 데이터셋 구축

| 데이터 소스 | 설명 |
|------------|------|
| DB 모티프 서열 | 4,162개 생리활성 펩타이드 DB에서 추출 |
| 식품 유래 기능성 펩타이드 | ACE 억제, 항산화, 항균 등 100+ 펩타이드 |
| Markov 생성 서열 | 각 샘플 조성에서 생성한 서열 |
| 직접 입력 | 사용자가 직접 추가한 서열 |

#### Step 2: Fine-tuning

| 파라미터 | 범위 | 권장값 |
|---------|------|--------|
| Epochs | 3 ~ 50 | 10~20 |
| Learning Rate | 1e-6 ~ 1e-4 | 5e-5 |
| Freeze Layers | 0 ~ 5 | 2~3 |
| Masking Ratio | 5% ~ 30% | 15% |

- 실시간 학습 진행바
- Training/Validation Loss 라이브 차트
- 학습 완료 후 이름 지정하여 모델 저장

### Tab 2 - Before vs After 비교

| 비교 항목 | 설명 |
|----------|------|
| **Perplexity** | 범용 vs Fine-tuned 혼란도 비교 (낮을수록 좋음) |
| **임베딩 변화** | 코사인 유사도/L2 거리 산점도, PCA 전후 비교 |
| **변이 예측** | 동일 변이에 대한 범용 vs Fine-tuned 스코어 비교 |

개선율 5% 이상이면 "특화 효과 확인" 판정

### Tab 3 - 설명
- Fine-tuning 개념 교육 자료
- 범용 vs 특화 모델 비교표
- 적용 가능 시나리오

---

## 페이지 간 연동 흐름도

```
Page 1 (조성 분석)          [독립]
  │
  ▼
Page 2 (서열 생성) ─────── 🤖 Top 서열 ──→ Page 6 Tab 1 (임베딩)
  │                        📦 배치 서열 ──→ Page 6 Tab 5 (배치)
  │
  ▼
Page 3 (생리활성 검색) ──── 🤖 선택 서열 ──→ Page 6 Tab 4 (활성 분석)
  │                        🔬 선택 서열 ──→ Page 5     (3D 구조)
  │                        📦 전체 서열 ──→ Page 6 Tab 5 (배치)
  │
  ▼
Page 4 (2D 시각화)          [독립]
  │
  ▼
Page 5 (3D 구조) ◀────────── Page 3에서 서열 수신
  │
  ▼
Page 6 (AI 분석) ◀────────── Page 2, 3에서 서열 수신
  │               ◀────────── Page 7에서 Fine-tuned 모델 수신
  │
  ▼
Page 7 (Fine-tuning) ──────→ Page 6 사이드바 (학습 모델 제공)
```

---

## 권장 사용 시나리오

### 시나리오 1: 펩톤 샘플의 생리활성 잠재력 평가

```
Page 1 → 조성 확인
  ↓
Page 3 Tab 1 → Markov+ESM-2+DB 파이프라인 실행
  ↓
결과 확인 → 상위 서열 선택
  ↓
Page 6 Tab 4 → 더 큰 ESM-2 모델로 정밀 분석
```

### 시나리오 2: 특정 활성 펩타이드 후보 탐색

```
Page 3 Tab 1 → 파이프라인 실행 → ACE_inhibitor 등 타겟 활성 확인
  ↓
🔬 → Page 5 → 후보 서열 3D 구조 확인
  ↓
📦 → Page 6 Tab 5 → 배치 클러스터링으로 다양성 확인
```

### 시나리오 3: 샘플 간 비교 분석

```
Page 1 Compare → 조성 차이 확인
  ↓
Page 3 Tab 2 → 생리활성 프로파일 비교
  → "어떤 샘플이 ACE 억제 활성이 높은가?" 등
```

### 시나리오 4: 예측 정확도 향상

```
Page 7 Tab 1 → 펩톤 데이터로 ESM-2 Fine-tuning
  ↓
Page 7 Tab 2 → Before vs After 비교 확인
  ↓
Page 6 사이드바 → Fine-tuned 모델 선택하여 분석
```

---

## 기술 참고

### ESM-2 Fitness Score 해석
- **0.8 ~ 1.0**: 자연에서 발견될 가능성이 매우 높은 서열
- **0.6 ~ 0.8**: 타당한 서열
- **0.4 ~ 0.6**: 가능하지만 드문 서열
- **< 0.4**: 자연에서 발견되기 어려운 서열

> Fitness Score가 높다고 해서 반드시 펩톤 내에 존재한다는 의미는 아닙니다.
> 실제 존재 여부는 LC-MS/MS 등 분석 기기로 확인이 필요합니다.

### DB 매칭 방식
- 입력 서열 내에 DB의 생리활성 펩타이드가 **부분 문자열(substring)**로 포함되는지 검색
- 최소 모티프 길이 3 AA (디펩타이드 노이즈 방지)
- 매칭 결과에 IC50, 출처 단백질, 분자량 등 DB 정보 포함

### Railway 배포 환경
- **Hobby Plan**: CPU 전용, GPU 없음
- ESM-2 Fitness 평가: ~2-3분 (500개 서열 기준)
- ESMFold 3D 예측: 외부 API 사용 (무료)
- 메모리: 제한적 → esm2_t6_8M 권장
