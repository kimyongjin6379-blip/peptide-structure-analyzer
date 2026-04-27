# 🧬 Peptide Structure Analyzer

펩톤(Peptone) 조성 데이터를 기반으로 생리활성 펩타이드를 예측·분석·시각화하는 통합 웹 애플리케이션

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-orange)
![Deploy](https://img.shields.io/badge/deploy-Railway-blueviolet)

---

## 📋 프로젝트 개요

`composition_template.xlsx`에 담긴 펩톤 샘플의 아미노산 조성(TAA/FAA) 데이터를 입력으로 받아,

1. **Markov Chain** 으로 통계적 후보 서열 생성
2. **ESM-2** (Protein Language Model) 로 진화적 타당성 검증
3. **BIOPEP-UWM 기반 4,162개 DB** 와 매칭하여 생리활성 프로파일 도출
4. **ESMFold** 로 3D 구조 예측 및 시각화

까지 하나의 파이프라인으로 수행합니다.

---

## ✨ 주요 기능

| Page | 기능 | 설명 |
|------|------|------|
| 1 | 📊 **조성 분석** | TAA/FAA 조성, MW 분포, 물리화학적 특성 분석 |
| 2 | 🧬 **서열 예측** | Markov Chain + ESM-2 재순위 서열 생성, FASTA 내보내기 |
| 3 | 💊 **생리활성 검색** | Markov→ESM-2→DB 파이프라인, 25종 활성 레이더 차트 |
| 4 | 🎨 **2D 시각화** | 소수성 프로파일, 잔기 다이어그램, 물성 레이더 |
| 5 | 🔬 **3D 구조 예측** | ESMFold API 구조 예측, py3Dmol 인터랙티브 뷰어 |
| 6 | 🤖 **AI 심층 분석** | ESM-2 임베딩, Zero-shot 변이 예측, DL 서열 생성, 배치 분석 |
| 7 | 🎯 **Fine-tuning** | 펩톤 데이터 특화 ESM-2 재학습, Before/After 비교 |

---

## 🚀 빠른 시작

### 로컬 실행

```bash
# 1. 저장소 클론
git clone https://github.com/kimyongjin6379-blip/peptide-structure-analyzer.git
cd peptide-structure-analyzer

# 2. 의존성 설치
pip install -r requirements.txt

# 3. Streamlit 실행
streamlit run streamlit_app/app.py
```

브라우저에서 http://localhost:8501 자동 실행

### Docker (Railway 배포)

```bash
# Docker 이미지 빌드 (PyTorch CPU + ESM-2 포함)
docker build -t peptide-analyzer .
docker run -p 8501:8501 peptide-analyzer
```

> Railway에 GitHub 연결 시 push마다 자동 배포됩니다.

---

## 📁 프로젝트 구조

```
peptide-structure-analyzer/
│
├── streamlit_app/                     # 웹 UI
│   ├── app.py                         # 메인 진입점
│   └── pages/
│       ├── 1_peptide_analysis.py      # 조성 분석
│       ├── 2_sequence_prediction.py   # 서열 예측
│       ├── 3_bioactive_search.py      # 생리활성 검색 (핵심 파이프라인)
│       ├── 4_structure_2d.py          # 2D 시각화
│       ├── 5_structure_3d.py          # 3D 구조 예측
│       ├── 6_ai_analysis.py           # AI 심층 분석
│       └── 7_finetuning.py            # ESM-2 Fine-tuning
│
├── src/                               # 백엔드 소스
│   ├── data_loader.py                 # 조성 데이터 로딩 (xlsx)
│   ├── peptide_analyzer.py            # AA 조성·MW·물성 분석
│   ├── sequence_predictor.py          # Markov Chain 서열 생성
│   ├── bioactive_predictor.py         # DB 매칭 생리활성 예측
│   ├── plm_embedder.py                # ESM-2 임베딩·Fitness 스코어링
│   ├── plm_finetuner.py               # ESM-2 Fine-tuning
│   ├── fitness_predictor.py           # ML 기반 활성 예측
│   ├── deep_generator.py              # DL 서열 생성 (VAE, ProtGPT2)
│   ├── structure_builder.py           # ESMFold API 연동
│   ├── visualizer_2d.py               # Plotly 2D 차트
│   ├── visualizer_3d.py               # py3Dmol 3D 뷰어
│   └── utils.py                       # 공통 유틸리티
│
├── data/
│   ├── composition_template.xlsx                # 펩톤 샘플 조성 데이터 (입력)
│   ├── bioactive_peptide_db_comprehensive.json  # BIOPEP-UWM DB (4,162 peptides, 25 activities)
│   ├── bioactive_peptide_db.json                # 큐레이션 DB (52 motifs)
│   ├── raw_db/                                  # PepLab 원본 CSV (16개 활성 유형)
│   └── cache/structures/                        # ESMFold 구조 캐시 (30일)
│
├── scripts/
│   └── build_bioactive_db.py          # BIOPEP-UWM 스크래핑 → DB 빌드 스크립트
│
├── tests/
│   ├── test_peptide_analyzer.py
│   ├── test_sequence_predictor.py
│   ├── test_bioactive_predictor.py
│   ├── test_structure_builder.py
│   └── test_visualizers.py
│
├── docs/
│   ├── TOOL_GUIDE.md                  # 상세 기능 가이드 (MD)
│   ├── Peptide_Structure_Analyzer_Tool_Guide.docx  # Word 버전 가이드
│   └── build_guide_docx.py            # MD → DOCX 자동 변환 스크립트
│
├── .streamlit/config.toml             # Streamlit 설정
├── Dockerfile                         # Railway 배포용 (PyTorch CPU + ESM-2)
├── requirements.txt                   # Python 의존성
└── packages.txt                       # 시스템 패키지
```

---

## 🔬 핵심 파이프라인 (Page 3)

```
[입력] 펩톤 샘플 조성 (TAA/FAA %)
    │
    ▼
Step 1. Markov Chain 서열 생성
    │  AA 조성 비율 + 전하 반발/소수성 선호 전이 행렬
    │  → 100~1,000개 후보 서열
    │
    ▼
Step 2. ESM-2 Fitness 필터링
    │  임베딩+로짓 기반 스코어링 (배치 16, CPU ~2-3분)
    │  combined = likelihood × 0.4 + ESM-2_fitness × 0.6
    │  → 상위 N개만 통과
    │
    ▼
Step 3. 생리활성 DB 매칭
    │  4,162개 BIOPEP-UWM 펩타이드 substring 매칭
    │  25가지 활성 유형 집계 및 정규화
    │
    ▼
[출력] 레이더 차트 · 활성 테이블 · Hit 서열 · 모티프 상세 (IC50 포함)
```

### 페이지 간 전송 연동

```
Page 2 (서열 생성) ──── 🤖 ──→ Page 6 Tab 1 (임베딩 분석)
                   ──── 📦 ──→ Page 6 Tab 5 (배치 분석)

Page 3 (생리활성 검색) ── 🤖 ──→ Page 6 Tab 4 (활성 예측)
                       ── 🔬 ──→ Page 5     (3D 구조 예측)
                       ── 📦 ──→ Page 6 Tab 5 (배치 분석)

Page 7 (Fine-tuning) ──────────→ Page 6 사이드바 (학습 모델)
```

---

## 📦 의존성

### requirements.txt 주요 패키지

| 분류 | 패키지 |
|------|--------|
| 데이터 처리 | pandas, numpy, scipy, openpyxl |
| ML/DL | scikit-learn, xgboost, shap, botorch |
| 시각화 | plotly, streamlit, py3Dmol, kaleido |
| 생물정보학 | biopython, biotite, requests |
| 유틸리티 | tqdm, python-dotenv, joblib |

### Dockerfile 추가 설치 (Railway)

| 패키지 | 버전 | 용도 |
|--------|------|------|
| torch (CPU) | 2.2.0+cpu | PyTorch CPU |
| fair-esm | 2.0.0 | ESM-2 모델 |
| transformers | ≥4.36.0 | ProtGPT2 등 |

> ESM-2 모델은 첫 실행 시 자동 다운로드 (`/app/model_cache`)

---

## 🗄️ 생리활성 데이터베이스

| 항목 | 상세 |
|------|------|
| 총 펩타이드 수 | **4,162개** (고유 서열) |
| 활성 유형 | **25종** |
| 주요 출처 | BIOPEP-UWM (University of Warmia and Mazury) |
| 파일 | `data/bioactive_peptide_db_comprehensive.json` |

**활성 유형 상위 10 (DB 내 펩타이드 수)**

| 활성 | 수 |
|------|----|
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

DB 재빌드 (BIOPEP-UWM 재스크래핑):
```bash
python scripts/build_bioactive_db.py
```

---

## 🤖 ESM-2 모델 선택 (Page 6)

| 모델 | 파라미터 | 임베딩 차원 | Railway CPU 권장 |
|------|----------|------------|-----------------|
| esm2_t6_8M | 800만 | 320 | ✅ 권장 |
| esm2_t12_35M | 3,500만 | 480 | ⚠️ 느림 |
| esm2_t30_150M | 1.5억 | 1,280 | ❌ 매우 느림 |

---

## 📊 성능 기준 (Railway Hobby, CPU)

| 작업 | 예상 시간 |
|------|----------|
| 데이터 로딩 | < 1초 |
| Markov 서열 생성 (500개) | < 1초 |
| ESM-2 Fitness 평가 (500개) | ~2-3분 |
| DB 매칭 (4,162 peptides) | < 1초 |
| ESMFold 3D 예측 (첫 호출) | 2-10초 |
| ESMFold 3D 예측 (캐시) | < 0.1초 |

---

## 🧪 테스트

```bash
# 전체 테스트
pytest tests/ -v

# 특정 모듈
pytest tests/test_bioactive_predictor.py -v

# 커버리지
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 문서

상세 기능 가이드는 `docs/` 폴더를 참고하세요.

```bash
# Word 가이드 재생성 (TOOL_GUIDE.md 수정 후)
python docs/build_guide_docx.py
```

---

## 📝 버전 기록

- **v2.0.0** (2025-04)
  - BIOPEP-UWM 스크래핑으로 DB 52 motifs → **4,162 peptides, 25 activities** 확장
  - Markov → ESM-2 → DB 통합 파이프라인 구축
  - Page 6 AI 심층 분석 (임베딩·변이·DL 생성·배치) 추가
  - Page 7 ESM-2 Fine-tuning 추가
  - 페이지 간 서열 전송 연동 (→ AI / → 3D / → 배치)
  - ESM-2 Fitness 임베딩+로짓 기반 고속 스코어링
  - DOCX 자동 생성 가이드 추가

- **v1.0.0** (2026-01)
  - 기본 5개 페이지 구현 (조성·서열·생리활성·2D·3D)
  - 52개 큐레이션 모티프 DB
  - Streamlit 웹 UI 완성

---

## 🙏 참고 및 감사

- **ESMFold / ESM-2** (Meta AI) — 구조 예측 및 Protein Language Model
- **BIOPEP-UWM** (University of Warmia and Mazury) — 생리활성 펩타이드 DB
- **Streamlit** — 웹 UI 프레임워크
- **Railway** — 클라우드 배포 플랫폼

---

**Built with Python · Streamlit · ESM-2 · ESMFold**
