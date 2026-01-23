# 🧬 Peptide Structure Analyzer

펩타이드 조성 분석, 서열 예측, 생리활성 모티프 검색, 2D/3D 구조 시각화 통합 웹 애플리케이션

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📋 프로젝트 개요

composition_template.xlsx 데이터를 기반으로 펩타이드/단백질의 조성을 분석하고, 생리활성을 예측하며, 3D 구조를 시각화하는 통합 도구입니다.

### ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 📊 **조성 분석** | TAA/FAA 조성, 분자량 분포, 물리화학적 특성 분석 |
| 🧬 **서열 예측** | Markov chain 기반 통계적 서열 생성 |
| 💊 **생리활성 예측** | 6가지 활성 예측 + 알려진 모티프 검색 |
| 🎨 **2D 시각화** | Plotly 인터랙티브 차트, 펩타이드 다이어그램 |
| 🔬 **3D 구조** | ESMFold API 구조 예측, py3Dmol 뷰어 |
| 🌐 **웹 UI** | Streamlit 기반 사용자 친화적 인터페이스 |

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/peptide_structure_analyzer.git
cd peptide_structure_analyzer

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행

**Windows:**
```bash
run_streamlit.bat
```

**Linux/Mac:**
```bash
streamlit run streamlit_app/app.py
```

브라우저에서 http://localhost:8501 자동 열림

## 📦 의존성

### 핵심 패키지
- `pandas>=2.0.0` - 데이터 처리
- `numpy>=1.24.0` - 수치 계산
- `plotly>=5.17.0` - 인터랙티브 시각화
- `streamlit>=1.28.0` - 웹 UI
- `scikit-learn>=1.3.0` - 클러스터링

### 생물정보학
- `biopython>=1.81` - 서열 처리
- `requests>=2.31.0` - ESMFold API

### 선택적 (3D 시각화)
- `py3Dmol>=2.0.0` - 3D 구조 뷰어

## 📁 프로젝트 구조

```
peptide_structure_analyzer/
├── data/                              # 데이터
│   ├── composition_template.xlsx      # 48개 샘플 조성 데이터
│   ├── bioactive_peptide_db.json      # 생리활성 모티프 DB
│   └── cache/                         # ESMFold 구조 캐시
│
├── src/                               # 소스 코드
│   ├── utils.py                       # 유틸리티 함수
│   ├── data_loader.py                 # 데이터 로딩
│   ├── peptide_analyzer.py            # 펩타이드 분석
│   ├── sequence_predictor.py          # 서열 예측
│   ├── bioactive_predictor.py         # 생리활성 예측
│   ├── structure_builder.py           # ESMFold 구조 생성
│   ├── visualizer_2d.py               # 2D 시각화
│   └── visualizer_3d.py               # 3D 시각화
│
├── streamlit_app/                     # Streamlit 웹 UI
│   ├── app.py                         # 메인 앱
│   └── pages/
│       ├── 1_peptide_analysis.py      # 조성 분석
│       ├── 2_sequence_prediction.py   # 서열 예측
│       ├── 3_bioactive_search.py      # 생리활성 검색
│       ├── 4_structure_2d.py          # 2D 시각화
│       └── 5_structure_3d.py          # 3D 구조
│
├── tests/                             # 단위 테스트
│   ├── test_data_loader.py
│   ├── test_peptide_analyzer.py
│   ├── test_sequence_predictor.py
│   ├── test_bioactive_predictor.py
│   ├── test_structure_builder.py
│   └── test_visualizers.py
│
├── requirements.txt                   # 의존성
├── run_streamlit.bat                  # 실행 스크립트
└── README.md                          # 문서
```

## 💡 사용 예제

### Python 스크립트

```python
from data_loader import CompositionLoader
from bioactive_predictor import BioactivePredictor

# 데이터 로딩
loader = CompositionLoader()
loader.load_data()

# 생리활성 예측
predictor = BioactivePredictor(loader)
result = predictor.predict_comprehensive(
    sample_id='Sample_01',
    n_sequences=50,
    length_range=(5, 12)
)

# 결과 확인
print(f"생성 서열: {result['generated_sequences']}")
print(f"모티프 발견: {result['motif_findings']['total_motifs_found']}")
```

### 웹 UI

1. **조성 분석**: 아미노산 조성 바 차트, 물리화학적 특성 레이더 차트
2. **서열 예측**: Markov chain으로 50-100개 서열 생성, FASTA 내보내기
3. **생리활성 검색**: 6가지 활성 점수, 모티프 검색, 추천 서열
4. **2D 시각화**: 펩타이드 다이어그램, 소수성 프로파일
5. **3D 구조**: ESMFold 구조 예측, 인터랙티브 3D 뷰어

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 특정 모듈 테스트
pytest tests/test_bioactive_predictor.py -v

# 커버리지 확인
pytest tests/ --cov=src --cov-report=html
```

## 📊 주요 알고리즘

### 1. Markov Chain 서열 생성

```
아미노산 조성 → 전이 확률 행렬 → 서열 생성 → 가능성 점수
```

- 인접 아미노산 선호도 반영 (소수성-소수성 1.2배, 같은 전하 0.6배)
- 조성 기반 likelihood scoring

### 2. 생리활성 예측

**조성 기반 점수:**
- 항균성: K, R, W, F 비율
- 항고혈압: P, V, I, L 비율
- 항산화: H, Y, W 비율

**모티프 검색:**
- 52개 알려진 모티프 (BIOPEP, MBPDB 참조)
- 정규표현식 패턴 매칭

### 3. ESMFold 구조 예측

- REST API 호출: https://api.esmatlas.com/
- 로컬 캐싱 (30일 유효)
- 최대 400 residues

## 📈 성능

| 작업 | 시간 |
|------|------|
| 데이터 로딩 (48 샘플) | < 1초 |
| 조성 분석 | < 0.1초 |
| 서열 예측 (100개) | < 1초 |
| 모티프 검색 (50개 서열) | < 0.5초 |
| ESMFold 예측 (첫 호출) | 2-10초 |
| ESMFold 예측 (캐시) | < 0.1초 |

## 🔬 생리활성 데이터베이스

총 52개 모티프, 6가지 활성:

| 활성 | 모티프 예시 | 설명 |
|------|------------|------|
| 항균성 | RR, KK, WR | 양전하 + 방향족 |
| 항고혈압 | IPP, VPP, IKP | ACE 억제 |
| 항산화 | HH, YY, WW | 라디칼 소거 |
| Opioid | YGG, YGGF | N-말단 Tyr |
| 면역조절 | VEP, IEP | 전하 잔기 |
| 항염 | WKP, FKP | Pro-rich |

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 👥 작성자

Peptide Analysis Team

## 📝 버전 기록

- **v1.0.0** (2026-01-23)
  - Phase 1-8 완료
  - 모든 핵심 기능 구현
  - Streamlit 웹 UI 완성

## 🙏 감사의 말

- ESMFold (Meta AI) - 구조 예측
- BIOPEP, MBPDB - 생리활성 모티프 데이터
- Claude Sonnet 4.5 - 프로젝트 개발

## 📞 문의

프로젝트 관련 문의는 GitHub Issues를 이용해주세요.

---

**Built with ❤️ using Python, Streamlit, and Claude AI**
