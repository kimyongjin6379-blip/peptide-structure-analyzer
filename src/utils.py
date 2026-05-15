"""
유틸리티 함수 모듈
Utility functions for peptide analysis
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np


# 아미노산 속성 정의
AMINO_ACIDS = {
    'A': {'name': 'Alanine', 'mw': 89.09, 'hydrophobic': True, 'charged': False, 'aromatic': False},
    'R': {'name': 'Arginine', 'mw': 174.20, 'hydrophobic': False, 'charged': True, 'aromatic': False},
    'N': {'name': 'Asparagine', 'mw': 132.12, 'hydrophobic': False, 'charged': False, 'aromatic': False},
    'D': {'name': 'Aspartic acid', 'mw': 133.10, 'hydrophobic': False, 'charged': True, 'aromatic': False},
    'C': {'name': 'Cysteine', 'mw': 121.16, 'hydrophobic': False, 'charged': False, 'aromatic': False},
    'Q': {'name': 'Glutamine', 'mw': 146.15, 'hydrophobic': False, 'charged': False, 'aromatic': False},
    'E': {'name': 'Glutamic acid', 'mw': 147.13, 'hydrophobic': False, 'charged': True, 'aromatic': False},
    'G': {'name': 'Glycine', 'mw': 75.07, 'hydrophobic': False, 'charged': False, 'aromatic': False},
    'H': {'name': 'Histidine', 'mw': 155.16, 'hydrophobic': False, 'charged': True, 'aromatic': True},
    'I': {'name': 'Isoleucine', 'mw': 131.18, 'hydrophobic': True, 'charged': False, 'aromatic': False},
    'L': {'name': 'Leucine', 'mw': 131.18, 'hydrophobic': True, 'charged': False, 'aromatic': False},
    'K': {'name': 'Lysine', 'mw': 146.19, 'hydrophobic': False, 'charged': True, 'aromatic': False},
    'M': {'name': 'Methionine', 'mw': 149.21, 'hydrophobic': True, 'charged': False, 'aromatic': False},
    'F': {'name': 'Phenylalanine', 'mw': 165.19, 'hydrophobic': True, 'charged': False, 'aromatic': True},
    'P': {'name': 'Proline', 'mw': 115.13, 'hydrophobic': True, 'charged': False, 'aromatic': False},
    'S': {'name': 'Serine', 'mw': 105.09, 'hydrophobic': False, 'charged': False, 'aromatic': False},
    'T': {'name': 'Threonine', 'mw': 119.12, 'hydrophobic': False, 'charged': False, 'aromatic': False},
    'W': {'name': 'Tryptophan', 'mw': 204.23, 'hydrophobic': True, 'charged': False, 'aromatic': True},
    'Y': {'name': 'Tyrosine', 'mw': 181.19, 'hydrophobic': False, 'charged': False, 'aromatic': True},
    'V': {'name': 'Valine', 'mw': 117.15, 'hydrophobic': True, 'charged': False, 'aromatic': False},
}


def get_aa_properties(aa_code: str) -> Optional[Dict]:
    """
    아미노산 속성 반환

    Args:
        aa_code: 1-letter 아미노산 코드

    Returns:
        아미노산 속성 딕셔너리 또는 None
    """
    return AMINO_ACIDS.get(aa_code.upper())


# ─── 1-letter ↔ 3-letter AA 변환 ─────────────────────
AA_1_TO_3 = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
}

AA_3_TO_1 = {v: k for k, v in AA_1_TO_3.items()}


def aa_to_3letter(aa: str) -> str:
    """단일 아미노산 1-letter → 3-letter"""
    return AA_1_TO_3.get(aa.upper(), aa)


def seq_to_3letter(sequence: str, separator: str = '-') -> str:
    """
    서열 1-letter → 3-letter 변환

    Args:
        sequence: 1-letter 서열 (예: "EEEFD")
        separator: 잔기 구분자 (기본 "-")

    Returns:
        3-letter 서열 (예: "Glu-Glu-Glu-Phe-Asp")
    """
    if not sequence:
        return ""
    return separator.join(AA_1_TO_3.get(aa.upper(), aa) for aa in sequence)


def seq_with_tooltip(sequence: str, show_dotted: bool = True) -> str:
    """
    호버 시 3-letter를 보여주는 HTML span 반환

    Args:
        sequence: 1-letter 서열
        show_dotted: 점선 밑줄로 호버 가능함을 시각적으로 표시

    Returns:
        HTML span 문자열 (st.markdown(..., unsafe_allow_html=True)와 함께 사용)
    """
    if not sequence:
        return ""
    three_letter = seq_to_3letter(sequence)
    style = (
        'border-bottom: 1px dotted #888; cursor: help;'
        if show_dotted else 'cursor: help;'
    )
    # HTML escape: 서열엔 특수문자 없지만 안전을 위해
    return (f'<span title="{three_letter}" style="{style}">'
            f'{sequence}</span>')


def mutation_with_tooltip(mutation: str) -> str:
    """
    변이 표기 호버 툴팁 (예: "A5G" → 호버 시 "Ala5Gly")

    Args:
        mutation: 변이 표기 (예: "A5G", "L10V")

    Returns:
        HTML span 문자열
    """
    import re
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation.strip().upper())
    if not m:
        return mutation
    wt, pos, mt = m.groups()
    three = f"{AA_1_TO_3.get(wt, wt)}{pos}{AA_1_TO_3.get(mt, mt)}"
    return (f'<span title="{three}" '
            f'style="border-bottom: 1px dotted #888; cursor: help;">'
            f'{mutation}</span>')


def calculate_sequence_mw(sequence: str) -> float:
    """
    펩타이드 서열의 분자량 계산

    Args:
        sequence: 아미노산 서열 (1-letter code)

    Returns:
        분자량 (Da)
    """
    if not sequence:
        return 0.0

    mw = 0.0
    for aa in sequence.upper():
        props = get_aa_properties(aa)
        if props:
            mw += props['mw']

    # 펩타이드 결합으로 인한 물 분자 손실 (n-1개)
    mw -= 18.015 * (len(sequence) - 1)

    return mw


def classify_aa_by_property(aa_code: str) -> Dict[str, bool]:
    """
    아미노산을 여러 속성으로 분류

    Args:
        aa_code: 1-letter 아미노산 코드

    Returns:
        속성별 분류 딕셔너리
    """
    props = get_aa_properties(aa_code)
    if not props:
        return {'hydrophobic': False, 'charged': False, 'aromatic': False, 'polar': False}

    # 극성 아미노산 판단 (전하 또는 극성 측쇄)
    polar = aa_code.upper() in ['S', 'T', 'N', 'Q', 'C', 'Y']

    return {
        'hydrophobic': props['hydrophobic'],
        'charged': props['charged'],
        'aromatic': props['aromatic'],
        'polar': polar or props['charged']
    }


def hash_sequence(sequence: str) -> str:
    """
    서열 해시 생성 (캐싱용)

    Args:
        sequence: 아미노산 서열

    Returns:
        SHA256 해시 문자열
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    디렉토리 생성 (없으면)

    Args:
        directory: 디렉토리 경로

    Returns:
        Path 객체
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    JSON 파일 저장

    Args:
        data: 저장할 데이터
        filepath: 파일 경로
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Union[str, Path]) -> Optional[Dict]:
    """
    JSON 파일 로드

    Args:
        filepath: 파일 경로

    Returns:
        로드된 데이터 또는 None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 반환

    Returns:
        프로젝트 루트 Path
    """
    # src/utils.py의 부모의 부모 = 프로젝트 루트
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """
    데이터 디렉토리 경로 반환

    Returns:
        data 디렉토리 Path
    """
    return get_project_root() / 'data'


def get_cache_dir() -> Path:
    """
    캐시 디렉토리 경로 반환

    Returns:
        cache 디렉토리 Path
    """
    cache_dir = get_data_dir() / 'cache'
    ensure_dir(cache_dir)
    return cache_dir


def get_output_dir() -> Path:
    """
    출력 디렉토리 경로 반환

    Returns:
        output 디렉토리 Path
    """
    output_dir = get_project_root() / 'output'
    ensure_dir(output_dir)
    return output_dir


def normalize_composition(composition: Dict[str, float]) -> Dict[str, float]:
    """
    아미노산 조성 정규화 (합=1)

    Args:
        composition: 아미노산 조성 딕셔너리 {AA: percentage}

    Returns:
        정규화된 조성
    """
    total = sum(composition.values())
    if total == 0:
        return composition

    return {aa: val / total for aa, val in composition.items()}


def calculate_property_ratios(composition: Dict[str, float]) -> Dict[str, float]:
    """
    물리화학적 속성별 비율 계산

    Args:
        composition: 아미노산 조성 {AA: percentage}

    Returns:
        속성별 비율 딕셔너리
    """
    hydrophobic = 0.0
    charged = 0.0
    aromatic = 0.0
    polar = 0.0

    for aa, percentage in composition.items():
        props = classify_aa_by_property(aa)
        if props['hydrophobic']:
            hydrophobic += percentage
        if props['charged']:
            charged += percentage
        if props['aromatic']:
            aromatic += percentage
        if props['polar']:
            polar += percentage

    return {
        'hydrophobic_ratio': hydrophobic,
        'charged_ratio': charged,
        'aromatic_ratio': aromatic,
        'polar_ratio': polar
    }


def validate_sequence(sequence: str) -> bool:
    """
    아미노산 서열 유효성 검사

    Args:
        sequence: 아미노산 서열

    Returns:
        유효 여부
    """
    if not sequence:
        return False

    valid_aas = set(AMINO_ACIDS.keys())
    return all(aa.upper() in valid_aas for aa in sequence)


def format_sequence(sequence: str, line_length: int = 60) -> str:
    """
    서열을 FASTA 형식으로 포맷팅

    Args:
        sequence: 아미노산 서열
        line_length: 한 줄당 문자 수

    Returns:
        포맷팅된 서열
    """
    lines = []
    for i in range(0, len(sequence), line_length):
        lines.append(sequence[i:i+line_length])
    return '\n'.join(lines)


def save_fasta(sequences: List[tuple], filepath: Union[str, Path]) -> None:
    """
    서열을 FASTA 파일로 저장

    Args:
        sequences: [(header, sequence), ...] 리스트
        filepath: 저장 경로
    """
    with open(filepath, 'w') as f:
        for header, sequence in sequences:
            f.write(f'>{header}\n')
            f.write(format_sequence(sequence) + '\n\n')


if __name__ == '__main__':
    # 테스트
    print("아미노산 속성 테스트:")
    test_aas = ['A', 'R', 'W', 'P']
    for aa in test_aas:
        props = get_aa_properties(aa)
        print(f"{aa}: {props}")

    print("\n서열 분자량 계산 테스트:")
    test_seq = "ARWP"
    mw = calculate_sequence_mw(test_seq)
    print(f"{test_seq}: {mw:.2f} Da")

    print("\n속성 비율 계산 테스트:")
    composition = {'A': 10, 'R': 5, 'W': 3, 'P': 7}
    ratios = calculate_property_ratios(composition)
    print(ratios)
