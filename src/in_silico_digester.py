"""
In Silico Digester — 원료 단백질 + 효소 공정 기반 펩타이드 예측

회사 펩톤 제품의 실제 효소 공정을 시뮬레이션하여 펩타이드 후보를 생성합니다.
Markov chain 기반의 통계적 추정과 달리, 실제 효소 절단 규칙을 적용합니다.

파이프라인:
    원료 단백질 (UniProt FASTA)
       ↓
    전처리 효소 (예: Alcalase)
       ↓
    메인 효소 (예: ZF101 + Flavourzyme)
       ↓
    [BIO 시리즈] UF 3K dalton 컷오프 필터
       ↓
    예상 펩타이드 풀

사용 예:
    from in_silico_digester import InSilicoDigester
    digester = InSilicoDigester()
    peptides = digester.digest_product("SOY-1")
"""

import json
import re
import random
import hashlib
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from enzyme_processor import EnzymeProcessor, PeptoneProcess
except ImportError:
    from .enzyme_processor import EnzymeProcessor, PeptoneProcess


# ─── 데이터 클래스 ──────────────────────────────────────────
@dataclass
class DigestedPeptide:
    """분해 산물 펩타이드"""
    sequence: str
    length: int
    mw_da: float
    source_protein: str        # UniProt ID
    source_protein_name: str   # 단백질 이름
    source_material: str       # soy, rice, wheat, pea
    start_position: int        # 원본 단백질 내 시작 위치 (0-based)
    end_position: int
    digestion_stage: str       # "pretreatment", "main", "exo_trimming"


# ─── 효소 cleavage 규칙 ────────────────────────────────────
# 평균 AA MW = 110 Da (대략치)
AVG_AA_MW = 110.0
WATER_MW = 18.0  # 펩타이드 결합 분해 시 추가


def stable_sequence_seed(sequence: str, seed: int = 42) -> int:
    """Return a reproducible integer seed for a sequence across Python runs."""
    digest = hashlib.blake2b(sequence.encode("utf-8"), digest_size=8).hexdigest()
    return seed + (int(digest, 16) % 10000)


def calc_peptide_mw(seq: str) -> float:
    """간단한 분자량 계산 (잔기당 평균 + 물)"""
    aa_weights = {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
        'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
    }
    if not seq:
        return 0.0
    mw = sum(aa_weights.get(aa, AVG_AA_MW) for aa in seq) - WATER_MW * (len(seq) - 1)
    return round(mw, 1)


# ─── 효소별 절단 함수 ──────────────────────────────────────
def cleave_alcalase(sequence: str) -> List[Tuple[int, int]]:
    """
    Alcalase: F/Y/W/L/M/A 다음 절단

    Returns:
        [(start, end), ...] 분해 후 펩타이드 위치 리스트 (0-based, end exclusive)
    """
    # 절단 위치 (해당 AA 다음 = 다음 AA가 시작)
    cleave_after = {'F', 'Y', 'W', 'L', 'M', 'A'}
    cut_points = [0]
    for i, aa in enumerate(sequence):
        if aa in cleave_after and i + 1 < len(sequence):
            cut_points.append(i + 1)
    cut_points.append(len(sequence))

    return [(cut_points[i], cut_points[i + 1])
            for i in range(len(cut_points) - 1)]


def cleave_broad_random(sequence: str, target_avg_length: int = 9,
                         seed: int = 42) -> List[Tuple[int, int]]:
    """
    ZF101/Flavourzyme의 광범위 endo 활성 시뮬레이션
    평균 ~9 AA 펩타이드가 되도록 확률적 절단

    Args:
        sequence: 입력 서열
        target_avg_length: 목표 평균 펩타이드 길이 (~1000 Da)
        seed: 재현 가능성을 위한 시드

    Returns:
        [(start, end), ...]
    """
    if len(sequence) <= target_avg_length:
        return [(0, len(sequence))]

    rng = random.Random(stable_sequence_seed(sequence, seed))
    # 절단 확률 = 1 / target_avg_length
    cleavage_prob = 1.0 / target_avg_length

    cut_points = [0]
    for i in range(1, len(sequence)):
        if rng.random() < cleavage_prob:
            cut_points.append(i)
    cut_points.append(len(sequence))

    return [(cut_points[i], cut_points[i + 1])
            for i in range(len(cut_points) - 1)]


def trim_hydrophobic_termini(sequence: str,
                              targets: set = None,
                              probability: float = 0.8,
                              seed: int = 42) -> Tuple[str, int, int]:
    """
    ZF101/Flavourzyme의 exo 활성: 소수성 말단 잔기 제거 (debittering)

    Args:
        sequence: 입력 펩타이드
        targets: 제거 대상 AA (기본: L, I, V, F, Y)
        probability: 제거 확률
        seed: 재현 가능성

    Returns:
        (trimmed_seq, n_removed_N, n_removed_C)
    """
    if targets is None:
        targets = {'L', 'I', 'V', 'F', 'Y'}

    rng = random.Random(stable_sequence_seed(sequence, seed))

    # N-말단 trimming
    trim_n = 0
    while len(sequence) > 2 and sequence[0] in targets and rng.random() < probability:
        sequence = sequence[1:]
        trim_n += 1

    # C-말단 trimming
    trim_c = 0
    while len(sequence) > 2 and sequence[-1] in targets and rng.random() < probability:
        sequence = sequence[:-1]
        trim_c += 1

    return sequence, trim_n, trim_c


# ─── 메인 클래스 ────────────────────────────────────────────
class InSilicoDigester:
    """효소 공정 기반 펩타이드 예측 엔진"""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent

        self.base_dir = base_dir
        self.raw_proteins_dir = base_dir / "data" / "raw_proteins"

        # 효소 공정 메타데이터
        self.enzyme_processor = EnzymeProcessor(
            base_dir / "data" / "enzyme_processes.json"
        )

        # 원료 단백질 인덱스 로드
        index_path = self.raw_proteins_dir / "raw_proteins_index.json"
        if index_path.exists():
            with open(index_path, encoding='utf-8') as f:
                self.raw_proteins_index = json.load(f)
        else:
            self.raw_proteins_index = {"materials": {}}

    # ─── 원료 단백질 로드 ─────────────────────────────────
    def get_raw_proteins(self, material: str) -> List[Dict]:
        """원료별 단백질 리스트 반환"""
        return (self.raw_proteins_index.get("materials", {})
                .get(material, {}).get("proteins", []))

    # ─── 단일 단백질 분해 ─────────────────────────────────
    def digest_protein(self, protein_seq: str, process: PeptoneProcess,
                       source_id: str = "", source_name: str = "",
                       source_material: str = "",
                       min_length: int = 2, max_length: int = 50
                       ) -> List[DigestedPeptide]:
        """
        단일 단백질을 공정에 따라 분해

        Args:
            protein_seq: 원료 단백질 서열
            process: PeptoneProcess 메타데이터
            source_id: 원본 UniProt ID
            source_name: 원본 단백질 이름
            source_material: 원료 (soy/rice/wheat/pea)
            min_length: 최소 펩타이드 길이
            max_length: 최대 펩타이드 길이

        Returns:
            DigestedPeptide 리스트
        """
        if not protein_seq:
            return []

        # 사용 효소 파악
        enzymes_used = process.enzymes_used
        has_alcalase = "Alcalase" in enzymes_used
        has_exo = any(e in enzymes_used for e in ["ZF101", "Flavourzyme"])
        is_alcalase_only = enzymes_used == ["Alcalase"]

        peptides = []

        # ── 단계 1: 전처리 (Alcalase) ─────────────────
        # 위치 정보 유지하며 분해
        fragments_with_pos = [(protein_seq, 0)]  # (서열, 원본 내 시작 위치)

        if has_alcalase and process.pretreatment:
            new_fragments = []
            for frag, frag_start in fragments_with_pos:
                for s, e in cleave_alcalase(frag):
                    new_fragments.append((frag[s:e], frag_start + s))
            fragments_with_pos = new_fragments

        # SOY-P의 경우 메인도 Alcalase로 더 분해 (1.3% × 20h)
        if is_alcalase_only and process.main_hydrolysis:
            new_fragments = []
            for frag, frag_start in fragments_with_pos:
                # Alcalase 단독 메인: 더 강한 분해 (한번 더)
                for s, e in cleave_alcalase(frag):
                    new_fragments.append((frag[s:e], frag_start + s))
            fragments_with_pos = new_fragments

        stage = "alcalase_pretreatment"

        # ── 단계 2: 메인 효소 (ZF101 + Flavourzyme) ──
        if has_exo:
            stage = "main_hydrolysis"
            new_fragments = []
            for frag, frag_start in fragments_with_pos:
                # 광범위 endo 활성 (~1000 Da 목표 = ~9 AA)
                for s, e in cleave_broad_random(frag, target_avg_length=9):
                    new_fragments.append((frag[s:e], frag_start + s,
                                          "main_endo"))
            fragments_with_pos = [(f, p) for f, p, _ in new_fragments]

            # ── 단계 3: Exo 활성 (소수성 말단 제거) ──
            trimmed_fragments = []
            for frag, frag_start in fragments_with_pos:
                if len(frag) < 3:
                    continue
                trimmed, n_n, n_c = trim_hydrophobic_termini(frag)
                if len(trimmed) >= min_length:
                    trimmed_fragments.append(
                        (trimmed, frag_start + n_n,  # 시작 위치 보정
                         "main_exo_trimmed" if (n_n or n_c) else "main_endo")
                    )
            fragments_with_pos = [(f, p) for f, p, _ in trimmed_fragments]
            stage = "main_with_exo"

        # ── 단계 4: UF 필터 (BIO 시리즈) ──────────────
        if process.has_uf:
            cutoff_kda = process.uf_cutoff_kda
            max_aa = int(cutoff_kda * 1000 / AVG_AA_MW)  # ≈ 27 AA for 3K
            fragments_with_pos = [(f, p) for f, p in fragments_with_pos
                                  if len(f) <= max_aa]
            stage = f"uf_{cutoff_kda}k"

        # ── 결과 변환 ────────────────────────────────
        for frag, frag_start in fragments_with_pos:
            if min_length <= len(frag) <= max_length:
                peptides.append(DigestedPeptide(
                    sequence=frag,
                    length=len(frag),
                    mw_da=calc_peptide_mw(frag),
                    source_protein=source_id,
                    source_protein_name=source_name,
                    source_material=source_material,
                    start_position=frag_start,
                    end_position=frag_start + len(frag),
                    digestion_stage=stage
                ))

        return peptides

    # ─── 제품 단위 분해 ───────────────────────────────────
    def digest_product(self, product_id: str,
                       min_length: int = 3, max_length: int = 30,
                       n_top_proteins: Optional[int] = None
                       ) -> List[DigestedPeptide]:
        """
        제품 ID로 전체 in silico digestion 실행

        Args:
            product_id: 제품 ID (예: "SOY-1", "RICE-BIO")
            min_length: 최소 펩타이드 길이
            max_length: 최대 펩타이드 길이
            n_top_proteins: 상위 N개 단백질만 사용 (None = 전부)

        Returns:
            DigestedPeptide 리스트
        """
        # 공정 정보 가져오기
        process = self.enzyme_processor.get_process(product_id)
        if not process:
            return []

        # Blending 제품 (SOY-B): 베이스 제품의 펩타이드 + 미분해 단백질
        if process.is_blend:
            return self._digest_blend(process, min_length, max_length)

        # BIO 시리즈: 베이스 제품 분해 후 UF 적용
        if process.category == "2nd_gen_bio" and process.based_on:
            base_peptides = self.digest_product(
                process.based_on, min_length=min_length, max_length=max_length,
                n_top_proteins=n_top_proteins
            )
            # UF 필터 적용
            cutoff_kda = process.uf_cutoff_kda
            max_aa = int(cutoff_kda * 1000 / AVG_AA_MW)
            return [p for p in base_peptides if p.length <= max_aa]

        # 1세대 제품: 원료 단백질 직접 분해
        material = process.raw_material_id
        proteins = self.get_raw_proteins(material)
        if n_top_proteins:
            # 길이 내림차순 (이미 정렬되어 있음) 기준 상위 N개
            proteins = proteins[:n_top_proteins]

        all_peptides = []
        for protein in proteins:
            peptides = self.digest_protein(
                protein_seq=protein["sequence"],
                process=process,
                source_id=protein["uniprot_id"],
                source_name=protein["full_name"],
                source_material=material,
                min_length=min_length,
                max_length=max_length
            )
            all_peptides.extend(peptides)

        return all_peptides

    def _digest_blend(self, process: PeptoneProcess,
                      min_length: int, max_length: int) -> List[DigestedPeptide]:
        """Blending 제품 처리 (예: SOY-B = SOY-1 90% + 탈지대두 10%)"""
        # SOY-B = SOY-1 펩타이드 + 미분해 탈지대두 단편
        # 단순화: SOY-1 펩타이드만 반환 (미분해 단백질은 ESM-2/DB에서 처리)
        return self.digest_product("SOY-1", min_length, max_length)

    # ─── 유틸리티 ─────────────────────────────────────────
    def get_unique_peptides(self, peptides: List[DigestedPeptide]
                            ) -> List[DigestedPeptide]:
        """중복 제거 (동일 서열 → 첫 출처만 유지)"""
        seen = {}
        for p in peptides:
            if p.sequence not in seen:
                seen[p.sequence] = p
        return list(seen.values())

    def get_aa_composition_from_raw_proteins(self, material: str) -> Dict[str, float]:
        """
        원료 단백질 FASTA에서 AA 조성 계산 (Markov chain 입력용)

        Returns:
            {AA: percentage} 형식의 조성 비율
        """
        proteins = self.get_raw_proteins(material)
        if not proteins:
            return {}

        from collections import Counter
        all_aa = ''.join(p['sequence'] for p in proteins)
        counts = Counter(all_aa)
        total = sum(counts.values())
        if total == 0:
            return {}

        valid_aa = 'ACDEFGHIKLMNPQRSTVWY'
        return {aa: round(counts.get(aa, 0) / total * 100, 3)
                for aa in valid_aa}

    # ─── Process-Aware Markov (효소 공정 인식) ───────────
    def _build_process_aware_markov(self, peptide_pool: List[DigestedPeptide]) -> Dict:
        """
        In silico digestion 산물 풀에서 Markov 통계 학습

        학습 항목:
        - 길이 분포: 실제 분해 산물의 길이 빈도
        - 시작 AA 빈도: post-cleavage 위치의 AA 분포
        - 종료 AA 빈도: 절단 site 또는 exo-trimming 후 말단 AA
        - Dipeptide 전이 행렬: 펩타이드 내부의 AA-AA 연결 패턴
        """
        from collections import Counter, defaultdict

        if not peptide_pool:
            return {}

        sequences = [p.sequence for p in peptide_pool]

        # 1. 길이 분포
        length_dist = Counter(len(s) for s in sequences)

        # 2. 시작/종료 AA
        start_aa = Counter(s[0] for s in sequences if s)
        end_aa = Counter(s[-1] for s in sequences if s)

        # 3. Dipeptide 전이 (펩타이드 내부 연결)
        transitions = defaultdict(lambda: defaultdict(int))
        for s in sequences:
            for i in range(len(s) - 1):
                transitions[s[i]][s[i + 1]] += 1

        # 정규화
        trans_prob = {}
        for aa1, counts in transitions.items():
            total = sum(counts.values())
            if total > 0:
                trans_prob[aa1] = {aa2: c / total for aa2, c in counts.items()}

        return {
            'length_dist': dict(length_dist),
            'start_aa_freq': dict(start_aa),
            'end_aa_freq': dict(end_aa),
            'transition_prob': trans_prob,
            'n_training_peptides': len(sequences),
        }

    def _generate_process_aware_markov(self, markov_stats: Dict,
                                        n_sequences: int,
                                        min_length: int = 3,
                                        max_length: int = 20,
                                        seed: int = 12345) -> List[Tuple[str, float]]:
        """
        학습된 통계 기반 Markov 생성

        in silico digestion 산물의 분포 안에서 변이체를 샘플링하므로
        in silico 결과와 자연스럽게 겹치게 됨.
        """
        if not markov_stats:
            return []

        rng = random.Random(seed)
        length_dist = markov_stats['length_dist']
        start_freq = markov_stats['start_aa_freq']
        end_freq = markov_stats['end_aa_freq']
        trans_prob = markov_stats['transition_prob']

        # 분포 → 샘플링 가능한 형태로
        if not length_dist or not start_freq or not end_freq or not trans_prob:
            return []

        lengths = list(length_dist.keys())
        length_weights = list(length_dist.values())

        starts = list(start_freq.keys())
        start_weights = list(start_freq.values())

        # 펩타이드 가능성 점수 계산용 (조성)
        max_start = max(start_weights)
        max_end = max(end_freq.values())

        sequences = {}
        max_attempts = max(n_sequences * 6, n_sequences + 100)
        attempts = 0
        while len(sequences) < n_sequences and attempts < max_attempts:
            attempts += 1
            # 길이 샘플링
            target_len = rng.choices(lengths, weights=length_weights)[0]
            target_len = max(min_length, min(max_length, target_len))

            # 시작 AA 샘플링
            seq = [rng.choices(starts, weights=start_weights)[0]]
            start_score = start_freq[seq[0]] / max_start

            # 전이 행렬로 확장
            log_prob = 0.0
            for _ in range(target_len - 1):
                current = seq[-1]
                if current in trans_prob and trans_prob[current]:
                    next_aas = list(trans_prob[current].keys())
                    weights = list(trans_prob[current].values())
                    next_aa = rng.choices(next_aas, weights=weights)[0]
                    prob = trans_prob[current][next_aa]
                    log_prob += math.log(prob if prob > 0 else 1e-10)
                    seq.append(next_aa)
                else:
                    break

            seq_str = ''.join(seq)
            if min_length <= len(seq_str) <= max_length:
                end_count = end_freq.get(seq_str[-1], 0)
                if end_count <= 0:
                    continue

                terminal_acceptance = end_count / max_end
                if rng.random() > terminal_acceptance:
                    continue

                transition_score = math.exp(log_prob / max(len(seq_str) - 1, 1))
                end_score = end_count / max_end
                score = 0.6 * transition_score + 0.2 * start_score + 0.2 * end_score
                score = max(0.0, min(1.0, score))
                if seq_str not in sequences or score > sequences[seq_str]:
                    sequences[seq_str] = float(score)

        return sorted(sequences.items(), key=lambda x: x[1], reverse=True)

    def hybrid_digest_and_markov(self, product_id: str,
                                  min_length: int = 3,
                                  max_length: int = 20,
                                  n_top_proteins: int = 20,
                                  n_markov_sequences: int = 500) -> Dict:
        """
        Hybrid 모드: In Silico Digestion + Markov Chain 결합

        Args:
            product_id: 제품 ID (예: "SOY-1")
            min_length: 최소 펩타이드 길이
            max_length: 최대 펩타이드 길이
            n_top_proteins: in silico에 사용할 원료 상위 N개
            n_markov_sequences: Markov로 생성할 추가 서열 수

        Returns:
            {
              "in_silico_peptides": [DigestedPeptide, ...],
              "markov_sequences": [(seq, score), ...],
              "combined_sequences": [(seq, source, score), ...],
              "overlap_count": int,
              "aa_composition": {AA: %},
              "product_id": str
            }
        """
        # 1. In Silico Digestion
        in_silico_peptides = self.digest_product(
            product_id,
            min_length=min_length,
            max_length=max_length,
            n_top_proteins=n_top_proteins
        )
        unique_in_silico = self.get_unique_peptides(in_silico_peptides)
        in_silico_seqs = set(p.sequence for p in unique_in_silico)

        # 2. AA 조성 추출 (참고용 - 결과에 포함)
        process = self.enzyme_processor.get_process(product_id)
        material = process.raw_material_id if process else "soy"
        if process and process.based_on:
            base_process = self.enzyme_processor.get_process(process.based_on)
            if base_process:
                material = base_process.raw_material_id

        aa_composition = self.get_aa_composition_from_raw_proteins(material)

        # 3. Process-Aware Markov
        # 효소 분해 산물 풀에서 길이/시작AA/dipeptide 분포 학습 후 변이체 생성
        # → in silico 분포 안에서 샘플링하므로 overlap 비율이 자연스럽게 높아짐
        markov_stats = self._build_process_aware_markov(unique_in_silico)
        markov_sequences = self._generate_process_aware_markov(
            markov_stats,
            n_sequences=n_markov_sequences,
            min_length=min_length,
            max_length=max_length
        )

        markov_seqs = set(seq for seq, _ in markov_sequences)

        # 4. 결합 (Union + 소스 라벨링)
        # 'both'      : in silico와 markov 모두 생성 → 최고 신뢰도
        # 'in_silico' : in silico만 → 효소 분해 결정론적 산물
        # 'markov'    : markov만 → 통계적 후보
        combined = []

        # In silico 펩타이드 먼저 추가
        for p in unique_in_silico:
            source = "both" if p.sequence in markov_seqs else "in_silico"
            # 점수: in silico = 1.0 (결정론적), both = 1.3 (양쪽 확인)
            score = 1.3 if source == "both" else 1.0
            combined.append({
                "sequence": p.sequence,
                "length": p.length,
                "mw_da": p.mw_da,
                "source": source,
                "score": score,
                "source_protein": p.source_protein,
                "source_protein_name": p.source_protein_name,
                "digestion_stage": p.digestion_stage,
            })

        # Markov만 있는 서열 추가
        for seq, markov_score in markov_sequences:
            if seq not in in_silico_seqs:
                combined.append({
                    "sequence": seq,
                    "length": len(seq),
                    "mw_da": calc_peptide_mw(seq),
                    "source": "markov",
                    "score": 0.8 * markov_score,  # Markov는 가중치 낮춤
                    "source_protein": "",
                    "source_protein_name": "(Markov-generated)",
                    "digestion_stage": "markov",
                })

        # 점수 내림차순 정렬
        combined.sort(key=lambda x: x["score"], reverse=True)

        overlap = sum(1 for c in combined if c["source"] == "both")

        return {
            "in_silico_peptides": unique_in_silico,
            "markov_sequences": markov_sequences,
            "combined_sequences": combined,
            "overlap_count": overlap,
            "n_in_silico_only": sum(1 for c in combined if c["source"] == "in_silico"),
            "n_markov_only": sum(1 for c in combined if c["source"] == "markov"),
            "aa_composition": aa_composition,
            "product_id": product_id,
            "markov_method": "process_aware",  # 더 이상 단순 조성 기반 아님
            "markov_training_size": markov_stats.get('n_training_peptides', 0),
        }

    def summarize(self, peptides: List[DigestedPeptide]) -> Dict:
        """펩타이드 풀 요약 통계"""
        if not peptides:
            return {"n_peptides": 0}

        lengths = [p.length for p in peptides]
        mws = [p.mw_da for p in peptides]
        proteins = set(p.source_protein for p in peptides)

        from collections import Counter
        stage_counts = Counter(p.digestion_stage for p in peptides)

        return {
            "n_peptides": len(peptides),
            "n_unique": len(set(p.sequence for p in peptides)),
            "n_source_proteins": len(proteins),
            "length_min": min(lengths),
            "length_max": max(lengths),
            "length_avg": round(sum(lengths) / len(lengths), 1),
            "mw_min": min(mws),
            "mw_max": max(mws),
            "mw_avg": round(sum(mws) / len(mws), 1),
            "stages": dict(stage_counts),
        }


# ─── 셀프 테스트 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    digester = InSilicoDigester()

    test_products = ["SOY-1", "SOY-P", "RICE-1", "WHEAT-1", "PEA-1", "SOY-BIO"]

    print("=" * 80)
    print(" In Silico Digestion Test")
    print("=" * 80)

    for pid in test_products:
        print(f"\n[{pid}]")
        peptides = digester.digest_product(pid)
        unique = digester.get_unique_peptides(peptides)
        summary = digester.summarize(unique)

        print(f"  Total: {summary['n_peptides']} peptides ({summary['n_unique']} unique)")
        print(f"  From : {summary['n_source_proteins']} source proteins")
        print(f"  Length: {summary['length_min']}-{summary['length_max']} AA "
              f"(avg {summary['length_avg']})")
        print(f"  MW   : {summary['mw_min']:.0f}-{summary['mw_max']:.0f} Da "
              f"(avg {summary['mw_avg']:.0f})")
        print(f"  Stages: {summary['stages']}")

        # 샘플 펩타이드
        print(f"  Sample peptides:")
        for p in unique[:5]:
            print(f"    {p.sequence:20s} {p.length:3d} AA  "
                  f"{p.mw_da:7.1f} Da  from {p.source_protein}")
