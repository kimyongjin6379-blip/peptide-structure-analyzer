"""
펩타이드 서열 예측 모듈
Peptide sequence prediction based on amino acid composition
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    from .data_loader import CompositionLoader
    from .utils import validate_sequence, calculate_sequence_mw, AMINO_ACIDS
except ImportError:
    from data_loader import CompositionLoader
    from utils import validate_sequence, calculate_sequence_mw, AMINO_ACIDS


class SequenceGenerator:
    """
    통계적 서열 생성 (아미노산 조성 기반)
    """

    def __init__(self, aa_composition: Dict[str, float]):
        """
        초기화

        Args:
            aa_composition: 아미노산 조성 {AA: percentage}
        """
        self.composition = aa_composition
        self.probabilities = self._normalize_composition()
        self.amino_acids = list(self.probabilities.keys())
        self.weights = list(self.probabilities.values())

    def _normalize_composition(self) -> Dict[str, float]:
        """
        조성 정규화 (합 = 1)

        Returns:
            정규화된 확률
        """
        total = sum(self.composition.values())
        if total == 0:
            return {}

        return {aa: val / total for aa, val in self.composition.items()}

    def generate_sequences(self,
                          length_range: Tuple[int, int] = (3, 20),
                          n_sequences: int = 100,
                          method: str = 'markov') -> List[Tuple[str, float]]:
        """
        펩타이드 서열 생성

        Args:
            length_range: 길이 범위 (min, max)
            n_sequences: 생성할 서열 수
            method: 생성 방법 ('random', 'markov', 'frequent')

        Returns:
            [(서열, 가능성 점수), ...] 리스트
        """
        if not self.probabilities:
            return []

        sequences = []

        if method == 'random':
            sequences = self._generate_random(length_range, n_sequences)
        elif method == 'markov':
            sequences = self._generate_markov(length_range, n_sequences)
        elif method == 'frequent':
            sequences = self._generate_frequent(length_range, n_sequences)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 가능성 점수 계산 및 정렬
        scored_sequences = []
        for seq in sequences:
            score = self.score_sequence_likelihood(seq)
            scored_sequences.append((seq, score))

        # 점수로 정렬 (높은 순)
        scored_sequences.sort(key=lambda x: x[1], reverse=True)

        return scored_sequences

    def _generate_random(self, length_range: Tuple[int, int],
                        n_sequences: int) -> List[str]:
        """
        단순 확률 샘플링

        Args:
            length_range: 길이 범위
            n_sequences: 서열 수

        Returns:
            서열 리스트
        """
        sequences = []

        for _ in range(n_sequences):
            length = random.randint(length_range[0], length_range[1])
            seq = ''.join(random.choices(self.amino_acids,
                                        weights=self.weights,
                                        k=length))
            sequences.append(seq)

        return sequences

    def _generate_markov(self, length_range: Tuple[int, int],
                        n_sequences: int) -> List[str]:
        """
        1st-order Markov chain 기반 생성

        Args:
            length_range: 길이 범위
            n_sequences: 서열 수

        Returns:
            서열 리스트
        """
        # 전이 행렬 구축
        transition_matrix = self._build_transition_matrix()

        sequences = []

        for _ in range(n_sequences):
            length = random.randint(length_range[0], length_range[1])

            # 첫 아미노산 선택
            seq = [random.choices(self.amino_acids, weights=self.weights)[0]]

            # Markov chain으로 나머지 생성
            for _ in range(length - 1):
                current_aa = seq[-1]
                next_aas = list(transition_matrix[current_aa].keys())
                next_weights = list(transition_matrix[current_aa].values())

                if next_aas:
                    next_aa = random.choices(next_aas, weights=next_weights)[0]
                    seq.append(next_aa)
                else:
                    # fallback: 전체 확률로 선택
                    seq.append(random.choices(self.amino_acids,
                                             weights=self.weights)[0])

            sequences.append(''.join(seq))

        return sequences

    def _build_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        전이 행렬 구축

        인접 아미노산 선호도:
        - 소수성-소수성: 선호 (1.2배)
        - 같은 전하-같은 전하: 회피 (0.6배)
        - 극성-극성: 중립 (1.0배)

        Returns:
            전이 행렬 {AA1: {AA2: probability}}
        """
        hydrophobic = {'A', 'I', 'L', 'M', 'F', 'P', 'W', 'V'}
        positive = {'K', 'R', 'H'}
        negative = {'D', 'E'}

        transition_matrix = defaultdict(dict)

        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                # 기본 확률
                prob = self.probabilities[aa2]

                # 선호도 조정
                if aa1 in hydrophobic and aa2 in hydrophobic:
                    prob *= 1.2  # 소수성끼리 선호
                elif (aa1 in positive and aa2 in positive) or \
                     (aa1 in negative and aa2 in negative):
                    prob *= 0.6  # 같은 전하 회피

                transition_matrix[aa1][aa2] = prob

            # 정규화
            total = sum(transition_matrix[aa1].values())
            if total > 0:
                for aa2 in transition_matrix[aa1]:
                    transition_matrix[aa1][aa2] /= total

        return transition_matrix

    def _generate_frequent(self, length_range: Tuple[int, int],
                          n_sequences: int) -> List[str]:
        """
        가장 빈도 높은 아미노산 조합 우선

        Args:
            length_range: 길이 범위
            n_sequences: 서열 수

        Returns:
            서열 리스트
        """
        # 상위 아미노산 추출
        top_aas = sorted(self.probabilities.items(),
                        key=lambda x: x[1], reverse=True)[:5]
        top_aa_list = [aa for aa, _ in top_aas]

        sequences = []

        for _ in range(n_sequences):
            length = random.randint(length_range[0], length_range[1])

            # 상위 AA 위주로 생성 (80%), 나머지는 전체 확률로
            seq = []
            for _ in range(length):
                if random.random() < 0.8:
                    seq.append(random.choice(top_aa_list))
                else:
                    seq.append(random.choices(self.amino_acids,
                                             weights=self.weights)[0])

            sequences.append(''.join(seq))

        return sequences

    def score_sequence_likelihood(self, sequence: str) -> float:
        """
        서열 가능성 점수 계산 (0-1 범위, 기하평균 기반)

        점수 해석:
        - 1.0: 가장 흔한 아미노산만으로 구성
        - 0.5: 평균적인 조합
        - 0.0: 매우 희귀한 아미노산으로 구성

        Args:
            sequence: 아미노산 서열

        Returns:
            가능성 점수 (0-1)
        """
        if not sequence:
            return 0.0

        # 각 아미노산의 확률 곱 (로그 공간)
        log_sum = 0.0
        for aa in sequence:
            prob = self.probabilities.get(aa, 1e-10)
            log_sum += np.log(prob + 1e-10)

        # 기하평균 = exp(평균 log 확률)
        geometric_mean = np.exp(log_sum / len(sequence))

        # 정규화 범위 설정
        max_prob = max(self.probabilities.values())
        min_prob = 0.001  # 매우 희귀한 AA의 하한선

        # 로그 스케일로 부드럽게 변환
        log_geom = np.log(geometric_mean + 1e-10)
        log_max = np.log(max_prob)
        log_min = np.log(min_prob)

        # 0-1 범위로 선형 변환
        if log_max > log_min:
            score = (log_geom - log_min) / (log_max - log_min)
            score = max(0.0, min(1.0, score))
        else:
            score = 0.5

        return score

    def find_abundant_peptides(self, length: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        특정 길이의 가장 가능성 높은 펩타이드 찾기

        Args:
            length: 펩타이드 길이
            top_n: 상위 n개

        Returns:
            [(서열, 점수), ...] 리스트
        """
        # 더 많은 후보 생성 (top_n * 10)
        candidates = self.generate_sequences(
            length_range=(length, length),
            n_sequences=top_n * 10,
            method='markov'
        )

        # 중복 제거 및 상위 선택
        unique_candidates = {}
        for seq, score in candidates:
            if seq not in unique_candidates:
                unique_candidates[seq] = score

        # 점수로 정렬
        sorted_candidates = sorted(unique_candidates.items(),
                                  key=lambda x: x[1], reverse=True)

        return sorted_candidates[:top_n]


class AbundancePredictor:
    """
    펩타이드 풍부도 예측
    """

    def __init__(self, loader: CompositionLoader):
        """
        초기화

        Args:
            loader: CompositionLoader 인스턴스
        """
        self.loader = loader

    def predict_for_sample(self, sample_id: str,
                          length_range: Tuple[int, int] = (3, 15),
                          n_sequences: int = 50,
                          method: str = 'markov') -> Dict:
        """
        샘플의 풍부 펩타이드 예측

        Args:
            sample_id: 샘플 ID
            length_range: 길이 범위
            n_sequences: 생성 서열 수
            method: 생성 방법

        Returns:
            예측 결과 딕셔너리
        """
        # 펩타이드 조성 가져오기 (TAA - FAA, 정규화)
        taa_comp = self.loader.get_peptide_composition(sample_id, normalize=True)

        if not taa_comp:
            return {'error': f'샘플 {sample_id}를 찾을 수 없습니다'}

        # 서열 생성기 초기화
        generator = SequenceGenerator(taa_comp)

        # 서열 생성
        sequences = generator.generate_sequences(
            length_range=length_range,
            n_sequences=n_sequences,
            method=method
        )

        # 길이별 그룹화
        by_length = defaultdict(list)
        for seq, score in sequences:
            by_length[len(seq)].append((seq, score))

        # 각 길이별 상위 서열
        top_by_length = {}
        for length, seqs in by_length.items():
            top_by_length[length] = sorted(seqs, key=lambda x: x[1], reverse=True)[:5]

        # 분자량 계산
        sequences_with_mw = []
        for seq, score in sequences[:20]:  # 상위 20개
            mw = calculate_sequence_mw(seq)
            sequences_with_mw.append({
                'sequence': seq,
                'length': len(seq),
                'likelihood_score': score,
                'molecular_weight': mw
            })

        result = {
            'sample_id': sample_id,
            'method': method,
            'n_generated': len(sequences),
            'top_sequences': sequences[:10],
            'by_length': dict(top_by_length),
            'sequences_with_mw': sequences_with_mw,
            'composition': taa_comp
        }

        return result

    def compare_samples_sequences(self, sample_ids: List[str],
                                 length: int = 10) -> Dict:
        """
        여러 샘플의 펩타이드 서열 비교

        Args:
            sample_ids: 샘플 ID 리스트
            length: 비교할 펩타이드 길이

        Returns:
            비교 결과
        """
        results = {}

        for sample_id in sample_ids:
            taa_comp = self.loader.get_peptide_composition(sample_id, normalize=True)
            if not taa_comp:
                continue

            generator = SequenceGenerator(taa_comp)
            top_peptides = generator.find_abundant_peptides(length=length, top_n=5)
            results[sample_id] = top_peptides

        return {
            'length': length,
            'sample_results': results
        }


class PeptideEnumerator:
    """
    짧은 펩타이드 열거 (3-5 AA)
    """

    @staticmethod
    def enumerate_peptides(composition: Dict[str, float],
                          length: int,
                          max_count: int = 100) -> List[Tuple[str, float]]:
        """
        주어진 길이의 모든 가능한 펩타이드 열거

        Args:
            composition: 아미노산 조성
            length: 펩타이드 길이
            max_count: 최대 반환 수

        Returns:
            [(서열, 점수), ...] 리스트
        """
        if length > 5:
            raise ValueError("길이가 5보다 큰 경우 열거 방식은 비효율적입니다")

        # 상위 아미노산만 사용 (조합 폭발 방지)
        top_aas = sorted(composition.items(), key=lambda x: x[1], reverse=True)[:8]
        amino_acids = [aa for aa, _ in top_aas]

        # 가능한 모든 조합 생성
        from itertools import product

        all_seqs = [''.join(combo) for combo in product(amino_acids, repeat=length)]

        # 점수 계산
        generator = SequenceGenerator(composition)
        scored_seqs = [(seq, generator.score_sequence_likelihood(seq))
                      for seq in all_seqs]

        # 정렬 후 상위 반환
        scored_seqs.sort(key=lambda x: x[1], reverse=True)

        return scored_seqs[:max_count]


if __name__ == '__main__':
    # 테스트
    print("=== Sequence Predictor 테스트 ===\n")

    loader = CompositionLoader()
    loader.load_data()

    samples = loader.get_sample_list()
    test_sample = samples[0]

    print(f"테스트 샘플: {test_sample}\n")

    # 1. 서열 생성
    print("1. 서열 생성 테스트")
    predictor = AbundancePredictor(loader)

    result = predictor.predict_for_sample(
        test_sample,
        length_range=(5, 10),
        n_sequences=30,
        method='markov'
    )

    print(f"  생성된 서열 수: {result['n_generated']}")
    print(f"  상위 10개 서열:")
    for i, (seq, score) in enumerate(result['top_sequences'][:10], 1):
        mw = calculate_sequence_mw(seq)
        print(f"    {i}. {seq} (길이: {len(seq)}, 점수: {score:.4f}, MW: {mw:.1f} Da)")

    # 2. 길이별 서열
    print("\n2. 길이별 상위 서열")
    for length in sorted(result['by_length'].keys())[:3]:
        seqs = result['by_length'][length]
        print(f"  길이 {length}:")
        for seq, score in seqs[:3]:
            print(f"    {seq} ({score:.4f})")

    # 3. 특정 길이 풍부 펩타이드
    print("\n3. 특정 길이 풍부 펩타이드 (길이 7)")
    taa_comp = loader.get_peptide_composition(test_sample, normalize=True)
    generator = SequenceGenerator(taa_comp)
    abundant = generator.find_abundant_peptides(length=7, top_n=5)

    for i, (seq, score) in enumerate(abundant, 1):
        print(f"  {i}. {seq} ({score:.4f})")

    # 4. 짧은 펩타이드 열거
    print("\n4. 짧은 펩타이드 열거 (길이 3)")
    short_peptides = PeptideEnumerator.enumerate_peptides(
        taa_comp, length=3, max_count=10
    )

    for i, (seq, score) in enumerate(short_peptides[:10], 1):
        print(f"  {i}. {seq} ({score:.4f})")

    # 5. 샘플 간 비교
    if len(samples) >= 3:
        print("\n5. 샘플 간 서열 비교 (길이 8)")
        comparison = predictor.compare_samples_sequences(samples[:3], length=8)

        for sample_id, peptides in comparison['sample_results'].items():
            print(f"  {sample_id}:")
            for seq, score in peptides[:3]:
                print(f"    {seq} ({score:.4f})")

    print("\n[OK] Sequence Predictor 테스트 완료!")
