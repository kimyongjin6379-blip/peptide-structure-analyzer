"""
펩타이드 조성 분석 모듈
Peptide composition and molecular weight analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from .data_loader import CompositionLoader
    from .utils import calculate_property_ratios, AMINO_ACIDS
except ImportError:
    from data_loader import CompositionLoader
    from utils import calculate_property_ratios, AMINO_ACIDS


class PeptideCompositionAnalyzer:
    """
    아미노산 조성 기반 펩타이드 분석
    """

    def __init__(self, loader: CompositionLoader):
        """
        초기화

        Args:
            loader: CompositionLoader 인스턴스
        """
        self.loader = loader
        if self.loader.data is None:
            self.loader.load_data()

    def analyze_sample(self, sample_id: str) -> Dict:
        """
        단일 샘플 분석

        Args:
            sample_id: 샘플 ID

        Returns:
            분석 결과 딕셔너리
        """
        # 조성 데이터 가져오기
        taa_comp = self.loader.get_taa_composition(sample_id)
        faa_comp = self.loader.get_faa_composition(sample_id)

        if not taa_comp:
            return {'error': f'샘플 {sample_id}를 찾을 수 없습니다'}

        # 물리화학적 특성 계산
        taa_props = calculate_property_ratios(taa_comp)
        faa_props = calculate_property_ratios(faa_comp)

        # 조성 통계
        taa_values = list(taa_comp.values())
        faa_values = list(faa_comp.values())

        result = {
            'sample_id': sample_id,
            'taa_composition': taa_comp,
            'faa_composition': faa_comp,
            'taa_properties': taa_props,
            'faa_properties': faa_props,
            'taa_statistics': {
                'n_amino_acids': len(taa_comp),
                'total': sum(taa_values),
                'mean': np.mean(taa_values),
                'std': np.std(taa_values),
                'max_aa': max(taa_comp, key=taa_comp.get),
                'max_value': max(taa_values),
                'min_aa': min(taa_comp, key=taa_comp.get),
                'min_value': min(taa_values),
            },
            'faa_statistics': {
                'n_amino_acids': len(faa_comp),
                'total': sum(faa_values) if faa_values else 0,
                'mean': np.mean(faa_values) if faa_values else 0,
                'std': np.std(faa_values) if faa_values else 0,
            }
        }

        # TAA/FAA 비율
        result['taa_faa_ratio'] = self._calculate_taa_faa_ratio(taa_comp, faa_comp)

        return result

    def _calculate_taa_faa_ratio(self, taa_comp: Dict[str, float],
                                  faa_comp: Dict[str, float]) -> Dict[str, float]:
        """
        TAA/FAA 비율 계산

        Args:
            taa_comp: TAA 조성
            faa_comp: FAA 조성

        Returns:
            비율 딕셔너리
        """
        ratios = {}
        for aa in set(taa_comp.keys()) | set(faa_comp.keys()):
            taa_val = taa_comp.get(aa, 0)
            faa_val = faa_comp.get(aa, 0)
            if taa_val > 0:
                ratios[aa] = faa_val / taa_val
            else:
                ratios[aa] = 0

        return ratios

    def compare_samples(self, sample_ids: List[str]) -> Dict:
        """
        여러 샘플 비교 분석

        Args:
            sample_ids: 비교할 샘플 ID 리스트

        Returns:
            비교 결과 딕셔너리
        """
        results = {}

        # 각 샘플 분석
        sample_analyses = {}
        for sample_id in sample_ids:
            sample_analyses[sample_id] = self.analyze_sample(sample_id)

        # TAA 조성 비교 DataFrame
        taa_comparison = pd.DataFrame({
            sid: analysis['taa_composition']
            for sid, analysis in sample_analyses.items()
        }).T.fillna(0)

        # FAA 조성 비교 DataFrame
        faa_comparison = pd.DataFrame({
            sid: analysis['faa_composition']
            for sid, analysis in sample_analyses.items()
        }).T.fillna(0)

        # 물리화학적 특성 비교
        taa_props_comparison = pd.DataFrame({
            sid: analysis['taa_properties']
            for sid, analysis in sample_analyses.items()
        }).T

        results['taa_composition_df'] = taa_comparison
        results['faa_composition_df'] = faa_comparison
        results['taa_properties_df'] = taa_props_comparison
        results['sample_analyses'] = sample_analyses

        # 유사도 계산 (코사인 유사도)
        results['similarity_matrix'] = self._calculate_similarity(taa_comparison)

        return results

    def _calculate_similarity(self, composition_df: pd.DataFrame) -> pd.DataFrame:
        """
        샘플 간 조성 유사도 계산 (코사인 유사도)

        Args:
            composition_df: 조성 DataFrame

        Returns:
            유사도 행렬
        """
        from sklearn.metrics.pairwise import cosine_similarity

        similarity = cosine_similarity(composition_df.values)
        return pd.DataFrame(
            similarity,
            index=composition_df.index,
            columns=composition_df.index
        )

    def cluster_by_composition(self, n_clusters: int = 5,
                               feature: str = 'taa') -> Dict:
        """
        조성 기반 샘플 클러스터링

        Args:
            n_clusters: 클러스터 수
            feature: 'taa' 또는 'faa'

        Returns:
            클러스터링 결과
        """
        # 전체 샘플의 조성 데이터 수집
        samples = self.loader.get_sample_list()
        compositions = []

        for sample_id in samples:
            if feature == 'taa':
                comp = self.loader.get_taa_composition(sample_id)
            else:
                comp = self.loader.get_faa_composition(sample_id)

            compositions.append(comp)

        # DataFrame으로 변환
        comp_df = pd.DataFrame(compositions, index=samples).fillna(0)

        # 표준화
        scaler = StandardScaler()
        comp_scaled = scaler.fit_transform(comp_df.values)

        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(comp_scaled)

        # PCA로 2D 시각화용 좌표
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(comp_scaled)

        result = {
            'clusters': dict(zip(samples, clusters)),
            'cluster_centers': kmeans.cluster_centers_,
            'pca_coords': dict(zip(samples, pca_coords)),
            'pca_explained_variance': pca.explained_variance_ratio_,
            'composition_df': comp_df,
            'n_samples_per_cluster': pd.Series(clusters).value_counts().to_dict()
        }

        return result

    def get_representative_amino_acids(self, sample_id: str,
                                       top_n: int = 5) -> Dict:
        """
        샘플의 대표 아미노산 추출

        Args:
            sample_id: 샘플 ID
            top_n: 상위 n개

        Returns:
            대표 아미노산 정보
        """
        taa_comp = self.loader.get_taa_composition(sample_id)

        if not taa_comp:
            return {}

        # 상위 n개 아미노산
        sorted_aas = sorted(taa_comp.items(), key=lambda x: x[1], reverse=True)
        top_aas = sorted_aas[:top_n]

        # 각 아미노산 정보
        aa_info = []
        for aa, percentage in top_aas:
            props = AMINO_ACIDS.get(aa, {})
            aa_info.append({
                'code': aa,
                'name': props.get('name', aa),
                'percentage': percentage,
                'mw': props.get('mw', 0),
                'hydrophobic': props.get('hydrophobic', False),
                'charged': props.get('charged', False),
                'aromatic': props.get('aromatic', False),
            })

        return {
            'sample_id': sample_id,
            'top_amino_acids': aa_info,
            'total_percentage': sum(aa[1] for aa in top_aas)
        }


class MolecularWeightAnalyzer:
    """
    분자량 분포 분석 (250Da bins)
    """

    BINS = {
        'mw_pct_lt250Da': {'range': (0, 250), 'label': '<250 Da'},
        'mw_pct_250_500Da': {'range': (250, 500), 'label': '250-500 Da'},
        'mw_pct_500_750Da': {'range': (500, 750), 'label': '500-750 Da'},
        'mw_pct_750_1000Da': {'range': (750, 1000), 'label': '750-1000 Da'},
        'mw_pct_gt1000Da': {'range': (1000, float('inf')), 'label': '>1000 Da'},
    }

    AVG_AA_MW = 110.0  # 평균 아미노산 분자량 (Da)

    def __init__(self, loader: CompositionLoader):
        """
        초기화

        Args:
            loader: CompositionLoader 인스턴스
        """
        self.loader = loader
        if self.loader.data is None:
            self.loader.load_data()

    def analyze_distribution(self, sample_id: str) -> Dict:
        """
        분자량 분포 분석

        Args:
            sample_id: 샘플 ID

        Returns:
            분석 결과
        """
        mw_dist = self.loader.get_mw_distribution(sample_id)

        if not mw_dist:
            return {'error': f'샘플 {sample_id}의 MW 데이터를 찾을 수 없습니다'}

        # 평균 분자량 계산
        avg_mw = self._calculate_weighted_average_mw(mw_dist)

        # 분포 통계
        percentages = list(mw_dist.values())

        result = {
            'sample_id': sample_id,
            'distribution': mw_dist,
            'average_mw': avg_mw,
            'statistics': {
                'total_percentage': sum(percentages),
                'dominant_bin': max(mw_dist, key=mw_dist.get),
                'dominant_percentage': max(percentages),
            }
        }

        # 펩타이드 길이 추정
        result['peptide_length_estimates'] = self.predict_peptide_lengths(sample_id)

        return result

    def _calculate_weighted_average_mw(self, mw_dist: Dict[str, float]) -> float:
        """
        가중 평균 분자량 계산

        Args:
            mw_dist: 분자량 분포

        Returns:
            평균 분자량
        """
        total_weight = 0
        total_percentage = 0

        for bin_name, percentage in mw_dist.items():
            if bin_name in self.BINS:
                mw_range = self.BINS[bin_name]['range']
                # 구간 중간값 사용
                if mw_range[1] == float('inf'):
                    mid_mw = 1250  # >1000Da는 1250Da로 가정
                else:
                    mid_mw = (mw_range[0] + mw_range[1]) / 2

                total_weight += mid_mw * percentage
                total_percentage += percentage

        if total_percentage > 0:
            return total_weight / total_percentage
        return 0

    def predict_peptide_lengths(self, sample_id: str) -> Dict[str, List[int]]:
        """
        분자량 기반 펩타이드 길이 추정

        Args:
            sample_id: 샘플 ID

        Returns:
            구간별 가능한 펩타이드 길이
        """
        mw_dist = self.loader.get_mw_distribution(sample_id)

        length_estimates = {}

        for bin_name, percentage in mw_dist.items():
            if bin_name in self.BINS and percentage > 0:
                mw_range = self.BINS[bin_name]['range']

                # 길이 = MW / 평균 AA MW
                min_length = max(1, int(mw_range[0] / self.AVG_AA_MW))
                if mw_range[1] == float('inf'):
                    max_length = 20  # 최대 20개로 제한
                else:
                    max_length = int(mw_range[1] / self.AVG_AA_MW)

                length_estimates[self.BINS[bin_name]['label']] = {
                    'min_length': min_length,
                    'max_length': max_length,
                    'possible_lengths': list(range(min_length, max_length + 1)),
                    'percentage': percentage
                }

        return length_estimates

    def calculate_peptide_count_estimate(self, sample_id: str,
                                        total_protein_g: float = 1.0) -> Dict:
        """
        펩타이드 수 추정

        Args:
            sample_id: 샘플 ID
            total_protein_g: 총 단백질 양 (g)

        Returns:
            추정 펩타이드 수
        """
        mw_analysis = self.analyze_distribution(sample_id)
        avg_mw = mw_analysis['average_mw']

        if avg_mw == 0:
            return {'error': '평균 분자량을 계산할 수 없습니다'}

        # 몰수 계산
        # 1 mol = 평균 MW g
        # n mol = total_protein_g / avg_mw
        n_moles = total_protein_g / avg_mw

        # 분자 수 (Avogadro 수)
        avogadro = 6.022e23
        n_molecules = n_moles * avogadro

        return {
            'sample_id': sample_id,
            'total_protein_g': total_protein_g,
            'average_mw': avg_mw,
            'estimated_moles': n_moles,
            'estimated_molecules': n_molecules,
            'estimated_peptides_per_gram': n_molecules / total_protein_g
        }

    def compare_distributions(self, sample_ids: List[str]) -> pd.DataFrame:
        """
        여러 샘플의 MW 분포 비교

        Args:
            sample_ids: 샘플 ID 리스트

        Returns:
            비교 DataFrame
        """
        distributions = {}

        for sample_id in sample_ids:
            mw_dist = self.loader.get_mw_distribution(sample_id)
            distributions[sample_id] = mw_dist

        df = pd.DataFrame(distributions).T
        df = df.fillna(0)

        return df


class AminoAcidProfiler:
    """
    아미노산 프로파일링 및 특성 분석
    """

    def __init__(self, loader: CompositionLoader):
        """
        초기화

        Args:
            loader: CompositionLoader 인스턴스
        """
        self.loader = loader

    def get_essential_aa_profile(self, sample_id: str) -> Dict:
        """
        필수 아미노산 프로파일

        Args:
            sample_id: 샘플 ID

        Returns:
            필수 아미노산 정보
        """
        essential_aas = ['I', 'L', 'K', 'M', 'F', 'T', 'W', 'V', 'H']

        taa_comp = self.loader.get_taa_composition(sample_id)

        essential_profile = {}
        total_essential = 0

        for aa in essential_aas:
            value = taa_comp.get(aa, 0)
            essential_profile[aa] = value
            total_essential += value

        return {
            'sample_id': sample_id,
            'essential_amino_acids': essential_profile,
            'total_essential_percentage': total_essential,
            'essential_ratio': total_essential / sum(taa_comp.values()) if taa_comp else 0
        }

    def get_functional_groups(self, sample_id: str) -> Dict:
        """
        기능별 아미노산 그룹

        Args:
            sample_id: 샘플 ID

        Returns:
            기능별 그룹 정보
        """
        taa_comp = self.loader.get_taa_composition(sample_id)

        groups = {
            'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'P', 'W', 'V'],
            'polar_uncharged': ['S', 'T', 'N', 'Q', 'C', 'Y'],
            'positively_charged': ['K', 'R', 'H'],
            'negatively_charged': ['D', 'E'],
            'aromatic': ['F', 'Y', 'W', 'H'],
            'sulfur_containing': ['C', 'M'],
        }

        group_totals = {}
        for group_name, aas in groups.items():
            total = sum(taa_comp.get(aa, 0) for aa in aas)
            group_totals[group_name] = total

        return {
            'sample_id': sample_id,
            'functional_groups': group_totals,
            'composition': taa_comp
        }


if __name__ == '__main__':
    # 테스트
    print("=== Peptide Analyzer 테스트 ===\n")

    loader = CompositionLoader()
    loader.load_data()

    samples = loader.get_sample_list()
    test_sample = samples[0]

    print(f"테스트 샘플: {test_sample}\n")

    # 1. 조성 분석
    print("1. 조성 분석")
    comp_analyzer = PeptideCompositionAnalyzer(loader)
    analysis = comp_analyzer.analyze_sample(test_sample)

    print(f"  TAA 아미노산 수: {analysis['taa_statistics']['n_amino_acids']}")
    print(f"  최다 아미노산: {analysis['taa_statistics']['max_aa']} ({analysis['taa_statistics']['max_value']:.2f}%)")
    print(f"  물리화학적 특성:")
    for key, val in analysis['taa_properties'].items():
        print(f"    {key}: {val:.2f}%")

    # 2. 대표 아미노산
    print("\n2. 대표 아미노산 (Top 5)")
    rep_aas = comp_analyzer.get_representative_amino_acids(test_sample, top_n=5)
    for aa_info in rep_aas['top_amino_acids']:
        print(f"  {aa_info['code']} ({aa_info['name']}): {aa_info['percentage']:.2f}%")

    # 3. MW 분포 분석
    print("\n3. 분자량 분포 분석")
    mw_analyzer = MolecularWeightAnalyzer(loader)
    mw_analysis = mw_analyzer.analyze_distribution(test_sample)

    print(f"  평균 분자량: {mw_analysis['average_mw']:.2f} Da")
    print(f"  분포:")
    for bin_name, percentage in mw_analysis['distribution'].items():
        print(f"    {bin_name}: {percentage:.2f}%")

    # 4. 펩타이드 길이 추정
    print("\n4. 펩타이드 길이 추정")
    for label, info in mw_analysis['peptide_length_estimates'].items():
        print(f"  {label}: {info['min_length']}-{info['max_length']} AA ({info['percentage']:.2f}%)")

    # 5. 샘플 비교
    if len(samples) >= 3:
        print("\n5. 샘플 비교 (첫 3개)")
        comparison = comp_analyzer.compare_samples(samples[:3])
        print(f"  비교 샘플 수: {len(comparison['sample_analyses'])}")
        print(f"  유사도 행렬 shape: {comparison['similarity_matrix'].shape}")

    # 6. 클러스터링
    print("\n6. 클러스터링 (5개 클러스터)")
    clustering = comp_analyzer.cluster_by_composition(n_clusters=5)
    print(f"  클러스터별 샘플 수:")
    for cluster_id, count in clustering['n_samples_per_cluster'].items():
        print(f"    Cluster {cluster_id}: {count}개")

    print("\n[OK] Peptide Analyzer 테스트 완료!")
