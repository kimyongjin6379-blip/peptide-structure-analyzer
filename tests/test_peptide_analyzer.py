"""
peptide_analyzer 단위 테스트
"""

import sys
sys.path.insert(0, '../src')

import pytest
from data_loader import CompositionLoader
from peptide_analyzer import (
    PeptideCompositionAnalyzer,
    MolecularWeightAnalyzer,
    AminoAcidProfiler
)


@pytest.fixture
def loader():
    """CompositionLoader fixture"""
    loader = CompositionLoader()
    loader.load_data()
    return loader


@pytest.fixture
def test_sample(loader):
    """테스트용 샘플 ID"""
    samples = loader.get_sample_list()
    return samples[0] if samples else None


class TestPeptideCompositionAnalyzer:
    """PeptideCompositionAnalyzer 테스트"""

    def test_analyze_sample(self, loader, test_sample):
        """샘플 분석 테스트"""
        analyzer = PeptideCompositionAnalyzer(loader)
        result = analyzer.analyze_sample(test_sample)

        assert 'sample_id' in result
        assert 'taa_composition' in result
        assert 'taa_properties' in result
        assert 'taa_statistics' in result
        assert result['taa_statistics']['n_amino_acids'] > 0

    def test_compare_samples(self, loader):
        """샘플 비교 테스트"""
        samples = loader.get_sample_list()[:3]
        analyzer = PeptideCompositionAnalyzer(loader)
        result = analyzer.compare_samples(samples)

        assert 'taa_composition_df' in result
        assert 'similarity_matrix' in result
        assert result['taa_composition_df'].shape[0] == 3

    def test_cluster_by_composition(self, loader):
        """클러스터링 테스트"""
        analyzer = PeptideCompositionAnalyzer(loader)
        result = analyzer.cluster_by_composition(n_clusters=3)

        assert 'clusters' in result
        assert 'pca_coords' in result
        assert len(result['clusters']) == len(loader.get_sample_list())

    def test_get_representative_amino_acids(self, loader, test_sample):
        """대표 아미노산 추출 테스트"""
        analyzer = PeptideCompositionAnalyzer(loader)
        result = analyzer.get_representative_amino_acids(test_sample, top_n=5)

        assert 'top_amino_acids' in result
        assert len(result['top_amino_acids']) <= 5
        assert result['top_amino_acids'][0]['percentage'] > 0


class TestMolecularWeightAnalyzer:
    """MolecularWeightAnalyzer 테스트"""

    def test_analyze_distribution(self, loader, test_sample):
        """MW 분포 분석 테스트"""
        analyzer = MolecularWeightAnalyzer(loader)
        result = analyzer.analyze_distribution(test_sample)

        assert 'distribution' in result
        assert 'average_mw' in result
        assert result['average_mw'] > 0

    def test_predict_peptide_lengths(self, loader, test_sample):
        """펩타이드 길이 추정 테스트"""
        analyzer = MolecularWeightAnalyzer(loader)
        result = analyzer.predict_peptide_lengths(test_sample)

        assert len(result) > 0
        for label, info in result.items():
            assert 'min_length' in info
            assert 'max_length' in info
            assert info['min_length'] <= info['max_length']

    def test_compare_distributions(self, loader):
        """MW 분포 비교 테스트"""
        samples = loader.get_sample_list()[:3]
        analyzer = MolecularWeightAnalyzer(loader)
        result = analyzer.compare_distributions(samples)

        assert result.shape[0] == 3
        assert result.shape[1] > 0


class TestAminoAcidProfiler:
    """AminoAcidProfiler 테스트"""

    def test_get_essential_aa_profile(self, loader, test_sample):
        """필수 아미노산 프로파일 테스트"""
        profiler = AminoAcidProfiler(loader)
        result = profiler.get_essential_aa_profile(test_sample)

        assert 'essential_amino_acids' in result
        assert 'total_essential_percentage' in result
        assert result['total_essential_percentage'] > 0

    def test_get_functional_groups(self, loader, test_sample):
        """기능별 그룹 테스트"""
        profiler = AminoAcidProfiler(loader)
        result = profiler.get_functional_groups(test_sample)

        assert 'functional_groups' in result
        assert 'hydrophobic' in result['functional_groups']
        assert 'positively_charged' in result['functional_groups']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
