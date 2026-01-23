"""
bioactive_predictor 단위 테스트
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pytest
from data_loader import CompositionLoader
from bioactive_predictor import (
    BioactiveMotifFinder,
    ActivityScorer,
    BioactivePredictor
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


@pytest.fixture
def motif_finder():
    """BioactiveMotifFinder fixture"""
    return BioactiveMotifFinder()


@pytest.fixture
def activity_scorer():
    """ActivityScorer fixture"""
    return ActivityScorer()


class TestBioactiveMotifFinder:
    """BioactiveMotifFinder 테스트"""

    def test_initialization(self, motif_finder):
        """초기화 테스트"""
        assert motif_finder.motifs is not None
        assert motif_finder.activity_rules is not None
        assert len(motif_finder.motifs) > 0

    def test_find_motifs_in_sequence(self, motif_finder):
        """단일 서열 모티프 검색 테스트"""
        # 알려진 모티프 포함 서열
        test_sequences = [
            "ARRKKWRF",      # RR, KK, WR 포함
            "IPPVPPIKP",     # IPP, VPP, IKP 포함
            "YGGFLHHY",      # YGG, YGGF, HH, HHY 포함
            "VEPLEPFKP"      # VEP, LEP, FKP 포함
        ]

        for seq in test_sequences:
            motifs = motif_finder.find_motifs_in_sequence(seq)
            assert len(motifs) > 0

            # 모티프 구조 검증
            for motif in motifs:
                assert 'motif' in motif
                assert 'position' in motif
                assert 'activity' in motif
                assert 'description' in motif

    def test_find_motifs_empty_sequence(self, motif_finder):
        """빈 서열 테스트"""
        motifs = motif_finder.find_motifs_in_sequence("")
        assert motifs == []

    def test_find_motifs_no_match(self, motif_finder):
        """매칭 없는 서열 테스트"""
        motifs = motif_finder.find_motifs_in_sequence("AAA")
        assert len(motifs) == 0

    def test_find_motifs_in_sequences(self, motif_finder):
        """다중 서열 모티프 검색 테스트"""
        sequences = [
            ("ARRKKWRF", 0.8),
            ("IPPVPP", 0.7),
            ("AAA", 0.5)  # 모티프 없음
        ]

        results = motif_finder.find_motifs_in_sequences(sequences)

        # 모티프 있는 서열만 결과에 포함
        assert len(results) >= 2
        assert "AAA" not in results

        for seq, data in results.items():
            assert 'likelihood_score' in data
            assert 'motifs' in data
            assert 'n_motifs' in data
            assert 'activities' in data

    def test_get_motifs_by_activity(self, motif_finder):
        """활성별 모티프 추출 테스트"""
        activities = ['antimicrobial', 'antihypertensive', 'antioxidant']

        for activity in activities:
            motifs = motif_finder.get_motifs_by_activity(activity)
            assert len(motifs) > 0
            for motif in motifs:
                assert motif['activity'] == activity

    def test_summarize_motif_findings(self, motif_finder):
        """모티프 발견 요약 테스트"""
        sequences = [
            ("ARRKKWRF", 0.8),
            ("IPPVPPIKP", 0.7)
        ]

        motif_results = motif_finder.find_motifs_in_sequences(sequences)
        summary = motif_finder.summarize_motif_findings(motif_results)

        assert 'total_sequences_with_motifs' in summary
        assert 'total_motifs_found' in summary
        assert 'by_activity' in summary
        assert 'top_sequences' in summary

        assert summary['total_sequences_with_motifs'] > 0
        assert summary['total_motifs_found'] > 0


class TestActivityScorer:
    """ActivityScorer 테스트"""

    def test_initialization(self, activity_scorer):
        """초기화 테스트"""
        assert activity_scorer.activity_rules is not None
        assert len(activity_scorer.activity_rules) > 0

    def test_calculate_activity_scores(self, activity_scorer):
        """활성 점수 계산 테스트"""
        # 고 라이신/아르기닌 조성 (항균성 높음)
        composition = {
            'K': 20.0,
            'R': 15.0,
            'W': 10.0,
            'F': 8.0,
            'A': 10.0,
            'G': 10.0
        }

        scores = activity_scorer.calculate_activity_scores(composition)

        assert len(scores) > 0
        for activity, score in scores.items():
            assert 0 <= score <= 1.0

        # 항균성 점수가 높아야 함 (K, R, W, F가 많으므로)
        assert scores.get('antimicrobial', 0) > 0.1

    def test_predict_for_sample(self, activity_scorer, loader, test_sample):
        """샘플 예측 테스트"""
        result = activity_scorer.predict_for_sample(test_sample, loader)

        assert 'sample_id' in result
        assert 'activity_scores' in result
        assert 'activity_details' in result
        assert 'composition' in result

        # 모든 활성 검증
        for activity, details in result['activity_details'].items():
            assert 'score' in details
            assert 'description' in details
            assert 'threshold' in details
            assert 'above_threshold' in details
            assert 'contributing_amino_acids' in details

    def test_rank_activities(self, activity_scorer):
        """활성 순위 테스트"""
        scores = {
            'antimicrobial': 0.7,
            'antihypertensive': 0.5,
            'antioxidant': 0.8,
            'opioid': 0.3
        }

        ranked = activity_scorer.rank_activities(scores)

        assert len(ranked) == 4
        assert ranked[0][0] == 'antioxidant'  # 최고 점수
        assert ranked[0][1] == 0.8
        assert ranked[-1][0] == 'opioid'  # 최저 점수


class TestBioactivePredictor:
    """BioactivePredictor 통합 테스트"""

    def test_predict_comprehensive(self, loader, test_sample):
        """포괄적 예측 테스트"""
        predictor = BioactivePredictor(loader)
        result = predictor.predict_comprehensive(
            test_sample,
            n_sequences=20,
            length_range=(5, 10)
        )

        assert 'sample_id' in result
        assert 'composition_analysis' in result
        assert 'generated_sequences' in result
        assert 'motif_findings' in result
        assert 'sequences_with_motifs' in result
        assert 'recommendations' in result

        # 서열 생성 검증
        assert result['generated_sequences'] == 20

        # 모티프 발견 검증
        motif_findings = result['motif_findings']
        assert 'total_sequences_with_motifs' in motif_findings
        assert 'total_motifs_found' in motif_findings
        assert 'by_activity' in motif_findings

    def test_compare_samples_bioactivity(self, loader):
        """샘플 비교 테스트"""
        samples = loader.get_sample_list()[:3]
        predictor = BioactivePredictor(loader)
        result = predictor.compare_samples_bioactivity(samples)

        assert 'sample_scores' in result
        assert 'best_samples_by_activity' in result
        assert 'n_samples' in result

        assert result['n_samples'] == 3
        assert len(result['sample_scores']) == 3

        # 각 활성별 최고 샘플 검증
        for activity, data in result['best_samples_by_activity'].items():
            assert 'sample_id' in data
            assert 'score' in data
            assert data['sample_id'] in samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
