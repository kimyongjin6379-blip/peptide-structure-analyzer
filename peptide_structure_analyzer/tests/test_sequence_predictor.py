"""
sequence_predictor 단위 테스트
"""

import sys
sys.path.insert(0, '../src')

import pytest
from data_loader import CompositionLoader
from sequence_predictor import (
    SequenceGenerator,
    AbundancePredictor,
    PeptideEnumerator
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
def test_composition(loader, test_sample):
    """테스트용 아미노산 조성"""
    return loader.get_taa_composition(test_sample)


class TestSequenceGenerator:
    """SequenceGenerator 테스트"""

    def test_initialization(self, test_composition):
        """초기화 테스트"""
        generator = SequenceGenerator(test_composition)
        assert generator.probabilities is not None
        assert sum(generator.probabilities.values()) == pytest.approx(1.0, rel=1e-5)

    def test_generate_random(self, test_composition):
        """Random 생성 테스트"""
        generator = SequenceGenerator(test_composition)
        sequences = generator.generate_sequences(
            length_range=(5, 10),
            n_sequences=10,
            method='random'
        )

        assert len(sequences) == 10
        for seq, score in sequences:
            assert 5 <= len(seq) <= 10
            assert 0 <= score <= 1

    def test_generate_markov(self, test_composition):
        """Markov chain 생성 테스트"""
        generator = SequenceGenerator(test_composition)
        sequences = generator.generate_sequences(
            length_range=(5, 10),
            n_sequences=10,
            method='markov'
        )

        assert len(sequences) == 10
        for seq, score in sequences:
            assert 5 <= len(seq) <= 10

    def test_generate_frequent(self, test_composition):
        """Frequent 생성 테스트"""
        generator = SequenceGenerator(test_composition)
        sequences = generator.generate_sequences(
            length_range=(5, 10),
            n_sequences=10,
            method='frequent'
        )

        assert len(sequences) == 10

    def test_score_sequence_likelihood(self, test_composition):
        """가능성 점수 테스트"""
        generator = SequenceGenerator(test_composition)

        # 빈 서열
        assert generator.score_sequence_likelihood('') == 0.0

        # 정상 서열
        score = generator.score_sequence_likelihood('ARNDCEQ')
        assert 0 <= score <= 1

    def test_find_abundant_peptides(self, test_composition):
        """풍부 펩타이드 찾기 테스트"""
        generator = SequenceGenerator(test_composition)
        abundant = generator.find_abundant_peptides(length=7, top_n=5)

        assert len(abundant) <= 5
        for seq, score in abundant:
            assert len(seq) == 7
            assert 0 <= score <= 1


class TestAbundancePredictor:
    """AbundancePredictor 테스트"""

    def test_predict_for_sample(self, loader, test_sample):
        """샘플 예측 테스트"""
        predictor = AbundancePredictor(loader)
        result = predictor.predict_for_sample(
            test_sample,
            length_range=(5, 10),
            n_sequences=20
        )

        assert 'sample_id' in result
        assert 'top_sequences' in result
        assert 'by_length' in result
        assert len(result['top_sequences']) <= 20

    def test_compare_samples_sequences(self, loader):
        """샘플 비교 테스트"""
        samples = loader.get_sample_list()[:2]
        predictor = AbundancePredictor(loader)
        result = predictor.compare_samples_sequences(samples, length=8)

        assert 'sample_results' in result
        assert result['length'] == 8


class TestPeptideEnumerator:
    """PeptideEnumerator 테스트"""

    def test_enumerate_peptides(self, test_composition):
        """펩타이드 열거 테스트"""
        result = PeptideEnumerator.enumerate_peptides(
            test_composition,
            length=3,
            max_count=50
        )

        assert len(result) <= 50
        for seq, score in result:
            assert len(seq) == 3
            assert 0 <= score <= 1

    def test_enumerate_long_peptides_error(self, test_composition):
        """긴 펩타이드 열거 에러 테스트"""
        with pytest.raises(ValueError):
            PeptideEnumerator.enumerate_peptides(
                test_composition,
                length=6,
                max_count=10
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
