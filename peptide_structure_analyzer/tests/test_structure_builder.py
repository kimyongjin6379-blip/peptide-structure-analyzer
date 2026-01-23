"""
structure_builder 단위 테스트
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pytest
from structure_builder import (
    StructureCache,
    ESMFoldConnector,
    StructureBuilder
)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """임시 캐시 디렉토리 fixture"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def test_sequences():
    """테스트용 서열 fixture"""
    return {
        'short': 'ARNDCE',           # 6 AA
        'medium': 'GPGPGPGPGP',      # 10 AA
        'long': 'YGGFLARNDCEQGHIKLMSTV'  # 21 AA
    }


class TestStructureCache:
    """StructureCache 테스트"""

    def test_initialization(self, temp_cache_dir):
        """초기화 테스트"""
        cache = StructureCache(cache_dir=temp_cache_dir)
        assert cache.cache_dir == temp_cache_dir
        assert cache.cache_dir.exists()
        assert cache.metadata_file.exists()

    def test_sequence_hash(self, temp_cache_dir):
        """서열 해시 생성 테스트"""
        cache = StructureCache(cache_dir=temp_cache_dir)
        seq1 = "ARNDCE"
        seq2 = "ARNDCE"
        seq3 = "ARNDCF"

        hash1 = cache._get_sequence_hash(seq1)
        hash2 = cache._get_sequence_hash(seq2)
        hash3 = cache._get_sequence_hash(seq3)

        # 같은 서열은 같은 해시
        assert hash1 == hash2
        # 다른 서열은 다른 해시
        assert hash1 != hash3
        # 해시 길이 확인
        assert len(hash1) == 16

    def test_put_and_get(self, temp_cache_dir, test_sequences):
        """캐시 저장 및 조회 테스트"""
        cache = StructureCache(cache_dir=temp_cache_dir)
        sequence = test_sequences['short']
        pdb_content = "ATOM      1  CA  ALA A   1"

        # 저장
        cache.put(sequence, pdb_content)

        # 캐시 확인
        assert cache.is_cached(sequence)

        # 조회
        retrieved = cache.get(sequence)
        assert retrieved == pdb_content

    def test_is_cached_nonexistent(self, temp_cache_dir):
        """존재하지 않는 캐시 확인 테스트"""
        cache = StructureCache(cache_dir=temp_cache_dir)
        assert not cache.is_cached("NONEXISTENT")

    def test_get_stats(self, temp_cache_dir, test_sequences):
        """캐시 통계 테스트"""
        cache = StructureCache(cache_dir=temp_cache_dir)

        # 여러 개 저장
        for i, (name, seq) in enumerate(test_sequences.items()):
            cache.put(seq, f"PDB content {i}")

        stats = cache.get_stats()
        assert 'total_entries' in stats
        assert 'total_size_mb' in stats
        assert 'cache_dir' in stats
        assert stats['total_entries'] == 3


class TestESMFoldConnector:
    """ESMFoldConnector 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        connector = ESMFoldConnector(use_cache=False)
        assert connector.use_cache is False
        assert connector.cache is None

        connector_with_cache = ESMFoldConnector(use_cache=True)
        assert connector_with_cache.use_cache is True
        assert connector_with_cache.cache is not None

    def test_sequence_validation(self):
        """서열 검증 테스트"""
        connector = ESMFoldConnector(use_cache=False)

        # 유효하지 않은 서열
        invalid_seq = "ARNDCE123"
        pdb_content, meta = connector.predict_structure(invalid_seq)

        assert pdb_content is None
        assert 'error' in meta
        assert meta['error'] == 'Invalid sequence'

    def test_sequence_length_limit(self):
        """서열 길이 제한 테스트"""
        connector = ESMFoldConnector(use_cache=False)

        # 너무 긴 서열 (401 AA)
        long_seq = "A" * (connector.MAX_SEQUENCE_LENGTH + 1)
        pdb_content, meta = connector.predict_structure(long_seq)

        assert pdb_content is None
        assert 'error' in meta
        assert meta['error'] == 'Sequence too long'

    @pytest.mark.skip(reason="Requires network connection and API access")
    def test_predict_structure_api(self):
        """실제 API 호출 테스트 (네트워크 필요)"""
        connector = ESMFoldConnector(use_cache=False)

        # 짧은 서열로 테스트
        sequence = "ARNDCE"
        pdb_content, meta = connector.predict_structure(sequence)

        if pdb_content:
            assert 'source' in meta
            assert meta['source'] == 'api'
            assert 'sequence_length' in meta
            assert meta['sequence_length'] == len(sequence)
            assert 'elapsed_time' in meta
            assert pdb_content.startswith('ATOM')

    def test_parse_pdb_info(self):
        """PDB 파싱 테스트"""
        connector = ESMFoldConnector(use_cache=False)

        # 샘플 PDB 내용
        pdb_content = """ATOM      1  N   ALA A   1      11.104  12.766  13.756  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639  13.954  13.102  1.00  0.00           C
ATOM      3  C   ALA A   1      10.751  15.157  13.417  1.00  0.00           C
ATOM      4  N   ARG A   2      11.104  16.366  12.986  1.00  0.00           N
ATOM      5  CA  ARG A   2      10.394  17.583  13.378  1.00  0.00           C
"""

        info = connector.parse_pdb_info(pdb_content)

        assert 'n_atoms' in info
        assert info['n_atoms'] == 5
        assert 'n_residues' in info
        assert info['n_residues'] >= 2
        assert 'chains' in info
        assert 'A' in info['chains']

    def test_save_pdb(self, tmp_path):
        """PDB 저장 테스트"""
        connector = ESMFoldConnector(use_cache=False)

        pdb_content = "ATOM      1  CA  ALA A   1"
        output_path = tmp_path / "test.pdb"

        success = connector.save_pdb(pdb_content, output_path)

        assert success
        assert output_path.exists()

        with open(output_path, 'r') as f:
            saved_content = f.read()
        assert saved_content == pdb_content


class TestStructureBuilder:
    """StructureBuilder 통합 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        builder = StructureBuilder(use_cache=True)
        assert builder.connector is not None
        assert builder.connector.use_cache is True

    @pytest.mark.skip(reason="Requires network connection and API access")
    def test_build_from_sequence(self, tmp_path):
        """단일 서열 구조 생성 테스트 (네트워크 필요)"""
        builder = StructureBuilder(use_cache=False)

        sequence = "ARNDCE"
        save_path = tmp_path / "test_structure.pdb"

        pdb_content, meta = builder.build_from_sequence(sequence, save_path)

        if pdb_content:
            assert 'n_atoms' in meta
            assert 'n_residues' in meta
            assert meta['n_residues'] == len(sequence)
            assert save_path.exists()

    def test_build_from_sequences_validation(self):
        """여러 서열 검증 테스트 (API 호출 없음)"""
        builder = StructureBuilder(use_cache=False)

        # 유효하지 않은 서열 포함
        sequences = [
            ("ARNDCE", "valid_seq"),
            ("INVALID123", "invalid_seq")
        ]

        results = builder.build_from_sequences(sequences)

        assert 'total' in results
        assert results['total'] == 2
        assert 'success' in results
        assert 'failed' in results
        assert 'structures' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
