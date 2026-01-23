"""
단백질 구조 생성 모듈 (ESMFold API)
Protein structure prediction using ESMFold
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

import requests

try:
    from .utils import get_data_dir, validate_sequence
except ImportError:
    from utils import get_data_dir, validate_sequence


class StructureCache:
    """
    PDB 파일 캐시 관리
    """

    def __init__(self, cache_dir: Optional[Path] = None, cache_days: int = 30):
        """
        초기화

        Args:
            cache_dir: 캐시 디렉토리 (None이면 기본 경로)
            cache_days: 캐시 유효 기간 (일)
        """
        if cache_dir is None:
            cache_dir = get_data_dir() / 'cache' / 'structures'

        self.cache_dir = Path(cache_dir)
        self.cache_days = cache_days
        self.metadata_file = self.cache_dir / 'cache_metadata.json'

        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 메타데이터 로딩
        self.metadata = self._load_metadata()

        # 메타데이터 파일이 없으면 생성
        if not self.metadata_file.exists():
            self._save_metadata()

    def _load_metadata(self) -> Dict:
        """
        메타데이터 로딩

        Returns:
            메타데이터 딕셔너리
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """
        메타데이터 저장
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save metadata: {e}")

    def _get_sequence_hash(self, sequence: str) -> str:
        """
        서열 해시 생성

        Args:
            sequence: 아미노산 서열

        Returns:
            SHA256 해시 (16자)
        """
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]

    def _get_cache_path(self, sequence: str) -> Path:
        """
        캐시 파일 경로 생성

        Args:
            sequence: 아미노산 서열

        Returns:
            캐시 파일 경로
        """
        seq_hash = self._get_sequence_hash(sequence)
        return self.cache_dir / f"{seq_hash}.pdb"

    def is_cached(self, sequence: str) -> bool:
        """
        캐시 존재 여부 확인

        Args:
            sequence: 아미노산 서열

        Returns:
            캐시 존재 여부
        """
        cache_path = self._get_cache_path(sequence)

        if not cache_path.exists():
            return False

        # 메타데이터에서 생성 시간 확인
        seq_hash = self._get_sequence_hash(sequence)
        if seq_hash in self.metadata:
            created_at = datetime.fromisoformat(self.metadata[seq_hash]['created_at'])
            expiry_date = created_at + timedelta(days=self.cache_days)

            if datetime.now() > expiry_date:
                # 만료됨
                self._remove_cache(sequence)
                return False

        return True

    def get(self, sequence: str) -> Optional[str]:
        """
        캐시에서 PDB 내용 가져오기

        Args:
            sequence: 아미노산 서열

        Returns:
            PDB 파일 내용 또는 None
        """
        if not self.is_cached(sequence):
            return None

        cache_path = self._get_cache_path(sequence)
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[WARNING] Failed to read cache: {e}")
            return None

    def put(self, sequence: str, pdb_content: str):
        """
        PDB 내용을 캐시에 저장

        Args:
            sequence: 아미노산 서열
            pdb_content: PDB 파일 내용
        """
        cache_path = self._get_cache_path(sequence)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(pdb_content)

            # 메타데이터 업데이트
            seq_hash = self._get_sequence_hash(sequence)
            self.metadata[seq_hash] = {
                'sequence': sequence,
                'length': len(sequence),
                'created_at': datetime.now().isoformat(),
                'file_path': str(cache_path)
            }
            self._save_metadata()

            print(f"[OK] Cached structure for sequence (length {len(sequence)})")

        except Exception as e:
            print(f"[ERROR] Failed to cache structure: {e}")

    def _remove_cache(self, sequence: str):
        """
        캐시 제거

        Args:
            sequence: 아미노산 서열
        """
        cache_path = self._get_cache_path(sequence)
        if cache_path.exists():
            cache_path.unlink()

        seq_hash = self._get_sequence_hash(sequence)
        if seq_hash in self.metadata:
            del self.metadata[seq_hash]
            self._save_metadata()

    def clear_expired(self):
        """
        만료된 캐시 모두 제거
        """
        expired = []

        for seq_hash, data in self.metadata.items():
            created_at = datetime.fromisoformat(data['created_at'])
            expiry_date = created_at + timedelta(days=self.cache_days)

            if datetime.now() > expiry_date:
                expired.append(data['sequence'])

        for sequence in expired:
            self._remove_cache(sequence)

        print(f"[OK] Cleared {len(expired)} expired cache entries")

    def get_stats(self) -> Dict:
        """
        캐시 통계

        Returns:
            통계 딕셔너리
        """
        total_entries = len(self.metadata)
        total_size = sum(
            Path(data['file_path']).stat().st_size
            for data in self.metadata.values()
            if Path(data['file_path']).exists()
        )

        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


class ESMFoldConnector:
    """
    ESMFold API 연결
    """

    API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    MAX_SEQUENCE_LENGTH = 400
    TIMEOUT = 60  # 초

    def __init__(self, use_cache: bool = True, cache_dir: Optional[Path] = None):
        """
        초기화

        Args:
            use_cache: 캐시 사용 여부
            cache_dir: 캐시 디렉토리
        """
        self.use_cache = use_cache
        self.cache = StructureCache(cache_dir) if use_cache else None

    def predict_structure(self, sequence: str) -> Tuple[Optional[str], Dict]:
        """
        서열의 3D 구조 예측

        Args:
            sequence: 아미노산 서열

        Returns:
            (PDB 내용, 메타정보) 튜플
        """
        # 서열 검증
        if not validate_sequence(sequence):
            return None, {
                'error': 'Invalid sequence',
                'message': 'Sequence contains invalid amino acid codes'
            }

        # 길이 제한
        if len(sequence) > self.MAX_SEQUENCE_LENGTH:
            return None, {
                'error': 'Sequence too long',
                'message': f'Maximum length is {self.MAX_SEQUENCE_LENGTH} residues',
                'sequence_length': len(sequence)
            }

        # 캐시 확인
        if self.use_cache and self.cache.is_cached(sequence):
            pdb_content = self.cache.get(sequence)
            if pdb_content:
                return pdb_content, {
                    'source': 'cache',
                    'sequence_length': len(sequence),
                    'cached': True
                }

        # API 호출
        print(f"[INFO] Predicting structure via ESMFold API (length: {len(sequence)})...")
        start_time = time.time()

        try:
            response = requests.post(
                self.API_URL,
                data=sequence,
                headers={'Content-Type': 'text/plain'},
                timeout=self.TIMEOUT
            )

            if response.status_code == 200:
                pdb_content = response.text
                elapsed = time.time() - start_time

                # 캐시 저장
                if self.use_cache:
                    self.cache.put(sequence, pdb_content)

                return pdb_content, {
                    'source': 'api',
                    'sequence_length': len(sequence),
                    'elapsed_time': elapsed,
                    'cached': False
                }

            else:
                return None, {
                    'error': 'API error',
                    'status_code': response.status_code,
                    'message': response.text[:200]
                }

        except requests.exceptions.Timeout:
            return None, {
                'error': 'Timeout',
                'message': f'Request timed out after {self.TIMEOUT} seconds'
            }
        except requests.exceptions.RequestException as e:
            return None, {
                'error': 'Network error',
                'message': str(e)
            }

    def save_pdb(self, pdb_content: str, output_path: Path) -> bool:
        """
        PDB 파일 저장

        Args:
            pdb_content: PDB 파일 내용
            output_path: 저장 경로

        Returns:
            성공 여부
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pdb_content)
            print(f"[OK] Saved PDB file: {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save PDB: {e}")
            return False

    def parse_pdb_info(self, pdb_content: str) -> Dict:
        """
        PDB 파일에서 정보 추출

        Args:
            pdb_content: PDB 파일 내용

        Returns:
            정보 딕셔너리
        """
        lines = pdb_content.split('\n')

        info = {
            'n_atoms': 0,
            'n_residues': 0,
            'chains': set(),
            'residue_names': []
        }

        for line in lines:
            if line.startswith('ATOM'):
                info['n_atoms'] += 1
                chain = line[21:22].strip()
                if chain:
                    info['chains'].add(chain)

                # 잔기 정보
                residue_name = line[17:20].strip()
                residue_num = int(line[22:26].strip())

                if residue_num > info['n_residues']:
                    info['n_residues'] = residue_num
                    info['residue_names'].append(residue_name)

        info['chains'] = list(info['chains'])

        return info


class StructureBuilder:
    """
    구조 생성 통합 클래스
    """

    def __init__(self, use_cache: bool = True):
        """
        초기화

        Args:
            use_cache: 캐시 사용 여부
        """
        self.connector = ESMFoldConnector(use_cache=use_cache)

    def build_from_sequence(self, sequence: str,
                          save_path: Optional[Path] = None) -> Tuple[Optional[str], Dict]:
        """
        서열로부터 구조 생성

        Args:
            sequence: 아미노산 서열
            save_path: PDB 저장 경로 (None이면 저장 안 함)

        Returns:
            (PDB 내용, 메타정보) 튜플
        """
        # 구조 예측
        pdb_content, meta = self.connector.predict_structure(sequence)

        if pdb_content is None:
            return None, meta

        # PDB 정보 파싱
        pdb_info = self.connector.parse_pdb_info(pdb_content)
        meta.update(pdb_info)

        # 파일 저장
        if save_path:
            success = self.connector.save_pdb(pdb_content, save_path)
            meta['saved'] = success
            meta['save_path'] = str(save_path) if success else None

        return pdb_content, meta

    def build_from_sequences(self, sequences: list,
                           output_dir: Optional[Path] = None) -> Dict:
        """
        여러 서열의 구조 생성

        Args:
            sequences: 서열 리스트 또는 [(서열, 이름), ...] 리스트
            output_dir: 출력 디렉토리

        Returns:
            결과 딕셔너리
        """
        results = {
            'total': len(sequences),
            'success': 0,
            'failed': 0,
            'structures': []
        }

        for i, seq_data in enumerate(sequences, 1):
            # 서열과 이름 추출
            if isinstance(seq_data, tuple):
                sequence, name = seq_data
            else:
                sequence = seq_data
                name = f"sequence_{i}"

            print(f"\n[{i}/{len(sequences)}] Processing: {name}")

            # 저장 경로 설정
            save_path = None
            if output_dir:
                save_path = Path(output_dir) / f"{name}.pdb"

            # 구조 생성
            pdb_content, meta = self.build_from_sequence(sequence, save_path)

            if pdb_content:
                results['success'] += 1
                results['structures'].append({
                    'name': name,
                    'sequence': sequence,
                    'meta': meta,
                    'pdb_content': pdb_content if not save_path else None
                })
            else:
                results['failed'] += 1
                results['structures'].append({
                    'name': name,
                    'sequence': sequence,
                    'error': meta.get('error', 'Unknown error'),
                    'meta': meta
                })

        return results


if __name__ == '__main__':
    # 테스트
    print("=== Structure Builder Test ===\n")

    # 짧은 테스트 서열
    test_sequences = [
        "ARNDCEQ",           # 7 AA
        "GPGPGPGPGP",        # 10 AA (poly-Gly-Pro)
        "YGGFL"              # 5 AA (enkephalin)
    ]

    builder = StructureBuilder(use_cache=True)

    print("1. Single sequence prediction")
    sequence = test_sequences[0]
    print(f"   Sequence: {sequence} (length: {len(sequence)})")

    pdb_content, meta = builder.build_from_sequence(sequence)

    if pdb_content:
        print(f"   Source: {meta.get('source', 'unknown')}")
        print(f"   Atoms: {meta.get('n_atoms', 0)}")
        print(f"   Residues: {meta.get('n_residues', 0)}")
        print(f"   Chains: {meta.get('chains', [])}")

        if 'elapsed_time' in meta:
            print(f"   Time: {meta['elapsed_time']:.2f}s")
    else:
        print(f"   Error: {meta.get('error', 'Unknown')}")

    print("\n2. Multiple sequences prediction")
    sequences_with_names = [
        (test_sequences[0], "heptapeptide"),
        (test_sequences[1], "poly_gly_pro"),
        (test_sequences[2], "enkephalin")
    ]

    output_dir = get_data_dir().parent / 'output' / 'structures'
    results = builder.build_from_sequences(sequences_with_names, output_dir)

    print(f"\n   Total: {results['total']}")
    print(f"   Success: {results['success']}")
    print(f"   Failed: {results['failed']}")

    print("\n3. Cache statistics")
    if builder.connector.use_cache:
        stats = builder.connector.cache.get_stats()
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Total size: {stats['total_size_mb']:.2f} MB")
        print(f"   Cache dir: {stats['cache_dir']}")

    print("\n[OK] Structure Builder test complete!")
