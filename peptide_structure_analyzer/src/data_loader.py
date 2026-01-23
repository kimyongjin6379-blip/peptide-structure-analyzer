"""
데이터 로딩 모듈
Composition data loader from Excel template
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from .utils import get_data_dir, calculate_property_ratios
except ImportError:
    from utils import get_data_dir, calculate_property_ratios


# 아미노산 전체 이름 -> 1-letter 코드 매핑
AA_NAME_TO_CODE = {
    'Alanine': 'A',
    'Arginine': 'R',
    'Asparagine': 'N',
    'Aspartic acid': 'D',
    'Cysteine': 'C',
    'Glutamine': 'Q',
    'Glutamic acid': 'E',
    'Glycine': 'G',
    'Histidine': 'H',
    'Isoleucine': 'I',
    'Leucine': 'L',
    'Lysine': 'K',
    'Methionine': 'M',
    'Phenylalanine': 'F',
    'Proline': 'P',
    'Serine': 'S',
    'Threonine': 'T',
    'Tryptophan': 'W',
    'Tyrosine': 'Y',
    'Valine': 'V',
    'Hydroxyproline': 'O',  # 특수 아미노산
}


class CompositionLoader:
    """
    composition_template.xlsx 로딩 및 전처리 클래스
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        초기화

        Args:
            data_path: 데이터 파일 경로 (None이면 기본 경로 사용)
        """
        if data_path is None:
            self.data_path = get_data_dir() / 'composition_template.xlsx'
        else:
            self.data_path = Path(data_path)

        self.data = None
        self.sample_info = None
        self.taa_columns = []  # Total Amino Acid 컬럼
        self.faa_columns = []  # Free Amino Acid 컬럼
        self.mw_columns = []   # Molecular Weight 컬럼

    def load_data(self) -> pd.DataFrame:
        """
        엑셀 파일에서 데이터 로드

        Returns:
            로드된 DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")

        # 엑셀 파일 로드
        print(f"데이터 로딩 중: {self.data_path}")
        self.data = pd.read_excel(self.data_path)

        # 컬럼 분류
        self._classify_columns()

        # 기본 정보 추출
        self._extract_sample_info()

        print(f"[OK] {len(self.data)}개 샘플 로드 완료")
        print(f"[OK] 총 {len(self.data.columns)}개 컬럼")
        print(f"  - TAA: {len(self.taa_columns)}개")
        print(f"  - FAA: {len(self.faa_columns)}개")
        print(f"  - MW bins: {len(self.mw_columns)}개")

        return self.data

    def _classify_columns(self):
        """
        컬럼을 카테고리별로 분류
        """
        all_columns = self.data.columns.tolist()

        # TAA 컬럼 (Total Amino Acid)
        self.taa_columns = [col for col in all_columns if col.startswith('taa_')]

        # FAA 컬럼 (Free Amino Acid)
        self.faa_columns = [col for col in all_columns if col.startswith('faa_')]

        # MW 컬럼 (분자량 분포)
        self.mw_columns = [col for col in all_columns if col.startswith('mw_pct_')]

    def _extract_sample_info(self):
        """
        샘플 기본 정보 추출
        """
        # 실제 컬럼명 확인
        info_columns = ['sample_id', 'Sample_name', 'raw_material', 'manufacturer']
        available_info_cols = [col for col in info_columns if col in self.data.columns]

        if available_info_cols:
            self.sample_info = self.data[available_info_cols].copy()
        else:
            # 기본 정보 컬럼이 없으면 인덱스 기반으로 생성
            self.sample_info = pd.DataFrame({
                'sample_id': self.data.index,
                'Sample_name': [f'Sample_{i}' for i in self.data.index]
            })

    def get_sample_list(self) -> List[str]:
        """
        샘플 ID 리스트 반환

        Returns:
            샘플 ID 리스트
        """
        if self.data is None:
            self.load_data()

        if 'sample_id' in self.data.columns:
            return self.data['sample_id'].tolist()
        elif 'Sample_name' in self.data.columns:
            return self.data['Sample_name'].tolist()
        else:
            return [f'Sample_{i}' for i in range(len(self.data))]

    def get_sample_by_id(self, sample_id: str) -> Optional[pd.Series]:
        """
        샘플 ID로 데이터 조회

        Args:
            sample_id: 샘플 ID

        Returns:
            샘플 데이터 Series 또는 None
        """
        if self.data is None:
            self.load_data()

        if 'sample_id' in self.data.columns:
            mask = self.data['sample_id'] == sample_id
        elif 'Sample_name' in self.data.columns:
            mask = self.data['Sample_name'] == sample_id
        else:
            return None

        matches = self.data[mask]
        if len(matches) > 0:
            return matches.iloc[0]
        return None

    def get_sample_by_index(self, index: int) -> Optional[pd.Series]:
        """
        인덱스로 샘플 데이터 조회

        Args:
            index: 샘플 인덱스 (0-based)

        Returns:
            샘플 데이터 Series 또는 None
        """
        if self.data is None:
            self.load_data()

        if 0 <= index < len(self.data):
            return self.data.iloc[index]
        return None

    def get_taa_composition(self, sample_id: str) -> Dict[str, float]:
        """
        총 아미노산(TAA) 조성 반환

        Args:
            sample_id: 샘플 ID

        Returns:
            {AA: percentage} 딕셔너리
        """
        sample = self.get_sample_by_id(sample_id)
        if sample is None:
            return {}

        composition = {}
        for col in self.taa_columns:
            # 컬럼명에서 아미노산 이름 추출 (예: taa_Alanine -> Alanine)
            aa_name = col.replace('taa_', '')
            # 1-letter 코드로 변환
            aa_code = AA_NAME_TO_CODE.get(aa_name, aa_name[0].upper())
            value = sample[col]
            if pd.notna(value):
                try:
                    value_float = float(value)
                    if value_float > 0:
                        composition[aa_code] = value_float
                except (ValueError, TypeError):
                    continue

        return composition

    def get_faa_composition(self, sample_id: str) -> Dict[str, float]:
        """
        유리 아미노산(FAA) 조성 반환

        Args:
            sample_id: 샘플 ID

        Returns:
            {AA: percentage} 딕셔너리
        """
        sample = self.get_sample_by_id(sample_id)
        if sample is None:
            return {}

        composition = {}
        for col in self.faa_columns:
            # 컬럼명에서 아미노산 이름 추출 (예: faa_Alanine -> Alanine)
            aa_name = col.replace('faa_', '')
            # 1-letter 코드로 변환
            aa_code = AA_NAME_TO_CODE.get(aa_name, aa_name[0].upper())
            value = sample[col]
            if pd.notna(value):
                try:
                    value_float = float(value)
                    if value_float > 0:
                        composition[aa_code] = value_float
                except (ValueError, TypeError):
                    continue

        return composition

    def get_mw_distribution(self, sample_id: str) -> Dict[str, float]:
        """
        분자량 분포 반환

        Args:
            sample_id: 샘플 ID

        Returns:
            {bin_name: percentage} 딕셔너리
        """
        sample = self.get_sample_by_id(sample_id)
        if sample is None:
            return {}

        distribution = {}
        for col in self.mw_columns:
            # 컬럼명 정리 (예: mw_pct_250 -> 250-500Da)
            value = sample[col]
            if pd.notna(value):
                try:
                    distribution[col] = float(value)
                except (ValueError, TypeError):
                    continue

        return distribution

    def get_general_properties(self, sample_id: str) -> Dict[str, float]:
        """
        일반 특성 반환 (TN, AN, sugars 등)

        Args:
            sample_id: 샘플 ID

        Returns:
            특성 딕셔너리
        """
        sample = self.get_sample_by_id(sample_id)
        if sample is None:
            return {}

        property_keys = [
            'total_nitrogen', 'amino_nitrogen', 'total_sugars',
            'reducing_sugars', 'ash', 'moisture', 'crude_fat', 'salinity'
        ]

        # 실제 컬럼명 매핑
        column_mapping = {
            'total_nitrogen': 'general_TN',
            'amino_nitrogen': 'general_AN',
            'total_sugars': 'general_total_sugar',
            'reducing_sugars': 'general_reducing_sugar',
            'ash': 'general_ash',
            'moisture': 'general_moisture',
            'crude_fat': 'general_crude_fat',
            'salinity': 'general_salinity'
        }

        properties = {}
        for key, col in column_mapping.items():
            if col in sample:
                value = sample[col]
                if pd.notna(value):
                    try:
                        properties[key] = float(value)
                    except (ValueError, TypeError):
                        continue

        return properties

    def get_complete_profile(self, sample_id: str) -> Dict:
        """
        샘플의 전체 프로파일 반환

        Args:
            sample_id: 샘플 ID

        Returns:
            전체 정보 딕셔너리
        """
        sample = self.get_sample_by_id(sample_id)
        if sample is None:
            return {}

        taa_comp = self.get_taa_composition(sample_id)
        faa_comp = self.get_faa_composition(sample_id)

        return {
            'sample_id': sample_id,
            'sample_info': self._get_sample_info(sample),
            'taa_composition': taa_comp,
            'faa_composition': faa_comp,
            'taa_property_ratios': calculate_property_ratios(taa_comp),
            'faa_property_ratios': calculate_property_ratios(faa_comp),
            'mw_distribution': self.get_mw_distribution(sample_id),
            'general_properties': self.get_general_properties(sample_id)
        }

    def _get_sample_info(self, sample: pd.Series) -> Dict:
        """
        샘플 기본 정보 추출

        Args:
            sample: 샘플 Series

        Returns:
            정보 딕셔너리
        """
        info = {}
        # 실제 컬럼명 사용
        info_keys = ['Sample_name', 'raw_material', 'manufacturer', 'material_type']

        for key in info_keys:
            if key in sample:
                value = sample[key]
                if pd.notna(value):
                    info[key] = str(value)

        return info

    def compare_samples(self, sample_ids: List[str],
                       feature: str = 'taa') -> pd.DataFrame:
        """
        여러 샘플 비교

        Args:
            sample_ids: 비교할 샘플 ID 리스트
            feature: 비교 특성 ('taa', 'faa', 'mw')

        Returns:
            비교 DataFrame
        """
        comparison_data = []

        for sample_id in sample_ids:
            if feature == 'taa':
                comp = self.get_taa_composition(sample_id)
            elif feature == 'faa':
                comp = self.get_faa_composition(sample_id)
            elif feature == 'mw':
                comp = self.get_mw_distribution(sample_id)
            else:
                raise ValueError(f"Unknown feature: {feature}")

            comp['sample_id'] = sample_id
            comparison_data.append(comp)

        df = pd.DataFrame(comparison_data)
        df = df.set_index('sample_id')
        return df

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        전체 데이터 요약 통계

        Returns:
            요약 통계 DataFrame
        """
        if self.data is None:
            self.load_data()

        # TAA 요약
        taa_data = self.data[self.taa_columns]
        summary = pd.DataFrame({
            'mean': taa_data.mean(),
            'std': taa_data.std(),
            'min': taa_data.min(),
            'max': taa_data.max(),
            'median': taa_data.median()
        })

        return summary

    def validate_data(self) -> Dict[str, any]:
        """
        데이터 유효성 검사

        Returns:
            검사 결과 딕셔너리
        """
        if self.data is None:
            self.load_data()

        results = {
            'total_samples': len(self.data),
            'taa_columns': len(self.taa_columns),
            'faa_columns': len(self.faa_columns),
            'mw_columns': len(self.mw_columns),
            'missing_values': {},
            'warnings': []
        }

        # 결측치 확인
        for col in self.taa_columns + self.faa_columns + self.mw_columns:
            missing = self.data[col].isna().sum()
            if missing > 0:
                results['missing_values'][col] = missing

        # TAA 합계 검사 (100% 근처여야 함)
        try:
            # 숫자형으로 변환
            taa_data_numeric = self.data[self.taa_columns].apply(pd.to_numeric, errors='coerce')
            taa_sums = taa_data_numeric.sum(axis=1)
            abnormal_sums = ((taa_sums < 90) | (taa_sums > 110)).sum()
            if abnormal_sums > 0:
                results['warnings'].append(
                    f"{abnormal_sums}개 샘플의 TAA 합계가 90-110% 범위를 벗어남"
                )
        except Exception as e:
            results['warnings'].append(f"TAA 합계 검사 실패: {str(e)}")

        return results


if __name__ == '__main__':
    # 테스트
    print("=== CompositionLoader 테스트 ===\n")

    loader = CompositionLoader()

    try:
        # 데이터 로드
        data = loader.load_data()
        print(f"\n데이터 shape: {data.shape}")

        # 샘플 리스트
        samples = loader.get_sample_list()
        print(f"\n샘플 수: {len(samples)}")
        print(f"첫 5개 샘플: {samples[:5]}")

        # 첫 번째 샘플 상세 정보
        if len(samples) > 0:
            first_sample = samples[0]
            print(f"\n'{first_sample}' 샘플 프로파일:")

            profile = loader.get_complete_profile(first_sample)
            print(f"  - TAA 조성: {len(profile['taa_composition'])}개 아미노산")
            print(f"  - FAA 조성: {len(profile['faa_composition'])}개 아미노산")
            print(f"  - MW 분포: {len(profile['mw_distribution'])}개 구간")
            print(f"  - 물리화학적 특성:")
            for key, val in profile['taa_property_ratios'].items():
                print(f"    {key}: {val:.2f}%")

        # 데이터 검증
        print("\n=== 데이터 검증 ===")
        validation = loader.validate_data()
        print(f"총 샘플 수: {validation['total_samples']}")
        print(f"TAA 컬럼: {validation['taa_columns']}")
        print(f"FAA 컬럼: {validation['faa_columns']}")
        print(f"MW 컬럼: {validation['mw_columns']}")

        if validation['warnings']:
            print("\n경고:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        print("\n[OK] 데이터 로더 테스트 완료!")

    except FileNotFoundError as e:
        print(f"\n✗ 오류: {e}")
        print("composition_template.xlsx 파일이 data/ 폴더에 있는지 확인하세요.")
    except Exception as e:
        print(f"\n✗ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
