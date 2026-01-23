"""
데이터 로더 테스트 스크립트
"""

import sys
sys.path.insert(0, 'src')

from data_loader import CompositionLoader

if __name__ == '__main__':
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
            print(f"  - 샘플 정보: {profile.get('sample_info', {})}")
            print(f"  - TAA 조성: {len(profile['taa_composition'])}개 아미노산")

            # TAA 상위 5개 출력
            taa_sorted = sorted(profile['taa_composition'].items(),
                              key=lambda x: x[1], reverse=True)[:5]
            print(f"    상위 5개: {taa_sorted}")

            print(f"  - FAA 조성: {len(profile['faa_composition'])}개 아미노산")
            print(f"  - MW 분포: {len(profile['mw_distribution'])}개 구간")

            # MW 분포 출력
            print(f"    분포:")
            for bin_name, pct in profile['mw_distribution'].items():
                print(f"      {bin_name}: {pct:.2f}%")

            print(f"  - 물리화학적 특성 (TAA):")
            for key, val in profile['taa_property_ratios'].items():
                print(f"    {key}: {val:.2f}%")

            print(f"  - 일반 특성:")
            for key, val in profile['general_properties'].items():
                print(f"    {key}: {val:.2f}")

        # 데이터 검증
        print("\n=== 데이터 검증 ===")
        validation = loader.validate_data()
        print(f"총 샘플 수: {validation['total_samples']}")
        print(f"TAA 컬럼: {validation['taa_columns']}")
        print(f"FAA 컬럼: {validation['faa_columns']}")
        print(f"MW 컬럼: {validation['mw_columns']}")

        if validation['missing_values']:
            print(f"\n결측치: {len(validation['missing_values'])}개 컬럼")

        if validation['warnings']:
            print("\n경고:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        # 샘플 비교 테스트
        print("\n=== 샘플 비교 테스트 ===")
        if len(samples) >= 3:
            comparison = loader.compare_samples(samples[:3], feature='taa')
            print(f"비교 DataFrame shape: {comparison.shape}")
            print(f"컬럼: {comparison.columns.tolist()[:5]}...")

        print("\n[OK] 데이터 로더 테스트 완료!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] 오류: {e}")
        print("composition_template.xlsx 파일이 data/ 폴더에 있는지 확인하세요.")
    except Exception as e:
        print(f"\n[ERROR] 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
