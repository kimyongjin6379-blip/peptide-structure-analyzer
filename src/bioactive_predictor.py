"""
생리활성 펩타이드 예측 모듈
Bioactive peptide prediction based on motifs and composition
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

try:
    from .data_loader import CompositionLoader
    from .sequence_predictor import SequenceGenerator
    from .utils import get_data_dir, AMINO_ACIDS
except ImportError:
    from data_loader import CompositionLoader
    from sequence_predictor import SequenceGenerator
    from utils import get_data_dir, AMINO_ACIDS


class BioactiveMotifFinder:
    """
    알려진 생리활성 모티프 검색
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        초기화

        Args:
            db_path: 모티프 데이터베이스 경로 (None이면 기본 경로)
        """
        if db_path is None:
            # Prefer comprehensive DB if available
            comprehensive = get_data_dir() / 'bioactive_peptide_db_comprehensive.json'
            if comprehensive.exists():
                db_path = comprehensive
            else:
                db_path = get_data_dir() / 'bioactive_peptide_db.json'
        else:
            db_path = Path(db_path)

        self.db_path = db_path
        self.motifs = []
        self.activity_rules = {}
        self.is_comprehensive = False
        self._load_database()

    def _load_database(self):
        """
        데이터베이스 로딩 (기존 52개 DB 및 comprehensive DB 모두 지원)
        """
        if not self.db_path.exists():
            print(f"[WARNING] Database not found: {self.db_path}")
            return

        with open(self.db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Comprehensive DB format (from build_bioactive_db.py)
        if 'peptides' in data and 'metadata' in data:
            self.is_comprehensive = True
            self.motifs = []
            for p in data['peptides']:
                activities = p.get('activities', ['unknown'])
                primary_activity = activities[0] if activities else 'unknown'
                self.motifs.append({
                    'sequence': p['sequence'],
                    'activity': primary_activity,
                    'all_activities': activities,
                    'description': p.get('description', ''),
                    'references': p.get('references', []),
                    'source': p.get('source_protein', ''),
                    'IC50': p.get('ic50'),
                    'molecular_weight': p.get('molecular_weight'),
                    'length': p.get('length', len(p['sequence'])),
                    'db_sources': p.get('db_sources', []),
                })
            self.activity_rules = data.get('activity_rules', {})
            meta = data['metadata']
            print(f"[OK] Loaded {len(self.motifs)} peptides from comprehensive DB "
                  f"({meta.get('activity_types', '?')} activity types)")
        else:
            # Original format
            self.motifs = data.get('motifs', [])
            self.activity_rules = data.get('activity_rules', {})
            print(f"[OK] Loaded {len(self.motifs)} motifs from database")

    def find_motifs_in_sequence(self, sequence: str, min_motif_length: int = 3) -> List[Dict]:
        """
        서열에서 모티프 검색

        Args:
            sequence: 아미노산 서열
            min_motif_length: 최소 모티프 길이 (기본 3, 2글자 디펩타이드 노이즈 방지)

        Returns:
            발견된 모티프 리스트
        """
        if not sequence:
            return []

        found_motifs = []

        for motif_data in self.motifs:
            motif_seq = motif_data['sequence']

            # Skip too-short motifs to reduce noise
            if len(motif_seq) < min_motif_length:
                continue

            # 정규표현식으로 모든 위치 찾기
            for match in re.finditer(motif_seq, sequence):
                entry = {
                    'motif': motif_seq,
                    'position': match.start(),
                    'activity': motif_data['activity'],
                    'description': motif_data['description'],
                    'references': motif_data.get('references', [])
                }
                # Add comprehensive DB fields if available
                if motif_data.get('all_activities'):
                    entry['all_activities'] = motif_data['all_activities']
                if motif_data.get('IC50'):
                    entry['IC50'] = motif_data['IC50']
                if motif_data.get('source'):
                    entry['source'] = motif_data['source']
                if motif_data.get('molecular_weight'):
                    entry['molecular_weight'] = motif_data['molecular_weight']
                found_motifs.append(entry)

        return found_motifs

    def find_motifs_in_sequences(self, sequences: List[Tuple[str, float]]) -> Dict:
        """
        여러 서열에서 모티프 검색

        Args:
            sequences: [(서열, 점수), ...] 리스트

        Returns:
            서열별 모티프 딕셔너리
        """
        results = {}

        for seq, score in sequences:
            motifs_found = self.find_motifs_in_sequence(seq)

            if motifs_found:
                results[seq] = {
                    'likelihood_score': score,
                    'motifs': motifs_found,
                    'n_motifs': len(motifs_found),
                    'activities': list(set(m['activity'] for m in motifs_found))
                }

        return results

    def get_motifs_by_activity(self, activity: str) -> List[Dict]:
        """
        특정 활성의 모든 모티프 추출

        Args:
            activity: 활성 유형

        Returns:
            모티프 리스트
        """
        return [m for m in self.motifs if m['activity'] == activity]

    def summarize_motif_findings(self, motif_results: Dict) -> Dict:
        """
        모티프 발견 요약

        Args:
            motif_results: find_motifs_in_sequences() 결과

        Returns:
            요약 통계
        """
        if not motif_results:
            return {
                'total_sequences_with_motifs': 0,
                'total_motifs_found': 0,
                'by_activity': {},
                'top_sequences': []
            }

        # 활성별 집계
        activity_counts = defaultdict(int)
        total_motifs = 0

        for seq_data in motif_results.values():
            total_motifs += seq_data['n_motifs']
            for activity in seq_data['activities']:
                activity_counts[activity] += 1

        # 상위 서열 (모티프 많은 순)
        top_sequences = sorted(
            [(seq, data) for seq, data in motif_results.items()],
            key=lambda x: x[1]['n_motifs'],
            reverse=True
        )[:10]

        return {
            'total_sequences_with_motifs': len(motif_results),
            'total_motifs_found': total_motifs,
            'by_activity': dict(activity_counts),
            'top_sequences': [
                {
                    'sequence': seq,
                    'n_motifs': data['n_motifs'],
                    'activities': data['activities'],
                    'likelihood_score': data['likelihood_score']
                }
                for seq, data in top_sequences
            ]
        }


class ActivityScorer:
    """
    조성 기반 생리활성 점수 계산
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        초기화

        Args:
            db_path: 모티프 데이터베이스 경로
        """
        if db_path is None:
            db_path = get_data_dir() / 'bioactive_peptide_db.json'
        else:
            db_path = Path(db_path)

        self.db_path = db_path
        self.activity_rules = {}
        self._load_rules()

    def _load_rules(self):
        """
        활성 규칙 로딩
        """
        if not self.db_path.exists():
            print(f"[WARNING] Database not found: {self.db_path}")
            return

        with open(self.db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.activity_rules = data.get('activity_rules', {})

        print(f"[OK] Loaded {len(self.activity_rules)} activity rules")

    def calculate_activity_scores(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        조성 기반 활성 점수 계산

        Args:
            composition: 아미노산 조성 {AA: percentage}

        Returns:
            활성별 점수 {activity: score}
        """
        scores = {}

        for activity, rules in self.activity_rules.items():
            score = 0.0
            weights = rules.get('scoring_weights', {})

            # 가중 합산
            for aa, weight in weights.items():
                aa_percentage = composition.get(aa, 0.0)
                score += (aa_percentage / 100.0) * weight

            scores[activity] = min(score, 1.0)  # 0-1 범위로 제한

        return scores

    def predict_for_sample(self, sample_id: str, loader: CompositionLoader) -> Dict:
        """
        샘플의 생리활성 잠재력 예측

        Args:
            sample_id: 샘플 ID
            loader: CompositionLoader 인스턴스

        Returns:
            예측 결과 딕셔너리
        """
        # 펩타이드 조성 가져오기 (TAA - FAA, 정규화)
        taa_comp = loader.get_peptide_composition(sample_id, normalize=True)

        if not taa_comp:
            return {'error': f'Sample {sample_id} not found'}

        # 활성 점수 계산
        activity_scores = self.calculate_activity_scores(taa_comp)

        # 각 활성별 상세 정보
        activity_details = {}
        for activity, score in activity_scores.items():
            rules = self.activity_rules.get(activity, {})

            # 기여 아미노산
            contributing_aa = []
            weights = rules.get('scoring_weights', {})
            for aa, weight in weights.items():
                aa_percentage = taa_comp.get(aa, 0.0)
                if aa_percentage > 0:
                    contributing_aa.append({
                        'amino_acid': aa,
                        'percentage': aa_percentage,
                        'weight': weight,
                        'contribution': (aa_percentage / 100.0) * weight
                    })

            # 기여도 순 정렬
            contributing_aa.sort(key=lambda x: x['contribution'], reverse=True)

            activity_details[activity] = {
                'score': score,
                'description': rules.get('description', ''),
                'threshold': rules.get('threshold', 0.3),
                'above_threshold': score >= rules.get('threshold', 0.3),
                'contributing_amino_acids': contributing_aa[:5]  # 상위 5개
            }

        return {
            'sample_id': sample_id,
            'activity_scores': activity_scores,
            'activity_details': activity_details,
            'composition': taa_comp
        }

    def rank_activities(self, activity_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        활성 점수 순위

        Args:
            activity_scores: 활성별 점수

        Returns:
            [(활성, 점수), ...] 정렬된 리스트
        """
        ranked = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked


class BioactivePredictor:
    """
    통합 생리활성 예측
    """

    def __init__(self, loader: CompositionLoader, db_path: Optional[str] = None):
        """
        초기화

        Args:
            loader: CompositionLoader 인스턴스
            db_path: 모티프 데이터베이스 경로
        """
        self.loader = loader
        self.motif_finder = BioactiveMotifFinder(db_path)
        self.activity_scorer = ActivityScorer(db_path)

    def predict_comprehensive(self, sample_id: str,
                            n_sequences: int = 50,
                            length_range: Tuple[int, int] = (5, 15)) -> Dict:
        """
        포괄적 생리활성 예측

        Args:
            sample_id: 샘플 ID
            n_sequences: 생성할 서열 수
            length_range: 서열 길이 범위

        Returns:
            예측 결과 딕셔너리
        """
        # 1. 조성 기반 활성 점수
        composition_analysis = self.activity_scorer.predict_for_sample(
            sample_id, self.loader
        )

        # 2. 서열 생성 (펩타이드 조성 사용)
        taa_comp = self.loader.get_peptide_composition(sample_id, normalize=True)
        generator = SequenceGenerator(taa_comp)
        sequences = generator.generate_sequences(
            length_range=length_range,
            n_sequences=n_sequences,
            method='markov'
        )

        # 3. 모티프 검색
        motif_results = self.motif_finder.find_motifs_in_sequences(sequences)
        motif_summary = self.motif_finder.summarize_motif_findings(motif_results)

        # 4. 활성별 최적 서열 추천
        recommendations = self._generate_recommendations(
            sequences, motif_results, composition_analysis
        )

        return {
            'sample_id': sample_id,
            'composition_analysis': composition_analysis,
            'generated_sequences': len(sequences),
            'motif_findings': motif_summary,
            'sequences_with_motifs': motif_results,
            'recommendations': recommendations
        }

    def _generate_recommendations(self, sequences: List[Tuple[str, float]],
                                 motif_results: Dict,
                                 composition_analysis: Dict) -> Dict:
        """
        활성별 추천 서열 생성

        Args:
            sequences: 생성된 서열 리스트
            motif_results: 모티프 검색 결과
            composition_analysis: 조성 분석 결과

        Returns:
            활성별 추천 딕셔너리
        """
        recommendations = {}

        # 활성 점수가 높은 활성 추출
        activity_scores = composition_analysis.get('activity_scores', {})
        top_activities = sorted(activity_scores.items(),
                               key=lambda x: x[1], reverse=True)[:3]

        for activity, score in top_activities:
            # 해당 활성의 모티프를 가진 서열 찾기
            relevant_sequences = []

            for seq, seq_data in motif_results.items():
                if activity in seq_data['activities']:
                    relevant_sequences.append({
                        'sequence': seq,
                        'likelihood_score': seq_data['likelihood_score'],
                        'n_motifs': seq_data['n_motifs'],
                        'motifs': [m for m in seq_data['motifs']
                                  if m['activity'] == activity]
                    })

            # 가능성 점수로 정렬
            relevant_sequences.sort(key=lambda x: x['likelihood_score'], reverse=True)

            recommendations[activity] = {
                'composition_score': score,
                'n_candidates': len(relevant_sequences),
                'top_sequences': relevant_sequences[:5]  # 상위 5개
            }

        return recommendations

    def compare_samples_bioactivity(self, sample_ids: List[str],
                                     n_sequences: int = 200,
                                     length_range: Tuple[int, int] = (3, 12)) -> Dict:
        """
        여러 샘플의 생리활성 비교 (DB 매칭 기반)

        Args:
            sample_ids: 샘플 ID 리스트
            n_sequences: 샘플당 생성 서열 수
            length_range: 서열 길이 범위

        Returns:
            비교 결과
        """
        results = {}

        for sample_id in sample_ids:
            taa_comp = self.loader.get_peptide_composition(sample_id, normalize=True)
            if not taa_comp:
                continue

            generator = SequenceGenerator(taa_comp)
            sequences = generator.generate_sequences(
                length_range=length_range,
                n_sequences=n_sequences,
                method='markov'
            )

            # DB matching
            activity_hit_counts = defaultdict(int)
            for seq, score in sequences:
                motifs = self.motif_finder.find_motifs_in_sequence(seq, min_motif_length=3)
                for hit in motifs:
                    activities = hit.get('all_activities', [hit['activity']])
                    for act in activities:
                        if act != 'unknown':
                            activity_hit_counts[act] += 1

            # Normalize
            if activity_hit_counts:
                max_hits = max(activity_hit_counts.values())
                results[sample_id] = {
                    act: round(count / max_hits, 4)
                    for act, count in activity_hit_counts.items()
                }
            else:
                results[sample_id] = {}

        # 활성별 최고 샘플
        activities = set()
        for scores in results.values():
            activities.update(scores.keys())

        best_samples = {}
        for activity in activities:
            best_sample = max(results.items(),
                            key=lambda x: x[1].get(activity, 0.0))
            best_samples[activity] = {
                'sample_id': best_sample[0],
                'score': best_sample[1].get(activity, 0.0)
            }

        return {
            'sample_scores': results,
            'best_samples_by_activity': best_samples,
            'n_samples': len(sample_ids)
        }


if __name__ == '__main__':
    # 테스트
    print("=== Bioactive Predictor Test ===\n")

    loader = CompositionLoader()
    loader.load_data()

    samples = loader.get_sample_list()
    test_sample = samples[0]

    print(f"Test sample: {test_sample}\n")

    # 1. 조성 기반 활성 점수
    print("1. Composition-based activity scores")
    scorer = ActivityScorer()
    result = scorer.predict_for_sample(test_sample, loader)

    print(f"  Activity scores:")
    for activity, score in result['activity_scores'].items():
        print(f"    {activity}: {score:.3f}")

    print(f"\n  Top 3 activities:")
    ranked = scorer.rank_activities(result['activity_scores'])
    for i, (activity, score) in enumerate(ranked[:3], 1):
        details = result['activity_details'][activity]
        threshold_status = "ABOVE" if details['above_threshold'] else "below"
        print(f"    {i}. {activity}: {score:.3f} ({threshold_status} threshold)")

    # 2. 서열 생성 및 모티프 검색
    print("\n2. Sequence generation and motif search")
    predictor = BioactivePredictor(loader)
    comprehensive = predictor.predict_comprehensive(
        test_sample,
        n_sequences=30,
        length_range=(5, 12)
    )

    print(f"  Generated sequences: {comprehensive['generated_sequences']}")
    motif_summary = comprehensive['motif_findings']
    print(f"  Sequences with motifs: {motif_summary['total_sequences_with_motifs']}")
    print(f"  Total motifs found: {motif_summary['total_motifs_found']}")

    if motif_summary['by_activity']:
        print(f"\n  Motifs by activity:")
        for activity, count in motif_summary['by_activity'].items():
            print(f"    {activity}: {count} sequences")

    # 3. 상위 추천 서열
    print("\n3. Top recommended sequences")
    recommendations = comprehensive['recommendations']

    for i, (activity, rec) in enumerate(recommendations.items(), 1):
        print(f"\n  Activity {i}: {activity} (composition score: {rec['composition_score']:.3f})")
        print(f"    Candidates: {rec['n_candidates']}")

        if rec['top_sequences']:
            print(f"    Top sequence: {rec['top_sequences'][0]['sequence']}")
            print(f"      Likelihood: {rec['top_sequences'][0]['likelihood_score']:.4f}")
            print(f"      Motifs found:")
            for motif in rec['top_sequences'][0]['motifs'][:3]:
                print(f"        - {motif['motif']} at position {motif['position']}")

    # 4. 샘플 비교
    if len(samples) >= 3:
        print("\n4. Sample comparison (top 3 samples)")
        comparison = predictor.compare_samples_bioactivity(samples[:3])

        print(f"\n  Best samples by activity:")
        for activity, data in comparison['best_samples_by_activity'].items():
            print(f"    {activity}: {data['sample_id']} (score: {data['score']:.3f})")

    print("\n[OK] Bioactive Predictor test complete!")
