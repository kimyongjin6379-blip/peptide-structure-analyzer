"""
Page 3: Bioactive Motif Search (핵심 페이지)
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import CompositionLoader
from bioactive_predictor import BioactivePredictor, ActivityScorer
from visualizer_2d import BioactivityVisualizer
from plm_embedder import PLMEmbedder
from fitness_predictor import FitnessPredictor

st.set_page_config(page_title="Bioactive Search", page_icon="💊", layout="wide")


@st.cache_resource
def load_data():
    loader = CompositionLoader()
    loader.load_data()
    return loader


@st.cache_resource
def load_plm_embedder():
    embedder = PLMEmbedder(model_name="esm2_t6_8M", device="cpu")
    embedder.load_model()
    return embedder


@st.cache_resource
def load_fitness_predictor():
    predictor = FitnessPredictor(embedding_dim=320)
    return predictor


def main():
    st.title("💊 Bioactive Peptide Prediction")
    st.markdown("Identify bioactive motifs and predict biological activities")
    st.markdown("---")

    loader = load_data()

    # Get sample options with product names
    sample_options = loader.get_sample_options()

    # Sample selection with product names
    selected_display = st.selectbox(
        "Select Sample:",
        list(sample_options.keys()),
        index=0
    )

    # Get actual sample_id
    selected_sample = sample_options[selected_display]

    st.markdown("---")

    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "🎯 DB-based Activity Profile",
        "🔍 Motif Search",
        "📊 Compare Samples"
    ])

    with tab1:
        st.markdown("### Bioactivity Profile: 예측 → DB 매칭 → 스코어링")
        st.markdown(
            "펩톤 조성에서 서열을 생성하고, **4,162개 DB**와 매칭하여 "
            "활성 프로파일을 산출합니다."
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            profile_n_seq = st.number_input("생성 서열 수", 50, 500, 200, key="profile_n")
        with col_p2:
            profile_len = st.slider("서열 길이 범위", 3, 20, (3, 12), key="profile_len")

        if st.button("🎯 활성 프로파일 분석", type="primary", key="run_profile"):
            with st.spinner("서열 생성 → DB 매칭 → 활성 스코어링 중..."):
                from sequence_predictor import SequenceGenerator
                from collections import Counter

                # Step 1: 서열 생성
                taa_comp = loader.get_peptide_composition(selected_sample, normalize=True)
                if taa_comp:
                    generator = SequenceGenerator(taa_comp)
                    sequences = generator.generate_sequences(
                        length_range=profile_len,
                        n_sequences=profile_n_seq,
                        method='markov'
                    )

                    # Step 2: DB 매칭
                    finder = BioactivePredictor(loader).motif_finder
                    all_hits = []
                    seq_with_hits = 0
                    for seq, score in sequences:
                        motifs = finder.find_motifs_in_sequence(seq, min_motif_length=3)
                        if motifs:
                            seq_with_hits += 1
                            all_hits.extend(motifs)

                    # Step 3: 활성별 스코어링
                    activity_hit_counts = Counter()
                    activity_peptides = {}
                    for hit in all_hits:
                        activities = hit.get('all_activities', [hit['activity']])
                        for act in activities:
                            if act != 'unknown':
                                activity_hit_counts[act] += 1
                                if act not in activity_peptides:
                                    activity_peptides[act] = set()
                                activity_peptides[act].add(hit['motif'])

                    # Normalize to 0-1 scores
                    if activity_hit_counts:
                        max_hits = max(activity_hit_counts.values())
                        activity_scores = {
                            act: round(count / max_hits, 4)
                            for act, count in activity_hit_counts.items()
                        }
                    else:
                        activity_scores = {}

                    st.session_state['profile_result'] = {
                        'activity_scores': activity_scores,
                        'activity_hit_counts': dict(activity_hit_counts),
                        'activity_peptides': {k: list(v) for k, v in activity_peptides.items()},
                        'total_sequences': len(sequences),
                        'seq_with_hits': seq_with_hits,
                        'total_hits': len(all_hits),
                        'sample_id': selected_sample,
                    }

        # Display results from session_state
        if 'profile_result' in st.session_state:
            result = st.session_state['profile_result']
            activity_scores = result['activity_scores']

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("생성 서열", result['total_sequences'])
            with c2:
                st.metric("DB 매칭 서열", result['seq_with_hits'])
            with c3:
                hit_rate = result['seq_with_hits'] / max(result['total_sequences'], 1) * 100
                st.metric("매칭률", f"{hit_rate:.1f}%")
            with c4:
                st.metric("총 히트 수", result['total_hits'])

            if activity_scores:
                # Radar chart - top 12 activities for readability
                top_activities = dict(
                    sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)[:12]
                )

                fig = BioactivityVisualizer.plot_activity_scores(
                    top_activities,
                    title=f"DB-based Bioactivity Profile - {result['sample_id']}"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Detailed table
                st.markdown("### 활성별 상세 (전체)")
                rows = []
                for act, score in sorted(activity_scores.items(), key=lambda x: x[1], reverse=True):
                    hits = result['activity_hit_counts'].get(act, 0)
                    peptides = result['activity_peptides'].get(act, [])
                    top_peps = sorted(peptides, key=len, reverse=True)[:5]
                    rows.append({
                        'Activity': act.replace('_', ' ').title(),
                        'Score': f"{score:.3f}",
                        'DB Hits': hits,
                        'Unique Motifs': len(peptides),
                        'Top Matched Peptides': ', '.join(top_peps),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Top 3 activities detail
                st.markdown("### Top 3 활성 분석")
                ranked = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)

                for i, (activity, score) in enumerate(ranked[:3], 1):
                    peptides = result['activity_peptides'].get(activity, [])
                    hits = result['activity_hit_counts'].get(activity, 0)

                    with st.expander(
                        f"#{i} {activity.replace('_', ' ').title()} "
                        f"(Score: {score:.3f}, Hits: {hits})",
                        expanded=(i == 1)
                    ):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Score", f"{score:.3f}")
                            st.metric("DB Hits", hits)
                            st.metric("Unique Motifs", len(peptides))
                        with col2:
                            st.markdown("**매칭된 생리활성 펩타이드:**")
                            for pep in sorted(peptides, key=len, reverse=True)[:10]:
                                st.code(pep, language=None)
            else:
                st.warning("매칭된 생리활성 펩타이드가 없습니다. 서열 수를 늘려보세요.")

            # DL-based Activity Prediction (ESM-2 보강)
            st.markdown("---")
            st.markdown("### 🤖 ESM-2 DL 활성 예측 (앙상블)")
            st.markdown("DB 매칭 스코어를 ESM-2 딥러닝 모델로 보강합니다.")

            if st.button("🧬 DL 활성 예측 실행", key="run_dl_prediction"):
                embedder = load_plm_embedder()
                fitness_pred = load_fitness_predictor()

                from sequence_predictor import AbundancePredictor
                abundance_pred = AbundancePredictor(loader)
                gen_result = abundance_pred.predict_for_sample(
                    result['sample_id'],
                    length_range=(5, 12),
                    n_sequences=10,
                    method="markov"
                )

                top_seqs = [s['sequence'] for s in gen_result.get('sequences_with_mw', [])[:5]]

                if top_seqs:
                    with st.spinner("ESM-2 임베딩 추출 및 DL 활성 예측 중..."):
                        dl_results = []
                        for seq in top_seqs:
                            emb = embedder.get_sequence_embedding(seq)
                            dl_pred = fitness_pred.predict(emb)

                            ensemble_scores = {}
                            for act_name, db_score in activity_scores.items():
                                dl_score = dl_pred.get(act_name, {}).get('score', 0.5)
                                ensemble = db_score * 0.4 + dl_score * 0.6
                                both_high = db_score > 0.5 and dl_score > 0.5
                                both_low = db_score <= 0.5 and dl_score <= 0.5
                                if both_high or both_low:
                                    confidence = "★★★"
                                elif db_score > 0.6 or dl_score > 0.6:
                                    confidence = "★★"
                                else:
                                    confidence = "★"
                                ensemble_scores[act_name] = {
                                    'db': round(db_score, 4),
                                    'dl': round(dl_score, 4),
                                    'ensemble': round(ensemble, 4),
                                    'confidence': confidence
                                }

                            dl_results.append({
                                'sequence': seq,
                                'scores': ensemble_scores
                            })

                    st.session_state['dl_activity_results'] = dl_results

            if 'dl_activity_results' in st.session_state:
                dl_results = st.session_state['dl_activity_results']

                for seq_result in dl_results:
                    with st.expander(f"서열: `{seq_result['sequence']}`"):
                        rows = []
                        for act_name, scores in sorted(
                            seq_result['scores'].items(),
                            key=lambda x: x[1]['ensemble'],
                            reverse=True
                        ):
                            rows.append({
                                'Activity': act_name.replace('_', ' ').title(),
                                'DB Score': f"{scores['db']:.3f}",
                                'DL Score': f"{scores['dl']:.3f}",
                                'Ensemble (DB*0.4 + DL*0.6)': f"{scores['ensemble']:.3f}",
                                'Confidence': scores['confidence']
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Motif Search in Generated Sequences")

        # Show DB info
        temp_predictor = BioactivePredictor(loader)
        n_motifs = len(temp_predictor.motif_finder.motifs)
        is_comprehensive = getattr(temp_predictor.motif_finder, 'is_comprehensive', False)
        if is_comprehensive:
            st.info(f"📊 Comprehensive DB: **{n_motifs:,}** bioactive peptides (BIOPEP-UWM + curated)")
        else:
            st.info(f"📊 Database: **{n_motifs}** bioactive motifs")

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            n_sequences = st.number_input("Number of sequences to generate", 20, 100, 50)

        with col2:
            length_range = st.slider("Sequence length range", 5, 15, (5, 12))

        if st.button("🔍 Search Motifs", type="primary"):
            with st.spinner("Generating sequences and searching motifs..."):
                predictor = BioactivePredictor(loader)
                comprehensive = predictor.predict_comprehensive(
                    selected_sample,
                    n_sequences=n_sequences,
                    length_range=length_range
                )
                # session_state에 결과 저장 (페이지 리렌더링 시에도 유지)
                st.session_state['motif_comprehensive'] = comprehensive
                st.session_state['motif_sample'] = selected_sample

        # session_state에서 결과 표시
        if 'motif_comprehensive' in st.session_state:
            comprehensive = st.session_state['motif_comprehensive']
            motif_sample = st.session_state.get('motif_sample', '')

            st.success("Search complete!")

            # Motif findings
            motif_findings = comprehensive['motif_findings']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sequences Generated", comprehensive['generated_sequences'])

            with col2:
                st.metric("Sequences with Motifs", motif_findings['total_sequences_with_motifs'])

            with col3:
                st.metric("Total Motifs Found", motif_findings['total_motifs_found'])

            # Motifs by activity
            if motif_findings['by_activity']:
                st.markdown("#### Motifs by Activity")

                for activity, count in sorted(
                    motif_findings['by_activity'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    st.write(f"**{activity.title()}**: {count} sequences")

            # Top sequences with motifs
            st.markdown("#### Top Sequences with Motifs")

            top_seqs = motif_findings.get('top_sequences', [])[:10]

            for seq_data in top_seqs:
                with st.expander(
                    f"{seq_data['sequence']} ({seq_data['n_motifs']} motifs, "
                    f"score: {seq_data['likelihood_score']:.4f})"
                ):
                    st.write(f"**Activities:** {', '.join(seq_data['activities'])}")

                    # Show all motifs in this sequence
                    sequences_with_motifs = comprehensive.get('sequences_with_motifs', {})
                    if seq_data['sequence'] in sequences_with_motifs:
                        motifs = sequences_with_motifs[seq_data['sequence']]['motifs']

                        for motif in motifs:
                            st.write(
                                f"- **{motif['motif']}** at position {motif['position']} "
                                f"({motif['activity']}): {motif['description']}"
                            )

            # ---- AI 분석 연계 ----
            if top_seqs:
                st.markdown("---")
                st.markdown("#### 🤖 AI 심층 분석 연계")

                # session_state에 모티프 보유 서열 저장
                motif_sequences = [s['sequence'] for s in top_seqs]
                st.session_state['bioactive_sequences'] = motif_sequences
                st.session_state['bioactive_source'] = f"3번 페이지 ({motif_sample})"

                col_ai1, col_ai2 = st.columns(2)
                with col_ai1:
                    selected_motif_seq = st.selectbox(
                        "AI 분석할 서열 선택",
                        motif_sequences,
                        key="select_motif_seq"
                    )
                with col_ai2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🤖 이 서열 → AI 분석으로 보내기", key="send_motif_ai"):
                        st.session_state['ai_input_sequence'] = selected_motif_seq
                        st.session_state['ai_target_tab'] = 'prediction'
                        st.success(f"✅ `{selected_motif_seq}` → 6번 AI 분석 페이지로 이동하세요")

                if st.button("📦 모티프 보유 서열 전체 → AI 배치 분석", key="send_motif_batch"):
                    st.session_state['ai_batch_sequences'] = motif_sequences
                    st.session_state['ai_batch_source'] = f"{motif_sample} 모티프 후보 {len(motif_sequences)}개"
                    st.success(f"✅ {len(motif_sequences)}개 서열이 AI 분석 페이지로 전송되었습니다!")

                st.markdown("---")

            # ---- 유사 모티프 검색 (ESM-2 기반) ----
            st.markdown("---")
            st.markdown("#### 🔬 유사 모티프 검색 (ESM-2 Embedding)")
            st.markdown("정확히 일치하지 않지만 ESM-2 임베딩 기반으로 유사한 모티프를 탐지합니다.")

            if st.button("🧬 유사 모티프 검색 실행", key="run_similar_motif"):
                embedder = load_plm_embedder()

                # Get all known motifs
                motif_finder_predictor = BioactivePredictor(loader)
                known_motifs = motif_finder_predictor.motif_finder.motifs

                # Get generated sequences
                gen_sequences = [s['sequence'] for s in motif_findings.get('top_sequences', [])]
                all_gen_seqs = comprehensive.get('all_sequences', gen_sequences)
                if not all_gen_seqs:
                    all_gen_seqs = gen_sequences

                similar_results = []

                with st.spinner("ESM-2 임베딩으로 유사 모티프 검색 중..."):
                    # Pre-compute motif embeddings
                    motif_embeddings = {}
                    for motif_data in known_motifs:
                        motif_seq = motif_data['sequence']
                        if len(motif_seq) >= 3:
                            try:
                                motif_emb = embedder.get_motif_embedding(motif_seq)
                                motif_embeddings[motif_seq] = {
                                    'embedding': motif_emb,
                                    'activity': motif_data['activity'],
                                    'description': motif_data['description']
                                }
                            except Exception:
                                continue

                    # Sliding window search
                    for seq in all_gen_seqs[:20]:
                        for motif_seq, motif_info in motif_embeddings.items():
                            motif_len = len(motif_seq)
                            if motif_len > len(seq):
                                continue
                            if motif_seq in seq:
                                continue

                            for start in range(len(seq) - motif_len + 1):
                                subseq = seq[start:start + motif_len]
                                try:
                                    sub_emb = embedder.get_motif_embedding(subseq)
                                    dot = np.dot(sub_emb, motif_info['embedding'])
                                    norm1 = np.linalg.norm(sub_emb)
                                    norm2 = np.linalg.norm(motif_info['embedding'])
                                    sim = dot / (norm1 * norm2 + 1e-10)

                                    if sim > 0.85:
                                        similar_results.append({
                                            'sequence': seq,
                                            'subsequence': subseq,
                                            'position': start + 1,
                                            'known_motif': motif_seq,
                                            'similarity': round(float(sim), 4),
                                            'activity': motif_info['activity'],
                                            'description': motif_info['description']
                                        })
                                except Exception:
                                    continue

                st.session_state['similar_motif_results'] = similar_results

            if 'similar_motif_results' in st.session_state:
                similar_results = st.session_state['similar_motif_results']

                if similar_results:
                    st.success(f"유사 모티프 {len(similar_results)}개 발견!")

                    sim_df = pd.DataFrame(similar_results)
                    sim_df = sim_df.sort_values('similarity', ascending=False)
                    sim_df_display = sim_df[['sequence', 'subsequence', 'position',
                                              'known_motif', 'similarity', 'activity']].copy()
                    sim_df_display.columns = ['서열', '부분 서열', '위치', '유사 모티프',
                                               '유사도', '활성']
                    sim_df_display['유사도'] = sim_df_display['유사도'].apply(lambda x: f"{x:.4f}")

                    st.dataframe(sim_df_display, use_container_width=True, hide_index=True)

                    st.markdown("**활성별 유사 모티프 분포:**")
                    activity_counts = sim_df['activity'].value_counts()
                    for act, cnt in activity_counts.items():
                        st.write(f"- **{act.title()}**: {cnt}개 유사 모티프")
                else:
                    st.info("유사도 0.85 이상의 유사 모티프가 발견되지 않았습니다.")

            st.markdown("---")

            # Recommendations
            st.markdown("#### Activity-based Recommendations")

            recommendations = comprehensive.get('recommendations', {})

            for activity, rec in recommendations.items():
                with st.expander(
                    f"{activity.title()} "
                    f"(Composition score: {rec['composition_score']:.3f}, "
                    f"{rec['n_candidates']} candidates)"
                ):
                    rec_seqs = rec.get('top_sequences', [])

                    if rec_seqs:
                        st.write("**Top Candidate:**")
                        top = rec_seqs[0]
                        st.code(top['sequence'], language=None)
                        st.write(f"Likelihood: {top['likelihood_score']:.4f}")
                        st.write(f"Motifs found: {top['n_motifs']}")

                        for motif in top['motifs']:
                            st.write(f"  - {motif['motif']} at position {motif['position']}")
                    else:
                        st.info("No sequences with motifs found for this activity")

    with tab3:
        st.markdown("### Compare Bioactivity Profiles")

        # Create reverse mapping for multiselect
        display_to_id = sample_options
        id_to_display = {v: k for k, v in sample_options.items()}

        # Get default selections
        sample_ids = list(sample_options.values())
        default_displays = [id_to_display[sid] for sid in sample_ids[:3]] if len(sample_ids) >= 3 else [id_to_display[sid] for sid in sample_ids]

        selected_displays = st.multiselect(
            "Select samples to compare (max 5):",
            list(sample_options.keys()),
            default=default_displays,
            max_selections=5
        )

        # Convert display names back to sample IDs
        selected_samples = [display_to_id[display] for display in selected_displays]

        if len(selected_samples) >= 2:
            if st.button("📊 Compare Activities"):
                with st.spinner("DB 매칭 기반 비교 분석 중... (샘플당 200개 서열 생성)"):
                    predictor = BioactivePredictor(loader)
                    comparison = predictor.compare_samples_bioactivity(selected_samples)

                    st.session_state['compare_result'] = comparison

            if 'compare_result' in st.session_state:
                comparison = st.session_state['compare_result']
                samples_activities = comparison.get('sample_scores', {})

                if samples_activities:
                    # Use top 10 activities across all samples for radar chart
                    all_acts = {}
                    for scores in samples_activities.values():
                        for act, score in scores.items():
                            all_acts[act] = max(all_acts.get(act, 0), score)
                    top_acts = sorted(all_acts, key=all_acts.get, reverse=True)[:10]

                    # Filter to top activities
                    filtered = {}
                    for sid, scores in samples_activities.items():
                        filtered[sid] = {act: scores.get(act, 0) for act in top_acts}

                    fig = BioactivityVisualizer.plot_activity_comparison(
                        filtered,
                        title="DB-based Bioactivity Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Best samples by activity
                    st.markdown("#### Best Samples by Activity")
                    best_samples = comparison.get('best_samples_by_activity', {})

                    df = pd.DataFrame([
                        {
                            'Activity': act.replace('_', ' ').title(),
                            'Best Sample': data['sample_id'],
                            'Score': f"{data['score']:.3f}"
                        }
                        for act, data in sorted(best_samples.items())
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Detailed scores table
                    st.markdown("#### Detailed Scores")
                    detailed_df = pd.DataFrame(samples_activities).T
                    detailed_df = detailed_df.round(3)
                    st.dataframe(detailed_df, use_container_width=True)

        else:
            st.info("비교할 샘플을 2개 이상 선택하세요.")


if __name__ == "__main__":
    main()
