"""
Page 3: Bioactive Peptide Search
Markov → ESM-2 필터 → DB(4,162) 매칭 → 활성 프로파일 + AI 연계
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
from utils import seq_with_tooltip, inject_aa_tooltip_css
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


@st.cache_resource
def load_bioactive_predictor(_loader):
    return BioactivePredictor(_loader)


def main():
    inject_aa_tooltip_css(st)  # 3-letter 호버 툴팁 CSS
    st.title("💊 Bioactive Peptide Prediction")
    st.markdown("펩톤 조성 기반 생리활성 펩타이드 예측 및 DB 매칭")
    st.markdown("---")

    loader = load_data()
    predictor = load_bioactive_predictor(loader)

    # DB info
    n_motifs = len(predictor.motif_finder.motifs)
    is_comprehensive = getattr(predictor.motif_finder, 'is_comprehensive', False)
    if is_comprehensive:
        st.info(f"📊 Comprehensive DB: **{n_motifs:,}** bioactive peptides (BIOPEP-UWM + curated)")
    else:
        st.info(f"📊 Database: **{n_motifs}** bioactive motifs")

    # ─── 예측 방식 선택 ─────────────────────────────────
    try:
        from in_silico_digester import InSilicoDigester
        digester = InSilicoDigester()
        digester_available = True
    except Exception:
        digester = None
        digester_available = False

    col_mode1, col_mode2 = st.columns([2, 1])
    with col_mode1:
        mode_options = []
        if digester_available:
            mode_options.append("🔬 Hybrid: In Silico + Markov (권장)")
            mode_options.append("🧪 In Silico only")
        mode_options.append("📊 Markov only (기존)")

        mode = st.radio(
            "서열 생성 방식:",
            mode_options,
            index=0,
            horizontal=True,
            key="bioactive_mode",
            help=(
                "Hybrid: 효소 분해 + Markov 결합 (BOTH 우선, In Silico, Markov 순)\n"
                "In Silico only: 효소 공정 + 원료 단백질만\n"
                "Markov only: TAA 조성 비율 기반"
            )
        )
    use_hybrid = mode.startswith("🔬")
    use_in_silico_only = mode.startswith("🧪")
    use_in_silico = use_hybrid or use_in_silico_only

    # sample_options는 Compare Samples 탭에서 항상 필요 → 미리 로드
    sample_options = loader.get_sample_options()

    if use_in_silico and digester_available:
        # 제품 선택 (효소 공정 자료가 있는 제품들)
        products = digester.enzyme_processor.list_products()
        with col_mode2:
            selected_sample = st.selectbox(
                "제품 선택:",
                products,
                index=0,
                key="profile_product"
            )
    else:
        # 기존 샘플 선택 (TAA 조성 데이터)
        with col_mode2:
            selected_display = st.selectbox("Select Sample:",
                                            list(sample_options.keys()),
                                            index=0, key="profile_sample")
        selected_sample = sample_options[selected_display]

    st.markdown("---")

    # Two tabs: Activity Profile + Compare
    tab1, tab2 = st.tabs([
        "🎯 Activity Profile",
        "📊 Compare Samples"
    ])

    # ========== TAB 1: Activity Profile (통합) ==========
    with tab1:
        st.markdown("### Markov → ESM-2 필터 → DB 매칭 → 활성 프로파일")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            profile_n_seq = st.number_input("초기 생성 서열 수", 100, 1000, 500, step=100, key="profile_n")
        with col_p2:
            profile_len = st.slider("서열 길이 범위", 3, 20, (3, 12), key="profile_len")
        with col_p3:
            esm2_top_n = st.number_input("ESM-2 필터 후 상위 N개", 50, 500, 200, step=50, key="esm2_top")

        if st.button("🎯 활성 프로파일 분석", type="primary", key="run_profile"):
            from sequence_predictor import SequenceGenerator
            from collections import Counter

            # Step 1: 서열 생성 (Hybrid / In Silico only / Markov only)
            sequences = []
            hybrid_data = None

            if use_hybrid and digester_available:
                with st.spinner(f"Step 1/3: Hybrid 서열 생성 ({selected_sample})..."):
                    hybrid_data = digester.hybrid_digest_and_markov(
                        selected_sample,
                        min_length=profile_len[0],
                        max_length=profile_len[1],
                        n_top_proteins=20,
                        n_markov_sequences=profile_n_seq
                    )

                    # 계층화 샘플링: 다양성 유지하면서 cap (profile_n_seq)
                    # - BOTH는 무조건 모두 포함 (드물고 가장 신뢰도 높음)
                    # - 나머지를 In Silico 60% / Markov 40%로 분배
                    all_combined = hybrid_data['combined_sequences']
                    both_list = [c for c in all_combined if c['source'] == 'both']
                    is_list = [c for c in all_combined if c['source'] == 'in_silico']
                    mk_list = [c for c in all_combined if c['source'] == 'markov']

                    combined_top = list(both_list)
                    remaining = max(0, profile_n_seq - len(combined_top))
                    n_is = int(remaining * 0.6)
                    n_mk = remaining - n_is
                    combined_top.extend(is_list[:n_is])
                    combined_top.extend(mk_list[:n_mk])
                    # 점수 내림차순 재정렬
                    combined_top.sort(key=lambda x: x['score'], reverse=True)

                    sequences = [(c['sequence'], c['score']) for c in combined_top]

                    # 소스별 카운트 (필터링 후)
                    src_counts = {'both': 0, 'in_silico': 0, 'markov': 0}
                    for c in combined_top:
                        src_counts[c['source']] = src_counts.get(c['source'], 0) + 1

                    st.info(
                        f"📊 Hybrid 생성 결과: 총 {len(all_combined)}개 "
                        f"중 ESM-2 평가 대상 **{len(sequences)}개** (계층화 샘플링)  \n"
                        f"(🔬 BOTH {src_counts['both']} | "
                        f"🧪 In Silico {src_counts['in_silico']} | "
                        f"📊 Markov {src_counts['markov']})"
                    )

            elif use_in_silico_only and digester_available:
                with st.spinner(f"Step 1/3: In silico digestion 실행 중 ({selected_sample})..."):
                    peptides = digester.digest_product(
                        selected_sample,
                        min_length=profile_len[0],
                        max_length=profile_len[1],
                        n_top_proteins=20
                    )
                    unique_peptides = digester.get_unique_peptides(peptides)
                    sequences = [(p.sequence, 1.0 / (1 + abs(p.length - 8)))
                                 for p in unique_peptides]
                    st.info(f"📊 In silico digestion: {len(unique_peptides)}개 unique 펩타이드 생성")

            else:
                taa_comp = loader.get_peptide_composition(selected_sample, normalize=True)
                if taa_comp:
                    with st.spinner(f"Step 1/3: Markov 서열 {profile_n_seq}개 생성 중..."):
                        generator = SequenceGenerator(taa_comp)
                        sequences = generator.generate_sequences(
                            length_range=profile_len,
                            n_sequences=profile_n_seq,
                            method='markov'
                        )

            if sequences:

                # Step 2: ESM-2 임베딩 기반 fitness
                with st.spinner(f"Step 2/3: ESM-2 임베딩 기반 fitness 평가 중... ({len(sequences)}개 서열)"):
                    embedder = load_plm_embedder()
                    seq_list = [seq for seq, _ in sequences]
                    likelihood_list = [lik for _, lik in sequences]
                    fitness_scores = embedder.get_batch_fitness_scores(seq_list, batch_size=16)

                    scored_sequences = []
                    for seq, likelihood, fitness in zip(seq_list, likelihood_list, fitness_scores):
                        combined = likelihood * 0.4 + fitness * 0.6
                        scored_sequences.append({
                            'sequence': seq,
                            'likelihood': likelihood,
                            'esm2_fitness': fitness,
                            'combined_score': combined,
                        })

                    scored_sequences.sort(key=lambda x: x['combined_score'], reverse=True)
                    top_sequences = scored_sequences[:esm2_top_n]

                    avg_fitness_all = sum(s['esm2_fitness'] for s in scored_sequences) / len(scored_sequences)
                    avg_fitness_top = sum(s['esm2_fitness'] for s in top_sequences) / len(top_sequences)

                # Step 3: DB 매칭
                with st.spinner(f"Step 3/3: DB 매칭 ({len(top_sequences)}개 → {n_motifs:,}개 DB)..."):
                    finder = predictor.motif_finder
                    all_hits = []
                    seq_with_hits = 0
                    hit_sequences_detail = []

                    for s in top_sequences:
                        seq = s['sequence']
                        motifs = finder.find_motifs_in_sequence(seq, min_motif_length=3)
                        if motifs:
                            seq_with_hits += 1
                            all_hits.extend(motifs)
                            hit_sequences_detail.append({
                                'sequence': seq,
                                'esm2_fitness': s['esm2_fitness'],
                                'combined_score': s['combined_score'],
                                'motifs_found': len(motifs),
                                'motifs_detail': motifs,
                                'activities': list(set(
                                    act for m in motifs
                                    for act in m.get('all_activities', [m['activity']])
                                    if act != 'unknown'
                                )),
                            })

                    # Activity scoring
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
                        'total_generated': len(sequences),
                        'esm2_filtered': len(top_sequences),
                        'seq_with_hits': seq_with_hits,
                        'total_hits': len(all_hits),
                        'sample_id': selected_sample,
                        'avg_fitness_all': round(avg_fitness_all, 4),
                        'avg_fitness_top': round(avg_fitness_top, 4),
                        'hit_sequences': sorted(hit_sequences_detail,
                                                key=lambda x: x['combined_score'], reverse=True)[:20],
                    }

        # ---- Display results ----
        if 'profile_result' in st.session_state:
            result = st.session_state['profile_result']
            activity_scores = result['activity_scores']

            # Pipeline summary
            st.markdown("#### 파이프라인 요약")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Markov 생성", result['total_generated'])
            with c2:
                st.metric("ESM-2 필터", result['esm2_filtered'],
                          delta=f"avg fitness {result['avg_fitness_top']:.3f}")
            with c3:
                st.metric("DB 매칭 서열", result['seq_with_hits'])
            with c4:
                hit_rate = result['seq_with_hits'] / max(result['esm2_filtered'], 1) * 100
                st.metric("매칭률", f"{hit_rate:.1f}%")
            with c5:
                st.metric("총 히트 수", result['total_hits'])

            st.caption(
                f"📊 ESM-2 필터 효과: 전체 평균 fitness {result['avg_fitness_all']:.4f} → "
                f"상위 {result['esm2_filtered']}개 평균 {result['avg_fitness_top']:.4f} "
                f"(+{result['avg_fitness_top'] - result['avg_fitness_all']:.4f})"
            )

            if activity_scores:
                # ---- Radar chart ----
                top_activities = dict(
                    sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)[:12]
                )
                fig = BioactivityVisualizer.plot_activity_scores(
                    top_activities,
                    title=f"Bioactivity Profile - {result['sample_id']}"
                )
                st.plotly_chart(fig, use_container_width=True)

                # ---- Activity detail table ----
                st.markdown("### 활성별 상세")
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

                # ---- Top 3 activities ----
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

                # ---- Hit sequences table ----
                if result.get('hit_sequences'):
                    st.markdown("### 🧬 DB 매칭된 상위 서열 (ESM-2 스코어 포함)")
                    hit_rows = []
                    for hs in result['hit_sequences']:
                        hit_rows.append({
                            'Sequence': hs['sequence'],
                            'ESM-2 Fitness': f"{hs['esm2_fitness']:.4f}",
                            'Combined Score': f"{hs['combined_score']:.4f}",
                            'Motifs Found': hs['motifs_found'],
                            'Activities': ', '.join(hs['activities'][:5]),
                        })
                    st.dataframe(pd.DataFrame(hit_rows), use_container_width=True, hide_index=True)

                    # ---- Motif detail per sequence ----
                    st.markdown("### 서열별 모티프 상세")
                    st.caption("💡 서열에 마우스를 올리면 3-letter 표기가 표시됩니다")
                    for hs in result['hit_sequences'][:10]:
                        with st.expander(
                            f"`{hs['sequence']}` — {hs['motifs_found']} motifs, "
                            f"fitness {hs['esm2_fitness']:.4f}"
                        ):
                            # 서열 자체에 호버 툴팁 표시
                            st.markdown(
                                f"**서열**: {seq_with_tooltip(hs['sequence'])}",
                                unsafe_allow_html=True
                            )
                            for m in hs.get('motifs_detail', []):
                                ic50 = m.get('IC50', '')
                                ic50_str = f" | IC50: {ic50}" if ic50 else ""
                                acts = m.get('all_activities', [m['activity']])
                                st.markdown(
                                    f"- {seq_with_tooltip(m['motif'])} "
                                    f"@ pos {m['position']} — "
                                    f"{', '.join(acts[:3])}{ic50_str}",
                                    unsafe_allow_html=True
                                )

                    # ---- AI 분석 연계 ----
                    st.markdown("---")
                    st.markdown("### 🤖 AI 심층 분석 연계")

                    motif_sequences = [hs['sequence'] for hs in result['hit_sequences']]
                    st.session_state['bioactive_sequences'] = motif_sequences
                    st.session_state['bioactive_source'] = f"Bioactive Search ({result['sample_id']})"

                    selected_motif_seq = st.selectbox(
                        "전송할 서열 선택",
                        motif_sequences,
                        key="select_motif_seq"
                    )

                    col_ai1, col_ai2, col_ai3 = st.columns(3)
                    with col_ai1:
                        if st.button("🤖 → AI 활성 분석", key="send_motif_ai"):
                            st.session_state['ai_input_sequence'] = selected_motif_seq
                            st.session_state['ai_target_tab'] = 'prediction'
                            st.success(f"✅ `{selected_motif_seq}` → 6번 AI 분석 페이지로 이동하세요")
                    with col_ai2:
                        if st.button("🔬 → 3D 구조 예측", key="send_motif_3d"):
                            st.session_state['structure_sequence'] = selected_motif_seq
                            st.session_state['structure_from_bioactive'] = True
                            st.success(f"✅ `{selected_motif_seq}` → 5번 3D 구조 페이지로 이동하세요")
                    with col_ai3:
                        if st.button("📦 전체 → AI 배치", key="send_motif_batch"):
                            st.session_state['ai_batch_sequences'] = motif_sequences
                            st.session_state['ai_batch_source'] = (
                                f"{result['sample_id']} DB 매칭 후보 {len(motif_sequences)}개"
                            )
                            st.success(f"✅ {len(motif_sequences)}개 서열 → 6번 AI 배치 분석으로 이동하세요")

            else:
                st.warning("매칭된 생리활성 펩타이드가 없습니다. 서열 수를 늘려보세요.")

    # ========== TAB 2: Compare Samples ==========
    with tab2:
        st.markdown("### Compare Bioactivity Profiles")
        st.markdown("여러 펩톤의 DB 매칭 기반 활성 프로파일을 비교합니다.")

        display_to_id = sample_options
        id_to_display = {v: k for k, v in sample_options.items()}

        sample_ids = list(sample_options.values())
        default_displays = (
            [id_to_display[sid] for sid in sample_ids[:3]]
            if len(sample_ids) >= 3
            else [id_to_display[sid] for sid in sample_ids]
        )

        selected_displays = st.multiselect(
            "비교할 샘플 선택 (최대 5개):",
            list(sample_options.keys()),
            default=default_displays,
            max_selections=5
        )

        selected_samples = [display_to_id[display] for display in selected_displays]

        if len(selected_samples) >= 2:
            if st.button("📊 비교 분석 실행"):
                with st.spinner("DB 매칭 기반 비교 분석 중... (샘플당 200개 서열 생성)"):
                    comparison = predictor.compare_samples_bioactivity(selected_samples)
                    st.session_state['compare_result'] = comparison

            if 'compare_result' in st.session_state:
                comparison = st.session_state['compare_result']
                samples_activities = comparison.get('sample_scores', {})

                if samples_activities:
                    # Top 10 activities for radar
                    all_acts = {}
                    for scores in samples_activities.values():
                        for act, score in scores.items():
                            all_acts[act] = max(all_acts.get(act, 0), score)
                    top_acts = sorted(all_acts, key=all_acts.get, reverse=True)[:10]

                    filtered = {}
                    for sid, scores in samples_activities.items():
                        filtered[sid] = {act: scores.get(act, 0) for act in top_acts}

                    fig = BioactivityVisualizer.plot_activity_comparison(
                        filtered,
                        title="DB-based Bioactivity Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Best samples
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

                    # Detailed scores
                    st.markdown("#### Detailed Scores")
                    detailed_df = pd.DataFrame(samples_activities).T
                    detailed_df = detailed_df.round(3)
                    st.dataframe(detailed_df, use_container_width=True)

        else:
            st.info("비교할 샘플을 2개 이상 선택하세요.")


if __name__ == "__main__":
    main()
