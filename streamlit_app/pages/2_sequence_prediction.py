"""
Page 2: Sequence Prediction
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import CompositionLoader
from sequence_predictor import SequenceGenerator, AbundancePredictor
from visualizer_2d import PeptideDiagram
from utils import calculate_sequence_mw, save_fasta
from plm_embedder import PLMEmbedder

st.set_page_config(page_title="Sequence Prediction", page_icon="🧬", layout="wide")


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
def load_digester():
    """In silico digester 로드"""
    try:
        from in_silico_digester import InSilicoDigester
        return InSilicoDigester()
    except Exception as e:
        return None


def main():
    st.title("🧬 Peptide Sequence Prediction")
    st.markdown("Generate probable peptide sequences from peptone samples")
    st.markdown("---")

    loader = load_data()
    digester = load_digester()

    # ─── 예측 방식 선택 ─────────────────────────────────
    st.markdown("### 🎯 예측 방식 선택")

    mode_options = []
    if digester:
        mode_options.append("🔬 Hybrid: In Silico + Markov (권장)")
        mode_options.append("🧪 In Silico Digestion only")
    mode_options.append("📊 Markov Chain only (기존 방식)")

    mode = st.radio(
        "Prediction Mode:",
        mode_options,
        index=0,
        help=(
            "Hybrid: 효소 분해 시뮬레이션 + Markov 통계 생성 결합 (가장 풍부한 후보)\n"
            "In Silico only: 효소 공정 + 원료 단백질만 사용 (결정론적)\n"
            "Markov only: TAA 조성 비율 기반 통계 생성 (기존 방식)"
        )
    )

    use_hybrid = mode.startswith("🔬")
    use_in_silico_only = mode.startswith("🧪")
    use_in_silico = use_hybrid or use_in_silico_only

    if use_in_silico and not digester:
        st.error("In silico digester 로드 실패. Markov 방식으로 전환됩니다.")
        use_in_silico = False
        use_hybrid = False
        use_in_silico_only = False

    st.markdown("---")

    # ─── 모드별 설정 ──────────────────────────────────
    if use_in_silico:
        # In Silico Digestion 모드
        st.markdown("### 🧪 In Silico Digestion 설정")

        products = digester.enzyme_processor.list_products()

        col_p1, col_p2 = st.columns([2, 1])
        with col_p1:
            selected_product = st.selectbox(
                "제품 선택:",
                products,
                index=0,
                help="회사 효소 공정 자료가 있는 제품들"
            )
        with col_p2:
            process = digester.enzyme_processor.get_process(selected_product)
            if process:
                st.metric("원료", process.raw_material_id)
                st.metric("효소", ", ".join(process.enzymes_used) or "—")

        # 공정 상세 표시
        if process:
            with st.expander("📋 공정 상세 보기"):
                st.markdown(f"**원료**: {process.raw_material_form} "
                           f"({process.raw_concentration_pct}%)")
                if process.pretreatment:
                    e = process.pretreatment.enzymes[0]
                    st.markdown(f"**전처리**: {e['enzyme']} {e['concentration_pct']}% "
                               f"@ {process.pretreatment.temperature_c}°C × "
                               f"{process.pretreatment.duration_min}분")
                if process.main_hydrolysis:
                    enz_str = " + ".join(
                        f"{e['enzyme']} {e['concentration_pct']}%"
                        for e in process.main_hydrolysis.enzymes
                    )
                    st.markdown(f"**메인 분해**: {enz_str} @ "
                               f"{process.main_hydrolysis.temperature_c}°C × "
                               f"{process.main_hydrolysis.duration_hours}시간")
                if process.has_uf:
                    st.markdown(f"**UF 컷오프**: {process.uf_cutoff_kda} kDa")
                if process.notes:
                    st.caption(f"📝 {process.notes}")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_length = st.number_input("Min Length (AA)", min_value=2,
                                         max_value=30, value=3)
        with col2:
            max_length = st.number_input("Max Length (AA)", min_value=3,
                                         max_value=50, value=20)
        with col3:
            n_top_proteins = st.number_input(
                "원료 단백질 상위 N개",
                min_value=5, max_value=100, value=20,
                help="모든 원료 단백질 대신 길이 기준 상위 N개만 사용"
            )

    else:
        # Markov Chain 모드 (기존)
        sample_options = loader.get_sample_options()
        selected_display = st.selectbox(
            "Select Sample:",
            list(sample_options.keys()),
            index=0
        )
        selected_sample = sample_options[selected_display]

        st.markdown("### Markov Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_length = st.number_input("Min Length (AA)", min_value=3,
                                         max_value=20, value=5)
        with col2:
            max_length = st.number_input("Max Length (AA)", min_value=3,
                                         max_value=20, value=12)
        with col3:
            n_sequences = st.number_input("Number of Sequences",
                                          min_value=10, max_value=200, value=50)

        method = st.selectbox(
            "Generation Method:",
            ["markov", "random", "frequent"],
            index=0,
            help="markov: 전이 확률 기반 | random: 단순 무작위 | frequent: 빈도 우선"
        )

    # ─── 헬퍼: result에 표시용 필드 추가 ────────────
    def enrich_result_for_display(seq_score_list):
        """sequences_with_mw, by_length, composition 등 표시용 필드 생성"""
        from collections import defaultdict
        sequences_with_mw = []
        by_length = defaultdict(list)
        for seq, score in seq_score_list[:200]:  # 상위 200개만
            mw = calculate_sequence_mw(seq)
            sequences_with_mw.append({
                'sequence': seq,
                'length': len(seq),
                'likelihood_score': float(score),
                'molecular_weight': float(mw),
            })
            by_length[len(seq)].append((seq, score))
        return sequences_with_mw, dict(by_length)

    # ─── 생성 버튼 ────────────────────────────────────
    if st.button("🎲 Generate Sequences", type="primary"):
        with st.spinner("Generating sequences..."):
            if use_hybrid:
                # Hybrid: In Silico + Markov 결합
                hybrid_result = digester.hybrid_digest_and_markov(
                    selected_product,
                    min_length=min_length,
                    max_length=max_length,
                    n_top_proteins=n_top_proteins,
                    n_markov_sequences=500
                )

                seq_score_list = [
                    (c['sequence'], c['score'])
                    for c in hybrid_result['combined_sequences']
                ]
                sequences_with_mw, by_length = enrich_result_for_display(seq_score_list)
                summary = digester.summarize(hybrid_result['in_silico_peptides'])

                result = {
                    'n_generated': len(seq_score_list),
                    'sequences': seq_score_list,
                    'sequences_with_mw': sequences_with_mw,
                    'by_length': by_length,
                    'sample_id': selected_product,
                    'method': 'hybrid_in_silico_markov',
                    'composition_used': hybrid_result['aa_composition'],
                    'in_silico_peptides': hybrid_result['in_silico_peptides'],
                    'in_silico_summary': summary,
                    'hybrid_data': hybrid_result,
                }

                st.session_state['seq_gen_result'] = result
                st.session_state['seq_gen_sample'] = selected_product
                st.session_state['seq_gen_method'] = 'hybrid_in_silico_markov'

            elif use_in_silico_only:
                peptides = digester.digest_product(
                    selected_product,
                    min_length=min_length,
                    max_length=max_length,
                    n_top_proteins=n_top_proteins
                )
                unique_peptides = digester.get_unique_peptides(peptides)
                summary = digester.summarize(unique_peptides)

                seq_score_list = [
                    (p.sequence, 1.0 / (1 + abs(p.length - 8)))
                    for p in unique_peptides
                ]
                seq_score_list.sort(key=lambda x: x[1], reverse=True)
                sequences_with_mw, by_length = enrich_result_for_display(seq_score_list)

                result = {
                    'n_generated': len(unique_peptides),
                    'sequences': seq_score_list,
                    'sequences_with_mw': sequences_with_mw,
                    'by_length': by_length,
                    'sample_id': selected_product,
                    'method': 'in_silico_digestion',
                    'composition_used': {},
                    'in_silico_peptides': unique_peptides,
                    'in_silico_summary': summary,
                }

                st.session_state['seq_gen_result'] = result
                st.session_state['seq_gen_sample'] = selected_product
                st.session_state['seq_gen_method'] = 'in_silico_digestion'
            else:
                # 기존 Markov Chain
                predictor = AbundancePredictor(loader)
                result = predictor.predict_for_sample(
                    selected_sample,
                    length_range=(min_length, max_length),
                    n_sequences=n_sequences,
                    method=method
                )
                st.session_state['seq_gen_result'] = result
                st.session_state['seq_gen_sample'] = selected_sample
                st.session_state['seq_gen_method'] = method

    # ---- 결과 표시: session_state에서 읽음 (버튼 블록 바깥!) ----
    if 'seq_gen_result' in st.session_state:
        result = st.session_state['seq_gen_result']
        gen_sample = st.session_state.get('seq_gen_sample', '?')
        gen_method = st.session_state.get('seq_gen_method', '?')

        st.success(f"Generated {result['n_generated']} sequences! (Sample: {gen_sample}, Method: {gen_method})")

        # Hybrid 모드 요약 표시
        if gen_method == 'hybrid_in_silico_markov' and 'hybrid_data' in result:
            hd = result['hybrid_data']
            st.markdown("### 🔬 Hybrid 생성 결과 요약")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("총 펩타이드 (Union)", f"{len(hd['combined_sequences'])}개")
            with m2:
                st.metric("🔬 양쪽 확인 (BOTH)", f"{hd['overlap_count']}개",
                         help="In silico와 Markov가 모두 생성 → 최고 신뢰도")
            with m3:
                st.metric("🧪 In Silico only", f"{hd['n_in_silico_only']}개")
            with m4:
                st.metric("📊 Markov only", f"{hd['n_markov_only']}개")

            st.info(
                "💡 **점수 가중치**: BOTH (1.3) > In Silico (1.0) > Markov (0.8)  \n"
                "양쪽에서 동시에 생성된 서열은 결정론적 + 통계적 양쪽으로 검증된 것이라 신뢰도가 가장 높습니다."
            )

        # In Silico Digestion only 요약
        elif gen_method == 'in_silico_digestion' and 'in_silico_summary' in result:
            summary = result['in_silico_summary']
            st.markdown("### 🧪 In Silico Digestion 요약")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("총 펩타이드 (중복 제거)", f"{summary['n_unique']}개")
            with m2:
                st.metric("원료 단백질 수", f"{summary['n_source_proteins']}개")
            with m3:
                st.metric("평균 길이", f"{summary['length_avg']} AA")
            with m4:
                st.metric("평균 분자량", f"{summary['mw_avg']:.0f} Da")

            # 출처 단백질별 펩타이드 분포
            with st.expander("📊 원료 단백질별 펩타이드 분포"):
                from collections import Counter
                source_counts = Counter(
                    p.source_protein_name[:50]
                    for p in result['in_silico_peptides']
                )
                source_df = pd.DataFrame(
                    sorted(source_counts.items(), key=lambda x: -x[1])[:15],
                    columns=['원료 단백질', '생성 펩타이드 수']
                )
                st.dataframe(source_df, use_container_width=True, hide_index=True)

        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Top Sequences",
            "📊 By Length",
            "🔬 Detailed Analysis",
            "💾 Export"
        ])

        with tab1:
            st.markdown("### Top 20 Sequences")

            sequences_with_mw = result.get('sequences_with_mw', [])

            if sequences_with_mw:
                df = pd.DataFrame(sequences_with_mw)
                df['rank'] = range(1, len(df) + 1)
                df = df[['rank', 'sequence', 'length', 'likelihood_score', 'molecular_weight']]

                df['likelihood_score'] = df['likelihood_score'].apply(
                    lambda x: f"{x:.6f}" if x > 0.0001 else f"{x:.2e}"
                )
                df['molecular_weight'] = df['molecular_weight'].apply(lambda x: f"{x:.1f}")

                st.dataframe(df, use_container_width=True, hide_index=True)

                # Visualize top sequence
                st.markdown("#### Top Sequence Visualization")
                top_seq = sequences_with_mw[0]['sequence']

                fig = PeptideDiagram.plot_sequence_diagram(
                    top_seq,
                    title=f"Top Sequence: {top_seq}"
                )
                st.plotly_chart(fig, use_container_width=True)

                fig2 = PeptideDiagram.plot_hydrophobicity_profile(
                    top_seq,
                    title="Hydrophobicity Profile"
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("### Sequences by Length")

            by_length = result.get('by_length', {})

            for length in sorted(by_length.keys()):
                with st.expander(f"Length {length} AA ({len(by_length[length])} sequences)"):
                    seqs = by_length[length][:5]

                    for i, (seq, score) in enumerate(seqs, 1):
                        mw = calculate_sequence_mw(seq)
                        score_str = f"{score:.6f}" if score > 0.0001 else f"{score:.2e}"
                        st.write(f"{i}. **{seq}** (score: {score_str}, MW: {mw:.1f} Da)")

        with tab3:
            st.markdown("### Detailed Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Statistics")
                st.metric("Total Generated", result['n_generated'])
                st.metric("Method", result['method'])

                if sequences_with_mw:
                    lengths = [info['length'] for info in sequences_with_mw]
                    st.metric("Avg Length", f"{sum(lengths) / len(lengths):.1f} AA")
                    st.metric("Min Length", min(lengths))
                    st.metric("Max Length", max(lengths))

            with col2:
                st.markdown("#### Composition Used")
                composition = result.get('composition', {})

                sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)[:10]

                for aa, pct in sorted_comp:
                    st.write(f"{aa}: {pct:.2f}%")

            if sequences_with_mw:
                st.markdown("#### Score Distribution")
                scores = [info['likelihood_score'] for info in sequences_with_mw]

                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20)])
                fig.update_layout(
                    title="Likelihood Score Distribution",
                    xaxis_title="Score",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.markdown("### Export Sequences")

            top_sequences = result.get('top_sequences', [])[:50]

            fasta_content = ""
            for i, (seq, score) in enumerate(top_sequences, 1):
                mw = calculate_sequence_mw(seq)
                fasta_content += f">{gen_sample}_seq{i} | score={score:.4f} | MW={mw:.1f}Da\n"
                fasta_content += f"{seq}\n"

            st.download_button(
                label="📥 Download FASTA",
                data=fasta_content,
                file_name=f"{gen_sample}_predicted_sequences.fasta",
                mime="text/plain"
            )

            csv_data = pd.DataFrame([
                {
                    'sequence': seq,
                    'length': len(seq),
                    'likelihood_score': score,
                    'molecular_weight': calculate_sequence_mw(seq)
                }
                for seq, score in top_sequences
            ])

            st.download_button(
                label="📥 Download CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"{gen_sample}_predicted_sequences.csv",
                mime="text/csv"
            )

            st.info(f"Exporting top {len(top_sequences)} sequences")

        # ============================================================
        # ESM-2 Re-ranking
        # ============================================================
        st.markdown("---")
        with st.expander("🧬 ESM-2 Re-ranking 결과", expanded=False):
            st.markdown("ESM-2 단백질 언어 모델을 활용하여 생성된 서열의 진화적 적합성(fitness)을 평가하고 재순위화합니다.")

            sequences_for_rerank = result.get('sequences_with_mw', [])

            if sequences_for_rerank:
                if st.button("🔬 ESM-2 Re-ranking 실행", key="run_esm2_rerank"):
                    embedder = load_plm_embedder()

                    with st.spinner("ESM-2 fitness scoring 중... (서열당 수 초 소요)"):
                        rerank_data = []
                        progress_bar = st.progress(0)

                        for idx, seq_info in enumerate(sequences_for_rerank):
                            seq = seq_info['sequence']
                            likelihood = seq_info['likelihood_score']

                            fitness = embedder.get_fitness_score(seq)

                            combined = likelihood * 0.4 + fitness * 0.6

                            rerank_data.append({
                                'sequence': seq,
                                'length': seq_info['length'],
                                'likelihood_score': likelihood,
                                'esm2_fitness': fitness,
                                'combined_score': round(combined, 6),
                                'molecular_weight': seq_info['molecular_weight'],
                            })

                            progress_bar.progress((idx + 1) / len(sequences_for_rerank))

                        progress_bar.empty()

                    # Store in session state
                    st.session_state['esm2_rerank_data'] = rerank_data

                # Display results if available
                if 'esm2_rerank_data' in st.session_state:
                    rerank_data = st.session_state['esm2_rerank_data']

                    # Before/After comparison
                    st.markdown("#### Markov-only vs ESM-2 Re-ranked 비교")

                    col_before, col_after = st.columns(2)

                    # Markov-only ranking (original order)
                    markov_ranked = sorted(rerank_data, key=lambda x: x['likelihood_score'], reverse=True)
                    # ESM-2 re-ranked (combined score)
                    esm2_ranked = sorted(rerank_data, key=lambda x: x['combined_score'], reverse=True)

                    with col_before:
                        st.markdown("**Markov-only Ranking (기존)**")
                        for i, item in enumerate(markov_ranked[:10], 1):
                            score_str = f"{item['likelihood_score']:.6f}" if item['likelihood_score'] > 0.0001 else f"{item['likelihood_score']:.2e}"
                            st.write(f"{i}. `{item['sequence']}` (score: {score_str})")

                    with col_after:
                        st.markdown("**ESM-2 Re-ranked (개선)**")
                        for i, item in enumerate(esm2_ranked[:10], 1):
                            st.write(f"{i}. `{item['sequence']}` (combined: {item['combined_score']:.4f})")

                    # Full re-ranked table
                    st.markdown("#### ESM-2 Re-ranked 전체 결과")

                    df_rerank = pd.DataFrame(esm2_ranked)
                    df_rerank.insert(0, 'rank', range(1, len(df_rerank) + 1))

                    df_display = df_rerank[['rank', 'sequence', 'length', 'likelihood_score', 'esm2_fitness', 'combined_score', 'molecular_weight']].copy()
                    df_display.columns = ['Rank', 'Sequence', 'Length', 'Likelihood', 'ESM-2 Fitness', 'Combined Score', 'MW (Da)']

                    df_display['Likelihood'] = df_display['Likelihood'].apply(
                        lambda x: f"{x:.6f}" if x > 0.0001 else f"{x:.2e}"
                    )
                    df_display['ESM-2 Fitness'] = df_display['ESM-2 Fitness'].apply(lambda x: f"{x:.4f}")
                    df_display['Combined Score'] = df_display['Combined Score'].apply(lambda x: f"{x:.4f}")
                    df_display['MW (Da)'] = df_display['MW (Da)'].apply(lambda x: f"{x:.1f}")

                    st.dataframe(df_display, use_container_width=True, hide_index=True)

                    # Improvement suggestions for top sequences
                    st.markdown("#### 🔧 개선 제안 (Top 서열)")
                    st.markdown("ESM-2가 각 위치에서 더 적합한 아미노산을 제안합니다.")

                    top_for_suggestions = esm2_ranked[:3]
                    embedder = load_plm_embedder()

                    for rank_idx, item in enumerate(top_for_suggestions, 1):
                        seq = item['sequence']
                        with st.spinner(f"서열 #{rank_idx} 개선안 분석 중..."):
                            suggestions = embedder.get_improvement_suggestions(seq, top_n=3)

                        if suggestions:
                            st.markdown(f"**#{rank_idx} `{seq}`** (fitness: {item['esm2_fitness']:.4f})")
                            for sug in suggestions:
                                st.write(
                                    f"  - 위치 {sug['position']}: "
                                    f"**{sug['current_aa']}** -> **{sug['suggested_aa']}** "
                                    f"(확률: {sug['current_prob']:.3f} -> {sug['suggested_prob']:.3f}, "
                                    f"변이: `{sug['mutation']}`, gain: +{sug['probability_gain']:.3f})"
                                )
                        else:
                            st.markdown(f"**#{rank_idx} `{seq}`** - 이미 최적화된 서열입니다.")

        # ============================================================
        # AI 분석 연계 (결과 표시 영역 바깥, session_state 블록 안)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🤖 AI 심층 분석 연계")

        all_seqs = [s['sequence'] for s in result.get('sequences_with_mw', [])]

        if all_seqs:
            st.markdown(f"**생성된 서열**: {len(all_seqs)}개 | **Top 서열**: `{all_seqs[0]}`")

            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                if st.button("🧬 Top 서열 → 임베딩 분석", key="send_embed"):
                    st.session_state['ai_input_sequence'] = all_seqs[0]
                    st.session_state['ai_target_tab'] = 'embedding'
                    st.success(f"✅ 전송 완료! 사이드바에서 **AI 심층 분석** 페이지로 이동하세요.")
            with col_ai2:
                if st.button("📊 Top 서열 → 활성 예측", key="send_predict"):
                    st.session_state['ai_input_sequence'] = all_seqs[0]
                    st.session_state['ai_target_tab'] = 'prediction'
                    st.success(f"✅ 전송 완료! 사이드바에서 **AI 심층 분석** 페이지로 이동하세요.")

            # 배치 전송
            n_batch = st.slider("배치 분석할 서열 수", 1, min(20, len(all_seqs)), 5, key="batch_n")
            if st.button("📦 상위 N개 서열 일괄 전송", key="send_batch"):
                st.session_state['ai_batch_sequences'] = all_seqs[:n_batch]
                st.session_state['ai_batch_source'] = f"{gen_sample} Top {n_batch}"
                st.success(f"✅ {n_batch}개 서열 전송 완료! 사이드바에서 **AI 심층 분석** 페이지로 이동하세요.")


if __name__ == "__main__":
    main()
