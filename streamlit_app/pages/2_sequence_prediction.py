"""
Page 2: Sequence Prediction
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
from data_loader import CompositionLoader
from sequence_predictor import SequenceGenerator, AbundancePredictor
from visualizer_2d import PeptideDiagram
from utils import calculate_sequence_mw, save_fasta

st.set_page_config(page_title="Sequence Prediction", page_icon="🧬", layout="wide")


@st.cache_resource
def load_data():
    loader = CompositionLoader()
    loader.load_data()
    return loader


def main():
    st.title("🧬 Peptide Sequence Prediction")
    st.markdown("Generate probable peptide sequences based on amino acid composition")
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

    # Prediction parameters
    st.markdown("### Prediction Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_length = st.number_input("Min Length (AA)", min_value=3, max_value=20, value=5)

    with col2:
        max_length = st.number_input("Max Length (AA)", min_value=3, max_value=20, value=12)

    with col3:
        n_sequences = st.number_input("Number of Sequences", min_value=10, max_value=200, value=50)

    method = st.selectbox(
        "Generation Method:",
        ["markov", "random", "frequent"],
        index=0,
        help="markov: Uses transition probabilities | random: Simple random | frequent: Prefers abundant AAs"
    )

    # ---- 생성 버튼: 결과를 session_state에 저장 ----
    if st.button("🎲 Generate Sequences", type="primary"):
        with st.spinner("Generating sequences..."):
            taa_comp = loader.get_peptide_composition(selected_sample, normalize=True)

            predictor = AbundancePredictor(loader)
            result = predictor.predict_for_sample(
                selected_sample,
                length_range=(min_length, max_length),
                n_sequences=n_sequences,
                method=method
            )

            # session_state에 결과 저장 (리렌더링 후에도 유지)
            st.session_state['seq_gen_result'] = result
            st.session_state['seq_gen_sample'] = selected_sample
            st.session_state['seq_gen_method'] = method

    # ---- 결과 표시: session_state에서 읽음 (버튼 블록 바깥!) ----
    if 'seq_gen_result' in st.session_state:
        result = st.session_state['seq_gen_result']
        gen_sample = st.session_state.get('seq_gen_sample', selected_sample)
        gen_method = st.session_state.get('seq_gen_method', method)

        st.success(f"Generated {result['n_generated']} sequences! (Sample: {gen_sample}, Method: {gen_method})")

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
        # AI 분석 연계 (결과 표시 영역 바깥, session_state 블록 안)
        # ============================================================
        st.markdown("---")
        st.markdown("### 🤖 AI 심층 분석 연계")

        all_seqs = [s['sequence'] for s in result.get('sequences_with_mw', [])]

        if all_seqs:
            st.markdown(f"**생성된 서열**: {len(all_seqs)}개 | **Top 서열**: `{all_seqs[0]}`")

            col_ai1, col_ai2, col_ai3 = st.columns(3)
            with col_ai1:
                if st.button("🧬 Top 서열 → 임베딩 분석", key="send_embed"):
                    st.session_state['ai_input_sequence'] = all_seqs[0]
                    st.session_state['ai_target_tab'] = 'embedding'
                    st.success(f"✅ 전송 완료! 사이드바에서 **AI 심층 분석** 페이지로 이동하세요.")
            with col_ai2:
                if st.button("🔬 Top 서열 → 변이 예측", key="send_mutation"):
                    st.session_state['ai_input_sequence'] = all_seqs[0]
                    st.session_state['ai_target_tab'] = 'mutation'
                    st.success(f"✅ 전송 완료! 사이드바에서 **AI 심층 분석** 페이지로 이동하세요.")
            with col_ai3:
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

        # 결과 초기화 버튼
        st.markdown("---")
        if st.button("🔄 새로운 서열 생성하기", key="reset_results"):
            del st.session_state['seq_gen_result']
            st.rerun()


if __name__ == "__main__":
    main()
