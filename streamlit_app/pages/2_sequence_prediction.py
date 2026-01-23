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
    samples = loader.get_sample_list()

    # Sample selection
    selected_sample = st.selectbox(
        "Select Sample:",
        samples,
        index=0
    )

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

    if st.button("🎲 Generate Sequences", type="primary"):
        with st.spinner("Generating sequences..."):
            # Get composition
            taa_comp = loader.get_taa_composition(selected_sample)

            # Generate sequences
            predictor = AbundancePredictor(loader)
            result = predictor.predict_for_sample(
                selected_sample,
                length_range=(min_length, max_length),
                n_sequences=n_sequences,
                method=method
            )

            st.success(f"Generated {result['n_generated']} sequences!")

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Top Sequences",
                "📊 By Length",
                "🔬 Detailed Analysis",
                "💾 Export"
            ])

            with tab1:
                st.markdown("### Top 20 Sequences")

                # Display top sequences
                sequences_with_mw = result.get('sequences_with_mw', [])

                if sequences_with_mw:
                    df = pd.DataFrame(sequences_with_mw)
                    df['rank'] = range(1, len(df) + 1)
                    df = df[['rank', 'sequence', 'length', 'likelihood_score', 'molecular_weight']]

                    # Format columns
                    # Use scientific notation for very small scores
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
                        seqs = by_length[length][:5]  # Top 5 per length

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

                    # Length distribution
                    lengths = [info['length'] for info in sequences_with_mw]
                    st.metric("Avg Length", f"{sum(lengths) / len(lengths):.1f} AA")
                    st.metric("Min Length", min(lengths))
                    st.metric("Max Length", max(lengths))

                with col2:
                    st.markdown("#### Composition Used")
                    composition = result.get('composition', {})

                    # Top 10 amino acids in composition
                    sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)[:10]

                    for aa, pct in sorted_comp:
                        st.write(f"{aa}: {pct:.2f}%")

                # Score distribution
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

                # Prepare FASTA format
                top_sequences = result.get('top_sequences', [])[:50]  # Top 50

                fasta_content = ""
                for i, (seq, score) in enumerate(top_sequences, 1):
                    mw = calculate_sequence_mw(seq)
                    fasta_content += f">{selected_sample}_seq{i} | score={score:.4f} | MW={mw:.1f}Da\n"
                    fasta_content += f"{seq}\n"

                st.download_button(
                    label="📥 Download FASTA",
                    data=fasta_content,
                    file_name=f"{selected_sample}_predicted_sequences.fasta",
                    mime="text/plain"
                )

                # CSV export
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
                    file_name=f"{selected_sample}_predicted_sequences.csv",
                    mime="text/csv"
                )

                st.info(f"Exporting top {len(top_sequences)} sequences")


if __name__ == "__main__":
    main()
