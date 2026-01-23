"""
Page 4: 2D Structure Visualization
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
from data_loader import CompositionLoader
from sequence_predictor import SequenceGenerator
from visualizer_2d import PeptideDiagram, CompositionVisualizer

st.set_page_config(page_title="2D Structure", page_icon="🎨", layout="wide")


@st.cache_resource
def load_data():
    loader = CompositionLoader()
    loader.load_data()
    return loader


def main():
    st.title("🎨 2D Structure Visualization")
    st.markdown("Visualize peptide sequences and properties")
    st.markdown("---")

    loader = load_data()
    samples = loader.get_sample_list()

    # Input method
    input_method = st.radio(
        "Sequence Input Method:",
        ["Generate from Sample", "Custom Sequence"]
    )

    sequence = None

    if input_method == "Generate from Sample":
        selected_sample = st.selectbox("Select Sample:", samples, index=0)

        col1, col2 = st.columns(2)
        with col1:
            length = st.slider("Sequence Length:", 5, 20, 10)
        with col2:
            method = st.selectbox("Method:", ["markov", "random", "frequent"])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎲 Generate Random Sequence"):
                taa_comp = loader.get_taa_composition(selected_sample)
                generator = SequenceGenerator(taa_comp)

                sequences = generator.generate_sequences(
                    length_range=(length, length),
                    n_sequences=1,
                    method=method
                )

                if sequences:
                    sequence = sequences[0][0]
                    score = generator.score_sequence_likelihood(sequence)
                    st.session_state['sequence'] = sequence
                    st.session_state['sequence_score'] = score

        with col2:
            if st.button("⭐ Generate Best Sequence", type="primary"):
                taa_comp = loader.get_taa_composition(selected_sample)
                generator = SequenceGenerator(taa_comp)

                # Generate multiple sequences and pick the best
                sequences = generator.generate_sequences(
                    length_range=(length, length),
                    n_sequences=20,
                    method="markov"
                )

                # Extract sequence strings (sequences are tuples)
                seq_strings = [s[0] if isinstance(s, tuple) else s for s in sequences]

                # Score all sequences
                scored = [(seq, generator.score_sequence_likelihood(seq)) for seq in seq_strings]
                scored.sort(key=lambda x: x[1], reverse=True)

                if scored:
                    sequence = scored[0][0]
                    score = scored[0][1]
                    st.session_state['sequence'] = sequence
                    st.session_state['sequence_score'] = score
                    st.success(f"✅ Best sequence selected (score: {score:.4f})")

        # Use stored sequence if exists
        if 'sequence' in st.session_state:
            sequence = st.session_state['sequence']
            if 'sequence_score' in st.session_state:
                st.info(f"📊 Likelihood Score: {st.session_state['sequence_score']:.4f}")

    else:
        sequence = st.text_input(
            "Enter Sequence (1-letter codes):",
            value="ARNDCEQGH",
            max_chars=30
        ).upper()

    if sequence:
        st.markdown("---")
        st.markdown(f"### Visualizing: **{sequence}** (Length: {len(sequence)} AA)")

        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "🔗 Sequence Diagram",
            "📈 Hydrophobicity",
            "🎯 Properties"
        ])

        with tab1:
            st.markdown("#### Peptide Sequence Diagram")
            st.caption("Color code: 🟠 Hydrophobic | 🔵 Positive | 🔴 Negative | 🟢 Polar | ⚫ Other")

            fig = PeptideDiagram.plot_sequence_diagram(sequence)
            st.plotly_chart(fig, use_container_width=True)

            # Sequence details
            st.markdown("#### Sequence Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Length", len(sequence))

            with col2:
                from utils import calculate_sequence_mw
                mw = calculate_sequence_mw(sequence)
                st.metric("Molecular Weight", f"{mw:.1f} Da")

            with col3:
                # Count amino acid types
                from utils import AMINO_ACIDS
                hydrophobic_count = sum(1 for aa in sequence if AMINO_ACIDS.get(aa, {}).get('hydrophobic', False))
                st.metric("Hydrophobic AAs", hydrophobic_count)

        with tab2:
            st.markdown("#### Hydrophobicity Profile")
            st.caption("Based on Kyte-Doolittle scale (positive = hydrophobic, negative = hydrophilic)")

            fig = PeptideDiagram.plot_hydrophobicity_profile(sequence)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            hydrophobicity_scale = {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            }

            hydro_values = [hydrophobicity_scale.get(aa, 0) for aa in sequence]
            avg_hydro = sum(hydro_values) / len(hydro_values)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Hydrophobicity", f"{avg_hydro:.2f}")

            with col2:
                st.metric("Most Hydrophobic", f"{max(hydro_values):.2f}")

            with col3:
                st.metric("Most Hydrophilic", f"{min(hydro_values):.2f}")

        with tab3:
            st.markdown("#### Physicochemical Properties")

            # Create composition from sequence
            from collections import Counter
            aa_counts = Counter(sequence)
            total = len(sequence)
            composition = {aa: (count / total) * 100 for aa, count in aa_counts.items()}

            fig = CompositionVisualizer.plot_property_radar(composition)
            st.plotly_chart(fig, use_container_width=True)

            # Property details
            from utils import calculate_property_ratios
            properties = calculate_property_ratios(composition)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Property Ratios:**")
                for key, value in list(properties.items())[:6]:
                    st.write(f"{key.replace('_', ' ').title()}: {value:.2f}%")

            with col2:
                st.markdown("**Amino Acid Composition:**")
                for aa, pct in sorted(composition.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{aa}: {pct:.1f}%")

        # Export section
        st.markdown("---")
        st.markdown("### Export")

        col1, col2 = st.columns(2)

        with col1:
            # FASTA export
            fasta = f">Custom_Sequence | Length={len(sequence)} | MW={mw:.1f}Da\n{sequence}\n"

            st.download_button(
                label="📥 Download FASTA",
                data=fasta,
                file_name="sequence.fasta",
                mime="text/plain"
            )

        with col2:
            # Sequence info
            st.info(f"Sequence: {sequence}")

    else:
        st.info("👆 Please generate or enter a sequence to visualize")


if __name__ == "__main__":
    main()
