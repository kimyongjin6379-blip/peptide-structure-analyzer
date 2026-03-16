"""
Page 5: 3D Structure Visualization
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import streamlit.components.v1 as components
from data_loader import CompositionLoader
from sequence_predictor import SequenceGenerator
from structure_builder import StructureBuilder

st.set_page_config(page_title="3D Structure", page_icon="🔬", layout="wide")


@st.cache_resource
def load_data():
    loader = CompositionLoader()
    loader.load_data()
    return loader


def main():
    st.title("🔬 3D Structure Prediction & Visualization")
    st.markdown("Predict and visualize 3D protein structures using ESMFold")
    st.markdown("---")

    loader = load_data()

    # Input method
    input_method = st.radio(
        "Sequence Input Method:",
        ["Generate from Sample", "Custom Sequence"]
    )

    sequence = None

    if input_method == "Generate from Sample":
        # Get sample options with product names
        sample_options = loader.get_sample_options()

        selected_display = st.selectbox(
            "Select Sample:",
            list(sample_options.keys()),
            index=0
        )

        # Get actual sample_id
        selected_sample = sample_options[selected_display]

        col1, col2 = st.columns(2)
        with col1:
            length = st.slider("Sequence Length:", 5, 30, 12)
        with col2:
            method = st.selectbox("Method:", ["markov", "random", "frequent"])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎲 Generate Random Sequence"):
                taa_comp = loader.get_peptide_composition(selected_sample, normalize=True)
                generator = SequenceGenerator(taa_comp)

                sequences = generator.generate_sequences(
                    length_range=(length, length),
                    n_sequences=1,
                    method=method
                )

                if sequences:
                    sequence = sequences[0][0]
                    score = generator.score_sequence_likelihood(sequence)
                    st.session_state['structure_sequence'] = sequence
                    st.session_state['structure_sequence_score'] = score

        with col2:
            if st.button("⭐ Generate Best Sequence", type="primary"):
                taa_comp = loader.get_peptide_composition(selected_sample, normalize=True)
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
                    st.session_state['structure_sequence'] = sequence
                    st.session_state['structure_sequence_score'] = score
                    st.success(f"✅ Best sequence selected (score: {score:.4f})")

        if 'structure_sequence' in st.session_state:
            sequence = st.session_state['structure_sequence']
            if 'structure_sequence_score' in st.session_state:
                st.info(f"📊 Likelihood Score: {st.session_state['structure_sequence_score']:.4f}")

    else:
        sequence = st.text_input(
            "Enter Sequence (1-letter codes, max 30 AA):",
            value="ARNDCEQGH",
            max_chars=30
        ).upper()

    if sequence:
        st.markdown("---")
        st.markdown(f"### Sequence: **{sequence}** (Length: {len(sequence)} AA)")

        # Validation
        if len(sequence) > 30:
            st.warning("⚠️ For demo purposes, sequence length is limited to 30 amino acids")
            sequence = sequence[:30]

        # Display sequence info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Length", len(sequence))

        with col2:
            from utils import calculate_sequence_mw
            mw = calculate_sequence_mw(sequence)
            st.metric("Molecular Weight", f"{mw:.1f} Da")

        with col3:
            st.metric("Structure Status", "Ready to predict")

        # Predict structure button
        st.markdown("### Structure Prediction")

        col1, col2 = st.columns([1, 3])

        with col1:
            predict_button = st.button("🔮 Predict Structure", type="primary")

        with col2:
            st.info(
                "🌐 This will use ESMFold API to predict the 3D structure. "
                "First prediction takes 2-10 seconds, cached predictions are instant."
            )

        if predict_button:
            with st.spinner("Predicting structure via ESMFold API..."):
                try:
                    builder = StructureBuilder(use_cache=True)
                    pdb_content, meta = builder.build_from_sequence(sequence)

                    if pdb_content:
                        st.success(f"✅ Structure predicted successfully!")

                        # Display metadata
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            source = meta.get('source', 'unknown')
                            st.metric("Source", source.upper())

                        with col2:
                            st.metric("Atoms", meta.get('n_atoms', 0))

                        with col3:
                            st.metric("Residues", meta.get('n_residues', 0))

                        with col4:
                            if 'elapsed_time' in meta:
                                st.metric("Time", f"{meta['elapsed_time']:.1f}s")
                            else:
                                st.metric("Time", "Cached")

                        # Store in session state
                        st.session_state['pdb_content'] = pdb_content
                        st.session_state['pdb_meta'] = meta

                    else:
                        st.error(f"❌ Prediction failed: {meta.get('error', 'Unknown error')}")
                        st.write(meta)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # Display structure if available
        if 'pdb_content' in st.session_state:
            st.markdown("---")
            st.markdown("### 3D Visualization")

            try:
                # Try to use py3Dmol
                import py3Dmol

                pdb_content = st.session_state['pdb_content']

                # Style options
                col1, col2 = st.columns(2)

                with col1:
                    style = st.selectbox(
                        "Style:",
                        ["cartoon", "stick", "sphere", "line"],
                        index=0
                    )

                with col2:
                    color = st.selectbox(
                        "Color Scheme:",
                        ["spectrum", "residue", "chain"],
                        index=0
                    )

                # Create viewer
                from visualizer_3d import Py3DmolViewer

                viewer = Py3DmolViewer(width=800, height=600)
                viewer.load_pdb_string(pdb_content)
                viewer.set_style(style, color)
                viewer.render()

                # Get HTML
                html = viewer.get_html()

                # Display in iframe
                components.html(html, height=650, scrolling=False)

                # Download PDB
                st.download_button(
                    label="📥 Download PDB File",
                    data=pdb_content,
                    file_name=f"{sequence[:10]}_structure.pdb",
                    mime="text/plain"
                )

                # Structure info
                with st.expander("📋 Structure Information"):
                    meta = st.session_state.get('pdb_meta', {})

                    st.write(f"**Sequence:** {sequence}")
                    st.write(f"**Length:** {len(sequence)} AA")
                    st.write(f"**Atoms:** {meta.get('n_atoms', 'N/A')}")
                    st.write(f"**Residues:** {meta.get('n_residues', 'N/A')}")
                    st.write(f"**Chains:** {', '.join(meta.get('chains', ['N/A']))}")

            except ImportError:
                st.warning("⚠️ py3Dmol not installed. Showing PDB file instead.")

                pdb_content = st.session_state['pdb_content']

                # Show PDB content in text area
                st.text_area("PDB Content:", pdb_content, height=400)

                # Download button
                st.download_button(
                    label="📥 Download PDB File",
                    data=pdb_content,
                    file_name=f"{sequence[:10]}_structure.pdb",
                    mime="text/plain"
                )

                st.info(
                    "💡 Install py3Dmol to view interactive 3D structures: `pip install py3Dmol`"
                )

    else:
        st.info("👆 Please generate or enter a sequence to predict its structure")

    # Information section
    with st.expander("ℹ️ About ESMFold"):
        st.markdown("""
        **ESMFold** is a state-of-the-art protein structure prediction model developed by Meta AI.

        **Features:**
        - Fast prediction (2-10 seconds for short sequences)
        - High accuracy comparable to AlphaFold2
        - No multiple sequence alignment required
        - Suitable for sequences up to 400 amino acids

        **Limitations:**
        - Maximum length: 400 residues (30 AA limit in this demo)
        - Requires internet connection for first prediction
        - Predictions are cached locally for 30 days

        **Citation:**
        Lin, Z. et al. (2022). Language models of protein sequences at the scale of evolution enable accurate structure prediction. *bioRxiv*.
        """)


if __name__ == "__main__":
    main()
