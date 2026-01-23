"""
Page 1: Peptide Composition Analysis
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
from data_loader import CompositionLoader
from peptide_analyzer import (
    PeptideCompositionAnalyzer,
    MolecularWeightAnalyzer,
    AminoAcidProfiler
)
from visualizer_2d import CompositionVisualizer, MolecularWeightVisualizer

st.set_page_config(page_title="Peptide Analysis", page_icon="📊", layout="wide")


@st.cache_resource
def load_data():
    loader = CompositionLoader()
    loader.load_data()
    return loader


def main():
    st.title("📊 Peptide Composition Analysis")
    st.markdown("---")

    loader = load_data()
    samples = loader.get_sample_list()

    # Sample selection
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_sample = st.selectbox(
            "Select Sample:",
            samples,
            index=0
        )

    with col2:
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Sample", "Compare Samples"]
        )

    st.markdown("---")

    if analysis_mode == "Single Sample":
        # Single sample analysis
        st.markdown(f"## Analysis: {selected_sample}")

        analyzer = PeptideCompositionAnalyzer(loader)
        mw_analyzer = MolecularWeightAnalyzer(loader)
        profiler = AminoAcidProfiler(loader)

        # Get data
        analysis = analyzer.analyze_sample(selected_sample)
        mw_analysis = mw_analyzer.analyze_distribution(selected_sample)
        essential_aa = profiler.get_essential_aa_profile(selected_sample)
        functional_groups = profiler.get_functional_groups(selected_sample)

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Composition",
            "⚖️ Molecular Weight",
            "🔬 Properties",
            "📋 Statistics"
        ])

        with tab1:
            st.markdown("### Amino Acid Composition")

            # TAA composition bar chart
            taa_comp = analysis.get('taa_composition', {})
            if taa_comp:
                fig = CompositionVisualizer.plot_composition_bar(
                    taa_comp,
                    title=f"TAA Composition - {selected_sample}"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top amino acids
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Top 10 Amino Acids")
                rep_aas = analyzer.get_representative_amino_acids(selected_sample, top_n=10)
                for aa_info in rep_aas['top_amino_acids']:
                    st.write(f"**{aa_info['code']}** ({aa_info['name']}): {aa_info['percentage']:.2f}%")

            with col2:
                st.markdown("#### Essential Amino Acids")
                st.metric("Total Essential AA", f"{essential_aa['total_essential_percentage']:.2f}%")
                st.metric("Essential Ratio", f"{essential_aa['essential_ratio']:.1%}")

                for aa_info in essential_aa['essential_amino_acids'][:5]:
                    st.write(f"{aa_info['code']}: {aa_info['percentage']:.2f}%")

        with tab2:
            st.markdown("### Molecular Weight Distribution")

            mw_dist = mw_analysis.get('distribution', {})
            if mw_dist:
                fig = MolecularWeightVisualizer.plot_mw_distribution(
                    mw_dist,
                    title=f"MW Distribution - {selected_sample}"
                )
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Statistics")
                stats = mw_analysis.get('statistics', {})
                st.metric("Average MW", f"{mw_analysis.get('average_mw', 0):.2f} Da")
                st.metric("Dominant Bin", stats.get('dominant_bin', 'N/A'))

            with col2:
                st.markdown("#### Peptide Length Estimate")
                lengths = mw_analyzer.predict_peptide_lengths(selected_sample)
                for label, info in list(lengths.items())[:3]:
                    st.write(f"**{label}**: {info['min_length']}-{info['max_length']} AA ({info['percentage']:.1f}%)")

        with tab3:
            st.markdown("### Physicochemical Properties")

            # Property radar chart
            fig = CompositionVisualizer.plot_property_radar(
                taa_comp,
                title=f"Properties - {selected_sample}"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Property Ratios")
                props = analysis.get('taa_properties', {})
                for key, value in props.items():
                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}%")

            with col2:
                st.markdown("#### Functional Groups")
                for group_name, total in functional_groups['functional_groups'].items():
                    st.write(f"**{group_name}**: {total:.2f}%")

        with tab4:
            st.markdown("### Statistical Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### TAA Statistics")
                taa_stats = analysis.get('taa_statistics', {})
                st.metric("Number of AAs", taa_stats.get('n_amino_acids', 0))
                st.metric("Max AA", taa_stats.get('max_aa', 'N/A'))
                st.metric("Max Percentage", f"{taa_stats.get('max_percentage', 0):.2f}%")

            with col2:
                st.markdown("#### MW Statistics")
                st.metric("Average MW", f"{mw_analysis.get('average_mw', 0):.2f} Da")
                st.metric("Total Percentage", f"{sum(mw_dist.values()):.2f}%")

            with col3:
                st.markdown("#### Essential AAs")
                st.metric("Total Essential", f"{essential_aa['total_essential_percentage']:.2f}%")
                st.metric("Number of Essential", len(essential_aa['essential_amino_acids']))

    else:
        # Compare samples
        st.markdown("## Compare Multiple Samples")

        selected_samples = st.multiselect(
            "Select samples to compare (max 5):",
            samples,
            default=samples[:3] if len(samples) >= 3 else samples,
            max_selections=5
        )

        if len(selected_samples) < 2:
            st.warning("Please select at least 2 samples to compare")
            return

        analyzer = PeptideCompositionAnalyzer(loader)

        # Compare compositions
        comparison = analyzer.compare_samples(selected_samples)

        st.markdown("### Composition Comparison")

        # Get compositions for visualization
        compositions = {}
        for sample in selected_samples:
            compositions[sample] = loader.get_taa_composition(sample)

        fig = CompositionVisualizer.plot_composition_comparison(
            compositions,
            title="Sample Composition Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Similarity matrix
        st.markdown("### Similarity Matrix")
        st.dataframe(comparison['similarity_matrix'], use_container_width=True)

        # Composition DataFrame
        st.markdown("### Detailed Composition")
        st.dataframe(comparison['taa_composition_df'], use_container_width=True)


if __name__ == "__main__":
    main()
