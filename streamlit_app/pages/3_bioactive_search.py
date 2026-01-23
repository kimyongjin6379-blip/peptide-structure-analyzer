"""
Page 3: Bioactive Motif Search (핵심 페이지)
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
import pandas as pd
from data_loader import CompositionLoader
from bioactive_predictor import BioactivePredictor, ActivityScorer
from visualizer_2d import BioactivityVisualizer

st.set_page_config(page_title="Bioactive Search", page_icon="💊", layout="wide")


@st.cache_resource
def load_data():
    loader = CompositionLoader()
    loader.load_data()
    return loader


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
        "🎯 Composition-based Prediction",
        "🔍 Motif Search",
        "📊 Compare Samples"
    ])

    with tab1:
        st.markdown("### Activity Prediction from Composition")

        scorer = ActivityScorer()
        result = scorer.predict_for_sample(selected_sample, loader)

        # Activity scores radar chart
        activity_scores = result.get('activity_scores', {})

        fig = BioactivityVisualizer.plot_activity_scores(
            activity_scores,
            title=f"Bioactivity Profile - {selected_sample}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top 3 activities
        st.markdown("### Top 3 Predicted Activities")

        ranked = sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)

        for i, (activity, score) in enumerate(ranked[:3], 1):
            details = result['activity_details'][activity]

            with st.expander(f"#{i} {activity.title()} (Score: {score:.3f})", expanded=(i==1)):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Score", f"{score:.3f}")
                    st.metric("Threshold", f"{details['threshold']:.2f}")

                    if details['above_threshold']:
                        st.success("✅ Above threshold")
                    else:
                        st.warning("⚠️ Below threshold")

                with col2:
                    st.markdown(f"**Description:** {details['description']}")

                    st.markdown("**Contributing Amino Acids:**")
                    contrib_aas = details['contributing_amino_acids']

                    for aa_info in contrib_aas:
                        st.write(
                            f"- **{aa_info['amino_acid']}**: {aa_info['percentage']:.2f}% "
                            f"(weight: {aa_info['weight']}, contribution: {aa_info['contribution']:.3f})"
                        )

        # All activities table
        st.markdown("### All Activities")

        df = pd.DataFrame([
            {
                'Activity': act.title(),
                'Score': f"{score:.3f}",
                'Threshold': f"{result['activity_details'][act]['threshold']:.2f}",
                'Above Threshold': '✅' if result['activity_details'][act]['above_threshold'] else '❌'
            }
            for act, score in sorted(activity_scores.items(), key=lambda x: x[1], reverse=True)
        ])

        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Motif Search in Generated Sequences")

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

                # Recommendations
                st.markdown("#### Activity-based Recommendations")

                recommendations = comprehensive.get('recommendations', {})

                for activity, rec in recommendations.items():
                    with st.expander(
                        f"{activity.title()} "
                        f"(Composition score: {rec['composition_score']:.3f}, "
                        f"{rec['n_candidates']} candidates)"
                    ):
                        top_seqs = rec.get('top_sequences', [])

                        if top_seqs:
                            st.write("**Top Candidate:**")
                            top = top_seqs[0]
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
                with st.spinner("Analyzing samples..."):
                    predictor = BioactivePredictor(loader)
                    comparison = predictor.compare_samples_bioactivity(selected_samples)

                    # Radar chart comparison
                    samples_activities = comparison.get('sample_scores', {})

                    fig = BioactivityVisualizer.plot_activity_comparison(
                        samples_activities,
                        title="Bioactivity Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Best samples by activity
                    st.markdown("#### Best Samples by Activity")

                    best_samples = comparison.get('best_samples_by_activity', {})

                    df = pd.DataFrame([
                        {
                            'Activity': act.title(),
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
            st.info("Please select at least 2 samples to compare")


if __name__ == "__main__":
    main()
