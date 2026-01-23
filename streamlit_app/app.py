"""
Peptide Structure Analyzer - Streamlit Web UI
Main application entry point
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import streamlit as st
from data_loader import CompositionLoader

# Page configuration
st.set_page_config(
    page_title="Peptide Structure Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    """데이터 로더 초기화 (캐싱)"""
    loader = CompositionLoader()
    loader.load_data()
    return loader


def main():
    """메인 페이지"""

    # Header
    st.markdown('<div class="main-header">🧬 Peptide Structure Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced peptide composition analysis and structure visualization</div>', unsafe_allow_html=True)

    # Load data
    try:
        loader = load_data()
        samples = loader.get_sample_list()

        # Sidebar
        st.sidebar.title("📊 Sample Selection")
        st.sidebar.markdown("---")

        selected_sample = st.sidebar.selectbox(
            "Select a sample:",
            samples,
            index=0
        )

        # Display sample info
        if selected_sample:
            profile = loader.get_complete_profile(selected_sample)

            st.sidebar.markdown("### Sample Information")
            sample_info = profile.get('sample_info', {})

            if 'Sample_name' in sample_info:
                st.sidebar.write(f"**Name:** {sample_info['Sample_name']}")
            if 'raw_material' in sample_info:
                st.sidebar.write(f"**Material:** {sample_info['raw_material']}")
            if 'manufacturer' in sample_info:
                st.sidebar.write(f"**Manufacturer:** {sample_info['manufacturer']}")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📖 Navigation")
        st.sidebar.info("""
        Use the pages in the sidebar to navigate:

        1. **Peptide Analysis** - Composition & properties
        2. **Sequence Prediction** - Generate sequences
        3. **Bioactive Search** - Find bioactive motifs
        4. **2D Structure** - 2D visualizations
        5. **3D Structure** - 3D structure viewer
        """)

        # Main content
        st.markdown("## Welcome! 👋")

        st.markdown("""
        This application provides comprehensive analysis tools for peptide composition data:

        ### Features

        - **📊 Composition Analysis**: Analyze amino acid composition and physicochemical properties
        - **🧬 Sequence Prediction**: Generate probable peptide sequences using Markov chain models
        - **💊 Bioactive Prediction**: Identify bioactive motifs and predict biological activities
        - **🎨 2D Visualization**: Interactive charts and diagrams
        - **🔬 3D Structure**: Predict and visualize 3D structures using ESMFold

        ### Quick Stats
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Samples", len(samples))

        with col2:
            taa_comp = loader.get_taa_composition(selected_sample)
            st.metric("Amino Acids (TAA)", len(taa_comp))

        with col3:
            mw_dist = loader.get_mw_distribution(selected_sample)
            avg_mw = sum(float(v) * i for i, v in enumerate([250, 375, 625, 875, 1500])) / sum(mw_dist.values()) if mw_dist else 0
            st.metric("Avg MW (Da)", f"{avg_mw:.0f}")

        # Sample preview
        st.markdown("### Selected Sample Preview")

        if selected_sample and profile:
            tab1, tab2, tab3 = st.tabs(["TAA Composition", "MW Distribution", "Properties"])

            with tab1:
                taa_comp = profile.get('taa_composition', {})
                if taa_comp:
                    # Top 10 amino acids
                    sorted_taa = sorted(taa_comp.items(), key=lambda x: x[1], reverse=True)[:10]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top 10 Amino Acids**")
                        for aa, pct in sorted_taa:
                            st.write(f"{aa}: {pct:.2f}%")

                    with col2:
                        # Simple bar chart
                        import pandas as pd
                        df = pd.DataFrame(sorted_taa, columns=['AA', 'Percentage'])
                        st.bar_chart(df.set_index('AA'))

            with tab2:
                mw_dist = profile.get('mw_distribution', {})
                if mw_dist:
                    st.markdown("**Molecular Weight Distribution**")

                    # Simplified labels
                    mw_labels = {
                        'mw_pct_250': '<250 Da',
                        'mw_pct_250_500': '250-500 Da',
                        'mw_pct_500_750': '500-750 Da',
                        'mw_pct_750_1000': '750-1000 Da',
                        'mw_pct_1000': '>1000 Da'
                    }

                    for key, value in mw_dist.items():
                        label = mw_labels.get(key, key)
                        st.write(f"{label}: {value:.2f}%")

            with tab3:
                props = profile.get('general_properties', {})
                if props:
                    st.markdown("**General Properties**")
                    for key, value in props.items():
                        st.write(f"{key.replace('_', ' ').title()}: {value:.2f}")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Peptide Structure Analyzer v1.0</p>
            <p>Built with Streamlit | Powered by Claude Sonnet 4.5</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please make sure composition_template.xlsx is in the data/ directory")


if __name__ == "__main__":
    main()
