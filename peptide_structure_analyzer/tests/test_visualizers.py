"""
visualizers 단위 테스트
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pytest
import plotly.graph_objects as go
from visualizer_2d import (
    CompositionVisualizer,
    MolecularWeightVisualizer,
    BioactivityVisualizer,
    PeptideDiagram
)

# Try importing 3D visualizer (optional dependency)
try:
    from visualizer_3d import (
        Py3DmolViewer,
        StructureComparator,
        StylePresets,
        StructureAnnotator
    )
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    print("[WARNING] py3Dmol not available, skipping 3D tests")


@pytest.fixture
def test_composition():
    """테스트용 조성 데이터"""
    return {
        'A': 10.5, 'R': 8.2, 'N': 5.3, 'D': 7.1, 'C': 2.1,
        'E': 12.1, 'G': 6.5, 'H': 3.2, 'I': 4.8, 'L': 8.9
    }


@pytest.fixture
def test_mw_distribution():
    """테스트용 MW 분포"""
    return {
        'mw_pct_250': 17.6,
        'mw_pct_250_500': 18.6,
        'mw_pct_500_750': 16.6,
        'mw_pct_750_1000': 13.2,
        'mw_pct_1000': 34.0
    }


@pytest.fixture
def test_activities():
    """테스트용 생리활성 점수"""
    return {
        'antimicrobial': 0.36,
        'antihypertensive': 0.27,
        'antioxidant': 0.13,
        'opioid': 0.23,
        'immunomodulatory': 0.50,
        'anti-inflammatory': 0.25
    }


@pytest.fixture
def test_sequence():
    """테스트용 서열"""
    return "ARNDCEQGH"


@pytest.fixture
def sample_pdb():
    """테스트용 PDB 내용"""
    return """ATOM      1  N   ALA A   1      11.104  12.766  13.756  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.639  13.954  13.102  1.00  0.00           C
ATOM      3  C   ALA A   1      10.751  15.157  13.417  1.00  0.00           C
ATOM      4  O   ALA A   1      10.068  15.185  14.443  1.00  0.00           O
END
"""


class TestCompositionVisualizer:
    """CompositionVisualizer 테스트"""

    def test_plot_composition_bar(self, test_composition):
        """조성 바 차트 테스트"""
        fig = CompositionVisualizer.plot_composition_bar(test_composition)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "Amino Acid"
        assert fig.layout.yaxis.title.text == "Percentage (%)"

    def test_plot_composition_comparison(self, test_composition):
        """조성 비교 테스트"""
        compositions = {
            'Sample_01': test_composition,
            'Sample_02': {aa: val * 0.8 for aa, val in test_composition.items()}
        }

        fig = CompositionVisualizer.plot_composition_comparison(compositions)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 2개 샘플

    def test_plot_property_radar(self, test_composition):
        """특성 레이더 차트 테스트"""
        fig = CompositionVisualizer.plot_property_radar(test_composition)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'scatterpolar'


class TestMolecularWeightVisualizer:
    """MolecularWeightVisualizer 테스트"""

    def test_plot_mw_distribution(self, test_mw_distribution):
        """MW 분포 차트 테스트"""
        fig = MolecularWeightVisualizer.plot_mw_distribution(test_mw_distribution)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "Molecular Weight Range"

    def test_plot_mw_comparison(self):
        """MW 비교 테스트"""
        import pandas as pd

        # 샘플 데이터
        mw_data = pd.DataFrame({
            'mw_pct_250': [17.6, 18.2, 16.9],
            'mw_pct_250_500': [18.6, 19.1, 17.8],
            'mw_pct_500_750': [16.6, 15.9, 17.2]
        })

        fig = MolecularWeightVisualizer.plot_mw_comparison(mw_data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3개 컬럼


class TestBioactivityVisualizer:
    """BioactivityVisualizer 테스트"""

    def test_plot_activity_scores(self, test_activities):
        """활성 점수 차트 테스트"""
        fig = BioactivityVisualizer.plot_activity_scores(test_activities)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'scatterpolar'

    def test_plot_activity_comparison(self, test_activities):
        """활성 비교 테스트"""
        samples_activities = {
            'Sample_01': test_activities,
            'Sample_02': {act: score * 0.9 for act, score in test_activities.items()}
        }

        fig = BioactivityVisualizer.plot_activity_comparison(samples_activities)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # 2개 샘플


class TestPeptideDiagram:
    """PeptideDiagram 테스트"""

    def test_plot_sequence_diagram(self, test_sequence):
        """서열 다이어그램 테스트"""
        fig = PeptideDiagram.plot_sequence_diagram(test_sequence)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # 선 + 마커

    def test_plot_hydrophobicity_profile(self, test_sequence):
        """소수성 프로파일 테스트"""
        fig = PeptideDiagram.plot_hydrophobicity_profile(test_sequence)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "Position"


@pytest.mark.skipif(not PY3DMOL_AVAILABLE, reason="py3Dmol not installed")
class TestPy3DmolViewer:
    """Py3DmolViewer 테스트"""

    def test_initialization(self):
        """초기화 테스트"""
        viewer = Py3DmolViewer(width=600, height=400)

        assert viewer.width == 600
        assert viewer.height == 400
        assert viewer.viewer is None

    def test_create_viewer(self):
        """뷰어 생성 테스트"""
        viewer = Py3DmolViewer()
        py3d_viewer = viewer.create_viewer()

        assert py3d_viewer is not None
        assert viewer.viewer is not None

    def test_load_pdb_string(self, sample_pdb):
        """PDB 문자열 로딩 테스트"""
        viewer = Py3DmolViewer()
        viewer.load_pdb_string(sample_pdb)

        assert viewer.viewer is not None

    def test_set_style(self, sample_pdb):
        """스타일 설정 테스트"""
        viewer = Py3DmolViewer()
        viewer.load_pdb_string(sample_pdb)
        viewer.set_style('cartoon', 'spectrum')

        # 에러 없이 실행되면 성공
        assert True

    def test_get_html(self, sample_pdb):
        """HTML 생성 테스트"""
        viewer = Py3DmolViewer()
        viewer.load_pdb_string(sample_pdb)
        viewer.set_style('cartoon', 'spectrum')
        viewer.render()

        html = viewer.get_html()

        assert isinstance(html, str)
        assert len(html) > 0
        assert 'viewer' in html.lower() or 'mol' in html.lower()


@pytest.mark.skipif(not PY3DMOL_AVAILABLE, reason="py3Dmol not installed")
class TestStructureComparator:
    """StructureComparator 테스트"""

    def test_create_side_by_side_view(self, sample_pdb):
        """나란히 비교 뷰 테스트"""
        pdb_contents = [
            ('structure1', sample_pdb),
            ('structure2', sample_pdb)
        ]

        viewers = StructureComparator.create_side_by_side_view(pdb_contents)

        assert len(viewers) == 2

    def test_create_overlay_view(self, sample_pdb):
        """겹쳐서 비교 뷰 테스트"""
        pdb_contents = [
            ('structure1', sample_pdb),
            ('structure2', sample_pdb)
        ]

        viewer = StructureComparator.create_overlay_view(pdb_contents)

        assert viewer is not None


@pytest.mark.skipif(not PY3DMOL_AVAILABLE, reason="py3Dmol not installed")
class TestStylePresets:
    """StylePresets 테스트"""

    def test_apply_publication_style(self, sample_pdb):
        """출판용 스타일 테스트"""
        import py3Dmol
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(sample_pdb, 'pdb')

        styled_viewer = StylePresets.apply_publication_style(viewer)

        assert styled_viewer is not None

    def test_apply_presentation_style(self, sample_pdb):
        """프레젠테이션용 스타일 테스트"""
        import py3Dmol
        viewer = py3Dmol.view(width=400, height=400)
        viewer.addModel(sample_pdb, 'pdb')

        styled_viewer = StylePresets.apply_presentation_style(viewer)

        assert styled_viewer is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
